import gc
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from video_interpolation.adapters.ema_vfi import EMAVFIAdapter, EMAVFIAdapterConfig
from video_interpolation.data.datasets import UniversalTripletDataset
from video_interpolation.metrics import compute_psnr, compute_ssim
from video_interpolation.mlflow import MlflowLoggingError, MlflowRunConfig, configure_mlflow
from video_interpolation.settings import Settings, load_settings

ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True)
class EMATrainingConfig:
    train_manifest_path: Path
    val_manifest_path: Path
    output_dir: Path = Path("outputs/training/ema_vfi_small")
    model: EMAVFIAdapterConfig = field(default_factory=EMAVFIAdapterConfig)
    mode: str = "finetune"
    seed: int = 42
    batch_size: int = 1
    num_workers: int = 0
    max_epochs: int = 1
    max_steps: int | None = None
    learning_rate: float = 2e-5
    validate_every_steps: int = 100
    limit_train_samples: int | None = None
    limit_val_samples: int | None = None
    mlflow: MlflowRunConfig = field(default_factory=lambda: MlflowRunConfig(experiment_name="stage1-ema-training"))
    config_path: Path | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EMATrainingConfig":
        values = dict(data)
        values["train_manifest_path"] = Path(values["train_manifest_path"])
        values["val_manifest_path"] = Path(values["val_manifest_path"])
        if "output_dir" in values:
            values["output_dir"] = Path(values["output_dir"])
        values["model"] = EMAVFIAdapterConfig.from_mapping(values.get("model"))
        values["mlflow"] = MlflowRunConfig.from_mapping(values.get("mlflow"))
        if values.get("config_path") is not None:
            values["config_path"] = Path(values["config_path"])
        return cls(**values)

    def validate(self) -> None:
        if self.mode not in ("finetune", "eval_only", "scratch_train"):
            raise ValueError("mode must be one of: finetune, eval_only, scratch_train")
        if self.mode == "scratch_train":
            raise ValueError("scratch_train is represented in config but not implemented for Stage 1 EMA smoke path")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when set")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.validate_every_steps <= 0:
            raise ValueError("validate_every_steps must be positive")
        if self.limit_train_samples is not None and self.limit_train_samples <= 0:
            raise ValueError("limit_train_samples must be positive when set")
        if self.limit_val_samples is not None and self.limit_val_samples <= 0:
            raise ValueError("limit_val_samples must be positive when set")


@dataclass(frozen=True)
class EMATrainingResult:
    output_dir: Path
    best_checkpoint_path: Path | None
    last_checkpoint_path: Path | None
    steps_completed: int
    best_psnr: float | None
    mlflow_run_id: str | None


def run_ema_training(
    config: EMATrainingConfig,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> EMATrainingResult:
    config.validate()
    settings = settings or load_settings()
    output_dir = _resolve_project_path(settings, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = _resolve_project_path(settings, config.train_manifest_path)
    val_manifest = _resolve_project_path(settings, config.val_manifest_path)
    _seed_everything(config.seed)

    train_dataset = _limited_dataset(
        UniversalTripletDataset(manifest_path=train_manifest, settings=settings),
        config.limit_train_samples,
    )
    val_dataset = _limited_dataset(
        UniversalTripletDataset(manifest_path=val_manifest, settings=settings),
        config.limit_val_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.mode == "finetune",
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    adapter = EMAVFIAdapter(config.model, settings=settings)
    mlflow_run_id: str | None = None
    best_psnr: float | None = None
    best_checkpoint: Path | None = None
    last_checkpoint: Path | None = None
    step = 0
    total_steps = _planned_steps(config, len(train_loader))
    _emit_progress(progress_callback, "start", total=total_steps)

    run_ctx = None
    try:
        if config.mlflow.enabled:
            try:
                configure_mlflow(config.mlflow, settings=settings)
                run_ctx = mlflow.start_run(run_name=config.mlflow.run_name)
                run = run_ctx.__enter__()
                mlflow_run_id = run.info.run_id
                if config.mlflow.tags:
                    mlflow.set_tags(dict(config.mlflow.tags))
                mlflow.log_params(
                    {
                        "mode": config.mode,
                        "model_name": config.model.model_name,
                        "checkpoint_path": str(config.model.checkpoint_path),
                        "train_manifest_path": str(config.train_manifest_path),
                        "val_manifest_path": str(config.val_manifest_path),
                        "batch_size": config.batch_size,
                        "max_epochs": config.max_epochs,
                        "max_steps": config.max_steps,
                        "learning_rate": config.learning_rate,
                        "limit_train_samples": config.limit_train_samples,
                        "limit_val_samples": config.limit_val_samples,
                    }
                )
                for artifact in (
                    train_manifest,
                    val_manifest,
                    _resolve_project_path(settings, config.config_path) if config.config_path else None,
                ):
                    if artifact is not None and artifact.exists():
                        mlflow.log_artifact(str(artifact))
            except Exception as exc:  # pragma: no cover - MLflow service dependent.
                raise MlflowLoggingError(
                    "MLflow training logging failed. Ensure the tracking server from infra/mlflow is running "
                    "or rerun the tiny smoke command with --disable-mlflow."
                ) from exc

        adapter.load_checkpoint()
        if config.mode == "eval_only":
            metrics = _evaluate(adapter, val_loader)
            best_psnr = metrics["psnr_mean"]
            _log_metrics(metrics, step=0)
            _emit_progress(progress_callback, "validation_done", step=0, **metrics)
            return EMATrainingResult(output_dir, None, None, 0, best_psnr, mlflow_run_id)

        for _epoch in range(config.max_epochs):
            for batch in train_loader:
                step += 1
                _, loss = adapter.train_step(
                    batch["left"],
                    batch["middle"],
                    batch["right"],
                    learning_rate=config.learning_rate,
                )
                _log_metric("train.loss", loss, step=step)
                _emit_progress(progress_callback, "step_done", step=step, loss=loss)

                should_validate = step % config.validate_every_steps == 0
                reached_final_step = config.max_steps is not None and step >= config.max_steps
                if should_validate or reached_final_step:
                    metrics = _evaluate(adapter, val_loader)
                    _log_metrics(metrics, step=step)
                    _emit_progress(progress_callback, "validation_done", step=step, **metrics)
                    if best_psnr is None or metrics["psnr_mean"] > best_psnr:
                        best_psnr = metrics["psnr_mean"]
                        best_checkpoint = output_dir / "best_checkpoint.pkl"
                        adapter.save_checkpoint(best_checkpoint)
                        _log_artifact(best_checkpoint)

                _emit_progress(progress_callback, "step_advanced")
                if reached_final_step:
                    break
            if config.max_steps is not None and step >= config.max_steps:
                break

        if best_psnr is None:
            metrics = _evaluate(adapter, val_loader)
            best_psnr = metrics["psnr_mean"]
            _log_metrics(metrics, step=step)
            best_checkpoint = output_dir / "best_checkpoint.pkl"
            adapter.save_checkpoint(best_checkpoint)
            _log_artifact(best_checkpoint)

        last_checkpoint = output_dir / "last_checkpoint.pkl"
        adapter.save_checkpoint(last_checkpoint)
        _log_artifact(last_checkpoint)
    finally:
        adapter.close()
        if run_ctx is not None:
            try:
                run_ctx.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover - MLflow service dependent.
                raise MlflowLoggingError("MLflow training run close failed.") from exc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return EMATrainingResult(
        output_dir=output_dir,
        best_checkpoint_path=best_checkpoint,
        last_checkpoint_path=last_checkpoint,
        steps_completed=step,
        best_psnr=best_psnr,
        mlflow_run_id=mlflow_run_id,
    )


def with_training_config_path(config: EMATrainingConfig, config_path: Path) -> EMATrainingConfig:
    return replace(config, config_path=config_path)


def with_training_limit(config: EMATrainingConfig, max_steps: int | None) -> EMATrainingConfig:
    if max_steps is None:
        return config
    return replace(config, max_steps=max_steps)


def with_training_sample_limits(
    config: EMATrainingConfig,
    limit_train_samples: int | None,
    limit_val_samples: int | None,
) -> EMATrainingConfig:
    updates: dict[str, int] = {}
    if limit_train_samples is not None:
        updates["limit_train_samples"] = limit_train_samples
    if limit_val_samples is not None:
        updates["limit_val_samples"] = limit_val_samples
    if not updates:
        return config
    return replace(config, **updates)


def with_training_output_dir(config: EMATrainingConfig, output_dir: Path | None) -> EMATrainingConfig:
    if output_dir is None:
        return config
    return replace(config, output_dir=output_dir)


def with_training_checkpoint(config: EMATrainingConfig, checkpoint_path: Path | None) -> EMATrainingConfig:
    if checkpoint_path is None:
        return config
    return replace(config, model=replace(config.model, checkpoint_path=checkpoint_path))


def with_training_mlflow_disabled(config: EMATrainingConfig, disabled: bool) -> EMATrainingConfig:
    if not disabled:
        return config
    return replace(config, mlflow=replace(config.mlflow, enabled=False))


def _evaluate(adapter: EMAVFIAdapter, loader: DataLoader) -> dict[str, float]:
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    for batch in loader:
        prediction = adapter.eval_step(batch["left"], batch["middle"], batch["right"])
        target = batch["middle"]
        for index in range(prediction.shape[0]):
            psnr_values.append(compute_psnr(prediction[index], target[index]))
            ssim_values.append(compute_ssim(prediction[index], target[index]))
    if not psnr_values:
        return {"psnr_mean": 0.0, "ssim_mean": 0.0}
    return {
        "psnr_mean": float(np.mean(psnr_values)),
        "ssim_mean": float(np.mean(ssim_values)),
    }


def _limited_dataset(dataset: UniversalTripletDataset, limit: int | None) -> UniversalTripletDataset | Subset:
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def _planned_steps(config: EMATrainingConfig, loader_len: int) -> int:
    if config.mode == "eval_only":
        return 1
    epoch_steps = loader_len * config.max_epochs
    if config.max_steps is None:
        return epoch_steps
    return min(epoch_steps, config.max_steps)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log_metric(name: str, value: float, step: int) -> None:
    if mlflow.active_run() is not None:
        mlflow.log_metric(name, float(value), step=step)


def _log_metrics(metrics: Mapping[str, float], step: int) -> None:
    for name, value in metrics.items():
        _log_metric(f"val.{name}", value, step)


def _log_artifact(path: Path) -> None:
    if mlflow.active_run() is not None and path.exists():
        mlflow.log_artifact(str(path))


def _resolve_project_path(settings: Settings, path: Path | None) -> Path:
    if path is None:
        raise ValueError("path must not be None")
    if path.is_absolute():
        return path
    return settings.resolve_path(path)


def _emit_progress(
    callback: ProgressCallback | None,
    event: str,
    **payload: object,
) -> None:
    if callback is not None:
        callback(event, payload)
