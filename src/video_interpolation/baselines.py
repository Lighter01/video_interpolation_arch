import gc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from video_interpolation.data.datasets import UniversalTripletDataset
from video_interpolation.image_io import save_triplet_prediction_set, tensor_to_uint8_hwc, uint8_hwc_to_tensor
from video_interpolation.metrics import (
    LPIPSEvaluator,
    aggregate_metric_rows,
    compute_sample_metrics,
    write_metrics_csv,
    write_summary_csv,
)
from video_interpolation.mlflow import MlflowRunConfig, log_baseline_evaluation
from video_interpolation.settings import Settings, load_settings

ProgressCallback = Callable[[str, Mapping[str, object]], None]

BASELINE_NAMES: tuple[str, ...] = ("duplicate_left", "blend", "farneback")


@dataclass(frozen=True)
class BaselineEvaluationConfig:
    manifest_path: Path
    output_dir: Path = Path("outputs/baselines")
    baselines: tuple[str, ...] = BASELINE_NAMES
    limit_samples: int | None = None
    compute_lpips: bool = True
    lpips_device: str = "cpu"
    lpips_net: str = "alex"
    save_predictions: bool = True
    prediction_limit: int = 8
    mlflow: MlflowRunConfig = field(default_factory=MlflowRunConfig)
    config_path: Path | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BaselineEvaluationConfig":
        values = dict(data)
        values["manifest_path"] = Path(values["manifest_path"])
        if "output_dir" in values:
            values["output_dir"] = Path(values["output_dir"])
        values["baselines"] = tuple(values.get("baselines") or BASELINE_NAMES)
        values["mlflow"] = MlflowRunConfig.from_mapping(values.get("mlflow"))
        if values.get("config_path") is not None:
            values["config_path"] = Path(values["config_path"])
        return cls(**values)

    def validate(self) -> None:
        if self.limit_samples is not None and self.limit_samples <= 0:
            raise ValueError("limit_samples must be positive when set")
        if self.prediction_limit < 0:
            raise ValueError("prediction_limit must be non-negative")
        unsupported = [name for name in self.baselines if name not in BASELINE_NAMES]
        if unsupported:
            raise ValueError(f"Unsupported baseline(s): {', '.join(unsupported)}")


@dataclass(frozen=True)
class BaselineEvaluationResult:
    output_dir: Path
    metrics_csv_path: Path
    summary_csv_path: Path
    samples_evaluated: int
    predictions_evaluated: int
    baselines: tuple[str, ...]
    sample_predictions_written: int
    mlflow_run_id: str | None
    summary_rows: Sequence[Mapping[str, str]]


def evaluate_baselines(
    config: BaselineEvaluationConfig,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BaselineEvaluationResult:
    config.validate()
    settings = settings or load_settings()
    manifest_path = _resolve_project_path(settings, config.manifest_path)
    output_dir = _resolve_project_path(settings, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = UniversalTripletDataset(manifest_path=manifest_path, settings=settings)
    sample_count = len(dataset) if config.limit_samples is None else min(len(dataset), config.limit_samples)
    total_predictions = sample_count * len(config.baselines)
    _emit_progress(progress_callback, "start", total=total_predictions)

    lpips_evaluator: LPIPSEvaluator | None = None
    if config.compute_lpips:
        lpips_evaluator = LPIPSEvaluator(device=config.lpips_device, net=config.lpips_net)

    rows: list[dict[str, Any]] = []
    prediction_counts = {name: 0 for name in config.baselines}
    predictions_written = 0
    try:
        for sample_index in range(sample_count):
            sample = dataset[sample_index]
            for baseline_name in config.baselines:
                prediction = predict_middle_frame(baseline_name, sample["left"], sample["right"])
                metrics = compute_sample_metrics(prediction, sample["middle"], lpips_evaluator)
                prediction_path = ""
                if config.save_predictions and prediction_counts[baseline_name] < config.prediction_limit:
                    prediction_path = save_triplet_prediction_set(
                        output_dir,
                        baseline_name,
                        sample["metadata"]["sample_id"],
                        sample["left"],
                        sample["middle"],
                        prediction,
                        sample["right"],
                    )
                    prediction_counts[baseline_name] += 1
                    predictions_written += 1
                rows.append(
                    {
                        "sample_id": sample["metadata"]["sample_id"],
                        "source_group": sample["metadata"]["source_group"],
                        "source_dataset": sample["metadata"]["source_dataset"],
                        "source_video_id": sample["metadata"]["source_video_id"],
                        "prediction_name": baseline_name,
                        "psnr": metrics.psnr,
                        "ssim": metrics.ssim,
                        "lpips": metrics.lpips,
                        "prediction_path": prediction_path,
                    }
                )
                _emit_progress(progress_callback, "prediction_advanced")
    finally:
        if lpips_evaluator is not None:
            lpips_evaluator.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "metrics_summary.csv"
    summary_rows = aggregate_metric_rows(rows)
    write_metrics_csv(metrics_path, rows)
    write_summary_csv(summary_path, summary_rows)

    artifact_paths = [metrics_path, summary_path, manifest_path]
    if config.config_path is not None:
        artifact_paths.append(_resolve_project_path(settings, config.config_path))
    if config.save_predictions:
        artifact_paths.extend(path for path in (output_dir / "sample_predictions").rglob("*.png"))

    mlflow_run_id = log_baseline_evaluation(
        config.mlflow,
        params={
            "mode": "baseline_eval",
            "manifest_path": str(config.manifest_path),
            "baselines": list(config.baselines),
            "limit_samples": config.limit_samples,
            "compute_lpips": config.compute_lpips,
            "lpips_device": config.lpips_device,
            "lpips_net": config.lpips_net,
            "save_predictions": config.save_predictions,
            "prediction_limit": config.prediction_limit,
        },
        summary_rows=summary_rows,
        artifact_paths=artifact_paths,
        settings=settings,
    )

    return BaselineEvaluationResult(
        output_dir=output_dir,
        metrics_csv_path=metrics_path,
        summary_csv_path=summary_path,
        samples_evaluated=sample_count,
        predictions_evaluated=len(rows),
        baselines=config.baselines,
        sample_predictions_written=predictions_written,
        mlflow_run_id=mlflow_run_id,
        summary_rows=summary_rows,
    )


def predict_middle_frame(baseline_name: str, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if baseline_name == "duplicate_left":
        return left.clone()
    if baseline_name == "blend":
        return torch.clamp((left + right) / 2.0, 0.0, 1.0)
    if baseline_name == "farneback":
        return _farneback_prediction(left, right)
    raise ValueError(f"Unsupported baseline: {baseline_name}")


def with_baseline_manifest(
    config: BaselineEvaluationConfig,
    manifest_path: Path | None,
) -> BaselineEvaluationConfig:
    if manifest_path is None:
        return config
    return replace(config, manifest_path=manifest_path)


def with_baseline_output_dir(
    config: BaselineEvaluationConfig,
    output_dir: Path | None,
) -> BaselineEvaluationConfig:
    if output_dir is None:
        return config
    return replace(config, output_dir=output_dir)


def with_baseline_limit(
    config: BaselineEvaluationConfig,
    limit_samples: int | None,
) -> BaselineEvaluationConfig:
    if limit_samples is None:
        return config
    return replace(config, limit_samples=limit_samples)


def with_baseline_names(
    config: BaselineEvaluationConfig,
    baselines: Sequence[str] | None,
) -> BaselineEvaluationConfig:
    if not baselines:
        return config
    return replace(config, baselines=tuple(baselines))


def with_baseline_mlflow_disabled(
    config: BaselineEvaluationConfig,
    disabled: bool,
) -> BaselineEvaluationConfig:
    if not disabled:
        return config
    return replace(config, mlflow=replace(config.mlflow, enabled=False))


def with_baseline_lpips(
    config: BaselineEvaluationConfig,
    compute_lpips: bool | None,
) -> BaselineEvaluationConfig:
    if compute_lpips is None:
        return config
    return replace(config, compute_lpips=compute_lpips)


def with_baseline_save_predictions(
    config: BaselineEvaluationConfig,
    save_predictions: bool | None,
) -> BaselineEvaluationConfig:
    if save_predictions is None:
        return config
    return replace(config, save_predictions=save_predictions)


def with_baseline_config_path(
    config: BaselineEvaluationConfig,
    config_path: Path,
) -> BaselineEvaluationConfig:
    return replace(config, config_path=config_path)


def _farneback_prediction(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_u8 = tensor_to_uint8_hwc(left)
    right_u8 = tensor_to_uint8_hwc(right)
    try:
        gray_left = cv2.cvtColor(left_u8, cv2.COLOR_RGB2GRAY)
        gray_right = cv2.cvtColor(right_u8, cv2.COLOR_RGB2GRAY)
        flow_lr = cv2.calcOpticalFlowFarneback(
            gray_left,
            gray_right,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        flow_rl = cv2.calcOpticalFlowFarneback(
            gray_right,
            gray_left,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        height, width = gray_left.shape
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        left_map_x = (grid_x - 0.5 * flow_lr[..., 0]).astype(np.float32)
        left_map_y = (grid_y - 0.5 * flow_lr[..., 1]).astype(np.float32)
        right_map_x = (grid_x - 0.5 * flow_rl[..., 0]).astype(np.float32)
        right_map_y = (grid_y - 0.5 * flow_rl[..., 1]).astype(np.float32)
        warped_left = cv2.remap(left_u8, left_map_x, left_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        warped_right = cv2.remap(right_u8, right_map_x, right_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        prediction = ((warped_left.astype(np.float32) + warped_right.astype(np.float32)) / 2.0).astype(np.uint8)
        return uint8_hwc_to_tensor(prediction)
    except cv2.error:
        return torch.clamp((left + right) / 2.0, 0.0, 1.0)


def _resolve_project_path(settings: Settings, path: Path) -> Path:
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
