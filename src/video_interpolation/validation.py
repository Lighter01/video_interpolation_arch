import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from video_interpolation.adapters.base import ModelAdapter
from video_interpolation.adapters.ema_vfi import EMAVFIAdapter, EMAVFIAdapterConfig
from video_interpolation.contracts import CANDIDATE_VALIDATION_REPORT_FIELDS, validate_required_fields
from video_interpolation.data.datasets import UniversalTripletDataset
from video_interpolation.image_io import save_triplet_prediction_set
from video_interpolation.metrics import (
    LPIPSEvaluator,
    aggregate_metric_rows,
    compute_sample_metrics,
    write_metrics_csv,
    write_summary_csv,
)
from video_interpolation.mlflow import MlflowRunConfig, log_stage1_run
from video_interpolation.settings import Settings, load_settings

ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True)
class ValidationThresholds:
    min_psnr_mean: float | None = None
    min_ssim_mean: float | None = None
    max_lpips_mean: float | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ValidationThresholds":
        return cls(**dict(data or {}))


@dataclass(frozen=True)
class CandidateValidationConfig:
    candidate_id: str
    dataset_version_id: str
    test_manifest_path: Path
    output_dir: Path = Path("outputs/candidate_validation/ema_vfi_small")
    model: EMAVFIAdapterConfig = field(default_factory=EMAVFIAdapterConfig)
    thresholds: ValidationThresholds = field(default_factory=ValidationThresholds)
    limit_samples: int | None = None
    compute_lpips: bool = True
    lpips_device: str = "cpu"
    lpips_net: str = "alex"
    save_predictions: bool = True
    prediction_limit: int = 8
    mlflow: MlflowRunConfig = field(default_factory=lambda: MlflowRunConfig(experiment_name="stage1-ema-validation"))
    config_path: Path | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CandidateValidationConfig":
        values = dict(data)
        values["test_manifest_path"] = Path(values["test_manifest_path"])
        if "output_dir" in values:
            values["output_dir"] = Path(values["output_dir"])
        values["model"] = EMAVFIAdapterConfig.from_mapping(values.get("model"))
        values["thresholds"] = ValidationThresholds.from_mapping(values.get("thresholds"))
        values["mlflow"] = MlflowRunConfig.from_mapping(values.get("mlflow"))
        if values.get("config_path") is not None:
            values["config_path"] = Path(values["config_path"])
        return cls(**values)

    def validate(self) -> None:
        if self.limit_samples is not None and self.limit_samples <= 0:
            raise ValueError("limit_samples must be positive when set")
        if self.prediction_limit < 0:
            raise ValueError("prediction_limit must be non-negative")


@dataclass(frozen=True)
class CandidateValidationResult:
    output_dir: Path
    metrics_csv_path: Path
    summary_csv_path: Path
    report_path: Path
    samples_evaluated: int
    sample_predictions_written: int
    approved: bool
    decision_reasons: Sequence[str]
    mlflow_run_id: str | None


def validate_candidate(
    config: CandidateValidationConfig,
    settings: Settings | None = None,
    adapter: ModelAdapter | None = None,
    progress_callback: ProgressCallback | None = None,
) -> CandidateValidationResult:
    config.validate()
    settings = settings or load_settings()
    manifest_path = _resolve_project_path(settings, config.test_manifest_path)
    output_dir = _resolve_project_path(settings, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = UniversalTripletDataset(manifest_path=manifest_path, settings=settings)
    sample_count = len(dataset) if config.limit_samples is None else min(len(dataset), config.limit_samples)
    _emit_progress(progress_callback, "start", total=sample_count)

    owns_adapter = adapter is None
    model_adapter = adapter or EMAVFIAdapter(config.model, settings=settings)
    lpips_evaluator: LPIPSEvaluator | None = None
    if config.compute_lpips:
        lpips_evaluator = LPIPSEvaluator(device=config.lpips_device, net=config.lpips_net)

    rows: list[dict[str, Any]] = []
    predictions_written = 0
    try:
        if owns_adapter:
            model_adapter.load_checkpoint()
        for sample_index in range(sample_count):
            sample = dataset[sample_index]
            prediction = model_adapter.predict_pair(sample["left"], sample["right"])
            metrics = compute_sample_metrics(prediction, sample["middle"], lpips_evaluator)
            prediction_path = ""
            if config.save_predictions and predictions_written < config.prediction_limit:
                prediction_path = save_triplet_prediction_set(
                    output_dir,
                    config.candidate_id,
                    sample["metadata"]["sample_id"],
                    sample["left"],
                    sample["middle"],
                    prediction,
                    sample["right"],
                )
                predictions_written += 1

            rows.append(
                {
                    "sample_id": sample["metadata"]["sample_id"],
                    "source_group": sample["metadata"]["source_group"],
                    "source_dataset": sample["metadata"]["source_dataset"],
                    "source_video_id": sample["metadata"]["source_video_id"],
                    "prediction_name": config.candidate_id,
                    "psnr": metrics.psnr,
                    "ssim": metrics.ssim,
                    "lpips": metrics.lpips,
                    "prediction_path": prediction_path,
                }
            )
            _emit_progress(progress_callback, "sample_advanced")
    finally:
        if lpips_evaluator is not None:
            lpips_evaluator.close()
        if owns_adapter:
            model_adapter.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "metrics_summary.csv"
    summary_rows = aggregate_metric_rows(rows)
    write_metrics_csv(metrics_path, rows)
    write_summary_csv(summary_path, summary_rows)

    aggregate_metrics = _global_summary_metrics(summary_rows, config.candidate_id)
    approved, reasons = decide_candidate(aggregate_metrics, config.thresholds)
    report = {
        "candidate_id": config.candidate_id,
        "model_name": config.model.model_name,
        "model_adapter": "EMAVFIAdapter",
        "checkpoint": str(config.model.checkpoint_path),
        "dataset_version_id": config.dataset_version_id,
        "test_manifest_path": str(config.test_manifest_path),
        "validation_thresholds": {
            "min_psnr_mean": config.thresholds.min_psnr_mean,
            "min_ssim_mean": config.thresholds.min_ssim_mean,
            "max_lpips_mean": config.thresholds.max_lpips_mean,
        },
        "aggregate_metrics": aggregate_metrics,
        "approval_decision": "approved" if approved else "rejected",
        "decision_reasons": reasons,
        "artifact_paths": {
            "metrics_csv": str(metrics_path),
            "metrics_summary_csv": str(summary_path),
            "sample_predictions": str(output_dir / "sample_predictions"),
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    validate_required_fields(report, CANDIDATE_VALIDATION_REPORT_FIELDS, "candidate validation report")
    report_path = output_dir / "candidate_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    mlflow_artifacts = [metrics_path, summary_path, report_path, manifest_path]
    if config.config_path is not None:
        mlflow_artifacts.append(_resolve_project_path(settings, config.config_path))
    if config.save_predictions and (output_dir / "sample_predictions").exists():
        mlflow_artifacts.append(output_dir / "sample_predictions")

    mlflow_metrics = {
        f"candidate.{config.candidate_id}.approved": 1.0 if approved else 0.0,
    }
    if aggregate_metrics.get("psnr_mean") is not None:
        mlflow_metrics[f"candidate.{config.candidate_id}.psnr_mean"] = float(aggregate_metrics["psnr_mean"])
    if aggregate_metrics.get("ssim_mean") is not None:
        mlflow_metrics[f"candidate.{config.candidate_id}.ssim_mean"] = float(aggregate_metrics["ssim_mean"])
    if aggregate_metrics.get("lpips_mean") is not None:
        mlflow_metrics[f"candidate.{config.candidate_id}.lpips_mean"] = float(aggregate_metrics["lpips_mean"])

    mlflow_run_id = log_stage1_run(
        config.mlflow,
        params={
            "mode": "candidate_validation",
            "candidate_id": config.candidate_id,
            "model_name": config.model.model_name,
            "checkpoint_path": str(config.model.checkpoint_path),
            "dataset_version_id": config.dataset_version_id,
            "test_manifest_path": str(config.test_manifest_path),
            "limit_samples": config.limit_samples,
            "compute_lpips": config.compute_lpips,
        },
        metrics=mlflow_metrics,
        artifact_paths=mlflow_artifacts,
        settings=settings,
    )

    return CandidateValidationResult(
        output_dir=output_dir,
        metrics_csv_path=metrics_path,
        summary_csv_path=summary_path,
        report_path=report_path,
        samples_evaluated=sample_count,
        sample_predictions_written=predictions_written,
        approved=approved,
        decision_reasons=reasons,
        mlflow_run_id=mlflow_run_id,
    )


def decide_candidate(
    aggregate_metrics: Mapping[str, float | None],
    thresholds: ValidationThresholds,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    approved = True
    psnr = aggregate_metrics.get("psnr_mean")
    ssim = aggregate_metrics.get("ssim_mean")
    lpips = aggregate_metrics.get("lpips_mean")

    if thresholds.min_psnr_mean is not None and (psnr is None or psnr < thresholds.min_psnr_mean):
        approved = False
        reasons.append(f"psnr_mean below threshold: {psnr} < {thresholds.min_psnr_mean}")
    if thresholds.min_ssim_mean is not None and (ssim is None or ssim < thresholds.min_ssim_mean):
        approved = False
        reasons.append(f"ssim_mean below threshold: {ssim} < {thresholds.min_ssim_mean}")
    if thresholds.max_lpips_mean is not None and (lpips is None or lpips > thresholds.max_lpips_mean):
        approved = False
        reasons.append(f"lpips_mean above threshold: {lpips} > {thresholds.max_lpips_mean}")
    if approved:
        reasons.append("all configured thresholds passed")
    return approved, reasons


def with_validation_config_path(
    config: CandidateValidationConfig,
    config_path: Path,
) -> CandidateValidationConfig:
    return replace(config, config_path=config_path)


def with_validation_manifest(
    config: CandidateValidationConfig,
    manifest_path: Path | None,
) -> CandidateValidationConfig:
    if manifest_path is None:
        return config
    return replace(config, test_manifest_path=manifest_path)


def with_validation_output_dir(
    config: CandidateValidationConfig,
    output_dir: Path | None,
) -> CandidateValidationConfig:
    if output_dir is None:
        return config
    return replace(config, output_dir=output_dir)


def with_validation_limit(
    config: CandidateValidationConfig,
    limit_samples: int | None,
) -> CandidateValidationConfig:
    if limit_samples is None:
        return config
    return replace(config, limit_samples=limit_samples)


def with_validation_lpips(
    config: CandidateValidationConfig,
    compute_lpips: bool | None,
) -> CandidateValidationConfig:
    if compute_lpips is None:
        return config
    return replace(config, compute_lpips=compute_lpips)


def with_validation_checkpoint(
    config: CandidateValidationConfig,
    checkpoint_path: Path | None,
) -> CandidateValidationConfig:
    if checkpoint_path is None:
        return config
    return replace(config, model=replace(config.model, checkpoint_path=checkpoint_path))


def with_validation_mlflow_disabled(
    config: CandidateValidationConfig,
    disabled: bool,
) -> CandidateValidationConfig:
    if not disabled:
        return config
    return replace(config, mlflow=replace(config.mlflow, enabled=False))


def _global_summary_metrics(
    summary_rows: Sequence[Mapping[str, str]],
    prediction_name: str,
) -> dict[str, float | None]:
    for row in summary_rows:
        if (
            row["prediction_name"] == prediction_name
            and row["source_group"] == "__all__"
            and row["source_dataset"] == "__all__"
        ):
            return {
                "psnr_mean": _float_or_none(row.get("psnr_mean")),
                "ssim_mean": _float_or_none(row.get("ssim_mean")),
                "lpips_mean": _float_or_none(row.get("lpips_mean")),
            }
    return {"psnr_mean": None, "ssim_mean": None, "lpips_mean": None}


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    if value == "inf":
        return float("inf")
    return float(value)


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
