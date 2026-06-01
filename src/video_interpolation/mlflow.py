from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import mlflow

from video_interpolation.settings import Settings, load_settings


class MlflowLoggingError(RuntimeError):
    """Raised when an MLflow-backed workflow cannot log as requested."""


@dataclass(frozen=True)
class MlflowRunConfig:
    enabled: bool = True
    experiment_name: str = "stage1-baselines"
    run_name: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "MlflowRunConfig":
        values = dict(data or {})
        values["tags"] = values.get("tags") or {}
        return cls(**values)


@dataclass(frozen=True)
class MlflowSmokeResult:
    tracking_uri: str
    experiment_name: str
    run_id: str


def configure_mlflow(config: MlflowRunConfig, settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)


def log_baseline_evaluation(
    config: MlflowRunConfig,
    *,
    params: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, str]],
    artifact_paths: Sequence[Path],
    settings: Settings | None = None,
) -> str | None:
    if not config.enabled:
        return None

    try:
        configure_mlflow(config, settings=settings)
        with mlflow.start_run(run_name=config.run_name) as run:
            if config.tags:
                mlflow.set_tags(dict(config.tags))
            mlflow.log_params(_flatten_params(params))
            for metric_name, metric_value in _summary_metrics(summary_rows).items():
                mlflow.log_metric(metric_name, metric_value)
            for artifact_path in artifact_paths:
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))
            return run.info.run_id
    except Exception as exc:
        raise MlflowLoggingError(
            "MLflow logging failed. Ensure the tracking server from infra/mlflow is running "
            "or rerun the tiny smoke command with --disable-mlflow."
        ) from exc


def log_stage1_run(
    config: MlflowRunConfig,
    *,
    params: Mapping[str, Any],
    metrics: Mapping[str, float] | None = None,
    artifact_paths: Sequence[Path] = (),
    settings: Settings | None = None,
) -> str | None:
    if not config.enabled:
        return None

    try:
        configure_mlflow(config, settings=settings)
        with mlflow.start_run(run_name=config.run_name) as run:
            if config.tags:
                mlflow.set_tags(dict(config.tags))
            mlflow.log_params(_flatten_params(params))
            for metric_name, metric_value in (metrics or {}).items():
                mlflow.log_metric(metric_name, float(metric_value))
            for artifact_path in artifact_paths:
                if artifact_path.exists():
                    if artifact_path.is_dir():
                        mlflow.log_artifacts(str(artifact_path))
                    else:
                        mlflow.log_artifact(str(artifact_path))
            return run.info.run_id
    except Exception as exc:
        raise MlflowLoggingError(
            "MLflow logging failed. Ensure the tracking server from infra/mlflow is running "
            "or rerun the tiny smoke command with MLflow disabled when the workflow supports it."
        ) from exc


def log_benchmark_run(
    config: MlflowRunConfig,
    *,
    params: Mapping[str, Any],
    metrics: Mapping[str, float] | None = None,
    artifact_paths: Sequence[Path] = (),
    settings: Settings | None = None,
) -> str | None:
    """Log a benchmark run through the same bounded MLflow behavior as Stage 1 workflows."""
    return log_stage1_run(
        config,
        params=params,
        metrics=metrics,
        artifact_paths=artifact_paths,
        settings=settings,
    )


def run_mlflow_smoke(
    config: MlflowRunConfig,
    artifact_path: Path,
    settings: Settings | None = None,
) -> MlflowSmokeResult:
    if not config.enabled:
        raise ValueError("MLflow smoke logging requires enabled=True")
    settings = settings or load_settings()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("stage1 mlflow smoke\n", encoding="utf-8")

    try:
        configure_mlflow(config, settings=settings)
        with mlflow.start_run(run_name=config.run_name or "stage1-mlflow-smoke") as run:
            mlflow.log_param("mode", "mlflow_smoke")
            mlflow.log_metric("smoke_value", 1.0)
            mlflow.log_artifact(str(artifact_path))
            run_id = run.info.run_id
    except Exception as exc:
        raise MlflowLoggingError(
            "MLflow smoke logging failed. Start infra/mlflow/docker-compose.yml and check "
            "MLFLOW_TRACKING_URI, then retry."
        ) from exc

    return MlflowSmokeResult(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=config.experiment_name,
        run_id=run_id,
    )


def _flatten_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_params(value, full_key))
        elif isinstance(value, (list, tuple, set)):
            flattened[full_key] = ",".join(str(item) for item in value)
        else:
            flattened[full_key] = "" if value is None else str(value)
    return flattened


def _summary_metrics(summary_rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in summary_rows:
        prediction_name = _sanitize_metric_part(row["prediction_name"])
        source_group = _sanitize_metric_part(row["source_group"])
        source_dataset = _sanitize_metric_part(row["source_dataset"])
        scope = source_group if source_dataset == "__all__" else f"{source_group}.{source_dataset}"
        for metric_key in ("psnr_mean", "ssim_mean", "lpips_mean"):
            value = row.get(metric_key)
            if value in (None, "", "inf"):
                continue
            metrics[f"baseline.{prediction_name}.{scope}.{metric_key}"] = float(value)
    return metrics


def _sanitize_metric_part(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")
