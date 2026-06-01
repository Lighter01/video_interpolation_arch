from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any
import contextlib
import csv
import importlib
import json
import os
import sys

from omegaconf import OmegaConf
import torch

from video_interpolation.adapters.base import AdapterEnvironmentReport, ModelAdapter, ModelAdapterError
from video_interpolation.adapters.ema_vfi import EMAVFIAdapter, EMAVFIAdapterConfig
from video_interpolation.adapters.rife import PracticalRIFEAdapter, PracticalRIFEAdapterConfig
from video_interpolation.batch_inference import BatchInferenceVideo, discover_inference_videos
from video_interpolation.inference import (
    AdapterFactory,
    VideoInferenceConfig,
    VideoInferenceExecutionMode,
    VideoInferenceResult,
    run_video_inference,
)
from video_interpolation.inference_runtime import (
    EMAVFIOnnxRuntime,
    EMAVFIOnnxRuntimeConfig,
    FramePairRequest,
    FramePairResult,
    InferenceMode,
    ModelBatchRequest,
    ModelBatchResult,
    PracticalRIFEOnnxRuntime,
    PracticalRIFEOnnxRuntimeConfig,
    RuntimeBackendKind,
    resolve_interpolation_timesteps,
)
from video_interpolation.inference_runtime.backends import resolve_onnx_providers
from video_interpolation.inference_runtime.onnx_validation import resolve_preferred_onnx_artifact_path
from video_interpolation.inference_runtime.rife import validate_rife_scale
from video_interpolation.mlflow import MlflowRunConfig, log_benchmark_run
from video_interpolation.settings import Settings, load_settings


class BenchmarkExecutionMode(StrEnum):
    SEQUENTIAL = "sequential"
    BATCHED = "batched"


SUPPORTED_BENCHMARK_MODELS = ("ema_vfi_small", "practical_rife_v4_26")
DEFAULT_VIDEO_BENCHMARK_INPUT = Path("raw_data/tmp_test/DORA_cut.mp4")
DEFAULT_VIDEO_BENCHMARK_OUTPUT_DIR = Path("outputs/benchmarks/video")

VideoBenchmarkAdapterFactory = Callable[["VideoBenchmarkConfig", Settings], AdapterFactory]


@dataclass(frozen=True)
class VideoBenchmarkConfig:
    model_name: str
    backend: RuntimeBackendKind | str = RuntimeBackendKind.TORCH
    execution_mode: BenchmarkExecutionMode | str = BenchmarkExecutionMode.BATCHED
    input_path: Path | None = DEFAULT_VIDEO_BENCHMARK_INPUT
    input_dir: Path | None = None
    limit_videos: int | None = None
    limit_pairs: int | None = None
    interpolation_mode: InferenceMode | str = InferenceMode.FIXED_2X
    interpolation_factor: int = 2
    device: str = "cpu"
    providers: Sequence[str] = ("CPUExecutionProvider",)
    onnx_path: Path | None = None
    inference_batch_size: int | None = None
    rife_scale: float = 1.0
    repeat_runs: int = 1
    codec: str = "libx264"
    container: str | None = None
    pix_fmt: str = "yuv420p"
    frame_format: str = "rgb24"
    encoder_options: Mapping[str, Any] = field(default_factory=dict)
    output_dir: Path = DEFAULT_VIDEO_BENCHMARK_OUTPUT_DIR
    log_output_videos: bool = False
    mlflow: MlflowRunConfig = field(
        default_factory=lambda: MlflowRunConfig(
            experiment_name="stage2-5-inference-benchmarks",
            run_name=None,
            tags={"stage": "stage2.5", "workflow": "runtime_video_benchmark"},
        )
    )

    def __post_init__(self) -> None:
        if self.model_name not in SUPPORTED_BENCHMARK_MODELS:
            allowed = ", ".join(SUPPORTED_BENCHMARK_MODELS)
            raise ValueError(f"Unsupported benchmark model_name={self.model_name!r}. Allowed: {allowed}.")
        backend = RuntimeBackendKind(str(self.backend))
        execution_mode = BenchmarkExecutionMode(str(self.execution_mode))
        interpolation_mode = InferenceMode(str(self.interpolation_mode))
        resolve_interpolation_timesteps(interpolation_mode, self.interpolation_factor)
        if self.input_dir is None and self.input_path is None:
            raise ValueError("Either input_path or input_dir must be set.")
        if self.input_dir is not None and self.input_path is not None:
            raise ValueError("input_path and input_dir are mutually exclusive.")
        if self.input_dir is None and self.limit_videos is not None:
            raise ValueError("limit_videos is only valid with input_dir.")
        if self.limit_videos is not None and self.limit_videos <= 0:
            raise ValueError("limit_videos must be positive when set.")
        if self.limit_pairs is not None and self.limit_pairs <= 0:
            raise ValueError("limit_pairs must be positive when set.")
        if self.inference_batch_size is not None and self.inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive when set.")
        if self.repeat_runs <= 0:
            raise ValueError("repeat_runs must be positive.")
        if self.model_name.startswith("practical_rife"):
            validate_rife_scale(self.rife_scale)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "execution_mode", execution_mode)
        object.__setattr__(self, "interpolation_mode", interpolation_mode)
        object.__setattr__(self, "input_path", Path(self.input_path) if self.input_path is not None else None)
        object.__setattr__(self, "input_dir", Path(self.input_dir) if self.input_dir is not None else None)
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.onnx_path is not None:
            object.__setattr__(self, "onnx_path", Path(self.onnx_path))
        providers = (self.providers,) if isinstance(self.providers, str) else tuple(self.providers)
        object.__setattr__(self, "providers", tuple(str(provider) for provider in providers))
        object.__setattr__(self, "encoder_options", dict(self.encoder_options))

    @property
    def timesteps(self) -> tuple[float, ...]:
        return resolve_interpolation_timesteps(self.interpolation_mode, self.interpolation_factor)

    @property
    def flattened_rows_per_pair(self) -> int:
        return len(self.timesteps)


@dataclass(frozen=True)
class VideoBenchmarkRecord:
    model_name: str
    backend: str
    execution_mode: str
    interpolation_mode: str
    interpolation_factor: int
    inference_batch_size: int | None
    repeat_index: int
    input_video: Path
    relative_input_video: Path
    output_video: Path | None
    status: str
    error: str | None
    provider: str | None
    artifact_path: str | None
    rife_scale: float | None
    source_frames: int
    pairs_processed: int
    generated_frames: int
    frames_written: int
    input_fps: float | None
    output_fps: float | None
    decode_sec: float
    preprocessing_sec: float
    model_inference_sec: float
    postprocessing_sec: float
    encode_sec: float
    audio_remux_sec: float
    total_sec: float
    model_pairs_per_sec: float
    total_pairs_per_sec: float
    generated_frames_per_sec: float
    total_generated_frames_per_sec: float
    batch_chunks_processed: int
    model_batch_requests: int
    runtime_backend: str | None
    peak_vram_mb: float | None
    runtime_options: Mapping[str, object] = field(default_factory=dict)

    def to_row(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "execution_mode": self.execution_mode,
            "interpolation_mode": self.interpolation_mode,
            "interpolation_factor": self.interpolation_factor,
            "inference_batch_size": self.inference_batch_size,
            "repeat_index": self.repeat_index,
            "input_video": str(self.input_video),
            "relative_input_video": str(self.relative_input_video),
            "output_video": str(self.output_video) if self.output_video is not None else "",
            "status": self.status,
            "error": self.error or "",
            "provider": self.provider or "",
            "artifact_path": self.artifact_path or "",
            "rife_scale": _format_optional_float(self.rife_scale),
            "source_frames": self.source_frames,
            "pairs_processed": self.pairs_processed,
            "generated_frames": self.generated_frames,
            "frames_written": self.frames_written,
            "input_fps": _format_optional_float(self.input_fps),
            "output_fps": _format_optional_float(self.output_fps),
            "decode_sec": _format_float(self.decode_sec),
            "preprocessing_sec": _format_float(self.preprocessing_sec),
            "model_inference_sec": _format_float(self.model_inference_sec),
            "postprocessing_sec": _format_float(self.postprocessing_sec),
            "encode_sec": _format_float(self.encode_sec),
            "audio_remux_sec": _format_float(self.audio_remux_sec),
            "total_sec": _format_float(self.total_sec),
            "model_pairs_per_sec": _format_float(self.model_pairs_per_sec),
            "total_pairs_per_sec": _format_float(self.total_pairs_per_sec),
            "generated_frames_per_sec": _format_float(self.generated_frames_per_sec),
            "total_generated_frames_per_sec": _format_float(self.total_generated_frames_per_sec),
            "batch_chunks_processed": self.batch_chunks_processed,
            "model_batch_requests": self.model_batch_requests,
            "runtime_backend": self.runtime_backend or "",
            "peak_vram_mb": _format_optional_float(self.peak_vram_mb),
            "runtime_options": json.dumps(_json_safe(dict(self.runtime_options)), sort_keys=True),
        }


@dataclass(frozen=True)
class VideoBenchmarkArtifacts:
    report_path: Path
    csv_path: Path
    output_video_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class VideoBenchmarkResult:
    config: VideoBenchmarkConfig
    records: tuple[VideoBenchmarkRecord, ...]
    artifacts: VideoBenchmarkArtifacts
    mlflow_run_id: str | None

    @property
    def success(self) -> bool:
        return all(record.status == "ok" for record in self.records)


def run_video_benchmark(
    config: VideoBenchmarkConfig,
    *,
    settings: Settings | None = None,
    adapter_factory: VideoBenchmarkAdapterFactory | None = None,
) -> VideoBenchmarkResult:
    settings = settings or load_settings()
    output_dir = _resolve_path(settings, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = discover_benchmark_videos(config, settings=settings)
    if not videos:
        raise ValueError("No benchmark input videos were found.")

    artifact_path = _artifact_path_from_config(config, settings)
    records: list[VideoBenchmarkRecord] = []
    output_video_paths: list[Path] = []
    profile = benchmark_profile_slug(config)
    benchmark_adapter_factory = adapter_factory or _build_video_adapter_factory
    model_config = _model_config_for_benchmark(config, settings)
    resolved_adapter_factory = benchmark_adapter_factory(config, settings)

    for repeat_index in range(config.repeat_runs):
        for video in videos:
            output_path = video_benchmark_output_path(
                output_dir=output_dir,
                profile=profile,
                relative_input_path=video.relative_path,
                repeat_index=repeat_index,
                interpolation_factor=config.interpolation_factor,
            )
            run_config = _video_inference_config(
                config,
                input_path=video.input_path,
                output_path=output_path,
                model_config=model_config,
            )
            _reset_peak_vram(config)
            try:
                result = run_video_inference(
                    run_config,
                    adapter_factory=resolved_adapter_factory,
                    settings=settings,
                    mlflow_mode="benchmark_video_inference",
                )
                peak_vram_mb = _peak_vram_mb(config)
                record = _record_from_result(
                    config,
                    result,
                    video=video,
                    repeat_index=repeat_index,
                    artifact_path=artifact_path,
                    peak_vram_mb=peak_vram_mb,
                )
                output_video_paths.append(result.output_path)
            except Exception as exc:
                record = _failed_record(
                    config,
                    video=video,
                    output_path=output_path,
                    repeat_index=repeat_index,
                    artifact_path=artifact_path,
                    error=exc,
                )
            records.append(record)

    artifacts = write_video_benchmark_report(
        output_dir,
        config=config,
        records=tuple(records),
        artifact_path=artifact_path,
        output_video_paths=tuple(output_video_paths),
    )
    mlflow_artifacts = _benchmark_artifact_paths(artifacts, log_output_videos=config.log_output_videos)
    mlflow_run_id = log_benchmark_run(
        config.mlflow,
        params=benchmark_params(config, artifact_path=artifact_path),
        metrics=benchmark_metrics(tuple(records)),
        artifact_paths=mlflow_artifacts,
        settings=settings,
    )
    return VideoBenchmarkResult(
        config=config,
        records=tuple(records),
        artifacts=artifacts,
        mlflow_run_id=mlflow_run_id,
    )


def discover_benchmark_videos(
    config: VideoBenchmarkConfig,
    *,
    settings: Settings | None = None,
) -> tuple[BatchInferenceVideo, ...]:
    settings = settings or load_settings()
    if config.input_dir is not None:
        input_dir = _resolve_path(settings, config.input_dir)
        return tuple(discover_inference_videos(input_dir, limit_videos=config.limit_videos))

    if config.input_path is None:
        raise ValueError("input_path must be set when input_dir is omitted.")
    input_path = _resolve_path(settings, config.input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    return (BatchInferenceVideo(input_path=input_path, relative_path=Path(input_path.name)),)


def video_benchmark_output_path(
    *,
    output_dir: Path,
    profile: str,
    relative_input_path: Path,
    repeat_index: int,
    interpolation_factor: int,
) -> Path:
    suffix = relative_input_path.suffix or ".mp4"
    return (
        output_dir
        / "videos"
        / profile
        / relative_input_path.parent
        / f"{relative_input_path.stem}_repeat{repeat_index:02d}_{interpolation_factor}x{suffix}"
    )


def write_video_benchmark_report(
    output_dir: Path,
    *,
    config: VideoBenchmarkConfig,
    records: Sequence[VideoBenchmarkRecord],
    artifact_path: Path | None = None,
    output_video_paths: Sequence[Path] = (),
) -> VideoBenchmarkArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "benchmark_report.json"
    csv_path = output_dir / "benchmark_metrics.csv"
    report = {
        "config": _config_payload(config, artifact_path=artifact_path),
        "records": [_record_payload(record) for record in records],
        "summary": benchmark_metrics(records),
        "output_video_paths": [str(path) for path in output_video_paths],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = list(records[0].to_row()) if records else list(_empty_record_row())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
    return VideoBenchmarkArtifacts(
        report_path=report_path,
        csv_path=csv_path,
        output_video_paths=tuple(output_video_paths),
    )


def benchmark_metrics(records: Sequence[VideoBenchmarkRecord]) -> dict[str, float]:
    successful = [record for record in records if record.status == "ok"]
    total_pairs = float(sum(record.pairs_processed for record in successful))
    total_generated = float(sum(record.generated_frames for record in successful))
    total_model_sec = sum(record.model_inference_sec for record in successful)
    total_sec = sum(record.total_sec for record in successful)
    metrics = {
        "benchmark.records": float(len(records)),
        "benchmark.successful_records": float(len(successful)),
        "benchmark.failed_records": float(len(records) - len(successful)),
        "benchmark.pairs_processed": total_pairs,
        "benchmark.generated_frames": total_generated,
        "benchmark.decode_sec": sum(record.decode_sec for record in successful),
        "benchmark.preprocessing_sec": sum(record.preprocessing_sec for record in successful),
        "benchmark.model_inference_sec": total_model_sec,
        "benchmark.postprocessing_sec": sum(record.postprocessing_sec for record in successful),
        "benchmark.encode_sec": sum(record.encode_sec for record in successful),
        "benchmark.audio_remux_sec": sum(record.audio_remux_sec for record in successful),
        "benchmark.total_sec": total_sec,
        "benchmark.model_pairs_per_sec": _safe_rate(total_pairs, total_model_sec),
        "benchmark.total_pairs_per_sec": _safe_rate(total_pairs, total_sec),
        "benchmark.model_generated_frames_per_sec": _safe_rate(total_generated, total_model_sec),
        "benchmark.total_generated_frames_per_sec": _safe_rate(total_generated, total_sec),
        "benchmark.batch_chunks_processed": float(sum(record.batch_chunks_processed for record in successful)),
        "benchmark.model_batch_requests": float(sum(record.model_batch_requests for record in successful)),
    }
    peak_values = [record.peak_vram_mb for record in successful if record.peak_vram_mb is not None]
    if peak_values:
        metrics["benchmark.peak_vram_mb"] = max(peak_values)
    return metrics


def benchmark_params(
    config: VideoBenchmarkConfig,
    *,
    artifact_path: Path | None = None,
) -> dict[str, object]:
    return {
        "model_name": config.model_name,
        "backend": config.backend.value,
        "execution_mode": config.execution_mode.value,
        "input_path": str(config.input_path) if config.input_path is not None else None,
        "input_dir": str(config.input_dir) if config.input_dir is not None else None,
        "limit_videos": config.limit_videos,
        "limit_pairs": config.limit_pairs,
        "interpolation_mode": config.interpolation_mode.value,
        "interpolation_factor": config.interpolation_factor,
        "device": config.device,
        "providers": tuple(config.providers),
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "inference_batch_size": config.inference_batch_size,
        "rife_scale": config.rife_scale,
        "repeat_runs": config.repeat_runs,
        "codec": config.codec,
        "container": config.container,
        "pix_fmt": config.pix_fmt,
        "frame_format": config.frame_format,
        "log_output_videos": config.log_output_videos,
    }


class _OnnxBenchmarkAdapter(ModelAdapter):
    def __init__(self, runtime: Any, *, model_name: str) -> None:
        self.runtime = runtime
        self.model_name = model_name

    def validate_environment(self) -> AdapterEnvironmentReport:
        return AdapterEnvironmentReport(status="ok")

    def build_model(self) -> None:
        self.runtime.load()

    def load_checkpoint(self, checkpoint_path: Path | None = None) -> None:
        del checkpoint_path
        self.runtime.load()

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        raise ModelAdapterError(f"{self.__class__.__name__} does not support checkpoint saving.")

    def train(self) -> None:
        raise ModelAdapterError(f"{self.__class__.__name__} is inference-only.")

    def eval(self) -> None:
        return None

    def predict_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.predict_frame_pair(
            FramePairRequest(
                left=left,
                right=right,
                mode=InferenceMode.FIXED_2X,
                interpolation_factor=2,
                backend_kind=RuntimeBackendKind.ONNX,
            )
        ).middle_frame

    def predict_frame_pair(self, request: FramePairRequest) -> FramePairResult:
        return self.runtime.predict(_onnx_frame_pair_request(request))

    def predict_frame_pairs_batch(self, request: ModelBatchRequest) -> ModelBatchResult:
        return self.runtime.predict_batch(_onnx_model_batch_request(request))

    def close(self) -> None:
        self.runtime.close()


def benchmark_profile_slug(config: VideoBenchmarkConfig) -> str:
    provider = "torch" if config.backend is RuntimeBackendKind.TORCH else "_".join(_provider_slug(p) for p in config.providers)
    batch = "auto" if config.inference_batch_size is None else str(config.inference_batch_size)
    return (
        f"{config.model_name}_{config.backend.value}_{provider}_"
        f"{config.execution_mode.value}_{config.interpolation_mode.value}_"
        f"{config.interpolation_factor}x_b{batch}"
    )


def _build_video_adapter_factory(config: VideoBenchmarkConfig, settings: Settings) -> AdapterFactory:
    if config.backend is RuntimeBackendKind.TORCH:

        def torch_factory(model_config: Any, run_settings: Settings) -> ModelAdapter:
            if config.model_name == "ema_vfi_small":
                return EMAVFIAdapter(model_config, settings=run_settings)
            return PracticalRIFEAdapter(model_config, settings=run_settings)

        return torch_factory

    artifact_path = _artifact_path_from_config(config, settings)
    if artifact_path is None:
        raise ValueError("ONNX benchmark requires an artifact path.")

    def onnx_factory(model_config: Any, run_settings: Settings) -> ModelAdapter:
        del model_config, run_settings
        if config.model_name == "ema_vfi_small":
            runtime = EMAVFIOnnxRuntime(
                _load_ema_input_padder_cls(settings),
                EMAVFIOnnxRuntimeConfig(
                    artifact_path=artifact_path,
                    providers=tuple(config.providers),
                    inference_batch_size=config.inference_batch_size,
                ),
            )
            return _OnnxBenchmarkAdapter(runtime, model_name=config.model_name)
        runtime = PracticalRIFEOnnxRuntime(
            PracticalRIFEOnnxRuntimeConfig(
                artifact_path=artifact_path,
                providers=tuple(config.providers),
                scale=config.rife_scale,
                inference_batch_size=config.inference_batch_size,
            )
        )
        return _OnnxBenchmarkAdapter(runtime, model_name=config.model_name)

    return onnx_factory


def _model_config_for_benchmark(config: VideoBenchmarkConfig, settings: Settings) -> Any:
    model_config = _load_model_config(config.model_name, settings)
    if config.model_name == "ema_vfi_small":
        return replace(
            EMAVFIAdapterConfig.from_mapping(model_config),
            device=config.device,
            inference_batch_size=config.inference_batch_size,
        )
    return replace(
        PracticalRIFEAdapterConfig.from_mapping(model_config),
        device=config.device,
        scale=config.rife_scale,
        inference_batch_size=config.inference_batch_size,
    )


def _video_inference_config(
    config: VideoBenchmarkConfig,
    *,
    input_path: Path,
    output_path: Path,
    model_config: Any,
) -> VideoInferenceConfig:
    runtime_options: dict[str, object] = {}
    if config.model_name.startswith("practical_rife"):
        runtime_options["scale"] = config.rife_scale
    return VideoInferenceConfig(
        input_path=input_path,
        output_path=output_path,
        model=model_config,
        limit_pairs=config.limit_pairs,
        interpolation_mode=config.interpolation_mode,
        interpolation_factor=config.interpolation_factor,
        execution_mode=VideoInferenceExecutionMode(config.execution_mode.value),
        inference_batch_size=config.inference_batch_size,
        runtime_options=runtime_options,
        output_fps_multiplier=float(config.interpolation_factor),
        codec=config.codec,
        container=config.container,
        pix_fmt=config.pix_fmt,
        frame_format=config.frame_format,
        encoder_options=config.encoder_options,
        mlflow=replace(config.mlflow, enabled=False),
    )


def _record_from_result(
    config: VideoBenchmarkConfig,
    result: VideoInferenceResult,
    *,
    video: BatchInferenceVideo,
    repeat_index: int,
    artifact_path: Path | None,
    peak_vram_mb: float | None,
) -> VideoBenchmarkRecord:
    source_frames = result.pairs_processed + 1 if result.pairs_processed > 0 else 0
    generated_frames = max(result.frames_written - source_frames, 0)
    return VideoBenchmarkRecord(
        model_name=config.model_name,
        backend=config.backend.value,
        execution_mode=result.execution_mode,
        interpolation_mode=result.interpolation_mode,
        interpolation_factor=result.interpolation_factor,
        inference_batch_size=result.inference_batch_size,
        repeat_index=repeat_index,
        input_video=result.input_path,
        relative_input_video=video.relative_path,
        output_video=result.output_path,
        status="ok",
        error=None,
        provider=_record_provider(config, result.runtime_options),
        artifact_path=str(artifact_path) if artifact_path is not None else None,
        rife_scale=config.rife_scale if config.model_name.startswith("practical_rife") else None,
        source_frames=source_frames,
        pairs_processed=result.pairs_processed,
        generated_frames=generated_frames,
        frames_written=result.frames_written,
        input_fps=result.input_fps,
        output_fps=result.output_fps,
        decode_sec=result.timing.decode_sec,
        preprocessing_sec=result.timing.preprocessing_sec,
        model_inference_sec=result.model_inference_elapsed_sec,
        postprocessing_sec=result.timing.postprocessing_sec,
        encode_sec=result.timing.encode_sec,
        audio_remux_sec=result.timing.audio_remux_sec,
        total_sec=result.total_elapsed_sec,
        model_pairs_per_sec=result.model_pairs_per_sec,
        total_pairs_per_sec=result.total_pairs_per_sec,
        generated_frames_per_sec=_safe_rate(generated_frames, result.model_inference_elapsed_sec),
        total_generated_frames_per_sec=_safe_rate(generated_frames, result.total_elapsed_sec),
        batch_chunks_processed=result.batch_chunks_processed,
        model_batch_requests=result.model_batch_requests,
        runtime_backend=result.runtime_backend,
        peak_vram_mb=peak_vram_mb,
        runtime_options=dict(result.runtime_options),
    )


def _failed_record(
    config: VideoBenchmarkConfig,
    *,
    video: BatchInferenceVideo,
    output_path: Path,
    repeat_index: int,
    artifact_path: Path | None,
    error: Exception,
) -> VideoBenchmarkRecord:
    return VideoBenchmarkRecord(
        model_name=config.model_name,
        backend=config.backend.value,
        execution_mode=config.execution_mode.value,
        interpolation_mode=config.interpolation_mode.value,
        interpolation_factor=config.interpolation_factor,
        inference_batch_size=config.inference_batch_size,
        repeat_index=repeat_index,
        input_video=video.input_path,
        relative_input_video=video.relative_path,
        output_video=output_path,
        status="failed",
        error=str(error),
        provider=_record_provider(config, {}),
        artifact_path=str(artifact_path) if artifact_path is not None else None,
        rife_scale=config.rife_scale if config.model_name.startswith("practical_rife") else None,
        source_frames=0,
        pairs_processed=0,
        generated_frames=0,
        frames_written=0,
        input_fps=None,
        output_fps=None,
        decode_sec=0.0,
        preprocessing_sec=0.0,
        model_inference_sec=0.0,
        postprocessing_sec=0.0,
        encode_sec=0.0,
        audio_remux_sec=0.0,
        total_sec=0.0,
        model_pairs_per_sec=0.0,
        total_pairs_per_sec=0.0,
        generated_frames_per_sec=0.0,
        total_generated_frames_per_sec=0.0,
        batch_chunks_processed=0,
        model_batch_requests=0,
        runtime_backend=None,
        peak_vram_mb=None,
        runtime_options={},
    )


def _load_model_config(model_name: str, settings: Settings) -> Mapping[str, object]:
    path = _resolve_path(settings, _default_model_config_path(model_name))
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, Mapping):
        raise ValueError(f"Model config must be a mapping: {path}")
    return data


def _default_model_config_path(model_name: str) -> Path:
    if model_name == "ema_vfi_small":
        return Path("configs/models/ema_vfi_small.yaml")
    if model_name == "practical_rife_v4_26":
        return Path("configs/models/practical_rife_v4_26.yaml")
    raise ValueError(f"Unsupported benchmark model: {model_name}")


def _load_ema_input_padder_cls(settings: Settings) -> Any:
    repo_path = settings.model_repo_path("EMA-VFI")
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    tracked_modules = {
        name: sys.modules.get(name)
        for name in ("config", "Trainer", "model", "benchmark")
        if name in sys.modules
    }
    try:
        os.chdir(repo_path)
        sys.path.insert(0, str(repo_path))
        return importlib.import_module("benchmark.utils.padder").InputPadder
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        with contextlib.suppress(Exception):
            for name in ("config", "Trainer", "model", "benchmark"):
                if name in tracked_modules:
                    sys.modules[name] = tracked_modules[name]
                else:
                    sys.modules.pop(name, None)


def _artifact_path_from_config(config: VideoBenchmarkConfig, settings: Settings) -> Path | None:
    if config.backend is not RuntimeBackendKind.ONNX:
        return None
    if config.onnx_path is not None:
        return _resolve_path(settings, config.onnx_path)
    return resolve_preferred_onnx_artifact_path(config.model_name)


def _onnx_frame_pair_request(request: FramePairRequest) -> FramePairRequest:
    return FramePairRequest(
        left=request.left,
        right=request.right,
        mode=request.mode,
        interpolation_factor=request.interpolation_factor,
        timesteps=request.timesteps,
        backend_kind=RuntimeBackendKind.ONNX,
        device=request.device,
        backend_options=dict(request.backend_options),
    )


def _onnx_model_batch_request(request: ModelBatchRequest) -> ModelBatchRequest:
    return ModelBatchRequest(
        left=request.left,
        right=request.right,
        mode=request.mode,
        interpolation_factor=request.interpolation_factor,
        timesteps=request.timesteps,
        backend_kind=RuntimeBackendKind.ONNX,
        device=request.device,
        backend_options=dict(request.backend_options),
        metadata=dict(request.metadata),
    )


def _benchmark_artifact_paths(
    artifacts: VideoBenchmarkArtifacts,
    *,
    log_output_videos: bool,
) -> tuple[Path, ...]:
    paths = [artifacts.report_path, artifacts.csv_path]
    if log_output_videos:
        paths.extend(artifacts.output_video_paths)
    return tuple(paths)


def _config_payload(config: VideoBenchmarkConfig, *, artifact_path: Path | None = None) -> dict[str, object]:
    return {
        **benchmark_params(config, artifact_path=artifact_path),
        "output_dir": str(config.output_dir),
        "mlflow": {
            "enabled": config.mlflow.enabled,
            "experiment_name": config.mlflow.experiment_name,
            "run_name": config.mlflow.run_name,
            "tags": dict(config.mlflow.tags),
        },
    }


def _record_payload(record: VideoBenchmarkRecord) -> dict[str, object]:
    return {
        **record.to_row(),
        "runtime_options": _json_safe(dict(record.runtime_options)),
    }


def _empty_record_row() -> dict[str, object]:
    return VideoBenchmarkRecord(
        model_name="",
        backend="",
        execution_mode="",
        interpolation_mode="",
        interpolation_factor=0,
        inference_batch_size=None,
        repeat_index=0,
        input_video=Path(),
        relative_input_video=Path(),
        output_video=None,
        status="",
        error=None,
        provider=None,
        artifact_path=None,
        rife_scale=None,
        source_frames=0,
        pairs_processed=0,
        generated_frames=0,
        frames_written=0,
        input_fps=None,
        output_fps=None,
        decode_sec=0.0,
        preprocessing_sec=0.0,
        model_inference_sec=0.0,
        postprocessing_sec=0.0,
        encode_sec=0.0,
        audio_remux_sec=0.0,
        total_sec=0.0,
        model_pairs_per_sec=0.0,
        total_pairs_per_sec=0.0,
        generated_frames_per_sec=0.0,
        total_generated_frames_per_sec=0.0,
        batch_chunks_processed=0,
        model_batch_requests=0,
        runtime_backend=None,
        peak_vram_mb=None,
    ).to_row()


def _record_provider(config: VideoBenchmarkConfig, runtime_options: Mapping[str, object]) -> str | None:
    del runtime_options
    if config.backend is not RuntimeBackendKind.ONNX:
        return None
    with contextlib.suppress(Exception):
        return ",".join(resolve_onnx_providers(config.providers))
    return ",".join(config.providers)


def _resolve_path(settings: Settings, path: Path) -> Path:
    if path.is_absolute():
        return path
    return settings.resolve_path(path)


def _reset_peak_vram(config: VideoBenchmarkConfig) -> None:
    if config.backend is RuntimeBackendKind.TORCH and str(config.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(config.device))


def _peak_vram_mb(config: VideoBenchmarkConfig) -> float | None:
    if config.backend is RuntimeBackendKind.TORCH and str(config.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(config.device))
        return float(torch.cuda.max_memory_allocated(torch.device(config.device)) / (1024 * 1024))
    return None


def _provider_slug(provider: str) -> str:
    return provider.lower().replace("executionprovider", "").replace("_", "-")


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return value


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _format_float(value: float) -> str:
    return f"{value:.8f}"


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else _format_float(value)
