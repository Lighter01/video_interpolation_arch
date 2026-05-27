from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video_interpolation.adapters.base import ModelAdapter
from video_interpolation.adapters.ema_vfi import EMAVFIAdapter, EMAVFIAdapterConfig
from video_interpolation.image_io import tensor_to_uint8_hwc, uint8_hwc_to_tensor
from video_interpolation.mlflow import MlflowRunConfig, log_stage1_run
from video_interpolation.settings import Settings, load_settings

ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True)
class VideoInferenceConfig:
    input_path: Path
    output_path: Path
    model: EMAVFIAdapterConfig = field(default_factory=EMAVFIAdapterConfig)
    limit_pairs: int | None = None
    output_fps_multiplier: float = 2.0
    codec: str = "mp4v"
    mlflow: MlflowRunConfig = field(default_factory=lambda: MlflowRunConfig(experiment_name="stage1-ema-inference"))
    config_path: Path | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "VideoInferenceConfig":
        values = dict(data)
        values["input_path"] = Path(values["input_path"])
        values["output_path"] = Path(values["output_path"])
        values["model"] = EMAVFIAdapterConfig.from_mapping(values.get("model"))
        values["mlflow"] = MlflowRunConfig.from_mapping(values.get("mlflow"))
        if values.get("config_path") is not None:
            values["config_path"] = Path(values["config_path"])
        return cls(**values)

    def validate(self) -> None:
        if self.limit_pairs is not None and self.limit_pairs <= 0:
            raise ValueError("limit_pairs must be positive when set")
        if self.output_fps_multiplier <= 0:
            raise ValueError("output_fps_multiplier must be positive")


@dataclass(frozen=True)
class VideoInferenceResult:
    input_path: Path
    output_path: Path
    input_fps: float
    output_fps: float
    pairs_processed: int
    frames_written: int
    mlflow_run_id: str | None


def run_ema_video_inference(
    config: VideoInferenceConfig,
    settings: Settings | None = None,
    adapter: ModelAdapter | None = None,
    progress_callback: ProgressCallback | None = None,
) -> VideoInferenceResult:
    config.validate()
    settings = settings or load_settings()
    input_path = _resolve_project_path(settings, config.input_path)
    output_path = _resolve_project_path(settings, config.output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_pairs = max(frame_count - 1, 0) if frame_count else None
    if config.limit_pairs is not None and total_pairs is not None:
        total_pairs = min(total_pairs, config.limit_pairs)
    output_fps = fps * config.output_fps_multiplier

    owns_adapter = adapter is None
    model_adapter = adapter or EMAVFIAdapter(config.model, settings=settings)
    writer: cv2.VideoWriter | None = None
    pairs_processed = 0
    frames_written = 0
    _emit_progress(progress_callback, "start", total=total_pairs or 0)

    try:
        if owns_adapter:
            model_adapter.load_checkpoint()
        ok, previous_bgr = capture.read()
        if not ok:
            raise ValueError(f"Input video contains no readable frames: {input_path}")
        height, width = previous_bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*config.codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Could not open output video for writing: {output_path}")
        writer.write(previous_bgr)
        frames_written += 1

        while config.limit_pairs is None or pairs_processed < config.limit_pairs:
            ok, current_bgr = capture.read()
            if not ok:
                break
            left = _bgr_to_tensor(previous_bgr)
            right = _bgr_to_tensor(current_bgr)
            prediction = model_adapter.predict_pair(left, right)
            writer.write(_tensor_to_bgr(prediction))
            writer.write(current_bgr)
            frames_written += 2
            pairs_processed += 1
            previous_bgr = current_bgr
            _emit_progress(progress_callback, "pair_advanced")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if owns_adapter:
            model_adapter.close()

    artifact_paths = [output_path]
    if config.config_path is not None:
        artifact_paths.append(_resolve_project_path(settings, config.config_path))

    mlflow_run_id = log_stage1_run(
        config.mlflow,
        params={
            "mode": "ema_video_inference",
            "model_name": config.model.model_name,
            "checkpoint_path": str(config.model.checkpoint_path),
            "input_path": str(config.input_path),
            "output_path": str(config.output_path),
            "limit_pairs": config.limit_pairs,
            "output_fps_multiplier": config.output_fps_multiplier,
            "codec": config.codec,
        },
        metrics={
            "inference.pairs_processed": float(pairs_processed),
            "inference.frames_written": float(frames_written),
            "inference.input_fps": fps,
            "inference.output_fps": output_fps,
        },
        artifact_paths=artifact_paths,
        settings=settings,
    )

    return VideoInferenceResult(
        input_path=input_path,
        output_path=output_path,
        input_fps=fps,
        output_fps=output_fps,
        pairs_processed=pairs_processed,
        frames_written=frames_written,
        mlflow_run_id=mlflow_run_id,
    )


def with_inference_input(
    config: VideoInferenceConfig,
    input_path: Path | None,
) -> VideoInferenceConfig:
    if input_path is None:
        return config
    return replace(config, input_path=input_path)


def with_inference_output(
    config: VideoInferenceConfig,
    output_path: Path | None,
) -> VideoInferenceConfig:
    if output_path is None:
        return config
    return replace(config, output_path=output_path)


def with_inference_limit(
    config: VideoInferenceConfig,
    limit_pairs: int | None,
) -> VideoInferenceConfig:
    if limit_pairs is None:
        return config
    return replace(config, limit_pairs=limit_pairs)


def with_inference_checkpoint(
    config: VideoInferenceConfig,
    checkpoint_path: Path | None,
) -> VideoInferenceConfig:
    if checkpoint_path is None:
        return config
    return replace(config, model=replace(config.model, checkpoint_path=checkpoint_path))


def with_inference_config_path(
    config: VideoInferenceConfig,
    config_path: Path,
) -> VideoInferenceConfig:
    return replace(config, config_path=config_path)


def with_inference_mlflow_disabled(
    config: VideoInferenceConfig,
    disabled: bool,
) -> VideoInferenceConfig:
    if not disabled:
        return config
    return replace(config, mlflow=replace(config.mlflow, enabled=False))


def _bgr_to_tensor(frame: np.ndarray):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return uint8_hwc_to_tensor(rgb)


def _tensor_to_bgr(tensor) -> np.ndarray:
    rgb = tensor_to_uint8_hwc(tensor)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


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
