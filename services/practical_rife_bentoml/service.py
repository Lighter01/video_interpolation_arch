from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import bentoml
from bentoml.exceptions import BadInput, InternalServerError
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from video_interpolation.inference import VideoOutputPlaybackMode
from video_interpolation.inference_runtime import InferenceRequestValidationError
from video_interpolation.serving import (
    PracticalRIFEServingConfig,
    PracticalRIFEVideoInferenceRunner,
)

MVP_DEFAULT_DEVICE = "cuda"
MVP_DEFAULT_CODEC = "h264_nvenc"


def _read_non_empty_env(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return value


def build_service_config() -> PracticalRIFEServingConfig:
    return PracticalRIFEServingConfig(
        backend="torch",
        device=_read_non_empty_env("RIFE_DEVICE", MVP_DEFAULT_DEVICE),
        codec=_read_non_empty_env("RIFE_CODEC", MVP_DEFAULT_CODEC),
    )


SERVICE_CONFIG = build_service_config()


class InterpolateVideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_path: str
    output_path: str
    interpolation_factor: int
    scale: float = 1.0
    output_playback_mode: str = VideoOutputPlaybackMode.REAL_TIME.value
    quality_evaluation_enabled: bool = True
    quality_sample_count: int = 16

    @field_validator("input_path", "output_path", "output_playback_mode", mode="before")
    @classmethod
    def _validate_string(cls, value: object) -> object:
        if not isinstance(value, str) or not value:
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("interpolation_factor", "quality_sample_count", mode="before")
    @classmethod
    def _validate_integer(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("must be an integer")
        return value

    @field_validator("scale", mode="before")
    @classmethod
    def _validate_scale_type(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("must be a number")
        return float(value)

    @field_validator("quality_evaluation_enabled", mode="before")
    @classmethod
    def _validate_boolean(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("must be a boolean")
        return value


class ValidatedInterpolateVideoRequest:
    __slots__ = (
        "input_path",
        "output_path",
        "interpolation_factor",
        "scale",
        "output_playback_mode",
        "quality_evaluation_enabled",
        "quality_sample_count",
    )

    def __init__(
        self,
        *,
        input_path: Path,
        output_path: Path,
        interpolation_factor: int,
        scale: float,
        output_playback_mode: VideoOutputPlaybackMode,
        quality_evaluation_enabled: bool,
        quality_sample_count: int,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.interpolation_factor = interpolation_factor
        self.scale = scale
        self.output_playback_mode = output_playback_mode
        self.quality_evaluation_enabled = quality_evaluation_enabled
        self.quality_sample_count = quality_sample_count


class VideoInferenceRunner(Protocol):
    def load(self) -> None: ...

    def run(
        self,
        *,
        input_path: Path,
        output_path: Path,
        interpolation_factor: int,
        scale: float,
        output_playback_mode: str,
        enable_quality_evaluation: bool,
        quality_sample_count: int,
    ) -> Any: ...


def create_runner() -> PracticalRIFEVideoInferenceRunner:
    return PracticalRIFEVideoInferenceRunner(SERVICE_CONFIG)


def validate_interpolate_video_request(
    payload: InterpolateVideoRequest | Mapping[str, object],
    *,
    config: PracticalRIFEServingConfig = SERVICE_CONFIG,
) -> ValidatedInterpolateVideoRequest:
    request = _coerce_request(payload)
    input_path = Path(request.input_path)
    output_path = Path(request.output_path)

    if not input_path.is_absolute():
        raise BadInput("input_path must be an absolute path.")
    if not input_path.is_file():
        raise BadInput(f"input_path must point to an existing video file: {input_path}")
    if not output_path.is_absolute():
        raise BadInput("output_path must be an absolute path.")
    if output_path.exists() and output_path.is_dir():
        raise BadInput(f"output_path points to a directory, expected a video file path: {output_path}")

    try:
        interpolation_factor, scale = config.validate_request(
            interpolation_factor=request.interpolation_factor,
            scale=request.scale,
        )
    except (InferenceRequestValidationError, ValueError) as exc:
        raise BadInput(str(exc)) from exc

    try:
        output_playback_mode = VideoOutputPlaybackMode(request.output_playback_mode)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in VideoOutputPlaybackMode)
        raise BadInput(
            f"Unsupported output_playback_mode: {request.output_playback_mode!r}. Allowed: {allowed}."
        ) from exc

    if request.quality_sample_count < 0:
        raise BadInput("quality_sample_count must be a non-negative integer.")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BadInput(f"Could not create output directory {output_path.parent}: {exc}") from exc

    return ValidatedInterpolateVideoRequest(
        input_path=input_path,
        output_path=output_path,
        interpolation_factor=interpolation_factor,
        scale=scale,
        output_playback_mode=output_playback_mode,
        quality_evaluation_enabled=request.quality_evaluation_enabled,
        quality_sample_count=request.quality_sample_count,
    )


def run_interpolate_video_request(
    runner: VideoInferenceRunner,
    payload: InterpolateVideoRequest | Mapping[str, object],
) -> dict[str, object]:
    request = validate_interpolate_video_request(payload)
    start = perf_counter()
    try:
        result = runner.run(
            input_path=request.input_path,
            output_path=request.output_path,
            interpolation_factor=request.interpolation_factor,
            scale=request.scale,
            output_playback_mode=request.output_playback_mode.value,
            enable_quality_evaluation=request.quality_evaluation_enabled,
            quality_sample_count=request.quality_sample_count,
        )
    except BadInput:
        raise
    except Exception as exc:
        raise InternalServerError(f"Practical-RIFE inference failed: {exc}") from exc

    verify_output_file(request.output_path)
    return build_success_response(
        result,
        output_path=request.output_path,
        interpolation_factor=request.interpolation_factor,
        duration_seconds=perf_counter() - start,
    )


def verify_output_file(output_path: Path) -> None:
    if not output_path.is_file():
        raise InternalServerError(f"Practical-RIFE inference did not create output file: {output_path}")
    if output_path.stat().st_size <= 0:
        raise InternalServerError(f"Practical-RIFE inference created an empty output file: {output_path}")


def build_success_response(
    result: object,
    *,
    output_path: Path,
    interpolation_factor: int,
    duration_seconds: float | None,
) -> dict[str, object]:
    response: dict[str, object] = {
        "status": "completed",
        "output_path": str(output_path),
        "interpolation_factor": interpolation_factor,
    }
    if duration_seconds is None:
        duration_seconds = _optional_float_attr(result, "total_elapsed_sec")
    if duration_seconds is not None:
        response["duration_seconds"] = float(duration_seconds)

    psnr_mean = _optional_float_attr(result, "quality_psnr_mean")
    ssim_mean = _optional_float_attr(result, "quality_ssim_mean")
    if psnr_mean is not None:
        response["psnr_mean"] = psnr_mean
    if ssim_mean is not None:
        response["ssim_mean"] = ssim_mean
    return response


@bentoml.service(name="practical_rife_v4_26_mvp")
class PracticalRIFEInterpolationService:
    def __init__(self) -> None:
        self.runner = create_runner()
        self.runner.load()

    @bentoml.api(route="/interpolate_video")
    def interpolate_video(self, request: InterpolateVideoRequest) -> dict[str, object]:
        return run_interpolate_video_request(self.runner, request)


def _coerce_request(payload: InterpolateVideoRequest | Mapping[str, object]) -> InterpolateVideoRequest:
    if isinstance(payload, InterpolateVideoRequest):
        return payload
    try:
        return InterpolateVideoRequest.model_validate(payload)
    except ValidationError as exc:
        raise BadInput(f"Invalid interpolate_video request: {exc}") from exc


def _optional_float_attr(result: object, name: str) -> float | None:
    value = getattr(result, name, None)
    if value is None:
        return None
    return float(value)
