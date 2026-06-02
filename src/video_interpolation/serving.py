from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import torch

from video_interpolation.adapters.base import AdapterEnvironmentReport, ModelAdapter, ModelAdapterError
from video_interpolation.adapters.rife import PracticalRIFEAdapter, PracticalRIFEAdapterConfig
from video_interpolation.inference import (
    VideoInferenceConfig,
    VideoInferenceExecutionMode,
    VideoInferenceResult,
    run_video_inference,
)
from video_interpolation.inference_runtime import (
    FramePairRequest,
    FramePairResult,
    InferenceMode,
    InferenceRequestValidationError,
    ModelBatchRequest,
    ModelBatchResult,
    PracticalRIFEOnnxRuntime,
    PracticalRIFEOnnxRuntimeConfig,
    RuntimeBackendKind,
    resolve_preferred_onnx_artifact_path,
)
from video_interpolation.inference_runtime.rife import validate_rife_scale
from video_interpolation.mlflow import MlflowRunConfig
from video_interpolation.settings import Settings, load_settings
from video_interpolation.video_quality import VideoQualityEvaluationConfig

PRACTICAL_RIFE_SERVING_MODEL = "practical_rife_v4_26"
DEFAULT_SERVING_DEVICE = "cuda"
DEFAULT_SERVING_ONNX_PROVIDER = "CUDAExecutionProvider"
DEFAULT_SERVING_CODEC = "libx264"
MIN_SERVING_INTERPOLATION_FACTOR = 2
MAX_SERVING_INTERPOLATION_FACTOR = 4


@dataclass(frozen=True)
class PracticalRIFEServingConfig:
    """Serving-facing defaults for Practical-RIFE v4.26 video inference examples."""

    model_name: str = PRACTICAL_RIFE_SERVING_MODEL
    backend: RuntimeBackendKind | str = RuntimeBackendKind.TORCH
    device: str = DEFAULT_SERVING_DEVICE
    provider: str = DEFAULT_SERVING_ONNX_PROVIDER
    onnx_path: Path | None = None
    onnx_artifact_scale: float = 1.0
    default_scale: float = 1.0
    interpolation_mode: InferenceMode | str = InferenceMode.ARBITRARY_NX
    execution_mode: VideoInferenceExecutionMode | str = VideoInferenceExecutionMode.SEQUENTIAL
    min_interpolation_factor: int = MIN_SERVING_INTERPOLATION_FACTOR
    max_interpolation_factor: int = MAX_SERVING_INTERPOLATION_FACTOR
    codec: str = DEFAULT_SERVING_CODEC
    pix_fmt: str = "yuv420p"
    frame_format: str = "rgb24"
    mlflow_enabled: bool = False
    quality_evaluation_enabled: bool = True
    quality_sample_count: int = 16
    quality_max_image_side: int | None = 360
    quality_write_triplets: bool = False

    def __post_init__(self) -> None:
        backend = _coerce_backend(self.backend)
        mode = _coerce_mode(self.interpolation_mode)
        execution_mode = _coerce_execution_mode(self.execution_mode)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "interpolation_mode", mode)
        object.__setattr__(self, "execution_mode", execution_mode)
        if self.model_name != PRACTICAL_RIFE_SERVING_MODEL:
            raise ValueError(f"Practical-RIFE serving supports only {PRACTICAL_RIFE_SERVING_MODEL}.")
        if mode is not InferenceMode.ARBITRARY_NX:
            raise ValueError("Practical-RIFE serving examples support only interpolation_mode='arbitrary_nx'.")
        if execution_mode is not VideoInferenceExecutionMode.SEQUENTIAL:
            raise ValueError("Practical-RIFE serving examples support only execution_mode='sequential'.")
        if self.min_interpolation_factor != MIN_SERVING_INTERPOLATION_FACTOR:
            raise ValueError(f"min_interpolation_factor must be {MIN_SERVING_INTERPOLATION_FACTOR}.")
        if self.max_interpolation_factor != MAX_SERVING_INTERPOLATION_FACTOR:
            raise ValueError(f"max_interpolation_factor must be {MAX_SERVING_INTERPOLATION_FACTOR}.")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty torch device string.")
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("provider must be a non-empty ONNX Runtime provider string.")
        if not isinstance(self.quality_evaluation_enabled, bool):
            raise ValueError("quality_evaluation_enabled must be a boolean.")
        if (
            isinstance(self.quality_sample_count, bool)
            or not isinstance(self.quality_sample_count, int)
            or self.quality_sample_count < 0
        ):
            raise ValueError("quality_sample_count must be a non-negative integer.")
        if self.quality_max_image_side is not None:
            if (
                isinstance(self.quality_max_image_side, bool)
                or not isinstance(self.quality_max_image_side, int)
                or self.quality_max_image_side <= 0
            ):
                raise ValueError("quality_max_image_side must be a positive integer or None.")
        if not isinstance(self.quality_write_triplets, bool):
            raise ValueError("quality_write_triplets must be a boolean.")
        if self.onnx_path is not None:
            object.__setattr__(self, "onnx_path", Path(self.onnx_path))
        validate_rife_scale(self.default_scale)
        validate_rife_scale(self.onnx_artifact_scale)

    def validate_request(self, *, interpolation_factor: int, scale: float | None = None) -> tuple[int, float]:
        factor = validate_serving_interpolation_factor(
            interpolation_factor,
            minimum=self.min_interpolation_factor,
            maximum=self.max_interpolation_factor,
        )
        resolved_scale = validate_rife_scale(self.default_scale if scale is None else scale)
        if self.backend is RuntimeBackendKind.ONNX and resolved_scale != self.onnx_artifact_scale:
            raise InferenceRequestValidationError(
                "Practical-RIFE ONNX serving requires request scale to match the loaded artifact scale. "
                f"Request scale={resolved_scale:g}, artifact scale={self.onnx_artifact_scale:g}."
            )
        return factor, resolved_scale


class PracticalRIFEVideoInferenceRunner:
    """Persistent Practical-RIFE video inference runner for BentoML-style service processes."""

    def __init__(
        self,
        config: PracticalRIFEServingConfig | None = None,
        *,
        settings: Settings | None = None,
        adapter: ModelAdapter | None = None,
    ) -> None:
        self.config = config or PracticalRIFEServingConfig()
        self.settings = settings or load_settings()
        self._adapter = adapter
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def adapter(self) -> ModelAdapter | None:
        return self._adapter

    def load(self) -> None:
        if self._loaded:
            return
        if self._adapter is None:
            self._adapter = self._create_adapter()
        self._adapter.load_checkpoint()
        self._loaded = True

    def close(self) -> None:
        if self._adapter is not None:
            self._adapter.close()
        self._loaded = False

    def run(
        self,
        *,
        input_path: str | PathLike[str],
        output_path: str | PathLike[str],
        interpolation_factor: int = MIN_SERVING_INTERPOLATION_FACTOR,
        scale: float | None = None,
        limit_pairs: int | None = None,
        codec: str | None = None,
        encoder_options: Mapping[str, Any] | None = None,
        enable_quality_evaluation: bool | None = None,
        quality_triplet_output_dir: str | PathLike[str] | None = None,
        write_quality_triplets: bool | None = None,
    ) -> VideoInferenceResult:
        factor, resolved_scale = self.config.validate_request(
            interpolation_factor=interpolation_factor,
            scale=scale,
        )
        resolved_input = _resolve_serving_path(self.settings, Path(input_path))
        resolved_output = _resolve_serving_path(self.settings, Path(output_path))
        if not resolved_input.is_file():
            raise FileNotFoundError(f"Input video does not exist: {resolved_input}")
        if resolved_output.exists() and resolved_output.is_dir():
            raise ValueError(f"Output path points to a directory, expected a video file path: {resolved_output}")
        resolved_quality_dir = (
            None
            if quality_triplet_output_dir is None
            else _resolve_serving_path(self.settings, Path(quality_triplet_output_dir))
        )

        self.load()
        model_config = self._video_model_config(resolved_scale)
        quality_enabled = self.config.quality_evaluation_enabled if enable_quality_evaluation is None else enable_quality_evaluation
        quality_write_triplets = self.config.quality_write_triplets if write_quality_triplets is None else write_quality_triplets
        inference_config = VideoInferenceConfig(
            input_path=resolved_input,
            output_path=resolved_output,
            model=model_config,
            limit_pairs=limit_pairs,
            interpolation_mode=self.config.interpolation_mode,
            interpolation_factor=factor,
            execution_mode=self.config.execution_mode,
            inference_batch_size=1,
            runtime_options={"scale": resolved_scale},
            output_fps_multiplier=float(factor),
            codec=codec or self.config.codec,
            pix_fmt=self.config.pix_fmt,
            frame_format=self.config.frame_format,
            encoder_options=dict(encoder_options or {}),
            quality_evaluation=VideoQualityEvaluationConfig(
                enabled=quality_enabled,
                triplet_output_dir=resolved_quality_dir,
                sample_count=self.config.quality_sample_count,
                max_image_side=self.config.quality_max_image_side,
                write_triplets=quality_write_triplets,
                fail_policy="warn",
            ),
            mlflow=MlflowRunConfig(enabled=self.config.mlflow_enabled),
        )
        return run_video_inference(
            inference_config,
            adapter_factory=lambda _model_config, _settings: self._require_adapter(),
            settings=self.settings,
            adapter=self._require_adapter(),
            mlflow_mode="practical_rife_serving",
        )

    def _create_adapter(self) -> ModelAdapter:
        if self.config.backend is RuntimeBackendKind.TORCH:
            return PracticalRIFEAdapter(self._video_model_config(self.config.default_scale), settings=self.settings)

        artifact_path = self._resolve_onnx_path()
        runtime = PracticalRIFEOnnxRuntime(
            PracticalRIFEOnnxRuntimeConfig(
                artifact_path=artifact_path,
                providers=(self.config.provider,),
                scale=self.config.onnx_artifact_scale,
                inference_batch_size=1,
            )
        )
        return PracticalRIFEOnnxVideoAdapter(runtime, model_name=self.config.model_name)

    def _video_model_config(self, scale: float) -> PracticalRIFEAdapterConfig:
        return PracticalRIFEAdapterConfig(
            model_name=self.config.model_name,
            device=self.config.device,
            scale=scale,
            default_interpolation_factor=MIN_SERVING_INTERPOLATION_FACTOR,
            min_interpolation_factor=MIN_SERVING_INTERPOLATION_FACTOR,
            max_interpolation_factor=MAX_SERVING_INTERPOLATION_FACTOR,
            inference_batch_size=1,
        )

    def _resolve_onnx_path(self) -> Path:
        if self.config.onnx_path is not None:
            path = _resolve_serving_path(self.settings, self.config.onnx_path)
            if not path.is_file():
                raise FileNotFoundError(f"ONNX artifact does not exist: {path}")
            return path
        return resolve_preferred_onnx_artifact_path(self.config.model_name)

    def _require_adapter(self) -> ModelAdapter:
        if self._adapter is None:
            raise RuntimeError("Practical-RIFE serving adapter has not been created.")
        return self._adapter

    def __enter__(self) -> PracticalRIFEVideoInferenceRunner:
        self.load()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


class PracticalRIFEOnnxVideoAdapter(ModelAdapter):
    """ModelAdapter-compatible wrapper around the Practical-RIFE ONNX runtime."""

    def __init__(self, runtime: PracticalRIFEOnnxRuntime, *, model_name: str = PRACTICAL_RIFE_SERVING_MODEL) -> None:
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
        del checkpoint_path
        raise ModelAdapterError("PracticalRIFEOnnxVideoAdapter does not support checkpoint saving.")

    def train(self) -> None:
        raise ModelAdapterError("PracticalRIFEOnnxVideoAdapter is inference-only.")

    def eval(self) -> None:
        return None

    def predict_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.predict_frame_pair(
            FramePairRequest(
                left=left,
                right=right,
                mode=InferenceMode.ARBITRARY_NX,
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


def run_practical_rife_video_inference(
    *,
    input_path: str | PathLike[str],
    output_path: str | PathLike[str],
    backend: RuntimeBackendKind | str = RuntimeBackendKind.TORCH,
    device: str = DEFAULT_SERVING_DEVICE,
    provider: str = DEFAULT_SERVING_ONNX_PROVIDER,
    onnx_path: str | PathLike[str] | None = None,
    execution_mode: VideoInferenceExecutionMode | str = VideoInferenceExecutionMode.SEQUENTIAL,
    interpolation_mode: InferenceMode | str = InferenceMode.ARBITRARY_NX,
    interpolation_factor: int = MIN_SERVING_INTERPOLATION_FACTOR,
    scale: float = 1.0,
    limit_pairs: int | None = None,
    codec: str = DEFAULT_SERVING_CODEC,
    enable_quality_evaluation: bool = True,
    quality_triplet_output_dir: str | PathLike[str] | None = None,
    write_quality_triplets: bool = False,
    settings: Settings | None = None,
) -> VideoInferenceResult:
    config = PracticalRIFEServingConfig(
        backend=backend,
        device=device,
        provider=provider,
        onnx_path=None if onnx_path is None else Path(onnx_path),
        execution_mode=execution_mode,
        interpolation_mode=interpolation_mode,
        default_scale=scale,
        codec=codec,
    )
    runner = PracticalRIFEVideoInferenceRunner(config, settings=settings)
    try:
        return runner.run(
            input_path=input_path,
            output_path=output_path,
            interpolation_factor=interpolation_factor,
            scale=scale,
            limit_pairs=limit_pairs,
            codec=codec,
            enable_quality_evaluation=enable_quality_evaluation,
            quality_triplet_output_dir=quality_triplet_output_dir,
            write_quality_triplets=write_quality_triplets,
        )
    finally:
        runner.close()


def validate_serving_interpolation_factor(
    interpolation_factor: int,
    *,
    minimum: int = MIN_SERVING_INTERPOLATION_FACTOR,
    maximum: int = MAX_SERVING_INTERPOLATION_FACTOR,
) -> int:
    if isinstance(interpolation_factor, bool) or not isinstance(interpolation_factor, int):
        raise InferenceRequestValidationError("serving interpolation_factor must be an integer.")
    if not minimum <= interpolation_factor <= maximum:
        raise InferenceRequestValidationError(
            f"Practical-RIFE serving interpolation_factor must be in [{minimum}, {maximum}], "
            f"got {interpolation_factor}."
        )
    return interpolation_factor


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


def _coerce_backend(backend: RuntimeBackendKind | str) -> RuntimeBackendKind:
    try:
        return RuntimeBackendKind(str(backend))
    except ValueError as exc:
        allowed = ", ".join(kind.value for kind in RuntimeBackendKind)
        raise ValueError(f"Unsupported serving backend: {backend!r}. Allowed: {allowed}.") from exc


def _coerce_mode(mode: InferenceMode | str) -> InferenceMode:
    try:
        return InferenceMode(str(mode))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in InferenceMode)
        raise ValueError(f"Unsupported serving interpolation_mode: {mode!r}. Allowed: {allowed}.") from exc


def _coerce_execution_mode(mode: VideoInferenceExecutionMode | str) -> VideoInferenceExecutionMode:
    try:
        return VideoInferenceExecutionMode(str(mode))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in VideoInferenceExecutionMode)
        raise ValueError(f"Unsupported serving execution_mode: {mode!r}. Allowed: {allowed}.") from exc


def _resolve_serving_path(settings: Settings, path: Path) -> Path:
    if path.is_absolute():
        return path
    return settings.resolve_path(path)
