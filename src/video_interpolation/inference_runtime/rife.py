from dataclasses import dataclass
from os import PathLike
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F

from video_interpolation.inference_runtime.api import (
    FramePairRequest,
    FramePairResult,
    InferenceMode,
    InferenceRequestValidationError,
    ModelBatchRequest,
    ModelBatchResult,
    RuntimeBackendKind,
    RuntimeInputs,
)
from video_interpolation.inference_runtime.backends.base import RuntimeBackendError
from video_interpolation.inference_runtime.backends.onnx import OnnxRuntimeBackend, OnnxRuntimeBackendConfig
from video_interpolation.inference_runtime.backends.torch import TorchRuntimeBackend

ALLOWED_RIFE_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class PracticalRIFEPyTorchRuntimeConfig:
    model_name: str = "practical_rife_v4_26"
    device: str | torch.device = "cuda"
    timestep: float = 0.5
    scale: float = 1.0
    divisor: int = 128
    inference_batch_size: int | None = None

    def __post_init__(self) -> None:
        validate_rife_scale(self.scale)
        _validate_optional_batch_size(self.inference_batch_size)


@dataclass(frozen=True)
class PracticalRIFEOnnxRuntimeConfig:
    model_name: str = "practical_rife_v4_26"
    artifact_path: str | PathLike[str] = (
        "model_exports/onnx/practical_rife_v4_26/"
        "practical_rife_v4_26_dynamo_dynamic_batch_hw_opset18_h384w512.onnx"
    )
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    scale: float = 1.0
    divisor: int = 128
    inference_batch_size: int | None = None

    def __post_init__(self) -> None:
        validate_rife_scale(self.scale)
        _validate_optional_batch_size(self.inference_batch_size)


class PracticalRIFEPyTorchRuntime:
    """Prediction-only Practical-RIFE runtime backed by a project-owned PyTorch source copy."""

    def __init__(
        self,
        model: Any,
        config: PracticalRIFEPyTorchRuntimeConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or PracticalRIFEPyTorchRuntimeConfig()
        self.device = torch.device(self.config.device)
        self.backend = TorchRuntimeBackend(
            self._run_timestep,
            module=getattr(model, "flownet", None),
            name=f"{self.config.model_name}_torch",
        )

    def load(self) -> None:
        self.backend.load()

    def close(self) -> None:
        self.backend.close()

    @property
    def is_loaded(self) -> bool:
        return self.backend.is_loaded

    def predict(self, request: FramePairRequest) -> FramePairResult:
        if request.backend_kind is not RuntimeBackendKind.TORCH:
            raise InferenceRequestValidationError("PracticalRIFEPyTorchRuntime only supports the torch backend.")
        if request.mode not in (InferenceMode.FIXED_2X, InferenceMode.ARBITRARY_NX):
            raise InferenceRequestValidationError(f"Unsupported Practical-RIFE inference mode: {request.mode.value}.")

        started = perf_counter()
        left_batch = prepare_rife_image_tensor(request.left, self.device)
        right_batch = prepare_rife_image_tensor(request.right, self.device)
        left_padded, padding = pad_to_divisor(left_batch, self.config.divisor)
        right_padded, _ = pad_to_divisor(right_batch, self.config.divisor)
        scale = _resolve_request_scale(request, default_scale=self.config.scale)

        frames: list[torch.Tensor] = []
        for timestep in request.timesteps:
            outputs = self.backend.run(
                RuntimeInputs(
                    tensors={"left": left_padded, "right": right_padded},
                    timesteps=(timestep,),
                    metadata={
                        "mode": request.mode.value,
                        "interpolation_factor": request.interpolation_factor,
                        "model_name": self.config.model_name,
                        "scale": scale,
                    },
                )
            )
            prediction = unpad(outputs.primary_tensor, padding)
            frames.append(prediction[0].detach().cpu().clamp(0.0, 1.0).contiguous())

        return FramePairResult(
            intermediate_frames=tuple(frames),
            timesteps=request.timesteps,
            mode=request.mode,
            interpolation_factor=request.interpolation_factor,
            backend_kind=RuntimeBackendKind.TORCH,
            model_name=self.config.model_name,
            original_shape=request.original_shape,
            padded_shape=tuple(left_padded.shape),
            elapsed_sec=perf_counter() - started,
            metadata={"scale": scale, "default_scale": self.config.scale, "timestep": self.config.timestep},
        )

    def predict_batch(self, request: ModelBatchRequest) -> ModelBatchResult:
        if request.backend_kind is not RuntimeBackendKind.TORCH:
            raise InferenceRequestValidationError("PracticalRIFEPyTorchRuntime only supports the torch backend.")
        if request.mode not in (InferenceMode.FIXED_2X, InferenceMode.ARBITRARY_NX):
            raise InferenceRequestValidationError(f"Unsupported Practical-RIFE inference mode: {request.mode.value}.")

        started = perf_counter()
        flattened = request.flatten_pair_timesteps()
        left_batch = prepare_rife_image_tensor(flattened.left, self.device)
        right_batch = prepare_rife_image_tensor(flattened.right, self.device)
        left_padded, padding = pad_to_divisor(left_batch, self.config.divisor)
        right_padded, _ = pad_to_divisor(right_batch, self.config.divisor)
        scale = _resolve_request_scale(request, default_scale=self.config.scale)
        effective_batch_size = _resolve_effective_batch_size(
            request,
            default_batch_size=self.config.inference_batch_size,
            flattened_size=flattened.flattened_size,
        )

        chunks: list[torch.Tensor] = []
        model_call_count = 0
        for start_index in range(0, flattened.flattened_size, effective_batch_size):
            end_index = min(start_index + effective_batch_size, flattened.flattened_size)
            outputs = self.backend.run(
                RuntimeInputs(
                    tensors={
                        "left": left_padded[start_index:end_index],
                        "right": right_padded[start_index:end_index],
                    },
                    timesteps=flattened.timesteps[start_index:end_index],
                    metadata={
                        "mode": request.mode.value,
                        "interpolation_factor": request.interpolation_factor,
                        "model_name": self.config.model_name,
                        "scale": scale,
                        "flattening_order": request.flattening_order,
                        "effective_batch_size": effective_batch_size,
                    },
                )
            )
            prediction = unpad(outputs.primary_tensor, padding)
            chunks.append(prediction.detach().cpu().clamp(0.0, 1.0).contiguous())
            model_call_count += 1

        flattened_outputs = torch.cat(chunks, dim=0)
        return ModelBatchResult.from_flattened_outputs(
            flattened_outputs,
            request,
            model_name=self.config.model_name,
            padded_shape=tuple(left_padded.shape),
            elapsed_sec=perf_counter() - started,
            metadata={
                "scale": scale,
                "default_scale": self.config.scale,
                "effective_batch_size": effective_batch_size,
                "model_call_count": model_call_count,
            },
        )

    def _run_timestep(self, inputs: RuntimeInputs) -> torch.Tensor:
        left = inputs.tensors["left"]
        right = inputs.tensors["right"]
        timestep = _model_timestep_value(inputs.timesteps, left)
        scale = validate_rife_scale(inputs.metadata.get("scale", self.config.scale))
        return self.model.inference(left, right, timestep=timestep, scale=scale)


def validate_rife_scale(scale: object) -> float:
    if isinstance(scale, bool) or not isinstance(scale, int | float):
        raise InferenceRequestValidationError("Practical-RIFE scale must be numeric.")
    resolved = float(scale)
    if resolved not in ALLOWED_RIFE_SCALES:
        allowed = ", ".join(f"{allowed:g}" for allowed in ALLOWED_RIFE_SCALES)
        raise InferenceRequestValidationError(f"Practical-RIFE scale must be one of: {allowed}.")
    return resolved


def _resolve_request_scale(request: FramePairRequest | ModelBatchRequest, *, default_scale: float) -> float:
    return validate_rife_scale(request.backend_options.get("scale", default_scale))


def prepare_rife_image_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"Expected image tensor shape CxHxW or BxCxHxW with 3 channels, got {tuple(tensor.shape)}")
    return tensor.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)


def pad_to_divisor(tensor: torch.Tensor, divisor: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    height = tensor.shape[-2]
    width = tensor.shape[-1]
    pad_height = ((height - 1) // divisor + 1) * divisor - height
    pad_width = ((width - 1) // divisor + 1) * divisor - width
    padding = (0, pad_width, 0, pad_height)
    if pad_height == 0 and pad_width == 0:
        return tensor, padding
    return F.pad(tensor, padding), padding


def unpad(tensor: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    _, pad_right, _, pad_bottom = padding
    height = tensor.shape[-2] - pad_bottom
    width = tensor.shape[-1] - pad_right
    return tensor[..., :height, :width]


def _model_timestep_value(timesteps: tuple[float, ...], reference: torch.Tensor) -> float | torch.Tensor:
    if len(timesteps) == 1:
        return timesteps[0]
    if len(timesteps) != reference.shape[0]:
        raise InferenceRequestValidationError(
            f"Expected either one timestep or one timestep per batch row, got {len(timesteps)} "
            f"for batch size {reference.shape[0]}."
        )
    return torch.tensor(timesteps, device=reference.device, dtype=reference.dtype).reshape(-1, 1, 1, 1)


def _validate_optional_batch_size(batch_size: int | None) -> int | None:
    if batch_size is None:
        return None
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise InferenceRequestValidationError("inference_batch_size must be a positive integer or None.")
    if batch_size <= 0:
        raise InferenceRequestValidationError("inference_batch_size must be positive.")
    return batch_size


def _resolve_effective_batch_size(
    request: ModelBatchRequest,
    *,
    default_batch_size: int | None,
    flattened_size: int,
) -> int:
    requested = request.backend_options.get("inference_batch_size", default_batch_size)
    resolved = _validate_optional_batch_size(requested)
    return flattened_size if resolved is None else min(resolved, flattened_size)


class PracticalRIFEOnnxRuntime:
    """Prediction-only Practical-RIFE runtime backed by an exported ONNX neural core."""

    def __init__(
        self,
        config: PracticalRIFEOnnxRuntimeConfig,
    ) -> None:
        self.config = config
        self.backend = OnnxRuntimeBackend(
            OnnxRuntimeBackendConfig(
                artifact_path=config.artifact_path,
                providers=config.providers,
            )
        )

    def load(self) -> None:
        self.backend.load()
        _require_dynamic_batch_onnx_artifact(self.backend, artifact_path=self.backend.config.artifact_path)

    def close(self) -> None:
        self.backend.close()

    @property
    def is_loaded(self) -> bool:
        return self.backend.is_loaded

    def predict(self, request: FramePairRequest) -> FramePairResult:
        if request.backend_kind is not RuntimeBackendKind.ONNX:
            raise InferenceRequestValidationError("PracticalRIFEOnnxRuntime only supports the onnx backend.")
        if request.mode not in (InferenceMode.FIXED_2X, InferenceMode.ARBITRARY_NX):
            raise InferenceRequestValidationError(f"Unsupported Practical-RIFE inference mode: {request.mode.value}.")

        requested_scale = _resolve_request_scale(request, default_scale=self.config.scale)
        if requested_scale != self.config.scale:
            raise InferenceRequestValidationError(
                "Practical-RIFE ONNX artifacts bake scale into the exported graph. "
                f"Request scale={requested_scale:g} requires an ONNX artifact exported with the same scale; "
                f"loaded artifact scale={self.config.scale:g}."
            )

        started = perf_counter()
        device = torch.device("cpu")
        left_batch = prepare_rife_image_tensor(request.left, device)
        right_batch = prepare_rife_image_tensor(request.right, device)
        left_padded, padding = pad_to_divisor(left_batch, self.config.divisor)
        right_padded, _ = pad_to_divisor(right_batch, self.config.divisor)

        frames: list[torch.Tensor] = []
        session_providers: tuple[str, ...] = ()
        for timestep in request.timesteps:
            timestep_tensor = torch.full(
                (left_padded.shape[0], 1, 1, 1),
                float(timestep),
                dtype=left_padded.dtype,
            )
            outputs = self.backend.run(
                RuntimeInputs(
                    tensors={"left": left_padded, "right": right_padded, "timestep": timestep_tensor},
                    timesteps=(timestep,),
                    metadata={
                        "mode": request.mode.value,
                        "interpolation_factor": request.interpolation_factor,
                        "model_name": self.config.model_name,
                        "scale": self.config.scale,
                    },
                )
            )
            session_providers = tuple(str(value) for value in outputs.metadata.get("session_providers", ()))
            prediction = unpad(outputs.primary_tensor, padding)
            frames.append(prediction[0].detach().cpu().clamp(0.0, 1.0).contiguous())

        return FramePairResult(
            intermediate_frames=tuple(frames),
            timesteps=request.timesteps,
            mode=request.mode,
            interpolation_factor=request.interpolation_factor,
            backend_kind=RuntimeBackendKind.ONNX,
            model_name=self.config.model_name,
            original_shape=request.original_shape,
            padded_shape=tuple(left_padded.shape),
            elapsed_sec=perf_counter() - started,
            metadata={
                "artifact_path": str(self.backend.config.artifact_path),
                "requested_providers": self.backend.config.providers,
                "session_providers": session_providers,
                "scale": self.config.scale,
            },
        )

    def predict_batch(self, request: ModelBatchRequest) -> ModelBatchResult:
        if request.backend_kind is not RuntimeBackendKind.ONNX:
            raise InferenceRequestValidationError("PracticalRIFEOnnxRuntime only supports the onnx backend.")
        if request.mode not in (InferenceMode.FIXED_2X, InferenceMode.ARBITRARY_NX):
            raise InferenceRequestValidationError(f"Unsupported Practical-RIFE inference mode: {request.mode.value}.")

        requested_scale = _resolve_request_scale(request, default_scale=self.config.scale)
        if requested_scale != self.config.scale:
            raise InferenceRequestValidationError(
                "Practical-RIFE ONNX artifacts bake scale into the exported graph. "
                f"Request scale={requested_scale:g} requires an ONNX artifact exported with the same scale; "
                f"loaded artifact scale={self.config.scale:g}."
            )

        started = perf_counter()
        device = torch.device("cpu")
        flattened = request.flatten_pair_timesteps()
        left_batch = prepare_rife_image_tensor(flattened.left, device)
        right_batch = prepare_rife_image_tensor(flattened.right, device)
        left_padded, padding = pad_to_divisor(left_batch, self.config.divisor)
        right_padded, _ = pad_to_divisor(right_batch, self.config.divisor)

        requested_effective_batch_size = _resolve_effective_batch_size(
            request,
            default_batch_size=self.config.inference_batch_size,
            flattened_size=flattened.flattened_size,
        )
        effective_batch_size = requested_effective_batch_size

        chunks: list[torch.Tensor] = []
        session_providers: tuple[str, ...] = ()
        model_call_count = 0
        for start_index in range(0, flattened.flattened_size, effective_batch_size):
            end_index = min(start_index + effective_batch_size, flattened.flattened_size)
            timestep_tensor = torch.tensor(
                flattened.timesteps[start_index:end_index],
                dtype=left_padded.dtype,
            ).reshape(-1, 1, 1, 1)
            outputs = self.backend.run(
                RuntimeInputs(
                    tensors={
                        "left": left_padded[start_index:end_index],
                        "right": right_padded[start_index:end_index],
                        "timestep": timestep_tensor,
                    },
                    timesteps=flattened.timesteps[start_index:end_index],
                    metadata={
                        "mode": request.mode.value,
                        "interpolation_factor": request.interpolation_factor,
                        "model_name": self.config.model_name,
                        "scale": self.config.scale,
                        "flattening_order": request.flattening_order,
                        "effective_batch_size": effective_batch_size,
                    },
                )
            )
            session_providers = tuple(str(value) for value in outputs.metadata.get("session_providers", ()))
            prediction = unpad(outputs.primary_tensor, padding)
            chunks.append(prediction.detach().cpu().clamp(0.0, 1.0).contiguous())
            model_call_count += 1

        flattened_outputs = torch.cat(chunks, dim=0)
        return ModelBatchResult.from_flattened_outputs(
            flattened_outputs,
            request,
            model_name=self.config.model_name,
            padded_shape=tuple(left_padded.shape),
            elapsed_sec=perf_counter() - started,
            metadata={
                "artifact_path": str(self.backend.config.artifact_path),
                "requested_providers": self.backend.config.providers,
                "session_providers": session_providers,
                "scale": self.config.scale,
                "effective_batch_size": effective_batch_size,
                "requested_effective_batch_size": requested_effective_batch_size,
                "model_call_count": model_call_count,
                "batch_execution": "dynamic_batch",
            },
        )


def _require_dynamic_batch_onnx_artifact(backend: OnnxRuntimeBackend, *, artifact_path: PathLike[str]) -> None:
    io_shapes = {**backend.input_shapes, **backend.output_shapes}
    fixed_batch_entries: list[str] = []
    for name in ("left", "right", "timestep", "intermediate_frame"):
        shape = io_shapes.get(name)
        if not shape:
            continue
        batch_dim = shape[0]
        if isinstance(batch_dim, int):
            fixed_batch_entries.append(f"{name}{tuple(shape)}")
    if fixed_batch_entries:
        joined = ", ".join(fixed_batch_entries)
        raise RuntimeBackendError(
            "Practical-RIFE ONNX runtime requires a dynamic-batch artifact. "
            f"Fixed batch dimension(s) found in {artifact_path}: {joined}."
        )
