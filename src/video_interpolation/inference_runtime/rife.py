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
    RuntimeBackendKind,
    RuntimeInputs,
)
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

    def __post_init__(self) -> None:
        validate_rife_scale(self.scale)


@dataclass(frozen=True)
class PracticalRIFEOnnxRuntimeConfig:
    model_name: str = "practical_rife_v4_26"
    artifact_path: str | PathLike[str] = (
        "model_exports/onnx/practical_rife_v4_26/practical_rife_v4_26_dynamic_hw_opset17.simplified.onnx"
    )
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    scale: float = 1.0
    divisor: int = 128

    def __post_init__(self) -> None:
        validate_rife_scale(self.scale)


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

    def _run_timestep(self, inputs: RuntimeInputs) -> torch.Tensor:
        left = inputs.tensors["left"]
        right = inputs.tensors["right"]
        timestep = inputs.timesteps[0]
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


def _resolve_request_scale(request: FramePairRequest, *, default_scale: float) -> float:
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
