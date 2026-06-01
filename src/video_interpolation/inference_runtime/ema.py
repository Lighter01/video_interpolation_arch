from collections.abc import Callable
from dataclasses import dataclass
from os import PathLike
from time import perf_counter
from typing import Any

import torch

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
from video_interpolation.inference_runtime.backends.onnx import OnnxRuntimeBackend, OnnxRuntimeBackendConfig
from video_interpolation.inference_runtime.backends.torch import TorchRuntimeBackend


@dataclass(frozen=True)
class EMAVFIPyTorchRuntimeConfig:
    model_name: str = "ema_vfi_small"
    device: str | torch.device = "cuda"
    divisor: int = 32
    tta: bool = False
    fast_tta: bool = False
    inference_batch_size: int | None = None

    def __post_init__(self) -> None:
        _validate_optional_batch_size(self.inference_batch_size)


@dataclass(frozen=True)
class EMAVFIOnnxRuntimeConfig:
    model_name: str = "ema_vfi_small"
    artifact_path: str | PathLike[str] = (
        "model_exports/onnx/ema_vfi_small/ema_vfi_small_dynamo_dynamic_hw_opset18_h336w560.onnx"
    )
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    divisor: int = 112
    inference_batch_size: int | None = None

    def __post_init__(self) -> None:
        _validate_optional_batch_size(self.inference_batch_size)


class EMAVFIPyTorchRuntime:
    """Prediction-only EMA-VFI runtime backed by the upstream PyTorch model."""

    def __init__(
        self,
        model: Any,
        input_padder_cls: Callable[..., Any],
        config: EMAVFIPyTorchRuntimeConfig | None = None,
    ) -> None:
        self.model = model
        self.input_padder_cls = input_padder_cls
        self.config = config or EMAVFIPyTorchRuntimeConfig()
        self.device = torch.device(self.config.device)
        self.backend = TorchRuntimeBackend(
            self._run_timestep,
            module=getattr(model, "net", None),
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
            raise InferenceRequestValidationError("EMAVFIPyTorchRuntime only supports the torch backend.")
        if request.mode not in (InferenceMode.FIXED_2X, InferenceMode.ARBITRARY_NX):
            raise InferenceRequestValidationError(f"Unsupported EMA-VFI inference mode: {request.mode.value}.")

        started = perf_counter()
        left_batch = prepare_ema_image_tensor(request.left, self.device)
        right_batch = prepare_ema_image_tensor(request.right, self.device)
        padder = self.input_padder_cls(left_batch.shape, divisor=self.config.divisor)
        left_padded, right_padded = padder.pad(left_batch, right_batch)

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
                    },
                )
            )
            prediction = padder.unpad(outputs.primary_tensor)
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
            metadata={"tta": self.config.tta, "fast_tta": self.config.fast_tta},
        )

    def predict_batch(self, request: ModelBatchRequest) -> ModelBatchResult:
        if request.backend_kind is not RuntimeBackendKind.TORCH:
            raise InferenceRequestValidationError("EMAVFIPyTorchRuntime only supports the torch backend.")
        if request.mode not in (InferenceMode.FIXED_2X, InferenceMode.ARBITRARY_NX):
            raise InferenceRequestValidationError(f"Unsupported EMA-VFI inference mode: {request.mode.value}.")

        started = perf_counter()
        flattened = request.flatten_pair_timesteps()
        left_batch = prepare_ema_image_tensor(flattened.left, self.device)
        right_batch = prepare_ema_image_tensor(flattened.right, self.device)
        padder = self.input_padder_cls(left_batch.shape, divisor=self.config.divisor)
        left_padded, right_padded = padder.pad(left_batch, right_batch)
        effective_batch_size = _resolve_effective_batch_size(
            request,
            default_batch_size=self.config.inference_batch_size,
            flattened_size=flattened.flattened_size,
            force_single_batch=self.config.fast_tta,
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
                        "flattening_order": request.flattening_order,
                        "effective_batch_size": effective_batch_size,
                    },
                )
            )
            prediction = padder.unpad(outputs.primary_tensor)
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
                "tta": self.config.tta,
                "fast_tta": self.config.fast_tta,
                "effective_batch_size": effective_batch_size,
                "model_call_count": model_call_count,
            },
        )

    def _run_timestep(self, inputs: RuntimeInputs) -> torch.Tensor:
        left = inputs.tensors["left"]
        right = inputs.tensors["right"]
        timestep = _model_timestep_value(inputs.timesteps, left)
        return self.model.inference(
            left,
            right,
            TTA=self.config.tta,
            timestep=timestep,
            fast_TTA=self.config.fast_tta,
        )


def prepare_ema_image_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"Expected image tensor shape CxHxW or BxCxHxW with 3 channels, got {tuple(tensor.shape)}")
    return tensor.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)


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
    force_single_batch: bool = False,
) -> int:
    requested = request.backend_options.get("inference_batch_size", default_batch_size)
    resolved = _validate_optional_batch_size(requested)
    effective = flattened_size if resolved is None else min(resolved, flattened_size)
    if force_single_batch:
        return 1
    return effective


class EMAVFIOnnxRuntime:
    """Prediction-only EMA-VFI runtime backed by an exported ONNX neural core."""

    def __init__(
        self,
        input_padder_cls: Callable[..., Any],
        config: EMAVFIOnnxRuntimeConfig,
    ) -> None:
        self.input_padder_cls = input_padder_cls
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
            raise InferenceRequestValidationError("EMAVFIOnnxRuntime only supports the onnx backend.")
        if request.mode not in (InferenceMode.FIXED_2X, InferenceMode.ARBITRARY_NX):
            raise InferenceRequestValidationError(f"Unsupported EMA-VFI inference mode: {request.mode.value}.")

        started = perf_counter()
        device = torch.device("cpu")
        left_batch = prepare_ema_image_tensor(request.left, device)
        right_batch = prepare_ema_image_tensor(request.right, device)
        padder = self.input_padder_cls(left_batch.shape, divisor=self.config.divisor)
        left_padded, right_padded = padder.pad(left_batch, right_batch)

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
                    },
                )
            )
            session_providers = tuple(str(value) for value in outputs.metadata.get("session_providers", ()))
            prediction = padder.unpad(outputs.primary_tensor)
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
            },
        )

    def predict_batch(self, request: ModelBatchRequest) -> ModelBatchResult:
        if request.backend_kind is not RuntimeBackendKind.ONNX:
            raise InferenceRequestValidationError("EMAVFIOnnxRuntime only supports the onnx backend.")
        if request.mode not in (InferenceMode.FIXED_2X, InferenceMode.ARBITRARY_NX):
            raise InferenceRequestValidationError(f"Unsupported EMA-VFI inference mode: {request.mode.value}.")

        started = perf_counter()
        device = torch.device("cpu")
        flattened = request.flatten_pair_timesteps()
        left_batch = prepare_ema_image_tensor(flattened.left, device)
        right_batch = prepare_ema_image_tensor(flattened.right, device)
        padder = self.input_padder_cls(left_batch.shape, divisor=self.config.divisor)
        left_padded, right_padded = padder.pad(left_batch, right_batch)
        effective_batch_size = _resolve_effective_batch_size(
            request,
            default_batch_size=self.config.inference_batch_size,
            flattened_size=flattened.flattened_size,
        )

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
                        "flattening_order": request.flattening_order,
                        "effective_batch_size": effective_batch_size,
                    },
                )
            )
            session_providers = tuple(str(value) for value in outputs.metadata.get("session_providers", ()))
            prediction = padder.unpad(outputs.primary_tensor)
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
                "effective_batch_size": effective_batch_size,
                "model_call_count": model_call_count,
            },
        )
