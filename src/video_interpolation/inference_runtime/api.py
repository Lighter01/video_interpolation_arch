from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import isclose
from types import MappingProxyType

import torch

MIN_INTERPOLATION_FACTOR = 2
MAX_INTERPOLATION_FACTOR = 8
FIXED_2X_TIMESTEP = 0.5


class InferenceRequestValidationError(ValueError):
    """Raised when an inference request is internally inconsistent."""


class InferenceMode(StrEnum):
    FIXED_2X = "fixed_2x"
    ARBITRARY_NX = "arbitrary_nx"


class RuntimeBackendKind(StrEnum):
    TORCH = "torch"
    ONNX = "onnx"


@dataclass(frozen=True)
class RuntimeInputs:
    """Already formatted tensors passed to a runtime backend."""

    tensors: Mapping[str, torch.Tensor]
    timesteps: Sequence[float] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", _immutable_tensor_mapping(self.tensors, "tensors"))
        object.__setattr__(self, "timesteps", _coerce_timesteps(self.timesteps, allow_empty=True))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def primary_tensor(self) -> torch.Tensor:
        if "input" in self.tensors:
            return self.tensors["input"]
        if len(self.tensors) == 1:
            return next(iter(self.tensors.values()))
        raise InferenceRequestValidationError(
            "RuntimeInputs.primary_tensor is only available for a single tensor or an 'input' tensor."
        )


@dataclass(frozen=True)
class RuntimeOutputs:
    """Raw tensors produced by a runtime backend."""

    tensors: Mapping[str, torch.Tensor]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", _immutable_tensor_mapping(self.tensors, "tensors"))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def primary_tensor(self) -> torch.Tensor:
        if "output" in self.tensors:
            return self.tensors["output"]
        if len(self.tensors) == 1:
            return next(iter(self.tensors.values()))
        raise InferenceRequestValidationError(
            "RuntimeOutputs.primary_tensor is only available for a single tensor or an 'output' tensor."
        )


@dataclass(frozen=True)
class FramePairRequest:
    """Request to interpolate one left/right frame tensor pair."""

    left: torch.Tensor
    right: torch.Tensor
    mode: InferenceMode | str = InferenceMode.FIXED_2X
    interpolation_factor: int = MIN_INTERPOLATION_FACTOR
    timesteps: Sequence[float] | None = None
    backend_kind: RuntimeBackendKind | str = RuntimeBackendKind.TORCH
    device: str | torch.device | None = None
    backend_options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_frame_pair(self.left, self.right)
        mode = _coerce_enum(self.mode, InferenceMode, "mode")
        backend_kind = _coerce_enum(self.backend_kind, RuntimeBackendKind, "backend_kind")
        factor, timesteps = _resolve_mode_timesteps(mode, self.interpolation_factor, self.timesteps)

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "backend_kind", backend_kind)
        object.__setattr__(self, "interpolation_factor", factor)
        object.__setattr__(self, "timesteps", timesteps)
        object.__setattr__(self, "backend_options", MappingProxyType(dict(self.backend_options)))

    @property
    def num_intermediate_frames(self) -> int:
        return len(self.timesteps)

    @property
    def original_shape(self) -> tuple[int, ...]:
        return tuple(self.left.shape)

    @property
    def is_batched(self) -> bool:
        return self.left.ndim == 4

    def to_runtime_inputs(self) -> RuntimeInputs:
        return RuntimeInputs(
            tensors={"left": self.left, "right": self.right},
            timesteps=self.timesteps,
            metadata={
                "mode": self.mode.value,
                "interpolation_factor": self.interpolation_factor,
                "backend_kind": self.backend_kind.value,
                "device": str(self.device) if self.device is not None else None,
            },
        )


@dataclass(frozen=True)
class FramePairResult:
    """Generated intermediate frames for one left/right frame tensor pair."""

    intermediate_frames: Sequence[torch.Tensor]
    timesteps: Sequence[float]
    mode: InferenceMode | str
    interpolation_factor: int
    backend_kind: RuntimeBackendKind | str
    model_name: str | None = None
    original_shape: Sequence[int] | None = None
    padded_shape: Sequence[int] | None = None
    elapsed_sec: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frames = tuple(self.intermediate_frames)
        if not frames:
            raise InferenceRequestValidationError("FramePairResult requires at least one intermediate frame.")
        for index, frame in enumerate(frames):
            if not isinstance(frame, torch.Tensor):
                raise InferenceRequestValidationError(
                    f"FramePairResult frame {index} must be a torch.Tensor, got {type(frame).__name__}."
                )

        mode = _coerce_enum(self.mode, InferenceMode, "mode")
        backend_kind = _coerce_enum(self.backend_kind, RuntimeBackendKind, "backend_kind")
        factor, timesteps = _resolve_mode_timesteps(mode, self.interpolation_factor, self.timesteps)
        if len(frames) != len(timesteps):
            raise InferenceRequestValidationError(
                f"Expected {len(timesteps)} intermediate frame(s) for timesteps {timesteps}, got {len(frames)}."
            )

        object.__setattr__(self, "intermediate_frames", frames)
        object.__setattr__(self, "timesteps", timesteps)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "backend_kind", backend_kind)
        object.__setattr__(self, "interpolation_factor", factor)
        object.__setattr__(
            self,
            "original_shape",
            tuple(self.original_shape) if self.original_shape is not None else None,
        )
        object.__setattr__(
            self,
            "padded_shape",
            tuple(self.padded_shape) if self.padded_shape is not None else None,
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def middle_frame(self) -> torch.Tensor:
        if len(self.intermediate_frames) != 1:
            raise InferenceRequestValidationError("middle_frame is only available for single-frame results.")
        return self.intermediate_frames[0]


@dataclass(frozen=True)
class PairTimestepIndex:
    """Stable mapping from a flattened model row to output[pair][timestep]."""

    flat_index: int
    pair_index: int
    timestep_index: int
    timestep: float


@dataclass(frozen=True)
class FlattenedFramePairBatch:
    """Pair-major flattened tensors for one true model batch call."""

    left: torch.Tensor
    right: torch.Tensor
    timesteps: Sequence[float]
    indices: Sequence[PairTimestepIndex]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_bchw_pair(self.left, self.right)
        timesteps = _coerce_timesteps(self.timesteps)
        indices = tuple(self.indices)
        if len(indices) != self.left.shape[0]:
            raise InferenceRequestValidationError(
                f"Flattened index count must match flattened batch size, got {len(indices)} and {self.left.shape[0]}."
            )
        if len(timesteps) != len(indices):
            raise InferenceRequestValidationError(
                f"Flattened timestep count must match flattened batch size, got {len(timesteps)} and {len(indices)}."
            )
        for expected_flat_index, index in enumerate(indices):
            if index.flat_index != expected_flat_index:
                raise InferenceRequestValidationError(
                    "Flattened indices must be contiguous and ordered by flat_index."
                )
            if not isclose(timesteps[expected_flat_index], index.timestep, rel_tol=0.0, abs_tol=1e-8):
                raise InferenceRequestValidationError(
                    "Flattened timesteps must match their PairTimestepIndex timestep values."
                )

        object.__setattr__(self, "timesteps", timesteps)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def flattened_size(self) -> int:
        return self.left.shape[0]

    def to_runtime_inputs(self) -> RuntimeInputs:
        return RuntimeInputs(
            tensors={"left": self.left, "right": self.right},
            timesteps=self.timesteps,
            metadata={
                **self.metadata,
                "flattening_order": "pair_major_timestep_minor",
                "flat_pair_indices": tuple(index.pair_index for index in self.indices),
                "flat_timestep_indices": tuple(index.timestep_index for index in self.indices),
            },
        )


@dataclass(frozen=True)
class ModelBatchRequest:
    """Request to interpolate a BCHW batch of left/right frame pairs."""

    left: torch.Tensor
    right: torch.Tensor
    mode: InferenceMode | str = InferenceMode.FIXED_2X
    interpolation_factor: int = MIN_INTERPOLATION_FACTOR
    timesteps: Sequence[float] | None = None
    backend_kind: RuntimeBackendKind | str = RuntimeBackendKind.TORCH
    device: str | torch.device | None = None
    backend_options: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_bchw_pair(self.left, self.right)
        mode = _coerce_enum(self.mode, InferenceMode, "mode")
        backend_kind = _coerce_enum(self.backend_kind, RuntimeBackendKind, "backend_kind")
        factor, timesteps = _resolve_mode_timesteps(mode, self.interpolation_factor, self.timesteps)

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "backend_kind", backend_kind)
        object.__setattr__(self, "interpolation_factor", factor)
        object.__setattr__(self, "timesteps", timesteps)
        object.__setattr__(self, "backend_options", MappingProxyType(dict(self.backend_options)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def pair_count(self) -> int:
        return self.left.shape[0]

    @property
    def batch_size(self) -> int:
        return self.pair_count

    @property
    def timesteps_per_pair(self) -> int:
        return len(self.timesteps)

    @property
    def num_intermediate_frames(self) -> int:
        return self.timesteps_per_pair

    @property
    def flattened_size(self) -> int:
        return self.pair_count * self.timesteps_per_pair

    @property
    def original_shape(self) -> tuple[int, ...]:
        return tuple(self.left.shape)

    @property
    def flattening_order(self) -> str:
        return "pair_major_timestep_minor"

    def flatten_pair_timesteps(self) -> FlattenedFramePairBatch:
        repeat_count = self.timesteps_per_pair
        flattened_left = self.left.repeat_interleave(repeat_count, dim=0)
        flattened_right = self.right.repeat_interleave(repeat_count, dim=0)
        flat_timesteps: list[float] = []
        indices: list[PairTimestepIndex] = []

        flat_index = 0
        for pair_index in range(self.pair_count):
            for timestep_index, timestep in enumerate(self.timesteps):
                flat_timesteps.append(timestep)
                indices.append(
                    PairTimestepIndex(
                        flat_index=flat_index,
                        pair_index=pair_index,
                        timestep_index=timestep_index,
                        timestep=timestep,
                    )
                )
                flat_index += 1

        return FlattenedFramePairBatch(
            left=flattened_left,
            right=flattened_right,
            timesteps=tuple(flat_timesteps),
            indices=tuple(indices),
            metadata={
                **self.metadata,
                "mode": self.mode.value,
                "interpolation_factor": self.interpolation_factor,
                "backend_kind": self.backend_kind.value,
                "device": str(self.device) if self.device is not None else None,
                "backend_options": dict(self.backend_options),
                "pair_count": self.pair_count,
                "timesteps_per_pair": self.timesteps_per_pair,
                "flattening_order": self.flattening_order,
                "source_batch_shape": self.original_shape,
            },
        )


@dataclass(frozen=True)
class ModelBatchResult:
    """Generated intermediate frames indexed as outputs[pair_index][timestep_index]."""

    outputs: Sequence[Sequence[torch.Tensor]]
    timesteps: Sequence[float]
    mode: InferenceMode | str
    interpolation_factor: int
    backend_kind: RuntimeBackendKind | str
    model_name: str | None = None
    original_shape: Sequence[int] | None = None
    padded_shape: Sequence[int] | None = None
    elapsed_sec: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = _coerce_enum(self.mode, InferenceMode, "mode")
        backend_kind = _coerce_enum(self.backend_kind, RuntimeBackendKind, "backend_kind")
        factor, timesteps = _resolve_mode_timesteps(mode, self.interpolation_factor, self.timesteps)
        outputs = _coerce_batch_outputs(self.outputs, expected_timesteps=len(timesteps))

        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "timesteps", timesteps)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "backend_kind", backend_kind)
        object.__setattr__(self, "interpolation_factor", factor)
        object.__setattr__(
            self,
            "original_shape",
            tuple(self.original_shape) if self.original_shape is not None else None,
        )
        object.__setattr__(
            self,
            "padded_shape",
            tuple(self.padded_shape) if self.padded_shape is not None else None,
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_flattened_outputs(
        cls,
        flattened_outputs: torch.Tensor,
        request: ModelBatchRequest,
        *,
        model_name: str | None = None,
        padded_shape: Sequence[int] | None = None,
        elapsed_sec: float | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "ModelBatchResult":
        _validate_flattened_outputs(flattened_outputs, request)
        reconstructed: list[tuple[torch.Tensor, ...]] = []
        cursor = 0
        for _pair_index in range(request.pair_count):
            pair_outputs = tuple(
                flattened_outputs[cursor + timestep_index]
                for timestep_index in range(request.timesteps_per_pair)
            )
            reconstructed.append(pair_outputs)
            cursor += request.timesteps_per_pair

        result_metadata = {
            **(metadata or {}),
            "flattening_order": request.flattening_order,
            "pair_count": request.pair_count,
            "timesteps_per_pair": request.timesteps_per_pair,
            "flattened_size": request.flattened_size,
            "source_batch_shape": request.original_shape,
        }
        return cls(
            outputs=tuple(reconstructed),
            timesteps=request.timesteps,
            mode=request.mode,
            interpolation_factor=request.interpolation_factor,
            backend_kind=request.backend_kind,
            model_name=model_name,
            original_shape=request.original_shape,
            padded_shape=padded_shape,
            elapsed_sec=elapsed_sec,
            metadata=result_metadata,
        )

    @property
    def pair_count(self) -> int:
        return len(self.outputs)

    @property
    def timesteps_per_pair(self) -> int:
        return len(self.timesteps)

    @property
    def flattened_outputs(self) -> tuple[torch.Tensor, ...]:
        return tuple(frame for pair_outputs in self.outputs for frame in pair_outputs)

    @property
    def middle_frames(self) -> tuple[torch.Tensor, ...]:
        if len(self.timesteps) != 1:
            raise InferenceRequestValidationError("middle_frames is only available for single-timestep results.")
        return tuple(pair_outputs[0] for pair_outputs in self.outputs)


def validate_interpolation_factor(interpolation_factor: int) -> int:
    if isinstance(interpolation_factor, bool) or not isinstance(interpolation_factor, int):
        raise InferenceRequestValidationError("interpolation_factor must be an integer.")
    if not MIN_INTERPOLATION_FACTOR <= interpolation_factor <= MAX_INTERPOLATION_FACTOR:
        raise InferenceRequestValidationError(
            "interpolation_factor must be in "
            f"[{MIN_INTERPOLATION_FACTOR}, {MAX_INTERPOLATION_FACTOR}], got {interpolation_factor}."
        )
    return interpolation_factor


def generate_interpolation_timesteps(interpolation_factor: int) -> tuple[float, ...]:
    factor = validate_interpolation_factor(interpolation_factor)
    return tuple(step / factor for step in range(1, factor))


def resolve_interpolation_timesteps(
    mode: InferenceMode | str,
    interpolation_factor: int,
    timesteps: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Return validated timesteps for a public interpolation mode/factor pair."""
    resolved_mode = _coerce_enum(mode, InferenceMode, "mode")
    _, resolved_timesteps = _resolve_mode_timesteps(resolved_mode, interpolation_factor, timesteps)
    return resolved_timesteps


def _resolve_mode_timesteps(
    mode: InferenceMode,
    interpolation_factor: int,
    timesteps: Sequence[float] | None,
) -> tuple[int, tuple[float, ...]]:
    factor = validate_interpolation_factor(interpolation_factor)

    if mode is InferenceMode.FIXED_2X:
        if factor != MIN_INTERPOLATION_FACTOR:
            raise InferenceRequestValidationError("fixed_2x mode requires interpolation_factor=2.")
        resolved = (FIXED_2X_TIMESTEP,) if timesteps is None else _coerce_timesteps(timesteps)
        if len(resolved) != 1 or not isclose(resolved[0], FIXED_2X_TIMESTEP, rel_tol=0.0, abs_tol=1e-8):
            raise InferenceRequestValidationError("fixed_2x mode requires exactly one timestep: 0.5.")
        return factor, resolved

    if mode is InferenceMode.ARBITRARY_NX:
        resolved = generate_interpolation_timesteps(factor) if timesteps is None else _coerce_timesteps(timesteps)
        if len(resolved) != factor - 1:
            raise InferenceRequestValidationError(
                f"arbitrary_nx mode with interpolation_factor={factor} requires {factor - 1} timestep(s)."
            )
        _validate_arbitrary_timesteps(resolved)
        return factor, resolved

    raise InferenceRequestValidationError(f"Unsupported inference mode: {mode}.")


def _validate_arbitrary_timesteps(timesteps: tuple[float, ...]) -> None:
    previous: float | None = None
    for timestep in timesteps:
        if not 0.0 < timestep < 1.0:
            raise InferenceRequestValidationError("timesteps must be strictly inside the open interval (0, 1).")
        if previous is not None and timestep <= previous:
            raise InferenceRequestValidationError("timesteps must be strictly increasing.")
        previous = timestep


def _coerce_timesteps(timesteps: Sequence[float], *, allow_empty: bool = False) -> tuple[float, ...]:
    if isinstance(timesteps, str | bytes):
        raise InferenceRequestValidationError("timesteps must be a sequence of numbers, not a string.")
    try:
        resolved = tuple(float(timestep) for timestep in timesteps)
    except TypeError as exc:
        raise InferenceRequestValidationError("timesteps must be an iterable sequence of numbers.") from exc
    except ValueError as exc:
        raise InferenceRequestValidationError("timesteps must contain only numeric values.") from exc
    if not resolved and not allow_empty:
        raise InferenceRequestValidationError("timesteps must not be empty.")
    return resolved


def _coerce_enum(value: object, enum_type: type[InferenceMode] | type[RuntimeBackendKind], label: str):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise InferenceRequestValidationError(f"Unsupported {label}: {value!r}. Allowed values: {allowed}.") from exc


def _validate_frame_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    if not isinstance(left, torch.Tensor):
        raise InferenceRequestValidationError(f"left must be a torch.Tensor, got {type(left).__name__}.")
    if not isinstance(right, torch.Tensor):
        raise InferenceRequestValidationError(f"right must be a torch.Tensor, got {type(right).__name__}.")
    if tuple(left.shape) != tuple(right.shape):
        raise InferenceRequestValidationError(
            f"left and right tensors must have the same shape, got {tuple(left.shape)} and {tuple(right.shape)}."
        )
    if left.ndim not in (3, 4):
        raise InferenceRequestValidationError(
            f"Expected CHW or NCHW frame tensors, got shape {tuple(left.shape)}."
        )
    if any(size <= 0 for size in left.shape):
        raise InferenceRequestValidationError(f"Frame tensor dimensions must be positive, got {tuple(left.shape)}.")


def _validate_bchw_pair(left: torch.Tensor, right: torch.Tensor) -> None:
    if not isinstance(left, torch.Tensor):
        raise InferenceRequestValidationError(f"left must be a torch.Tensor, got {type(left).__name__}.")
    if not isinstance(right, torch.Tensor):
        raise InferenceRequestValidationError(f"right must be a torch.Tensor, got {type(right).__name__}.")
    if left.ndim != 4 or right.ndim != 4:
        raise InferenceRequestValidationError(
            f"Expected BCHW frame batch tensors, got {tuple(left.shape)} and {tuple(right.shape)}."
        )
    if left.shape[0] != right.shape[0]:
        raise InferenceRequestValidationError(
            f"left and right batch sizes must match, got {left.shape[0]} and {right.shape[0]}."
        )
    if tuple(left.shape) != tuple(right.shape):
        raise InferenceRequestValidationError(
            f"left and right BCHW tensors must have the same shape, got {tuple(left.shape)} and {tuple(right.shape)}."
        )
    if any(size <= 0 for size in left.shape):
        raise InferenceRequestValidationError(
            f"BCHW frame batch tensor dimensions must be positive, got {tuple(left.shape)}."
        )


def _coerce_batch_outputs(
    outputs: Sequence[Sequence[torch.Tensor]],
    *,
    expected_timesteps: int,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    if isinstance(outputs, torch.Tensor):
        raise InferenceRequestValidationError(
            "ModelBatchResult outputs must be indexed as outputs[pair_index][timestep_index], not a tensor."
        )
    pair_outputs = tuple(tuple(pair_output) for pair_output in outputs)
    if not pair_outputs:
        raise InferenceRequestValidationError("ModelBatchResult requires at least one pair output.")
    for pair_index, timestep_outputs in enumerate(pair_outputs):
        if len(timestep_outputs) != expected_timesteps:
            raise InferenceRequestValidationError(
                f"Pair {pair_index} must contain {expected_timesteps} timestep output(s), "
                f"got {len(timestep_outputs)}."
            )
        for timestep_index, frame in enumerate(timestep_outputs):
            if not isinstance(frame, torch.Tensor):
                raise InferenceRequestValidationError(
                    f"outputs[{pair_index}][{timestep_index}] must be a torch.Tensor, "
                    f"got {type(frame).__name__}."
                )
            if frame.ndim != 3:
                raise InferenceRequestValidationError(
                    f"outputs[{pair_index}][{timestep_index}] must be a CHW tensor, got {tuple(frame.shape)}."
                )
            if any(size <= 0 for size in frame.shape):
                raise InferenceRequestValidationError(
                    f"outputs[{pair_index}][{timestep_index}] dimensions must be positive, got {tuple(frame.shape)}."
                )
    return pair_outputs


def _validate_flattened_outputs(flattened_outputs: torch.Tensor, request: ModelBatchRequest) -> None:
    if not isinstance(flattened_outputs, torch.Tensor):
        raise InferenceRequestValidationError(
            f"flattened_outputs must be a torch.Tensor, got {type(flattened_outputs).__name__}."
        )
    if flattened_outputs.ndim != 4:
        raise InferenceRequestValidationError(
            f"flattened_outputs must be a BCHW tensor, got {tuple(flattened_outputs.shape)}."
        )
    if flattened_outputs.shape[0] != request.flattened_size:
        raise InferenceRequestValidationError(
            "flattened_outputs batch size must equal pair_count * timesteps_per_pair, "
            f"got {flattened_outputs.shape[0]} and {request.flattened_size}."
        )
    if any(size <= 0 for size in flattened_outputs.shape):
        raise InferenceRequestValidationError(
            f"flattened_outputs dimensions must be positive, got {tuple(flattened_outputs.shape)}."
        )


def _immutable_tensor_mapping(
    tensors: Mapping[str, torch.Tensor],
    field_name: str,
) -> Mapping[str, torch.Tensor]:
    resolved = dict(tensors)
    if not resolved:
        raise InferenceRequestValidationError(f"{field_name} must contain at least one tensor.")
    for name, tensor in resolved.items():
        if not isinstance(name, str) or not name:
            raise InferenceRequestValidationError(f"{field_name} keys must be non-empty strings.")
        if not isinstance(tensor, torch.Tensor):
            raise InferenceRequestValidationError(
                f"{field_name}[{name!r}] must be a torch.Tensor, got {type(tensor).__name__}."
            )
    return MappingProxyType(resolved)
