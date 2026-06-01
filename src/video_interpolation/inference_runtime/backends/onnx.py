from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import numpy as np
import torch

from video_interpolation.inference_runtime.api import RuntimeBackendKind, RuntimeInputs, RuntimeOutputs
from video_interpolation.inference_runtime.backends.base import RuntimeBackend, RuntimeBackendError

LOGGER = logging.getLogger(__name__)

DEFAULT_ONNX_PROVIDERS = ("CPUExecutionProvider",)
DEFAULT_ONNX_INPUT_NAMES = ("left", "right", "timestep")
DEFAULT_ONNX_OUTPUT_NAMES = ("intermediate_frame",)

_PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}


@dataclass(frozen=True)
class OnnxRuntimeBackendConfig:
    """Configuration for a loaded ONNX Runtime session."""

    artifact_path: Path
    providers: Sequence[str] = DEFAULT_ONNX_PROVIDERS
    input_names: Sequence[str] = DEFAULT_ONNX_INPUT_NAMES
    output_names: Sequence[str] = DEFAULT_ONNX_OUTPUT_NAMES

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_path", Path(self.artifact_path).expanduser().resolve())
        object.__setattr__(self, "providers", _normalise_providers(self.providers))
        object.__setattr__(self, "input_names", _normalise_names(self.input_names, "input_names"))
        object.__setattr__(self, "output_names", _normalise_names(self.output_names, "output_names"))


class OnnxRuntimeBackend(RuntimeBackend):
    """ONNX Runtime backend for already prepared model-core tensors."""

    kind = RuntimeBackendKind.ONNX

    def __init__(self, config: OnnxRuntimeBackendConfig) -> None:
        self.config = config
        self._session: Any | None = None
        self._input_names: tuple[str, ...] = ()
        self._output_names: tuple[str, ...] = ()
        self._input_shapes: dict[str, tuple[int | str | None, ...]] = {}
        self._output_shapes: dict[str, tuple[int | str | None, ...]] = {}
        self._session_providers: tuple[str, ...] = ()

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def session_providers(self) -> tuple[str, ...]:
        return self._session_providers

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    @property
    def input_shapes(self) -> dict[str, tuple[int | str | None, ...]]:
        return dict(self._input_shapes)

    @property
    def output_shapes(self) -> dict[str, tuple[int | str | None, ...]]:
        return dict(self._output_shapes)

    def load(self) -> None:
        if not self.config.artifact_path.is_file():
            raise RuntimeBackendError(f"ONNX model does not exist: {self.config.artifact_path}")

        ort = _import_onnxruntime()
        providers = resolve_onnx_providers(self.config.providers)
        try:
            session = ort.InferenceSession(str(self.config.artifact_path), providers=list(providers))
        except Exception as exc:  # pragma: no cover - provider/model loader errors vary by runtime build.
            raise RuntimeBackendError(
                f"Failed to load ONNX model {self.config.artifact_path} with providers {providers}: {exc}"
            ) from exc

        self._session = session
        input_metas = tuple(session.get_inputs())
        output_metas = tuple(session.get_outputs())
        self._input_names = tuple(input_meta.name for input_meta in input_metas)
        self._output_names = tuple(output_meta.name for output_meta in output_metas)
        self._input_shapes = {input_meta.name: _normalise_meta_shape(input_meta.shape) for input_meta in input_metas}
        self._output_shapes = {output_meta.name: _normalise_meta_shape(output_meta.shape) for output_meta in output_metas}
        self._session_providers = tuple(session.get_providers())
        _require_names(self.config.input_names, self._input_names, "input", self.config.artifact_path)
        _require_names(self.config.output_names, self._output_names, "output", self.config.artifact_path)
        LOGGER.info(
            "Loaded ONNX Runtime session artifact=%s providers=%s",
            self.config.artifact_path,
            self._session_providers,
        )

    def run(self, inputs: RuntimeInputs) -> RuntimeOutputs:
        self.ensure_loaded()
        assert self._session is not None
        missing = [name for name in self.config.input_names if name not in inputs.tensors]
        if missing:
            raise RuntimeBackendError(f"ONNX runtime inputs are missing required tensor(s): {missing}")

        feed = {name: _tensor_to_ort_array(inputs.tensors[name], name) for name in self.config.input_names}
        try:
            output_arrays = self._session.run(list(self.config.output_names), feed)
        except Exception as exc:  # pragma: no cover - shape/operator errors vary by graph/runtime.
            shape_detail = {name: tuple(array.shape) for name, array in feed.items()}
            raise RuntimeBackendError(
                f"ONNX Runtime inference failed for {self.config.artifact_path} "
                f"with input shapes {shape_detail}: {exc}"
            ) from exc

        output_tensors = {
            name: torch.from_numpy(np.array(array, copy=True))
            for name, array in zip(self.config.output_names, output_arrays, strict=True)
        }
        return RuntimeOutputs(
            tensors=output_tensors,
            metadata={
                "artifact_path": str(self.config.artifact_path),
                "requested_providers": self.config.providers,
                "session_providers": self._session_providers,
                "input_shapes": self.input_shapes,
                "output_shapes": self.output_shapes,
            },
        )

    def close(self) -> None:
        self._session = None
        self._input_names = ()
        self._output_names = ()
        self._input_shapes = {}
        self._output_shapes = {}
        self._session_providers = ()


def available_onnx_providers() -> tuple[str, ...]:
    return tuple(_import_onnxruntime().get_available_providers())


def resolve_onnx_providers(providers: Sequence[str] | str | None = None) -> tuple[str, ...]:
    requested = _normalise_providers(DEFAULT_ONNX_PROVIDERS if providers is None else providers)
    available = available_onnx_providers()
    missing = [provider for provider in requested if provider not in available]
    if missing:
        raise RuntimeBackendError(
            f"Requested ONNX Runtime provider(s) are unavailable: {missing}. Available providers: {available}."
        )
    return requested


def _import_onnxruntime():
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeBackendError(
            "onnxruntime is not installed. Install the project ONNX Runtime dependencies before using backend=onnx."
        ) from exc
    return ort


def _tensor_to_ort_array(tensor: torch.Tensor, name: str) -> np.ndarray:
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeBackendError(f"ONNX input {name!r} must be a torch.Tensor, got {type(tensor).__name__}.")
    if tensor.dtype not in (torch.float16, torch.float32, torch.float64):
        raise RuntimeBackendError(f"ONNX input {name!r} must be floating point, got dtype={tensor.dtype}.")
    return tensor.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)


def _normalise_meta_shape(shape: Sequence[Any]) -> tuple[int | str | None, ...]:
    dims: list[int | str | None] = []
    for dim in shape:
        if dim is None or isinstance(dim, int | str):
            dims.append(dim)
        else:
            dims.append(str(dim))
    return tuple(dims)


def _normalise_providers(providers: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(providers, str):
        raw = (providers,)
    else:
        raw = tuple(providers)
    if not raw:
        raise RuntimeBackendError("At least one ONNX Runtime provider must be configured.")

    resolved: list[str] = []
    for provider in raw:
        if not isinstance(provider, str) or not provider:
            raise RuntimeBackendError("ONNX Runtime provider names must be non-empty strings.")
        resolved.append(_PROVIDER_ALIASES.get(provider.lower(), provider))
    return tuple(resolved)


def _normalise_names(names: Sequence[str] | str, label: str) -> tuple[str, ...]:
    if isinstance(names, str):
        raw = (names,)
    else:
        raw = tuple(names)
    if not raw:
        raise RuntimeBackendError(f"{label} must contain at least one name.")
    if any(not isinstance(name, str) or not name for name in raw):
        raise RuntimeBackendError(f"{label} must contain only non-empty strings.")
    return raw


def _require_names(expected: Sequence[str], actual: Sequence[str], label: str, artifact_path: Path) -> None:
    missing = [name for name in expected if name not in actual]
    if missing:
        raise RuntimeBackendError(
            f"ONNX model {artifact_path} is missing expected {label} name(s) {missing}; "
            f"available {label} names: {tuple(actual)}."
        )
