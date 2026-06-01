from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any
import traceback

import torch

from video_interpolation.inference_runtime.rife import validate_rife_scale

DEFAULT_ONNX_EXPORT_ROOT = Path("model_exports/onnx")
DEFAULT_ONNX_OPSET_VERSION = 17
ONNX_INPUT_NAMES = ("left", "right", "timestep")
ONNX_OUTPUT_NAMES = ("intermediate_frame",)


class OnnxExportValidationError(ValueError):
    """Raised when ONNX export options are internally inconsistent."""


class OnnxShapeMode(StrEnum):
    """Shape policy for exported ONNX graph inputs."""

    DYNAMIC_HW = "dynamic_hw"
    STATIC = "static"


class OnnxExporterKind(StrEnum):
    """Torch ONNX exporter implementation to use."""

    LEGACY = "legacy"
    DYNAMO = "dynamo"


@dataclass(frozen=True)
class OnnxArtifactPaths:
    output_dir: Path
    original_path: Path
    simplified_path: Path

    @property
    def preferred_path_if_simplified(self) -> Path:
        return self.simplified_path


@dataclass(frozen=True)
class OnnxExportConfig:
    """Configuration for exporting an already loaded PyTorch neural core."""

    model_name: str
    output_dir: Path = DEFAULT_ONNX_EXPORT_ROOT
    sample_input_shape: tuple[int, int, int, int] = (1, 3, 32, 32)
    opset_version: int = DEFAULT_ONNX_OPSET_VERSION
    shape_mode: OnnxShapeMode | str = OnnxShapeMode.DYNAMIC_HW
    exporter: OnnxExporterKind | str = OnnxExporterKind.LEGACY
    dynamic_hw_multiple: int | None = None
    device: str | torch.device = "cpu"
    timestep: float = 0.5
    simplify: bool = True
    artifact_stem: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_name", _validate_artifact_name(self.model_name, "model_name"))
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        object.__setattr__(self, "sample_input_shape", _validate_sample_input_shape(self.sample_input_shape))
        object.__setattr__(self, "opset_version", _validate_opset_version(self.opset_version))
        object.__setattr__(self, "shape_mode", _coerce_shape_mode(self.shape_mode))
        object.__setattr__(self, "exporter", _coerce_exporter_kind(self.exporter))
        object.__setattr__(self, "dynamic_hw_multiple", _validate_optional_positive_int(self.dynamic_hw_multiple, "dynamic_hw_multiple"))
        object.__setattr__(self, "device", torch.device(self.device))
        object.__setattr__(self, "timestep", _validate_timestep(self.timestep))
        if self.artifact_stem is not None:
            object.__setattr__(self, "artifact_stem", _validate_artifact_name(self.artifact_stem, "artifact_stem"))


@dataclass(frozen=True)
class OnnxExportResult:
    """Structured status for ONNX export and optional simplification."""

    success: bool
    model_name: str
    shape_mode: OnnxShapeMode
    exporter: OnnxExporterKind
    opset_version: int
    sample_input_shape: tuple[int, int, int, int]
    original_path: Path
    simplified_path: Path
    preferred_path: Path | None
    dynamic_axes: Mapping[str, Mapping[int, str]] | None
    simplify_requested: bool
    simplification_status: str
    checker_status: str
    error: str | None = None
    simplification_error: str | None = None
    checker_error: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dynamic_axes = None if self.dynamic_axes is None else MappingProxyType(dict(self.dynamic_axes))
        object.__setattr__(self, "dynamic_axes", dynamic_axes)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class EMAVFIOnnxWrapper(torch.nn.Module):
    """EMA-VFI neural-core ONNX boundary.

    Inputs are already prepared/padded NCHW tensors and a timestep tensor.
    Padding, timestep loops, image/video conversion, and output interleaving stay outside ONNX.
    """

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, left: torch.Tensor, right: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        images = torch.cat((left, right), dim=1)
        outputs = self.net(images, timestep=timestep)
        return outputs[-1]


class PracticalRIFEOnnxWrapper(torch.nn.Module):
    """Practical-RIFE neural-core ONNX boundary around the loaded IFNet/flownet."""

    def __init__(self, flownet: torch.nn.Module, *, scale: float = 1.0) -> None:
        super().__init__()
        self.flownet = flownet
        self.scale = validate_rife_scale(scale)

    def forward(self, left: torch.Tensor, right: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        images = torch.cat((left, right), dim=1)
        scale_list = [16 / self.scale, 8 / self.scale, 4 / self.scale, 2 / self.scale, 1 / self.scale]
        _, _, merged = self.flownet(images, timestep, scale_list)
        return merged[-1]


def resolve_onnx_artifact_paths(config: OnnxExportConfig) -> OnnxArtifactPaths:
    model_dir = config.output_dir / config.model_name
    stem = config.artifact_stem or f"{config.model_name}_{config.shape_mode.value}_opset{config.opset_version}"
    return OnnxArtifactPaths(
        output_dir=model_dir,
        original_path=model_dir / f"{stem}.onnx",
        simplified_path=model_dir / f"{stem}.simplified.onnx",
    )


def dynamic_axes_for_config(config: OnnxExportConfig) -> Mapping[str, Mapping[int, str]] | None:
    if config.shape_mode is OnnxShapeMode.STATIC:
        return None
    return {
        "left": {0: "batch", 2: "height", 3: "width"},
        "right": {0: "batch", 2: "height", 3: "width"},
        "timestep": {0: "batch"},
        "intermediate_frame": {0: "batch", 2: "height", 3: "width"},
    }


def dynamic_shapes_for_config(config: OnnxExportConfig) -> object | None:
    if config.shape_mode is OnnxShapeMode.STATIC:
        return None
    batch = torch.export.Dim("batch", min=1)
    if config.dynamic_hw_multiple is None:
        height = torch.export.Dim("height", min=1)
        width = torch.export.Dim("width", min=1)
    else:
        height = config.dynamic_hw_multiple * torch.export.Dim("height_units", min=1)
        width = config.dynamic_hw_multiple * torch.export.Dim("width_units", min=1)
    return (
        {0: batch, 2: height, 3: width},
        {0: batch, 2: height, 3: width},
        {0: batch},
    )


def create_sample_onnx_inputs(config: OnnxExportConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(config.device)
    batch = config.sample_input_shape[0]
    left = torch.zeros(config.sample_input_shape, dtype=torch.float32, device=device)
    right = torch.ones(config.sample_input_shape, dtype=torch.float32, device=device)
    timestep = torch.full((batch, 1, 1, 1), config.timestep, dtype=torch.float32, device=device)
    return left, right, timestep


def export_torch_module_to_onnx(module: torch.nn.Module, config: OnnxExportConfig) -> OnnxExportResult:
    """Export a project-owned ONNX wrapper and return structured status.

    The function does not perform ONNX Runtime inference. That belongs to the next milestone.
    """
    paths = resolve_onnx_artifact_paths(config)
    dynamic_axes = dynamic_axes_for_config(config)
    dynamic_shapes = dynamic_shapes_for_config(config) if config.exporter is OnnxExporterKind.DYNAMO else None
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    preferred_path: Path | None = None
    checker_status = "not_run"
    checker_error: str | None = None
    simplification_status = "not_requested" if not config.simplify else "not_run"
    simplification_error: str | None = None

    try:
        device = torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise OnnxExportValidationError("CUDA export requested but torch.cuda.is_available() is false.")

        module = module.to(device=device)
        module.eval()
        sample_inputs = create_sample_onnx_inputs(config)
        # Some upstream inference cores lazily create shape-specific buffers/caches
        # during the first forward pass. Initialize them before tracing so the
        # exporter does not see a mutated state_dict during trace validation.
        with torch.no_grad():
            module(*sample_inputs)
        with torch.no_grad():
            if config.exporter is OnnxExporterKind.DYNAMO:
                torch.onnx.export(
                    module,
                    sample_inputs,
                    str(paths.original_path),
                    input_names=list(ONNX_INPUT_NAMES),
                    output_names=list(ONNX_OUTPUT_NAMES),
                    dynamic_shapes=dynamic_shapes,
                    opset_version=config.opset_version,
                    do_constant_folding=True,
                    dynamo=True,
                )
            else:
                torch.onnx.export(
                    module,
                    sample_inputs,
                    str(paths.original_path),
                    input_names=list(ONNX_INPUT_NAMES),
                    output_names=list(ONNX_OUTPUT_NAMES),
                    dynamic_axes=dynamic_axes,
                    opset_version=config.opset_version,
                    do_constant_folding=True,
                    dynamo=False,
                )

        checker_status, checker_error = _check_onnx_model(paths.original_path)
        if checker_status == "failed":
            return _export_result(
                config,
                paths,
                success=False,
                preferred_path=None,
                dynamic_axes=dynamic_axes,
                simplification_status=simplification_status,
                checker_status=checker_status,
                checker_error=checker_error,
                error=f"ONNX checker failed: {checker_error}",
            )

        preferred_path = paths.original_path
        if config.simplify:
            simplification_status, simplification_error = _simplify_onnx_model(
                paths.original_path,
                paths.simplified_path,
            )
            if simplification_status == "succeeded":
                preferred_path = paths.preferred_path_if_simplified

        return _export_result(
            config,
            paths,
            success=True,
            preferred_path=preferred_path,
            dynamic_axes=dynamic_axes,
            simplification_status=simplification_status,
            simplification_error=simplification_error,
            checker_status=checker_status,
            checker_error=checker_error,
        )
    except Exception as exc:  # pragma: no cover - exact exporter exceptions vary by torch/onnx version.
        return _export_result(
            config,
            paths,
            success=False,
            preferred_path=preferred_path,
            dynamic_axes=dynamic_axes,
            simplification_status=simplification_status,
            simplification_error=simplification_error,
            checker_status=checker_status,
            checker_error=checker_error,
            error=_format_exception(exc),
        )


def export_ema_runtime_onnx(runtime: Any, config: OnnxExportConfig) -> OnnxExportResult:
    net = getattr(getattr(runtime, "model", None), "net", None)
    if not isinstance(net, torch.nn.Module):
        raise OnnxExportValidationError("EMA runtime does not expose a torch.nn.Module at runtime.model.net.")
    return export_torch_module_to_onnx(EMAVFIOnnxWrapper(net), config)


def export_rife_runtime_onnx(
    runtime: Any,
    config: OnnxExportConfig,
    *,
    scale: float | None = None,
) -> OnnxExportResult:
    flownet = getattr(getattr(runtime, "model", None), "flownet", None)
    if not isinstance(flownet, torch.nn.Module):
        raise OnnxExportValidationError("Practical-RIFE runtime does not expose a torch.nn.Module at runtime.model.flownet.")
    resolved_scale = validate_rife_scale(scale if scale is not None else getattr(runtime.config, "scale", 1.0))
    return export_torch_module_to_onnx(PracticalRIFEOnnxWrapper(flownet, scale=resolved_scale), config)


def _export_result(
    config: OnnxExportConfig,
    paths: OnnxArtifactPaths,
    *,
    success: bool,
    preferred_path: Path | None,
    dynamic_axes: Mapping[str, Mapping[int, str]] | None,
    simplification_status: str,
    checker_status: str,
    error: str | None = None,
    simplification_error: str | None = None,
    checker_error: str | None = None,
) -> OnnxExportResult:
    return OnnxExportResult(
        success=success,
        model_name=config.model_name,
        shape_mode=config.shape_mode,
        exporter=config.exporter,
        opset_version=config.opset_version,
        sample_input_shape=config.sample_input_shape,
        original_path=paths.original_path,
        simplified_path=paths.simplified_path,
        preferred_path=preferred_path,
        dynamic_axes=dynamic_axes,
        simplify_requested=config.simplify,
        simplification_status=simplification_status,
        checker_status=checker_status,
        error=error,
        simplification_error=simplification_error,
        checker_error=checker_error,
        metadata={"dynamic_hw_multiple": config.dynamic_hw_multiple},
    )


def _check_onnx_model(path: Path) -> tuple[str, str | None]:
    try:
        import onnx
    except ModuleNotFoundError as exc:
        return "skipped", _format_exception(exc)

    try:
        onnx.checker.check_model(str(path))
        return "succeeded", None
    except Exception as exc:  # pragma: no cover - checker messages depend on onnx version.
        return "failed", _format_exception(exc)


def _simplify_onnx_model(input_path: Path, output_path: Path) -> tuple[str, str | None]:
    try:
        import onnx
        from onnxsim import simplify
    except ModuleNotFoundError as exc:
        return "skipped", _format_exception(exc)

    try:
        model = onnx.load(str(input_path))
        simplified_model, check = simplify(model)
        if not check:
            return "failed", "onnx-simplifier returned check=False."
        onnx.save(simplified_model, str(output_path))
        return "succeeded", None
    except Exception as exc:  # pragma: no cover - simplifier support varies by graph/operator.
        return "failed", _format_exception(exc)


def _validate_sample_input_shape(shape: object) -> tuple[int, int, int, int]:
    if isinstance(shape, str | bytes):
        raise OnnxExportValidationError("sample_input_shape must be a B,C,H,W integer sequence.")
    try:
        resolved = tuple(int(value) for value in shape)  # type: ignore[arg-type]
    except TypeError as exc:
        raise OnnxExportValidationError("sample_input_shape must be a B,C,H,W integer sequence.") from exc
    except ValueError as exc:
        raise OnnxExportValidationError("sample_input_shape must contain only integers.") from exc
    if len(resolved) != 4:
        raise OnnxExportValidationError("sample_input_shape must have exactly four dimensions: B,C,H,W.")
    if any(value <= 0 for value in resolved):
        raise OnnxExportValidationError("sample_input_shape dimensions must be positive.")
    if resolved[1] != 3:
        raise OnnxExportValidationError("sample_input_shape channel dimension must be 3.")
    return resolved


def _validate_opset_version(opset_version: object) -> int:
    if isinstance(opset_version, bool) or not isinstance(opset_version, int):
        raise OnnxExportValidationError("opset_version must be an integer.")
    if opset_version < 11:
        raise OnnxExportValidationError("opset_version must be at least 11.")
    return opset_version


def _validate_optional_positive_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise OnnxExportValidationError(f"{label} must be a positive integer.")
    if value <= 0:
        raise OnnxExportValidationError(f"{label} must be a positive integer.")
    return value


def _validate_timestep(timestep: object) -> float:
    if isinstance(timestep, bool) or not isinstance(timestep, int | float):
        raise OnnxExportValidationError("timestep must be numeric.")
    resolved = float(timestep)
    if not 0.0 < resolved < 1.0:
        raise OnnxExportValidationError("timestep must be strictly inside the open interval (0, 1).")
    return resolved


def _coerce_shape_mode(shape_mode: OnnxShapeMode | str) -> OnnxShapeMode:
    if isinstance(shape_mode, OnnxShapeMode):
        return shape_mode
    try:
        return OnnxShapeMode(str(shape_mode))
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in OnnxShapeMode)
        raise OnnxExportValidationError(f"Unsupported ONNX shape mode: {shape_mode!r}. Allowed: {allowed}.") from exc


def _coerce_exporter_kind(exporter: OnnxExporterKind | str) -> OnnxExporterKind:
    if isinstance(exporter, OnnxExporterKind):
        return exporter
    try:
        return OnnxExporterKind(str(exporter))
    except ValueError as exc:
        allowed = ", ".join(kind.value for kind in OnnxExporterKind)
        raise OnnxExportValidationError(f"Unsupported ONNX exporter: {exporter!r}. Allowed: {allowed}.") from exc


def _validate_artifact_name(value: str, label: str) -> str:
    if not value:
        raise OnnxExportValidationError(f"{label} must not be empty.")
    if Path(value).name != value or value in {".", ".."}:
        raise OnnxExportValidationError(f"{label} must be a simple file/directory stem, got {value!r}.")
    return value


def _format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()
