from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
import csv
import json
import traceback

from PIL import Image
import torch

from video_interpolation.image_io import tensor_to_uint8_hwc
from video_interpolation.inference_runtime.api import (
    FramePairRequest,
    FramePairResult,
    InferenceMode,
    RuntimeBackendKind,
)
from video_interpolation.inference_runtime.onnx_export import (
    DEFAULT_ONNX_EXPORT_ROOT,
    DEFAULT_ONNX_OPSET_VERSION,
    OnnxShapeMode,
)

DEFAULT_EQUIVALENCE_ATOL = 1e-3
DEFAULT_EQUIVALENCE_RTOL = 1e-3

FramePairPredictor = Callable[[FramePairRequest], FramePairResult]


@dataclass(frozen=True)
class TensorEquivalenceMetrics:
    mae: float
    max_abs_error: float
    mse: float
    allclose: bool
    atol: float = DEFAULT_EQUIVALENCE_ATOL
    rtol: float = DEFAULT_EQUIVALENCE_RTOL


@dataclass(frozen=True)
class OnnxEquivalenceRecord:
    model_name: str
    input_shape: tuple[int, int, int]
    timestep: float | None
    interpolation_factor: int
    artifact_path: Path
    providers: tuple[str, ...]
    session_providers: tuple[str, ...] = ()
    torch_padded_shape: tuple[int, ...] | None = None
    onnx_padded_shape: tuple[int, ...] | None = None
    torch_output_shape: tuple[int, ...] | None = None
    onnx_output_shape: tuple[int, ...] | None = None
    metrics: TensorEquivalenceMetrics | None = None
    error: str | None = None
    torch_sample_path: Path | None = None
    onnx_sample_path: Path | None = None
    absdiff_sample_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.metrics is not None and self.metrics.allclose


@dataclass(frozen=True)
class OnnxEquivalenceCheckResult:
    model_name: str
    artifact_path: Path
    providers: tuple[str, ...]
    input_shapes: tuple[tuple[int, int, int], ...]
    mode: InferenceMode
    interpolation_factor: int
    atol: float
    rtol: float
    records: tuple[OnnxEquivalenceRecord, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_path", Path(self.artifact_path))
        object.__setattr__(self, "providers", tuple(self.providers))
        object.__setattr__(self, "input_shapes", tuple(tuple(shape) for shape in self.input_shapes))
        object.__setattr__(self, "mode", InferenceMode(self.mode))
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def success(self) -> bool:
        return bool(self.records) and all(record.ok for record in self.records)

    @property
    def max_abs_error(self) -> float | None:
        values = [record.metrics.max_abs_error for record in self.records if record.metrics is not None]
        return max(values) if values else None

    @property
    def mae(self) -> float | None:
        values = [record.metrics.mae for record in self.records if record.metrics is not None]
        return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class OnnxValidationArtifacts:
    report_path: Path
    metrics_csv_path: Path


def compute_tensor_equivalence_metrics(
    torch_tensor: torch.Tensor,
    onnx_tensor: torch.Tensor,
    *,
    atol: float = DEFAULT_EQUIVALENCE_ATOL,
    rtol: float = DEFAULT_EQUIVALENCE_RTOL,
) -> TensorEquivalenceMetrics:
    if tuple(torch_tensor.shape) != tuple(onnx_tensor.shape):
        raise ValueError(
            f"Cannot compare tensors with different shapes: {tuple(torch_tensor.shape)} and {tuple(onnx_tensor.shape)}."
        )
    torch_cpu = torch_tensor.detach().cpu().float()
    onnx_cpu = onnx_tensor.detach().cpu().float()
    diff = torch_cpu - onnx_cpu
    abs_diff = diff.abs()
    return TensorEquivalenceMetrics(
        mae=float(abs_diff.mean().item()),
        max_abs_error=float(abs_diff.max().item()),
        mse=float((diff * diff).mean().item()),
        allclose=bool(torch.allclose(torch_cpu, onnx_cpu, atol=atol, rtol=rtol)),
        atol=atol,
        rtol=rtol,
    )


def run_frame_pair_equivalence_check(
    *,
    model_name: str,
    torch_predictor: FramePairPredictor,
    onnx_predictor: FramePairPredictor,
    artifact_path: Path,
    providers: Sequence[str],
    input_shapes: Sequence[tuple[int, int, int]],
    mode: InferenceMode | str = InferenceMode.FIXED_2X,
    interpolation_factor: int = 2,
    backend_options: Mapping[str, object] | None = None,
    sample_output_root: Path | None = None,
    seed: int = 20260531,
    atol: float = DEFAULT_EQUIVALENCE_ATOL,
    rtol: float = DEFAULT_EQUIVALENCE_RTOL,
) -> OnnxEquivalenceCheckResult:
    resolved_mode = InferenceMode(mode)
    resolved_shapes = tuple(_validate_input_shape(shape) for shape in input_shapes)
    resolved_providers = tuple(providers)
    resolved_backend_options = dict(backend_options or {})
    records: list[OnnxEquivalenceRecord] = []

    for index, input_shape in enumerate(resolved_shapes):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + index)
        left = torch.rand(input_shape, generator=generator, dtype=torch.float32)
        right = torch.rand(input_shape, generator=generator, dtype=torch.float32)

        torch_request = FramePairRequest(
            left=left,
            right=right,
            mode=resolved_mode,
            interpolation_factor=interpolation_factor,
            backend_kind=RuntimeBackendKind.TORCH,
            backend_options=resolved_backend_options,
        )
        onnx_request = FramePairRequest(
            left=left,
            right=right,
            mode=resolved_mode,
            interpolation_factor=interpolation_factor,
            backend_kind=RuntimeBackendKind.ONNX,
            backend_options=resolved_backend_options,
        )
        try:
            torch_result = torch_predictor(torch_request)
            onnx_result = onnx_predictor(onnx_request)
            if len(torch_result.intermediate_frames) != len(onnx_result.intermediate_frames):
                raise ValueError(
                    "PyTorch and ONNX result frame counts differ: "
                    f"{len(torch_result.intermediate_frames)} vs {len(onnx_result.intermediate_frames)}."
                )
            session_providers = tuple(str(value) for value in onnx_result.metadata.get("session_providers", ()))
            for timestep, torch_frame, onnx_frame in zip(
                torch_result.timesteps,
                torch_result.intermediate_frames,
                onnx_result.intermediate_frames,
                strict=True,
            ):
                metrics = compute_tensor_equivalence_metrics(
                    torch_frame,
                    onnx_frame,
                    atol=atol,
                    rtol=rtol,
                )
                sample_paths = (None, None, None)
                if sample_output_root is not None and not metrics.allclose:
                    sample_paths = _write_sample_images(
                        sample_output_root,
                        model_name=model_name,
                        input_shape=input_shape,
                        timestep=timestep,
                        torch_frame=torch_frame,
                        onnx_frame=onnx_frame,
                    )
                records.append(
                    OnnxEquivalenceRecord(
                        model_name=model_name,
                        input_shape=input_shape,
                        timestep=timestep,
                        interpolation_factor=interpolation_factor,
                        artifact_path=artifact_path,
                        providers=resolved_providers,
                        session_providers=session_providers,
                        torch_padded_shape=(
                            None if torch_result.padded_shape is None else tuple(int(value) for value in torch_result.padded_shape)
                        ),
                        onnx_padded_shape=(
                            None if onnx_result.padded_shape is None else tuple(int(value) for value in onnx_result.padded_shape)
                        ),
                        torch_output_shape=tuple(int(value) for value in torch_frame.shape),
                        onnx_output_shape=tuple(int(value) for value in onnx_frame.shape),
                        metrics=metrics,
                        torch_sample_path=sample_paths[0],
                        onnx_sample_path=sample_paths[1],
                        absdiff_sample_path=sample_paths[2],
                    )
                )
        except Exception as exc:  # pragma: no cover - real ORT/model errors vary by graph/provider.
            records.append(
                OnnxEquivalenceRecord(
                    model_name=model_name,
                    input_shape=input_shape,
                    timestep=None,
                    interpolation_factor=interpolation_factor,
                    artifact_path=artifact_path,
                    providers=resolved_providers,
                    error=_format_exception(exc),
                )
            )

    return OnnxEquivalenceCheckResult(
        model_name=model_name,
        artifact_path=artifact_path,
        providers=resolved_providers,
        input_shapes=resolved_shapes,
        mode=resolved_mode,
        interpolation_factor=interpolation_factor,
        atol=atol,
        rtol=rtol,
        records=tuple(records),
        metadata={"seed": seed, "artifact_io": _describe_onnx_model_io(artifact_path)},
    )


def write_onnx_equivalence_report(
    result: OnnxEquivalenceCheckResult,
    output_root: Path = Path("outputs/onnx_validation"),
) -> OnnxValidationArtifacts:
    output_dir = Path(output_root).expanduser().resolve() / result.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "equivalence_report.json"
    metrics_csv_path = output_dir / "equivalence_metrics.csv"

    report_path.write_text(json.dumps(_result_to_json_dict(result), indent=2, sort_keys=True) + "\n")
    with metrics_csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model_name",
                "input_shape",
                "timestep",
                "interpolation_factor",
                "artifact_path",
                "providers",
                "session_providers",
                "torch_padded_shape",
                "onnx_padded_shape",
                "torch_output_shape",
                "onnx_output_shape",
                "mae",
                "max_abs_error",
                "mse",
                "allclose",
                "atol",
                "rtol",
                "error",
                "torch_sample_path",
                "onnx_sample_path",
                "absdiff_sample_path",
            ],
        )
        writer.writeheader()
        for record in result.records:
            writer.writerow(_record_to_csv_dict(record))
    return OnnxValidationArtifacts(report_path=report_path, metrics_csv_path=metrics_csv_path)


def resolve_preferred_onnx_artifact_path(
    model_name: str,
    *,
    artifact_root: Path = DEFAULT_ONNX_EXPORT_ROOT,
    artifact_path: Path | None = None,
    prefer_simplified: bool = True,
    shape_mode: OnnxShapeMode | str = OnnxShapeMode.DYNAMIC_HW,
    opset_version: int = DEFAULT_ONNX_OPSET_VERSION,
) -> Path:
    if artifact_path is not None:
        resolved = Path(artifact_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"ONNX artifact does not exist: {resolved}")
        return resolved

    resolved_shape_mode = OnnxShapeMode(shape_mode)
    stem = f"{model_name}_{resolved_shape_mode.value}_opset{opset_version}"
    model_dir = Path(artifact_root).expanduser().resolve() / model_name
    original_path = model_dir / f"{stem}.onnx"
    simplified_path = model_dir / f"{stem}.simplified.onnx"
    candidates = (simplified_path, original_path) if prefer_simplified else (original_path, simplified_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No ONNX artifact found for {model_name} under {model_dir}. "
        f"Checked: {simplified_path}, {original_path}."
    )


def _validate_input_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    resolved = tuple(int(value) for value in shape)
    if len(resolved) != 3:
        raise ValueError("Input shape must be C,H,W.")
    if resolved[0] != 3:
        raise ValueError("Input shape channel dimension must be 3.")
    if any(value <= 0 for value in resolved):
        raise ValueError("Input shape dimensions must be positive.")
    return resolved


def _describe_onnx_model_io(path: Path) -> dict[str, object]:
    try:
        import onnx
    except ModuleNotFoundError as exc:
        return {"status": "skipped", "error": _format_exception(exc)}
    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:  # pragma: no cover - exact ONNX load errors vary by artifact.
        return {"status": "failed", "error": _format_exception(exc)}
    return {
        "status": "ok",
        "inputs": [_describe_value_info(value) for value in model.graph.input],
        "outputs": [_describe_value_info(value) for value in model.graph.output],
    }


def _describe_value_info(value_info: object) -> dict[str, object]:
    tensor_type = value_info.type.tensor_type  # type: ignore[attr-defined]
    dims: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append(None)
    return {"name": value_info.name, "dims": dims}


def _result_to_json_dict(result: OnnxEquivalenceCheckResult) -> dict[str, object]:
    return {
        "success": result.success,
        "model_name": result.model_name,
        "artifact_path": str(result.artifact_path),
        "providers": list(result.providers),
        "input_shapes": ["x".join(str(value) for value in shape) for shape in result.input_shapes],
        "mode": result.mode.value,
        "interpolation_factor": result.interpolation_factor,
        "atol": result.atol,
        "rtol": result.rtol,
        "mae": result.mae,
        "max_abs_error": result.max_abs_error,
        "metadata": dict(result.metadata),
        "records": [_record_to_json_dict(record) for record in result.records],
    }


def _record_to_json_dict(record: OnnxEquivalenceRecord) -> dict[str, object]:
    data = {
        "ok": record.ok,
        "model_name": record.model_name,
        "input_shape": "x".join(str(value) for value in record.input_shape),
        "timestep": record.timestep,
        "interpolation_factor": record.interpolation_factor,
        "artifact_path": str(record.artifact_path),
        "providers": list(record.providers),
        "session_providers": list(record.session_providers),
        "torch_padded_shape": None if record.torch_padded_shape is None else "x".join(str(value) for value in record.torch_padded_shape),
        "onnx_padded_shape": None if record.onnx_padded_shape is None else "x".join(str(value) for value in record.onnx_padded_shape),
        "torch_output_shape": None if record.torch_output_shape is None else "x".join(str(value) for value in record.torch_output_shape),
        "onnx_output_shape": None if record.onnx_output_shape is None else "x".join(str(value) for value in record.onnx_output_shape),
        "error": record.error,
        "torch_sample_path": None if record.torch_sample_path is None else str(record.torch_sample_path),
        "onnx_sample_path": None if record.onnx_sample_path is None else str(record.onnx_sample_path),
        "absdiff_sample_path": None if record.absdiff_sample_path is None else str(record.absdiff_sample_path),
    }
    if record.metrics is not None:
        data.update(
            {
                "mae": record.metrics.mae,
                "max_abs_error": record.metrics.max_abs_error,
                "mse": record.metrics.mse,
                "allclose": record.metrics.allclose,
                "atol": record.metrics.atol,
                "rtol": record.metrics.rtol,
            }
        )
    return data


def _record_to_csv_dict(record: OnnxEquivalenceRecord) -> dict[str, object]:
    metrics = record.metrics
    return {
        "model_name": record.model_name,
        "input_shape": "x".join(str(value) for value in record.input_shape),
        "timestep": "" if record.timestep is None else f"{record.timestep:.8g}",
        "interpolation_factor": record.interpolation_factor,
        "artifact_path": str(record.artifact_path),
        "providers": ";".join(record.providers),
        "session_providers": ";".join(record.session_providers),
        "torch_padded_shape": "" if record.torch_padded_shape is None else "x".join(str(value) for value in record.torch_padded_shape),
        "onnx_padded_shape": "" if record.onnx_padded_shape is None else "x".join(str(value) for value in record.onnx_padded_shape),
        "torch_output_shape": "" if record.torch_output_shape is None else "x".join(str(value) for value in record.torch_output_shape),
        "onnx_output_shape": "" if record.onnx_output_shape is None else "x".join(str(value) for value in record.onnx_output_shape),
        "mae": "" if metrics is None else f"{metrics.mae:.10g}",
        "max_abs_error": "" if metrics is None else f"{metrics.max_abs_error:.10g}",
        "mse": "" if metrics is None else f"{metrics.mse:.10g}",
        "allclose": "" if metrics is None else str(metrics.allclose),
        "atol": "" if metrics is None else f"{metrics.atol:.10g}",
        "rtol": "" if metrics is None else f"{metrics.rtol:.10g}",
        "error": record.error or "",
        "torch_sample_path": "" if record.torch_sample_path is None else str(record.torch_sample_path),
        "onnx_sample_path": "" if record.onnx_sample_path is None else str(record.onnx_sample_path),
        "absdiff_sample_path": "" if record.absdiff_sample_path is None else str(record.absdiff_sample_path),
    }


def _write_sample_images(
    output_root: Path,
    *,
    model_name: str,
    input_shape: tuple[int, int, int],
    timestep: float,
    torch_frame: torch.Tensor,
    onnx_frame: torch.Tensor,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_root).expanduser().resolve() / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    shape_label = "x".join(str(value) for value in input_shape)
    timestep_label = f"{timestep:.6g}".replace(".", "_")
    stem = f"shape_{shape_label}_t{timestep_label}"
    torch_path = output_dir / f"{stem}_torch.png"
    onnx_path = output_dir / f"{stem}_onnx.png"
    absdiff_path = output_dir / f"{stem}_absdiff_normalized.png"

    Image.fromarray(tensor_to_uint8_hwc(torch_frame)).save(torch_path)
    Image.fromarray(tensor_to_uint8_hwc(onnx_frame)).save(onnx_path)
    absdiff = (torch_frame.detach().cpu().float() - onnx_frame.detach().cpu().float()).abs()
    max_value = float(absdiff.max().item())
    if max_value > 0.0:
        absdiff = absdiff / max_value
    Image.fromarray(tensor_to_uint8_hwc(absdiff)).save(absdiff_path)
    return torch_path, onnx_path, absdiff_path


def _format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()
