from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
import csv
import json
import traceback

import numpy as np
from PIL import Image
import torch

from video_interpolation.image_io import tensor_to_uint8_hwc, uint8_hwc_to_tensor
from video_interpolation.inference_runtime.api import (
    FramePairRequest,
    FramePairResult,
    InferenceMode,
    RuntimeBackendKind,
)
from video_interpolation.inference_runtime.onnx_export import (
    DEFAULT_ONNX_EXPORT_ROOT,
    DEFAULT_ONNX_OPSET_VERSION,
    OnnxExporterKind,
    OnnxShapeMode,
    default_onnx_artifact_stem,
)
from video_interpolation.metrics import compute_psnr, compute_ssim

DEFAULT_EQUIVALENCE_ATOL = 1e-3
DEFAULT_EQUIVALENCE_RTOL = 1e-3

FramePairPredictor = Callable[[FramePairRequest], FramePairResult]


@dataclass(frozen=True)
class TensorEquivalenceMetrics:
    mae: float
    max_abs_error: float
    mse: float
    allclose: bool
    psnr: float | None = None
    ssim: float | None = None
    atol: float = DEFAULT_EQUIVALENCE_ATOL
    rtol: float = DEFAULT_EQUIVALENCE_RTOL


@dataclass(frozen=True)
class RealImagePair:
    pair_id: str
    pair_dir: Path
    left_path: Path
    right_path: Path
    input_shape: tuple[int, int, int]


@dataclass(frozen=True)
class OnnxEquivalenceRecord:
    model_name: str
    input_shape: tuple[int, int, int]
    timestep: float | None
    interpolation_factor: int
    artifact_path: Path
    providers: tuple[str, ...]
    input_group: str = "synthetic"
    pair_id: str | None = None
    left_input_path: Path | None = None
    right_input_path: Path | None = None
    session_providers: tuple[str, ...] = ()
    torch_padded_shape: tuple[int, ...] | None = None
    onnx_padded_shape: tuple[int, ...] | None = None
    torch_output_shape: tuple[int, ...] | None = None
    onnx_output_shape: tuple[int, ...] | None = None
    metrics: TensorEquivalenceMetrics | None = None
    error: str | None = None
    left_sample_path: Path | None = None
    right_sample_path: Path | None = None
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

    @property
    def mse(self) -> float | None:
        values = [record.metrics.mse for record in self.records if record.metrics is not None]
        return sum(values) / len(values) if values else None

    @property
    def psnr(self) -> float | None:
        values = [record.metrics.psnr for record in self.records if record.metrics is not None and record.metrics.psnr is not None]
        return sum(values) / len(values) if values else None

    @property
    def ssim(self) -> float | None:
        values = [record.metrics.ssim for record in self.records if record.metrics is not None and record.metrics.ssim is not None]
        return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class OnnxValidationArtifacts:
    report_path: Path
    metrics_csv_path: Path


def compute_tensor_equivalence_metrics(
    torch_tensor: torch.Tensor,
    onnx_tensor: torch.Tensor,
    *,
    include_image_metrics: bool = False,
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
    psnr = compute_psnr(onnx_cpu, torch_cpu) if include_image_metrics else None
    ssim = compute_ssim(onnx_cpu, torch_cpu) if include_image_metrics else None
    return TensorEquivalenceMetrics(
        mae=float(abs_diff.mean().item()),
        max_abs_error=float(abs_diff.max().item()),
        mse=float((diff * diff).mean().item()),
        allclose=bool(torch.allclose(torch_cpu, onnx_cpu, atol=atol, rtol=rtol)),
        psnr=psnr,
        ssim=ssim,
        atol=atol,
        rtol=rtol,
    )


def discover_real_image_pairs(pair_root: Path, *, limit: int | None = None) -> tuple[RealImagePair, ...]:
    root = Path(pair_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Real-image pair root does not exist or is not a directory: {root}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided.")

    pairs: list[RealImagePair] = []
    for pair_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        left_path = pair_dir / "frame1.png"
        right_path = pair_dir / "frame2.png"
        missing = [path.name for path in (left_path, right_path) if not path.is_file()]
        if missing:
            raise ValueError(f"Malformed real-image pair {pair_dir}: missing {', '.join(missing)}.")
        left_shape = _image_file_shape(left_path)
        right_shape = _image_file_shape(right_path)
        if left_shape != right_shape:
            raise ValueError(
                f"Malformed real-image pair {pair_dir}: frame1 shape {left_shape} differs from frame2 shape {right_shape}."
            )
        pairs.append(
            RealImagePair(
                pair_id=pair_dir.name,
                pair_dir=pair_dir,
                left_path=left_path,
                right_path=right_path,
                input_shape=left_shape,
            )
        )
        if limit is not None and len(pairs) >= limit:
            break
    if not pairs:
        raise ValueError(f"No real-image pair directories found under {root}.")
    return tuple(pairs)


def load_real_image_pair(pair: RealImagePair) -> tuple[torch.Tensor, torch.Tensor]:
    left = _load_rgb_tensor(pair.left_path)
    right = _load_rgb_tensor(pair.right_path)
    if tuple(left.shape) != pair.input_shape:
        raise ValueError(f"Loaded frame1 shape {tuple(left.shape)} does not match discovered shape {pair.input_shape}.")
    if tuple(right.shape) != pair.input_shape:
        raise ValueError(f"Loaded frame2 shape {tuple(right.shape)} does not match discovered shape {pair.input_shape}.")
    return left, right


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
    metadata: Mapping[str, object] | None = None,
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

    result_metadata = {
        "seed": seed,
        "input_group": "synthetic",
        "artifact_io": _describe_onnx_model_io(artifact_path),
        **dict(metadata or {}),
    }
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
        metadata=result_metadata,
    )


def run_real_pair_equivalence_check(
    *,
    model_name: str,
    torch_predictor: FramePairPredictor,
    onnx_predictor: FramePairPredictor,
    artifact_path: Path,
    providers: Sequence[str],
    pairs: Sequence[RealImagePair],
    mode: InferenceMode | str = InferenceMode.FIXED_2X,
    interpolation_factor: int = 2,
    backend_options: Mapping[str, object] | None = None,
    sample_output_root: Path | None = None,
    atol: float = DEFAULT_EQUIVALENCE_ATOL,
    rtol: float = DEFAULT_EQUIVALENCE_RTOL,
    metadata: Mapping[str, object] | None = None,
) -> OnnxEquivalenceCheckResult:
    resolved_mode = InferenceMode(mode)
    resolved_pairs = tuple(pairs)
    if not resolved_pairs:
        raise ValueError("At least one real-image pair is required.")
    resolved_providers = tuple(providers)
    resolved_backend_options = dict(backend_options or {})
    records: list[OnnxEquivalenceRecord] = []

    for pair in resolved_pairs:
        try:
            left, right = load_real_image_pair(pair)
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
            torch_result = torch_predictor(torch_request)
            onnx_result = onnx_predictor(onnx_request)
            if len(torch_result.intermediate_frames) != len(onnx_result.intermediate_frames):
                raise ValueError(
                    "PyTorch and ONNX result frame counts differ: "
                    f"{len(torch_result.intermediate_frames)} vs {len(onnx_result.intermediate_frames)}."
                )
            session_providers = tuple(str(value) for value in onnx_result.metadata.get("session_providers", ()))
            single_timestep = len(torch_result.timesteps) == 1
            for timestep, torch_frame, onnx_frame in zip(
                torch_result.timesteps,
                torch_result.intermediate_frames,
                onnx_result.intermediate_frames,
                strict=True,
            ):
                metrics = compute_tensor_equivalence_metrics(
                    torch_frame,
                    onnx_frame,
                    include_image_metrics=True,
                    atol=atol,
                    rtol=rtol,
                )
                sample_paths = (None, None, None, None, None)
                if sample_output_root is not None:
                    sample_paths = _write_real_pair_images(
                        sample_output_root,
                        model_name=model_name,
                        providers=resolved_providers,
                        pair_id=pair.pair_id,
                        left=left,
                        right=right,
                        timestep=timestep,
                        torch_frame=torch_frame,
                        onnx_frame=onnx_frame,
                        single_timestep=single_timestep,
                    )
                records.append(
                    OnnxEquivalenceRecord(
                        model_name=model_name,
                        input_shape=pair.input_shape,
                        timestep=timestep,
                        interpolation_factor=interpolation_factor,
                        artifact_path=artifact_path,
                        providers=resolved_providers,
                        input_group="real_pairs",
                        pair_id=pair.pair_id,
                        left_input_path=pair.left_path,
                        right_input_path=pair.right_path,
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
                        left_sample_path=sample_paths[0],
                        right_sample_path=sample_paths[1],
                        torch_sample_path=sample_paths[2],
                        onnx_sample_path=sample_paths[3],
                        absdiff_sample_path=sample_paths[4],
                    )
                )
        except Exception as exc:  # pragma: no cover - real ORT/model errors vary by graph/provider.
            records.append(
                OnnxEquivalenceRecord(
                    model_name=model_name,
                    input_shape=pair.input_shape,
                    timestep=None,
                    interpolation_factor=interpolation_factor,
                    artifact_path=artifact_path,
                    providers=resolved_providers,
                    input_group="real_pairs",
                    pair_id=pair.pair_id,
                    left_input_path=pair.left_path,
                    right_input_path=pair.right_path,
                    error=_format_exception(exc),
                )
            )

    result_metadata = {
        "input_group": "real_pairs",
        "pair_root": str(Path(resolved_pairs[0].pair_dir).parent),
        "pair_ids": [pair.pair_id for pair in resolved_pairs],
        "artifact_io": _describe_onnx_model_io(artifact_path),
        **dict(metadata or {}),
    }
    return OnnxEquivalenceCheckResult(
        model_name=model_name,
        artifact_path=artifact_path,
        providers=resolved_providers,
        input_shapes=tuple(pair.input_shape for pair in resolved_pairs),
        mode=resolved_mode,
        interpolation_factor=interpolation_factor,
        atol=atol,
        rtol=rtol,
        records=tuple(records),
        metadata=result_metadata,
    )


def write_onnx_equivalence_report(
    result: OnnxEquivalenceCheckResult,
    output_root: Path = Path("outputs/onnx_validation"),
) -> OnnxValidationArtifacts:
    output_dir = Path(output_root).expanduser().resolve() / result.model_name
    if result.metadata.get("input_group") == "real_pairs":
        output_dir = output_dir / _provider_label(result.providers)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "equivalence_report.json"
    metrics_csv_path = output_dir / "equivalence_metrics.csv"

    report_path.write_text(json.dumps(_result_to_json_dict(result), indent=2, sort_keys=True) + "\n")
    with metrics_csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model_name",
                "input_group",
                "pair_id",
                "input_shape",
                "timestep",
                "interpolation_factor",
                "artifact_path",
                "left_input_path",
                "right_input_path",
                "providers",
                "session_providers",
                "torch_padded_shape",
                "onnx_padded_shape",
                "torch_output_shape",
                "onnx_output_shape",
                "mae",
                "max_abs_error",
                "mse",
                "psnr",
                "ssim",
                "allclose",
                "atol",
                "rtol",
                "error",
                "left_sample_path",
                "right_sample_path",
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
    artifact_exporter: OnnxExporterKind | str = OnnxExporterKind.DYNAMO,
) -> Path:
    if artifact_path is not None:
        resolved = Path(artifact_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"ONNX artifact does not exist: {resolved}")
        return resolved

    resolved_shape_mode = OnnxShapeMode(shape_mode)
    resolved_exporter = OnnxExporterKind(artifact_exporter)
    stem = default_onnx_artifact_stem(
        model_name,
        shape_mode=resolved_shape_mode,
        opset_version=opset_version,
        exporter=resolved_exporter,
    )
    model_dir = Path(artifact_root).expanduser().resolve() / model_name
    original_path = model_dir / f"{stem}.onnx"
    simplified_path = model_dir / f"{stem}.simplified.onnx"
    if resolved_exporter is OnnxExporterKind.DYNAMO:
        dynamic_batch_stem = f"{model_name}_{resolved_exporter.value}_dynamic_batch_hw_opset{opset_version}"
        dynamic_batch_path = model_dir / f"{dynamic_batch_stem}.onnx"
        dynamic_batch_candidates = tuple(
            sorted(
                path
                for path in model_dir.glob(f"{dynamic_batch_stem}*.onnx")
                if path.is_file() and ".simplified." not in path.name
            )
        )
        candidates = (dynamic_batch_path, *dynamic_batch_candidates, original_path)
    else:
        candidates = (simplified_path, original_path) if prefer_simplified else (original_path, simplified_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if resolved_exporter is OnnxExporterKind.DYNAMO:
        dynamo_candidates = sorted(
            path
            for path in model_dir.glob(f"{stem}*.onnx")
            if path.is_file() and ".simplified." not in path.name
        )
        if len(dynamo_candidates) == 1:
            return dynamo_candidates[0]
    current_candidates = sorted(path for path in model_dir.glob("*.onnx") if path.is_file())
    if len(current_candidates) == 1:
        return current_candidates[0]
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
        "mse": result.mse,
        "psnr": result.psnr,
        "ssim": result.ssim,
        "aggregate_metrics": {
            "count": sum(1 for record in result.records if record.metrics is not None),
            "mae_mean": result.mae,
            "max_abs_error": result.max_abs_error,
            "mse_mean": result.mse,
            "psnr_mean": result.psnr,
            "ssim_mean": result.ssim,
        },
        "metadata": dict(result.metadata),
        "records": [_record_to_json_dict(record) for record in result.records],
    }


def _record_to_json_dict(record: OnnxEquivalenceRecord) -> dict[str, object]:
    data = {
        "ok": record.ok,
        "model_name": record.model_name,
        "input_group": record.input_group,
        "pair_id": record.pair_id,
        "input_shape": "x".join(str(value) for value in record.input_shape),
        "timestep": record.timestep,
        "interpolation_factor": record.interpolation_factor,
        "artifact_path": str(record.artifact_path),
        "left_input_path": None if record.left_input_path is None else str(record.left_input_path),
        "right_input_path": None if record.right_input_path is None else str(record.right_input_path),
        "providers": list(record.providers),
        "session_providers": list(record.session_providers),
        "torch_padded_shape": None if record.torch_padded_shape is None else "x".join(str(value) for value in record.torch_padded_shape),
        "onnx_padded_shape": None if record.onnx_padded_shape is None else "x".join(str(value) for value in record.onnx_padded_shape),
        "torch_output_shape": None if record.torch_output_shape is None else "x".join(str(value) for value in record.torch_output_shape),
        "onnx_output_shape": None if record.onnx_output_shape is None else "x".join(str(value) for value in record.onnx_output_shape),
        "error": record.error,
        "left_sample_path": None if record.left_sample_path is None else str(record.left_sample_path),
        "right_sample_path": None if record.right_sample_path is None else str(record.right_sample_path),
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
                "psnr": record.metrics.psnr,
                "ssim": record.metrics.ssim,
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
        "input_group": record.input_group,
        "pair_id": record.pair_id or "",
        "input_shape": "x".join(str(value) for value in record.input_shape),
        "timestep": "" if record.timestep is None else f"{record.timestep:.8g}",
        "interpolation_factor": record.interpolation_factor,
        "artifact_path": str(record.artifact_path),
        "left_input_path": "" if record.left_input_path is None else str(record.left_input_path),
        "right_input_path": "" if record.right_input_path is None else str(record.right_input_path),
        "providers": ";".join(record.providers),
        "session_providers": ";".join(record.session_providers),
        "torch_padded_shape": "" if record.torch_padded_shape is None else "x".join(str(value) for value in record.torch_padded_shape),
        "onnx_padded_shape": "" if record.onnx_padded_shape is None else "x".join(str(value) for value in record.onnx_padded_shape),
        "torch_output_shape": "" if record.torch_output_shape is None else "x".join(str(value) for value in record.torch_output_shape),
        "onnx_output_shape": "" if record.onnx_output_shape is None else "x".join(str(value) for value in record.onnx_output_shape),
        "mae": "" if metrics is None else f"{metrics.mae:.10g}",
        "max_abs_error": "" if metrics is None else f"{metrics.max_abs_error:.10g}",
        "mse": "" if metrics is None else f"{metrics.mse:.10g}",
        "psnr": "" if metrics is None or metrics.psnr is None else f"{metrics.psnr:.10g}",
        "ssim": "" if metrics is None or metrics.ssim is None else f"{metrics.ssim:.10g}",
        "allclose": "" if metrics is None else str(metrics.allclose),
        "atol": "" if metrics is None else f"{metrics.atol:.10g}",
        "rtol": "" if metrics is None else f"{metrics.rtol:.10g}",
        "error": record.error or "",
        "left_sample_path": "" if record.left_sample_path is None else str(record.left_sample_path),
        "right_sample_path": "" if record.right_sample_path is None else str(record.right_sample_path),
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


def _write_real_pair_images(
    output_root: Path,
    *,
    model_name: str,
    providers: Sequence[str],
    pair_id: str,
    left: torch.Tensor,
    right: torch.Tensor,
    timestep: float,
    torch_frame: torch.Tensor,
    onnx_frame: torch.Tensor,
    single_timestep: bool,
) -> tuple[Path, Path, Path, Path, Path]:
    output_dir = Path(output_root).expanduser().resolve() / model_name / _provider_label(providers) / pair_id
    output_dir.mkdir(parents=True, exist_ok=True)
    left_path = output_dir / "left.png"
    right_path = output_dir / "right.png"
    timestep_label = f"{timestep:.6g}".replace(".", "_")
    suffix = "" if single_timestep else f"_t{timestep_label}"
    torch_path = output_dir / f"pytorch_generated{suffix}.png"
    onnx_path = output_dir / f"onnx_generated{suffix}.png"
    absdiff_path = output_dir / f"absdiff{suffix}.png"

    Image.fromarray(tensor_to_uint8_hwc(left)).save(left_path)
    Image.fromarray(tensor_to_uint8_hwc(right)).save(right_path)
    Image.fromarray(tensor_to_uint8_hwc(torch_frame)).save(torch_path)
    Image.fromarray(tensor_to_uint8_hwc(onnx_frame)).save(onnx_path)
    absdiff = (torch_frame.detach().cpu().float() - onnx_frame.detach().cpu().float()).abs()
    max_value = float(absdiff.max().item())
    if max_value > 0.0:
        absdiff = absdiff / max_value
    Image.fromarray(tensor_to_uint8_hwc(absdiff)).save(absdiff_path)
    return left_path, right_path, torch_path, onnx_path, absdiff_path


def _provider_label(providers: Sequence[str]) -> str:
    if not providers:
        return "provider_default"
    return "_".join(_safe_label(str(provider).replace("ExecutionProvider", "")) for provider in providers)


def _safe_label(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in safe.split("_") if part) or "unknown"


def _image_file_shape(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
    return (3, height, width)


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        array = np.asarray(rgb_image.copy())
    return uint8_hwc_to_tensor(array)


def _format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()
