from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from video_interpolation.inference_runtime import FramePairRequest, InferenceMode, RuntimeBackendKind
from video_interpolation.inference_runtime.api import FramePairResult, ModelBatchRequest
from video_interpolation.inference_runtime.backends.base import RuntimeBackendError
from video_interpolation.inference_runtime.backends.onnx import (
    OnnxRuntimeBackend,
    OnnxRuntimeBackendConfig,
    resolve_onnx_providers,
)
from video_interpolation.inference_runtime.ema import EMAVFIOnnxRuntime, EMAVFIOnnxRuntimeConfig
from video_interpolation.inference_runtime.onnx_export import (
    OnnxExportConfig,
    OnnxExporterKind,
    OnnxShapeMode,
    export_torch_module_to_onnx,
)
from video_interpolation.inference_runtime.onnx_validation import (
    compute_tensor_equivalence_metrics,
    discover_real_image_pairs,
    run_real_pair_equivalence_check,
    resolve_preferred_onnx_artifact_path,
    run_frame_pair_equivalence_check,
    write_onnx_equivalence_report,
)
from video_interpolation.inference_runtime.rife import PracticalRIFEOnnxRuntime, PracticalRIFEOnnxRuntimeConfig


def test_resolve_preferred_onnx_artifact_path_prefers_simplified(tmp_path: Path) -> None:
    model_dir = tmp_path / "unit_model"
    model_dir.mkdir()
    original = model_dir / "unit_model_dynamic_hw_opset17.onnx"
    simplified = model_dir / "unit_model_dynamic_hw_opset17.simplified.onnx"
    original.write_bytes(b"original")
    simplified.write_bytes(b"simplified")

    assert (
        resolve_preferred_onnx_artifact_path(
            "unit_model",
            artifact_root=tmp_path,
            artifact_exporter="legacy",
            opset_version=17,
        )
        == simplified
    )
    assert (
        resolve_preferred_onnx_artifact_path(
            "unit_model",
            artifact_root=tmp_path,
            prefer_simplified=False,
            artifact_exporter="legacy",
            opset_version=17,
        )
        == original
    )


def test_resolve_preferred_onnx_artifact_path_prefers_dynamo_by_default(tmp_path: Path) -> None:
    model_dir = tmp_path / "unit_model"
    model_dir.mkdir()
    (model_dir / "unit_model_dynamic_hw_opset17.onnx").write_bytes(b"legacy")
    (model_dir / "unit_model_dynamic_hw_opset17.simplified.onnx").write_bytes(b"legacy-simplified")
    dynamo = model_dir / "unit_model_dynamo_dynamic_hw_opset18.onnx"
    dynamo.write_bytes(b"dynamo")

    assert resolve_preferred_onnx_artifact_path("unit_model", artifact_root=tmp_path) == dynamo


def test_resolve_preferred_onnx_artifact_path_accepts_custom_dynamo_stem(tmp_path: Path) -> None:
    model_dir = tmp_path / "unit_model"
    model_dir.mkdir()
    current = model_dir / "unit_model_dynamo_dynamic_hw_opset18_h128w256.onnx"
    current.write_bytes(b"current")

    assert resolve_preferred_onnx_artifact_path("unit_model", artifact_root=tmp_path) == current


def test_resolve_preferred_onnx_artifact_path_prefers_dynamic_batch_dynamo(tmp_path: Path) -> None:
    model_dir = tmp_path / "unit_model"
    model_dir.mkdir()
    (model_dir / "unit_model_dynamo_dynamic_hw_opset18_h128w256.onnx").write_bytes(b"old")
    current = model_dir / "unit_model_dynamo_dynamic_batch_hw_opset18_h128w256.onnx"
    current.write_bytes(b"current")

    assert resolve_preferred_onnx_artifact_path("unit_model", artifact_root=tmp_path) == current


def test_resolve_preferred_onnx_artifact_path_reports_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No ONNX artifact found"):
        resolve_preferred_onnx_artifact_path("missing_model", artifact_root=tmp_path)


def test_resolve_preferred_onnx_artifact_path_accepts_single_current_artifact(tmp_path: Path) -> None:
    model_dir = tmp_path / "unit_model"
    model_dir.mkdir()
    current = model_dir / "unit_model_dynamo_dynamic_hw_opset18_h336w560.onnx"
    current.write_bytes(b"current")

    assert resolve_preferred_onnx_artifact_path("unit_model", artifact_root=tmp_path) == current


def test_resolve_onnx_providers_accepts_cpu_alias_and_rejects_unavailable_provider() -> None:
    assert resolve_onnx_providers("cpu") == ("CPUExecutionProvider",)

    with pytest.raises(RuntimeBackendError, match="unavailable"):
        resolve_onnx_providers(("DefinitelyMissingProvider",))


def test_onnx_runtime_backend_runs_tiny_exported_model(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    export_result = export_torch_module_to_onnx(
        _TinyOnnxModule(),
        OnnxExportConfig(
            model_name="unit_model",
            output_dir=tmp_path,
            sample_input_shape=(1, 3, 4, 4),
            simplify=False,
        ),
    )
    assert export_result.success

    backend = OnnxRuntimeBackend(
        OnnxRuntimeBackendConfig(
            artifact_path=export_result.preferred_path or export_result.original_path,
            providers=("CPUExecutionProvider",),
        )
    )
    left = torch.zeros(1, 3, 4, 4)
    right = torch.ones(1, 3, 4, 4)
    timestep = torch.full((1, 1, 1, 1), 0.25)

    backend.load()
    assert backend.input_shapes["left"][0] != 1
    assert backend.input_shapes["timestep"][0] != 1
    assert backend.output_shapes["intermediate_frame"][0] != 1
    output = backend.run(
        _runtime_inputs(left=left, right=right, timestep=timestep)
    )
    assert torch.allclose(output.primary_tensor, torch.full_like(left, 0.25), atol=1e-5)
    assert output.metadata["session_providers"] == ("CPUExecutionProvider",)
    assert output.metadata["input_shapes"]["left"] == backend.input_shapes["left"]
    batched_output = backend.run(
        _runtime_inputs(
            left=torch.zeros(2, 3, 4, 4),
            right=torch.ones(2, 3, 4, 4),
            timestep=torch.tensor((0.25, 0.75), dtype=torch.float32).reshape(2, 1, 1, 1),
        )
    )
    assert batched_output.primary_tensor.shape == (2, 3, 4, 4)
    assert float(batched_output.primary_tensor[0].mean()) == pytest.approx(0.25, abs=1e-5)
    assert float(batched_output.primary_tensor[1].mean()) == pytest.approx(0.75, abs=1e-5)
    backend.close()


def test_onnx_runtime_backend_reports_missing_input(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    export_result = export_torch_module_to_onnx(
        _TinyOnnxModule(),
        OnnxExportConfig(model_name="unit_model", output_dir=tmp_path, sample_input_shape=(1, 3, 4, 4)),
    )
    backend = OnnxRuntimeBackend(
        OnnxRuntimeBackendConfig(
            artifact_path=export_result.preferred_path or export_result.original_path,
            providers=("CPUExecutionProvider",),
        )
    )
    backend.load()
    with pytest.raises(RuntimeBackendError, match="missing required"):
        backend.run(_runtime_inputs(left=torch.zeros(1, 3, 4, 4), right=torch.ones(1, 3, 4, 4)))
    backend.close()


def test_ema_onnx_runtime_uses_request_result_api(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    export_result = export_torch_module_to_onnx(
        _TinyOnnxModule(),
        OnnxExportConfig(
            model_name="ema_unit",
            output_dir=tmp_path,
            sample_input_shape=(1, 3, 4, 4),
            simplify=False,
        ),
    )
    runtime = EMAVFIOnnxRuntime(
        _IdentityPadder,
        EMAVFIOnnxRuntimeConfig(
            model_name="ema_unit",
            artifact_path=export_result.preferred_path or export_result.original_path,
            providers=("CPUExecutionProvider",),
            divisor=1,
        ),
    )
    request = FramePairRequest(
        left=torch.zeros(3, 4, 4),
        right=torch.ones(3, 4, 4),
        backend_kind=RuntimeBackendKind.ONNX,
    )

    runtime.load()
    result = runtime.predict(request)
    runtime.close()

    assert result.backend_kind is RuntimeBackendKind.ONNX
    assert torch.allclose(result.middle_frame, torch.full((3, 4, 4), 0.5), atol=1e-5)
    assert result.metadata["session_providers"] == ("CPUExecutionProvider",)


def test_ema_onnx_runtime_predict_batch_reconstructs_fixed_2x_outputs(tmp_path: Path) -> None:
    runtime = _ema_tiny_onnx_runtime(tmp_path)
    request = ModelBatchRequest(
        left=torch.stack((torch.zeros(3, 4, 4), torch.full((3, 4, 4), 0.25))),
        right=torch.stack((torch.ones(3, 4, 4), torch.full((3, 4, 4), 0.75))),
        backend_kind=RuntimeBackendKind.ONNX,
    )

    runtime.load()
    result = runtime.predict_batch(request)
    runtime.close()

    assert result.backend_kind is RuntimeBackendKind.ONNX
    assert result.pair_count == 2
    assert result.timesteps == (0.5,)
    assert result.metadata["flattening_order"] == "pair_major_timestep_minor"
    assert result.metadata["effective_batch_size"] == 2
    assert result.metadata["model_call_count"] == 1
    assert result.metadata["session_providers"] == ("CPUExecutionProvider",)
    assert torch.allclose(result.outputs[0][0], torch.full((3, 4, 4), 0.5), atol=1e-5)
    assert torch.allclose(result.outputs[1][0], torch.full((3, 4, 4), 0.5), atol=1e-5)


def test_ema_onnx_runtime_predict_batch_preserves_nx_pair_timestep_order(tmp_path: Path) -> None:
    runtime = _ema_tiny_onnx_runtime(tmp_path)
    request = ModelBatchRequest(
        left=torch.stack((torch.zeros(3, 4, 4), torch.full((3, 4, 4), 0.2))),
        right=torch.stack((torch.ones(3, 4, 4), torch.full((3, 4, 4), 0.8))),
        mode=InferenceMode.ARBITRARY_NX,
        interpolation_factor=4,
        backend_kind=RuntimeBackendKind.ONNX,
        backend_options={"inference_batch_size": 3},
    )

    runtime.load()
    result = runtime.predict_batch(request)
    runtime.close()

    assert result.timesteps == (0.25, 0.5, 0.75)
    assert result.metadata["effective_batch_size"] == 3
    assert result.metadata["model_call_count"] == 2
    pair0_values = [float(frame.mean()) for frame in result.outputs[0]]
    pair1_values = [float(frame.mean()) for frame in result.outputs[1]]
    assert pair0_values == pytest.approx([0.25, 0.5, 0.75], abs=1e-5)
    assert pair1_values == pytest.approx([0.35, 0.5, 0.65], abs=1e-5)


def test_ema_onnx_runtime_batch_matches_sequential_fixed_2x(tmp_path: Path) -> None:
    runtime = _ema_tiny_onnx_runtime(tmp_path)
    left = torch.zeros(3, 4, 4)
    right = torch.ones(3, 4, 4)
    sequential_request = FramePairRequest(left=left, right=right, backend_kind=RuntimeBackendKind.ONNX)
    batch_request = ModelBatchRequest(
        left=left.unsqueeze(0),
        right=right.unsqueeze(0),
        backend_kind=RuntimeBackendKind.ONNX,
    )

    runtime.load()
    sequential = runtime.predict(sequential_request)
    batched = runtime.predict_batch(batch_request)
    runtime.close()

    assert torch.allclose(sequential.middle_frame, batched.outputs[0][0], atol=1e-5)


def test_ema_onnx_runtime_rejects_torch_batch_request(tmp_path: Path) -> None:
    runtime = _ema_tiny_onnx_runtime(tmp_path)
    request = ModelBatchRequest(left=torch.zeros(1, 3, 4, 4), right=torch.ones(1, 3, 4, 4))

    with pytest.raises(ValueError, match="onnx backend"):
        runtime.predict_batch(request)


def test_rife_onnx_runtime_rejects_request_scale_that_differs_from_export_scale(tmp_path: Path) -> None:
    runtime = PracticalRIFEOnnxRuntime(
        PracticalRIFEOnnxRuntimeConfig(
            artifact_path=tmp_path / "missing.onnx",
            providers=("CPUExecutionProvider",),
            scale=1.0,
        )
    )
    request = FramePairRequest(
        left=torch.zeros(3, 4, 4),
        right=torch.ones(3, 4, 4),
        backend_kind=RuntimeBackendKind.ONNX,
        backend_options={"scale": 0.5},
    )

    with pytest.raises(ValueError, match="scale=0.5 requires"):
        runtime.predict(request)


def test_rife_onnx_runtime_rejects_static_batch_artifact(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    export_result = export_torch_module_to_onnx(
        _TinyOnnxModule(),
        OnnxExportConfig(
            model_name="rife_unit",
            output_dir=tmp_path,
            sample_input_shape=(1, 3, 4, 4),
            shape_mode=OnnxShapeMode.STATIC,
            exporter=OnnxExporterKind.LEGACY,
            simplify=False,
        ),
    )
    assert export_result.success
    runtime = PracticalRIFEOnnxRuntime(
        PracticalRIFEOnnxRuntimeConfig(
            model_name="rife_unit",
            artifact_path=export_result.preferred_path or export_result.original_path,
            providers=("CPUExecutionProvider",),
            scale=1.0,
            divisor=1,
        )
    )

    with pytest.raises(RuntimeBackendError, match="dynamic-batch artifact"):
        runtime.load()


def test_rife_onnx_runtime_predict_batch_uses_dynamic_batch(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    export_result = _export_tiny_onnx(tmp_path, model_name="rife_unit")
    runtime = PracticalRIFEOnnxRuntime(
        PracticalRIFEOnnxRuntimeConfig(
            model_name="rife_unit",
            artifact_path=export_result.preferred_path or export_result.original_path,
            providers=("CPUExecutionProvider",),
            scale=1.0,
            divisor=1,
        )
    )
    request = ModelBatchRequest(
        left=torch.stack((torch.zeros(3, 4, 4), torch.full((3, 4, 4), 0.2))),
        right=torch.stack((torch.ones(3, 4, 4), torch.full((3, 4, 4), 0.8))),
        mode=InferenceMode.ARBITRARY_NX,
        interpolation_factor=4,
        backend_kind=RuntimeBackendKind.ONNX,
    )

    runtime.load()
    result = runtime.predict_batch(request)
    runtime.close()

    assert result.pair_count == 2
    assert result.timesteps == (0.25, 0.5, 0.75)
    assert result.metadata["effective_batch_size"] == 6
    assert result.metadata["model_call_count"] == 1
    assert result.metadata["batch_execution"] == "dynamic_batch"
    assert result.metadata["session_providers"] == ("CPUExecutionProvider",)
    assert [float(frame.mean()) for frame in result.outputs[0]] == pytest.approx([0.25, 0.5, 0.75], abs=1e-5)
    assert [float(frame.mean()) for frame in result.outputs[1]] == pytest.approx([0.35, 0.5, 0.65], abs=1e-5)


def test_rife_onnx_runtime_rejects_batch_request_scale_that_differs_from_export_scale(tmp_path: Path) -> None:
    runtime = PracticalRIFEOnnxRuntime(
        PracticalRIFEOnnxRuntimeConfig(
            artifact_path=tmp_path / "missing.onnx",
            providers=("CPUExecutionProvider",),
            scale=1.0,
        )
    )
    request = ModelBatchRequest(
        left=torch.zeros(1, 3, 4, 4),
        right=torch.ones(1, 3, 4, 4),
        backend_kind=RuntimeBackendKind.ONNX,
        backend_options={"scale": 0.5},
    )

    with pytest.raises(ValueError, match="scale=0.5 requires"):
        runtime.predict_batch(request)


def test_equivalence_metrics_and_report_writing(tmp_path: Path) -> None:
    left = torch.zeros(1, 3, 2, 2)
    right = torch.ones(1, 3, 2, 2)
    metrics = compute_tensor_equivalence_metrics(left, right, atol=1e-3, rtol=1e-3)
    assert metrics.mae == 1.0
    assert metrics.max_abs_error == 1.0
    assert not metrics.allclose

    result = run_frame_pair_equivalence_check(
        model_name="unit_model",
        torch_predictor=_fake_predictor,
        onnx_predictor=_fake_predictor,
        artifact_path=tmp_path / "unit.onnx",
        providers=("CPUExecutionProvider",),
        input_shapes=((3, 4, 4),),
    )
    artifacts = write_onnx_equivalence_report(result, tmp_path / "reports")

    assert result.success
    assert result.records[0].metrics is not None
    assert artifacts.report_path.is_file()
    assert artifacts.metrics_csv_path.is_file()


def test_discover_real_image_pairs_validates_expected_files(tmp_path: Path) -> None:
    pair_root = tmp_path / "pairs"
    pair_dir = pair_root / "001"
    pair_dir.mkdir(parents=True)
    _write_rgb_png(pair_dir / "frame1.png", value=32)
    _write_rgb_png(pair_dir / "frame2.png", value=64)

    pairs = discover_real_image_pairs(pair_root)

    assert len(pairs) == 1
    assert pairs[0].pair_id == "001"
    assert pairs[0].input_shape == (3, 4, 5)


def test_discover_real_image_pairs_reports_malformed_pair(tmp_path: Path) -> None:
    pair_root = tmp_path / "pairs"
    (pair_root / "001").mkdir(parents=True)
    _write_rgb_png(pair_root / "001" / "frame1.png", value=32)

    with pytest.raises(ValueError, match="missing frame2.png"):
        discover_real_image_pairs(pair_root)


def test_real_pair_equivalence_writes_metrics_and_visual_artifacts(tmp_path: Path) -> None:
    pair_root = tmp_path / "pairs"
    pair_dir = pair_root / "001"
    pair_dir.mkdir(parents=True)
    _write_rgb_png(pair_dir / "frame1.png", value=0)
    _write_rgb_png(pair_dir / "frame2.png", value=255)
    pairs = discover_real_image_pairs(pair_root)

    result = run_real_pair_equivalence_check(
        model_name="unit_model",
        torch_predictor=_fake_predictor,
        onnx_predictor=_fake_predictor,
        artifact_path=tmp_path / "unit.onnx",
        providers=("CPUExecutionProvider",),
        pairs=pairs,
        sample_output_root=tmp_path / "reports",
    )
    artifacts = write_onnx_equivalence_report(result, tmp_path / "reports")

    record = result.records[0]
    assert result.success
    assert record.input_group == "real_pairs"
    assert record.pair_id == "001"
    assert record.metrics is not None
    assert record.metrics.psnr == float("inf")
    assert record.metrics.ssim == 1.0
    assert record.left_sample_path is not None and record.left_sample_path.is_file()
    assert record.right_sample_path is not None and record.right_sample_path.is_file()
    assert record.torch_sample_path is not None and record.torch_sample_path.is_file()
    assert record.onnx_sample_path is not None and record.onnx_sample_path.is_file()
    assert record.absdiff_sample_path is not None and record.absdiff_sample_path.is_file()
    assert artifacts.report_path.parent.name == "cpu"
    assert artifacts.report_path.is_file()
    assert artifacts.metrics_csv_path.is_file()


class _TinyOnnxModule(torch.nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return left * (1.0 - timestep) + right * timestep


class _IdentityPadder:
    def __init__(self, _shape, divisor: int = 1) -> None:
        self.divisor = divisor

    def pad(self, *inputs):
        return list(inputs)

    def unpad(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def _fake_predictor(request: FramePairRequest) -> FramePairResult:
    frame = request.left * 0.25 + request.right * 0.75
    return FramePairResult(
        intermediate_frames=(frame,),
        timesteps=request.timesteps,
        mode=request.mode,
        interpolation_factor=request.interpolation_factor,
        backend_kind=request.backend_kind,
        model_name="unit_model",
        metadata={"session_providers": ("CPUExecutionProvider",)},
    )


def _runtime_inputs(**tensors: torch.Tensor):
    from video_interpolation.inference_runtime.api import RuntimeInputs

    return RuntimeInputs(tensors=tensors)


def _export_tiny_onnx(tmp_path: Path, *, model_name: str):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    export_result = export_torch_module_to_onnx(
        _TinyOnnxModule(),
        OnnxExportConfig(
            model_name=model_name,
            output_dir=tmp_path,
            sample_input_shape=(1, 3, 4, 4),
            exporter=OnnxExporterKind.LEGACY,
            opset_version=17,
            simplify=False,
        ),
    )
    assert export_result.success
    return export_result


def _ema_tiny_onnx_runtime(tmp_path: Path) -> EMAVFIOnnxRuntime:
    export_result = _export_tiny_onnx(tmp_path, model_name="ema_unit")
    return EMAVFIOnnxRuntime(
        _IdentityPadder,
        EMAVFIOnnxRuntimeConfig(
            model_name="ema_unit",
            artifact_path=export_result.preferred_path or export_result.original_path,
            providers=("CPUExecutionProvider",),
            divisor=1,
        ),
    )


def _write_rgb_png(path: Path, *, value: int) -> None:
    array = np.full((4, 5, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)
