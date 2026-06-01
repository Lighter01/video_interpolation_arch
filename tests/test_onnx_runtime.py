from pathlib import Path

import pytest
import torch

from video_interpolation.inference_runtime import FramePairRequest, RuntimeBackendKind
from video_interpolation.inference_runtime.api import FramePairResult
from video_interpolation.inference_runtime.backends.base import RuntimeBackendError
from video_interpolation.inference_runtime.backends.onnx import (
    OnnxRuntimeBackend,
    OnnxRuntimeBackendConfig,
    resolve_onnx_providers,
)
from video_interpolation.inference_runtime.ema import EMAVFIOnnxRuntime, EMAVFIOnnxRuntimeConfig
from video_interpolation.inference_runtime.onnx_export import (
    OnnxExportConfig,
    export_torch_module_to_onnx,
)
from video_interpolation.inference_runtime.onnx_validation import (
    compute_tensor_equivalence_metrics,
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

    assert resolve_preferred_onnx_artifact_path("unit_model", artifact_root=tmp_path) == simplified
    assert (
        resolve_preferred_onnx_artifact_path("unit_model", artifact_root=tmp_path, prefer_simplified=False)
        == original
    )


def test_resolve_preferred_onnx_artifact_path_reports_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No ONNX artifact found"):
        resolve_preferred_onnx_artifact_path("missing_model", artifact_root=tmp_path)


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
    output = backend.run(
        _runtime_inputs(left=left, right=right, timestep=timestep)
    )
    assert torch.allclose(output.primary_tensor, torch.full_like(left, 0.25), atol=1e-5)
    assert output.metadata["session_providers"] == ("CPUExecutionProvider",)
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
