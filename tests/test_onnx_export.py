from pathlib import Path

import pytest
import torch

from video_interpolation.inference_runtime.onnx_export import (
    EMAVFIOnnxWrapper,
    OnnxExportConfig,
    OnnxExportValidationError,
    OnnxExporterKind,
    OnnxShapeMode,
    PracticalRIFEOnnxWrapper,
    create_sample_onnx_inputs,
    default_onnx_artifact_stem,
    dynamic_axes_for_config,
    dynamic_shapes_for_config,
    export_torch_module_to_onnx,
    resolve_onnx_artifact_paths,
)


def test_onnx_export_config_validates_shape_mode_and_artifact_paths(tmp_path: Path) -> None:
    config = OnnxExportConfig(
        model_name="ema_vfi_small",
        output_dir=tmp_path,
        sample_input_shape=[1, 3, 32, 48],
        shape_mode="dynamic_hw",
        opset_version=17,
        simplify=False,
    )

    paths = resolve_onnx_artifact_paths(config)

    assert config.shape_mode is OnnxShapeMode.DYNAMIC_HW
    assert config.exporter is OnnxExporterKind.DYNAMO
    assert config.sample_input_shape == (1, 3, 32, 48)
    assert paths.output_dir == tmp_path / "ema_vfi_small"
    assert paths.original_path.name == "ema_vfi_small_dynamo_dynamic_hw_opset17.onnx"
    assert paths.simplified_path.name == "ema_vfi_small_dynamo_dynamic_hw_opset17.simplified.onnx"
    assert dynamic_axes_for_config(config)["left"] == {0: "batch", 2: "height", 3: "width"}

    static_config = OnnxExportConfig(
        model_name="ema_vfi_small",
        output_dir=tmp_path,
        shape_mode=OnnxShapeMode.STATIC,
    )
    assert dynamic_axes_for_config(static_config) is None
    assert dynamic_shapes_for_config(static_config) is None

    default_config = OnnxExportConfig(model_name="unit", output_dir=tmp_path)
    assert default_config.exporter is OnnxExporterKind.DYNAMO
    assert default_config.opset_version == 18
    assert not default_config.simplify

    dynamo_config = OnnxExportConfig(
        model_name="ema_vfi_small",
        output_dir=tmp_path,
        exporter="dynamo",
        dynamic_hw_multiple=112,
    )
    assert dynamo_config.exporter is OnnxExporterKind.DYNAMO
    assert dynamo_config.dynamic_hw_multiple == 112
    dynamic_shapes = dynamic_shapes_for_config(dynamo_config)
    assert dynamic_shapes is not None
    batch_dim = dynamic_shapes[0][0]
    assert dynamic_shapes[1][0] is batch_dim
    assert dynamic_shapes[2][0] is batch_dim
    assert set(dynamic_shapes[0]) == {0, 2, 3}
    assert set(dynamic_shapes[1]) == {0, 2, 3}
    assert set(dynamic_shapes[2]) == {0}

    legacy_config = OnnxExportConfig(model_name="ema_vfi_small", output_dir=tmp_path, exporter="legacy")
    legacy_paths = resolve_onnx_artifact_paths(legacy_config)
    assert legacy_paths.original_path.name == "ema_vfi_small_dynamic_hw_opset18.onnx"
    assert (
        default_onnx_artifact_stem(
            "ema_vfi_small",
            shape_mode=OnnxShapeMode.DYNAMIC_HW,
            opset_version=17,
            exporter=OnnxExporterKind.LEGACY,
        )
        == "ema_vfi_small_dynamic_hw_opset17"
    )


def test_onnx_export_config_resolves_relative_output_dir_before_later_cwd_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = OnnxExportConfig(model_name="unit_model", output_dir=Path("exports"))
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)

    paths = resolve_onnx_artifact_paths(config)

    assert paths.output_dir == tmp_path / "exports" / "unit_model"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_name": "../bad"}, "model_name"),
        ({"model_name": "bad/name"}, "model_name"),
        ({"model_name": "unit", "sample_input_shape": (1, 1, 8, 8)}, "channel"),
        ({"model_name": "unit", "sample_input_shape": (1, 3, 8)}, "four dimensions"),
        ({"model_name": "unit", "opset_version": 10}, "opset_version"),
        ({"model_name": "unit", "shape_mode": "batch_only"}, "shape mode"),
        ({"model_name": "unit", "exporter": "future"}, "exporter"),
        ({"model_name": "unit", "dynamic_hw_multiple": 0}, "dynamic_hw_multiple"),
        ({"model_name": "unit", "timestep": 1.0}, "timestep"),
        ({"model_name": "unit", "exporter": "dynamo", "simplify": True}, "simplification"),
    ],
)
def test_onnx_export_config_rejects_invalid_options(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(OnnxExportValidationError, match=message):
        OnnxExportConfig(**kwargs)


def test_create_sample_onnx_inputs_uses_bchw_shape_and_timestep(tmp_path: Path) -> None:
    config = OnnxExportConfig(
        model_name="unit",
        output_dir=tmp_path,
        sample_input_shape=(2, 3, 8, 10),
        timestep=0.25,
    )

    left, right, timestep = create_sample_onnx_inputs(config)

    assert left.shape == (2, 3, 8, 10)
    assert right.shape == (2, 3, 8, 10)
    assert timestep.shape == (2, 1, 1, 1)
    assert float(timestep[0, 0, 0, 0]) == 0.25


def test_ema_onnx_wrapper_returns_only_generated_frame() -> None:
    wrapper = EMAVFIOnnxWrapper(_FakeEMANet())
    left = torch.zeros(1, 3, 4, 4)
    right = torch.ones(1, 3, 4, 4)
    timestep = torch.full((1, 1, 1, 1), 0.25)

    output = wrapper(left, right, timestep)

    assert output.shape == left.shape
    assert torch.allclose(output, torch.full_like(left, 0.25))


def test_rife_onnx_wrapper_uses_flownet_core_and_constant_scale_list() -> None:
    flownet = _FakeRIFEFlowNet()
    wrapper = PracticalRIFEOnnxWrapper(flownet, scale=0.5)
    left = torch.zeros(1, 3, 4, 4)
    right = torch.ones(1, 3, 4, 4)
    timestep = torch.full((1, 1, 1, 1), 0.75)

    output = wrapper(left, right, timestep)

    assert torch.allclose(output, torch.full_like(left, 0.75))
    assert flownet.scale_lists == [(32.0, 16.0, 8.0, 4.0, 2.0)]


def test_export_torch_module_to_onnx_writes_original_artifact_without_simplification(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    config = OnnxExportConfig(
        model_name="unit_model",
        output_dir=tmp_path,
        sample_input_shape=(1, 3, 4, 4),
        shape_mode=OnnxShapeMode.DYNAMIC_HW,
        exporter=OnnxExporterKind.LEGACY,
        simplify=False,
    )

    result = export_torch_module_to_onnx(_TinyOnnxModule(), config)

    assert result.success
    assert result.exporter is OnnxExporterKind.LEGACY
    assert result.metadata["dynamic_hw_multiple"] is None
    assert result.original_path.is_file()
    assert result.preferred_path == result.original_path
    assert result.simplification_status == "not_requested"
    assert result.dynamic_axes["intermediate_frame"] == {0: "batch", 2: "height", 3: "width"}


def test_export_torch_module_to_onnx_writes_dynamo_artifact_by_default(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    config = OnnxExportConfig(
        model_name="unit_model",
        output_dir=tmp_path,
        sample_input_shape=(1, 3, 4, 4),
    )

    result = export_torch_module_to_onnx(_TinyOnnxModule(), config)

    assert result.success
    assert result.exporter is OnnxExporterKind.DYNAMO
    assert result.original_path.name == "unit_model_dynamo_dynamic_hw_opset18.onnx"
    assert result.preferred_path == result.original_path
    assert result.simplification_status == "not_requested"


def test_export_torch_module_to_onnx_reports_export_failure(tmp_path: Path) -> None:
    config = OnnxExportConfig(model_name="broken_model", output_dir=tmp_path, simplify=False)

    result = export_torch_module_to_onnx(_FailingOnnxModule(), config)

    assert not result.success
    assert "RuntimeError" in result.error
    assert "intentional export failure" in result.error


class _FakeEMANet(torch.nn.Module):
    def forward(self, images: torch.Tensor, timestep: torch.Tensor):
        left = images[:, :3]
        right = images[:, 3:6]
        output = left * (1.0 - timestep) + right * timestep
        return [], [], [], output


class _FakeRIFEFlowNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_lists: list[tuple[float, ...]] = []

    def forward(self, images: torch.Tensor, timestep: torch.Tensor, scale_list: list[float]):
        self.scale_lists.append(tuple(float(value) for value in scale_list))
        left = images[:, :3]
        right = images[:, 3:6]
        output = left * (1.0 - timestep) + right * timestep
        return [], None, [output]


class _TinyOnnxModule(torch.nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return left * (1.0 - timestep) + right * timestep


class _FailingOnnxModule(torch.nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("intentional export failure")
