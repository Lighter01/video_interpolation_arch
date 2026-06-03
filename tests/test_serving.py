from __future__ import annotations

from pathlib import Path
import importlib.util
import inspect

import av
import numpy as np
import pytest
import torch

from video_interpolation.adapters.base import AdapterEnvironmentReport, ModelAdapter, ModelAdapterError
from video_interpolation.inference_runtime import (
    FramePairRequest,
    FramePairResult,
    InferenceMode,
    RuntimeBackendKind,
)
from video_interpolation.serving import (
    DEFAULT_SERVING_CODEC,
    DEFAULT_SERVING_DEVICE,
    DEFAULT_SERVING_ONNX_PROVIDER,
    MAX_SERVING_INTERPOLATION_FACTOR,
    MIN_SERVING_INTERPOLATION_FACTOR,
    PracticalRIFEServingConfig,
    PracticalRIFEVideoInferenceRunner,
    run_practical_rife_video_inference,
)
from video_interpolation.settings import Settings


def test_practical_rife_serving_config_defaults_to_torch_cuda_sequential_nx() -> None:
    config = PracticalRIFEServingConfig()

    assert config.model_name == "practical_rife_v4_26"
    assert config.backend is RuntimeBackendKind.TORCH
    assert config.device == DEFAULT_SERVING_DEVICE
    assert config.provider == DEFAULT_SERVING_ONNX_PROVIDER
    assert config.execution_mode.value == "sequential"
    assert config.interpolation_mode is InferenceMode.ARBITRARY_NX
    assert config.codec == DEFAULT_SERVING_CODEC
    assert config.output_playback_mode.value == "real_time"
    assert config.min_interpolation_factor == MIN_SERVING_INTERPOLATION_FACTOR
    assert config.max_interpolation_factor == MAX_SERVING_INTERPOLATION_FACTOR
    assert config.default_scale == 1.0
    assert config.quality_evaluation_enabled
    assert config.quality_sample_count == 16
    assert config.quality_max_image_side == 360
    assert not config.quality_write_triplets


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_practical_rife_serving_accepts_service_factor_range(factor: int) -> None:
    assert PracticalRIFEServingConfig().validate_request(interpolation_factor=factor) == (factor, 1.0)


@pytest.mark.parametrize("factor", [1, 5, 8, True])
def test_practical_rife_serving_rejects_factor_outside_service_range(factor: int | bool) -> None:
    with pytest.raises(ValueError, match="interpolation_factor"):
        PracticalRIFEServingConfig().validate_request(interpolation_factor=factor)  # type: ignore[arg-type]


def test_practical_rife_serving_validates_scale_and_onnx_artifact_scale() -> None:
    assert PracticalRIFEServingConfig().validate_request(interpolation_factor=2, scale=0.5) == (2, 0.5)

    with pytest.raises(ValueError, match="scale"):
        PracticalRIFEServingConfig().validate_request(interpolation_factor=2, scale=0.75)

    onnx_config = PracticalRIFEServingConfig(backend="onnx", onnx_artifact_scale=1.0)
    with pytest.raises(ValueError, match="artifact scale"):
        onnx_config.validate_request(interpolation_factor=2, scale=0.5)


def test_practical_rife_serving_rejects_invalid_static_config_values() -> None:
    with pytest.raises(ValueError, match="backend"):
        PracticalRIFEServingConfig(backend="bad")
    with pytest.raises(ValueError, match="provider"):
        PracticalRIFEServingConfig(provider="")
    with pytest.raises(ValueError, match="sequential"):
        PracticalRIFEServingConfig(execution_mode="batched")
    with pytest.raises(ValueError, match="arbitrary_nx"):
        PracticalRIFEServingConfig(interpolation_mode="fixed_2x")
    with pytest.raises(ValueError, match="quality_write_triplets"):
        PracticalRIFEServingConfig(quality_write_triplets=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="quality_max_image_side"):
        PracticalRIFEServingConfig(quality_max_image_side=0)
    with pytest.raises(ValueError, match="output_playback_mode"):
        PracticalRIFEServingConfig(output_playback_mode="bad")


def test_practical_rife_serving_runner_fails_before_model_load_for_bad_paths(tmp_path: Path) -> None:
    adapter = _ServingRuntimeAdapter()
    runner = PracticalRIFEVideoInferenceRunner(adapter=adapter, settings=Settings())

    with pytest.raises(FileNotFoundError, match="Input video"):
        runner.run(
            input_path=tmp_path / "missing.mp4",
            output_path=tmp_path / "out.mp4",
            interpolation_factor=2,
        )

    assert adapter.load_count == 0

    input_path = tmp_path / "input.mp4"
    output_dir = tmp_path / "output_dir"
    output_dir.mkdir()
    _write_synthetic_input_video(input_path)

    with pytest.raises(ValueError, match="directory"):
        runner.run(input_path=input_path, output_path=output_dir, interpolation_factor=2)

    assert adapter.load_count == 0


def test_practical_rife_serving_onnx_runner_reports_missing_artifact(tmp_path: Path) -> None:
    runner = PracticalRIFEVideoInferenceRunner(
        PracticalRIFEServingConfig(backend="onnx", onnx_path=tmp_path / "missing.onnx"),
        settings=Settings(),
    )

    with pytest.raises(FileNotFoundError, match="ONNX artifact"):
        runner.load()


def test_practical_rife_serving_runner_reuses_loaded_adapter_across_requests(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    first_output = tmp_path / "first.mp4"
    second_output = tmp_path / "second.mp4"
    _write_synthetic_input_video(input_path)
    adapter = _ServingRuntimeAdapter()
    runner = PracticalRIFEVideoInferenceRunner(adapter=adapter, settings=Settings())

    first = runner.run(
        input_path=input_path,
        output_path=first_output,
        interpolation_factor=2,
        scale=0.5,
        output_playback_mode="slow_motion",
    )
    second = runner.run(input_path=input_path, output_path=second_output, interpolation_factor=4, scale=0.5)

    assert adapter.load_count == 1
    assert [request.interpolation_factor for request in adapter.requests] == [2, 4]
    assert first.frames_written == 3
    assert second.frames_written == 5
    assert first.execution_mode == "sequential"
    assert second.execution_mode == "sequential"
    assert first.output_playback_mode == "slow_motion"
    assert second.output_playback_mode == "real_time"
    assert first.output_fps == pytest.approx(first.input_fps)
    assert second.output_fps == pytest.approx(second.input_fps * 4)
    assert first.runtime_options["scale"] == 0.5
    assert second.runtime_options["scale"] == 0.5


def test_practical_rife_serving_convenience_function_uses_facade(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _write_synthetic_input_video(input_path)
    created: list[_ServingRuntimeAdapter] = []

    class _FakePracticalRIFEAdapter(_ServingRuntimeAdapter):
        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()
            created.append(self)

    monkeypatch.setattr("video_interpolation.serving.PracticalRIFEAdapter", _FakePracticalRIFEAdapter)

    result = run_practical_rife_video_inference(
        input_path=input_path,
        output_path=output_path,
        interpolation_factor=3,
        scale=0.5,
        output_playback_mode="slow_motion",
        settings=Settings(),
    )

    assert result.frames_written == 4
    assert result.interpolation_factor == 3
    assert result.output_playback_mode == "slow_motion"
    assert result.runtime_backend == RuntimeBackendKind.TORCH.value
    assert len(created) == 1
    assert created[0].load_count == 1
    assert created[0].close_count == 1


def test_bentoml_examples_import_and_expose_expected_defaults() -> None:
    torch_service = _load_example_module(
        Path("examples/bentoml/practical_rife_torch_service/service.py"),
        "practical_rife_torch_service_example",
    )
    onnx_service = _load_example_module(
        Path("examples/bentoml/practical_rife_onnx_service/service.py"),
        "practical_rife_onnx_service_example",
    )

    assert torch_service.SERVICE_CONFIG.backend is RuntimeBackendKind.TORCH
    assert torch_service.SERVICE_CONFIG.device == "cuda"
    assert torch_service.SERVICE_CONFIG.execution_mode.value == "sequential"
    assert torch_service.SERVICE_CONFIG.interpolation_mode is InferenceMode.ARBITRARY_NX
    assert onnx_service.SERVICE_CONFIG.backend is RuntimeBackendKind.ONNX
    assert onnx_service.SERVICE_CONFIG.provider == "CUDAExecutionProvider"

    torch_runner = torch_service.create_runner()
    onnx_runner = onnx_service.create_runner()

    assert isinstance(torch_runner, PracticalRIFEVideoInferenceRunner)
    assert isinstance(onnx_runner, PracticalRIFEVideoInferenceRunner)
    assert not torch_runner.is_loaded
    assert not onnx_runner.is_loaded
    assert "output_playback_mode" in inspect.signature(
        torch_service.PracticalRIFETorchService.apis["interpolate_video"].func
    ).parameters
    assert "output_playback_mode" in inspect.signature(
        onnx_service.PracticalRIFEOnnxService.apis["interpolate_video"].func
    ).parameters


class _ServingRuntimeAdapter(ModelAdapter):
    model_name = "serving_runtime_adapter"

    def __init__(self) -> None:
        self.load_count = 0
        self.close_count = 0
        self.requests: list[FramePairRequest] = []

    def validate_environment(self) -> AdapterEnvironmentReport:
        return AdapterEnvironmentReport(status="ok")

    def build_model(self) -> None:
        return None

    def load_checkpoint(self, checkpoint_path: Path | None = None) -> None:
        del checkpoint_path
        self.load_count += 1

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        del checkpoint_path
        raise ModelAdapterError("test adapter does not save checkpoints")

    def train(self) -> None:
        raise ModelAdapterError("test adapter does not train")

    def eval(self) -> None:
        return None

    def predict_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.clamp((left + right) / 2.0, 0.0, 1.0)

    def predict_frame_pair(self, request: FramePairRequest) -> FramePairResult:
        self.requests.append(request)
        frames = tuple(
            torch.clamp((1.0 - timestep) * request.left + timestep * request.right, 0.0, 1.0)
            for timestep in request.timesteps
        )
        return FramePairResult(
            intermediate_frames=frames,
            timesteps=request.timesteps,
            mode=request.mode,
            interpolation_factor=request.interpolation_factor,
            backend_kind=RuntimeBackendKind.TORCH,
            model_name=self.model_name,
            original_shape=request.original_shape,
        )

    def close(self) -> None:
        self.close_count += 1


def _write_synthetic_input_video(path: Path, *, frame_values: tuple[int, ...] = (32, 96)) -> None:
    with av.open(str(path), mode="w") as container:
        video_stream = container.add_stream("libx264", rate=24)
        video_stream.width = 64
        video_stream.height = 64
        video_stream.pix_fmt = "yuv420p"
        for offset in frame_values:
            frame = av.VideoFrame.from_ndarray(_solid_rgb_frame(offset), format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)


def _solid_rgb_frame(value: int) -> np.ndarray:
    return np.full((64, 64, 3), value, dtype=np.uint8)


def _load_example_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive import guard.
        raise ImportError(f"Could not load example module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
