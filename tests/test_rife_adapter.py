from pathlib import Path

import pytest
import torch

from video_interpolation.adapters.rife import (
    PracticalRIFEAdapter,
    PracticalRIFEAdapterConfig,
    _normalise_rife_state_dict,
)
from video_interpolation.inference_runtime import FramePairRequest, InferenceMode, ModelBatchRequest
from video_interpolation.inference_runtime.rife import (
    PracticalRIFEPyTorchRuntime,
    PracticalRIFEPyTorchRuntimeConfig,
    validate_rife_scale,
)
from video_interpolation.inference import VideoInferenceConfig
from video_interpolation.mlflow import MlflowRunConfig
from video_interpolation.validation import CandidateValidationConfig


def test_rife_adapter_predict_pair_returns_unbatched_middle_frame_and_unpads() -> None:
    adapter = PracticalRIFEAdapter(PracticalRIFEAdapterConfig(device="cpu", divisor=8))
    adapter._model = _AverageRIFEModel()

    left = torch.full((3, 5, 7), 0.25)
    right = torch.full((3, 5, 7), 0.75)

    prediction = adapter.predict_pair(left, right)

    assert prediction.shape == left.shape
    assert torch.allclose(prediction, torch.full_like(prediction, 0.5))
    assert adapter._runtime is not None
    assert adapter._model.inference_call_count == 1


def test_rife_runtime_predicts_fixed_2x_through_request_result_api() -> None:
    model = _AverageRIFEModel()
    runtime = PracticalRIFEPyTorchRuntime(
        model,
        PracticalRIFEPyTorchRuntimeConfig(model_name="rife_unit", device="cpu", divisor=8, scale=0.5),
    )
    runtime.load()
    left = torch.full((3, 5, 7), 0.25)
    right = torch.full((3, 5, 7), 0.75)

    result = runtime.predict(FramePairRequest(left=left, right=right))

    assert result.model_name == "rife_unit"
    assert result.timesteps == (0.5,)
    assert result.original_shape == (3, 5, 7)
    assert result.padded_shape == (1, 3, 8, 8)
    assert torch.allclose(result.middle_frame, torch.full_like(left, 0.5))
    assert model.inference_calls == [
        {
            "left_shape": (1, 3, 8, 8),
            "right_shape": (1, 3, 8, 8),
            "timestep": 0.5,
            "scale": 0.5,
            "grad_enabled": False,
        }
    ]


def test_rife_runtime_uses_request_scale_over_config_default() -> None:
    model = _AverageRIFEModel()
    runtime = PracticalRIFEPyTorchRuntime(
        model,
        PracticalRIFEPyTorchRuntimeConfig(model_name="rife_unit", device="cpu", divisor=8, scale=1.0),
    )
    runtime.load()

    result = runtime.predict(
        FramePairRequest(
            left=torch.zeros(3, 5, 7),
            right=torch.ones(3, 5, 7),
            backend_options={"scale": 0.5},
        )
    )

    assert result.metadata["scale"] == 0.5
    assert result.metadata["default_scale"] == 1.0
    assert model.inference_calls[0]["scale"] == 0.5


def test_rife_runtime_rejects_invalid_request_scale_before_model_call() -> None:
    model = _AverageRIFEModel()
    runtime = PracticalRIFEPyTorchRuntime(model, PracticalRIFEPyTorchRuntimeConfig(device="cpu"))
    runtime.load()

    with pytest.raises(ValueError, match="scale"):
        runtime.predict(
            FramePairRequest(
                left=torch.zeros(3, 5, 7),
                right=torch.ones(3, 5, 7),
                backend_options={"scale": 0.75},
            )
        )

    assert model.inference_calls == []


def test_rife_runtime_rejects_non_torch_backend() -> None:
    runtime = PracticalRIFEPyTorchRuntime(_AverageRIFEModel(), PracticalRIFEPyTorchRuntimeConfig(device="cpu"))
    runtime.load()

    with pytest.raises(ValueError, match="only supports the torch backend"):
        runtime.predict(FramePairRequest(left=torch.zeros(3, 5, 7), right=torch.ones(3, 5, 7), backend_kind="onnx"))


def test_rife_runtime_predicts_arbitrary_nx_frames_from_request_factor() -> None:
    model = _AverageRIFEModel()
    runtime = PracticalRIFEPyTorchRuntime(model, PracticalRIFEPyTorchRuntimeConfig(device="cpu", divisor=8))
    runtime.load()
    left = torch.zeros(3, 5, 7)
    right = torch.ones(3, 5, 7)

    result = runtime.predict(
        FramePairRequest(
            left=left,
            right=right,
            mode=InferenceMode.ARBITRARY_NX,
            interpolation_factor=4,
        )
    )

    assert result.mode is InferenceMode.ARBITRARY_NX
    assert result.interpolation_factor == 4
    assert result.timesteps == (0.25, 0.5, 0.75)
    assert len(result.intermediate_frames) == 3
    assert [float(frame.mean()) for frame in result.intermediate_frames] == [0.25, 0.5, 0.75]
    assert [call["timestep"] for call in model.inference_calls] == [0.25, 0.5, 0.75]
    assert all(call["grad_enabled"] is False for call in model.inference_calls)


def test_rife_runtime_predicts_fixed_2x_model_batch_in_one_call() -> None:
    model = _AverageRIFEModel()
    runtime = PracticalRIFEPyTorchRuntime(
        model,
        PracticalRIFEPyTorchRuntimeConfig(model_name="rife_unit", device="cpu", divisor=8, scale=0.5),
    )
    runtime.load()
    left = torch.stack((torch.full((3, 5, 7), 0.0), torch.full((3, 5, 7), 0.2)), dim=0)
    right = torch.stack((torch.full((3, 5, 7), 0.4), torch.full((3, 5, 7), 0.8)), dim=0)

    result = runtime.predict_batch(ModelBatchRequest(left=left, right=right))

    assert result.padded_shape == (2, 3, 8, 8)
    assert torch.allclose(result.outputs[0][0], torch.full((3, 5, 7), 0.2))
    assert torch.allclose(result.outputs[1][0], torch.full((3, 5, 7), 0.5))
    assert [frame.mean().item() for frame in result.middle_frames] == pytest.approx([0.2, 0.5])
    assert result.metadata["model_call_count"] == 1
    assert model.inference_call_count == 1
    assert model.inference_calls[0]["left_shape"] == (2, 3, 8, 8)
    assert model.inference_calls[0]["scale"] == 0.5
    assert _timestep_values(model.inference_calls[0]["timestep"]) == [0.5, 0.5]


def test_rife_runtime_predicts_nx_model_batch_pair_major_timestep_order() -> None:
    model = _AverageRIFEModel()
    runtime = PracticalRIFEPyTorchRuntime(model, PracticalRIFEPyTorchRuntimeConfig(device="cpu", divisor=8))
    runtime.load()
    left = torch.stack((torch.zeros(3, 5, 7), torch.full((3, 5, 7), 0.2)), dim=0)
    right = torch.stack((torch.ones(3, 5, 7), torch.full((3, 5, 7), 0.6)), dim=0)

    result = runtime.predict_batch(
        ModelBatchRequest(left=left, right=right, mode=InferenceMode.ARBITRARY_NX, interpolation_factor=4)
    )

    assert result.timesteps == (0.25, 0.5, 0.75)
    assert [frame.mean().item() for frame in result.flattened_outputs] == pytest.approx(
        [0.25, 0.5, 0.75, 0.3, 0.4, 0.5]
    )
    assert result.metadata["model_call_count"] == 1
    assert model.inference_call_count == 1
    assert model.inference_calls[0]["left_shape"] == (6, 3, 8, 8)
    assert _timestep_values(model.inference_calls[0]["timestep"]) == [0.25, 0.5, 0.75, 0.25, 0.5, 0.75]


def test_rife_runtime_batch_respects_effective_batch_size_config() -> None:
    model = _AverageRIFEModel()
    runtime = PracticalRIFEPyTorchRuntime(
        model,
        PracticalRIFEPyTorchRuntimeConfig(device="cpu", divisor=8, inference_batch_size=2),
    )
    runtime.load()
    request = ModelBatchRequest(
        left=torch.zeros(2, 3, 5, 7),
        right=torch.ones(2, 3, 5, 7),
        mode="arbitrary_nx",
        interpolation_factor=4,
    )

    result = runtime.predict_batch(request)

    assert result.metadata["effective_batch_size"] == 2
    assert result.metadata["model_call_count"] == 3
    assert [call["left_shape"][0] for call in model.inference_calls] == [2, 2, 2]


def test_rife_adapter_predict_intermediate_frames_uses_arbitrary_nx_request() -> None:
    adapter = PracticalRIFEAdapter(PracticalRIFEAdapterConfig(device="cpu", divisor=8))
    adapter._model = _AverageRIFEModel()
    left = torch.zeros(3, 5, 7)
    right = torch.ones(3, 5, 7)

    frames = adapter.predict_intermediate_frames(left, right, interpolation_factor=8, scale=0.5)

    assert len(frames) == 7
    assert torch.allclose(frames[0], torch.full_like(left, 0.125))
    assert torch.allclose(frames[-1], torch.full_like(left, 0.875))
    assert [call["timestep"] for call in adapter._model.inference_calls] == [
        0.125,
        0.25,
        0.375,
        0.5,
        0.625,
        0.75,
        0.875,
    ]
    assert {call["scale"] for call in adapter._model.inference_calls} == {0.5}


def test_rife_adapter_predict_frame_pairs_batch_wraps_runtime_batch_api() -> None:
    adapter = PracticalRIFEAdapter(PracticalRIFEAdapterConfig(device="cpu", divisor=8))
    adapter._model = _AverageRIFEModel()
    request = ModelBatchRequest(
        left=torch.stack((torch.zeros(3, 5, 7), torch.full((3, 5, 7), 0.2)), dim=0),
        right=torch.stack((torch.ones(3, 5, 7), torch.full((3, 5, 7), 0.6)), dim=0),
        mode="arbitrary_nx",
        interpolation_factor=4,
        backend_options={"scale": 0.5},
    )

    result = adapter.predict_frame_pairs_batch(request)

    assert [frame.mean().item() for frame in result.flattened_outputs] == pytest.approx(
        [0.25, 0.5, 0.75, 0.3, 0.4, 0.5]
    )
    assert result.metadata["scale"] == 0.5
    assert adapter._model.inference_call_count == 1


def test_rife_scale_validation_matches_upstream_allowed_values() -> None:
    assert [validate_rife_scale(scale) for scale in (0.25, 0.5, 1.0, 2.0, 4.0)] == [
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
    ]
    with pytest.raises(ValueError, match="scale"):
        validate_rife_scale(0.75)


def test_rife_configs_can_be_used_by_shared_inference_and_validation_configs() -> None:
    default_config = PracticalRIFEAdapterConfig()
    assert default_config.model_name == "practical_rife_v4_26"
    assert default_config.checkpoint_path == Path("Practical-RIFE/RIFEv4.26/train_log")
    assert default_config.inference_batch_size is None

    inference = VideoInferenceConfig.from_mapping(
        {
            "input_path": "raw_data/tmp_test/Dora.mp4",
            "output_path": "outputs/inference/practical_rife_v4_25/dora_2x.mp4",
            "model": {
                "model_name": "practical_rife_v4_25",
                "checkpoint_path": "Practical-RIFE/RIFEv4.25/train_log",
                "device": "cpu",
            },
            "mlflow": {"enabled": False},
        },
        model_config_factory=PracticalRIFEAdapterConfig.from_mapping,
    )
    validation = CandidateValidationConfig.from_mapping(
        {
            "candidate_id": "practical_rife_v4_25_candidate",
            "dataset_version_id": "unit",
            "test_manifest_path": "dataset_versions/unit/test_all.csv",
            "model": {
                "model_name": "practical_rife_v4_25",
                "checkpoint_path": "Practical-RIFE/RIFEv4.25/train_log",
                "device": "cpu",
            },
            "mlflow": {"enabled": False},
        },
        model_config_factory=PracticalRIFEAdapterConfig.from_mapping,
    )

    assert inference.model.checkpoint_path == Path("Practical-RIFE/RIFEv4.25/train_log")
    assert validation.model.model_name == "practical_rife_v4_25"
    assert isinstance(validation.mlflow, MlflowRunConfig)


def test_rife_checkpoint_normalisation_and_non_strict_load_accepts_extra_keys(tmp_path) -> None:
    checkpoint_dir = tmp_path / "train_log"
    checkpoint_dir.mkdir()
    torch.save(
        {
            "module.weight": torch.ones(1),
            "module.teacher.weight": torch.ones(1),
        },
        checkpoint_dir / "flownet.pkl",
    )
    adapter = PracticalRIFEAdapter(
        PracticalRIFEAdapterConfig(
            checkpoint_path=checkpoint_dir,
            device="cpu",
            strict_checkpoint=False,
        )
    )
    adapter._model = _LoadRecordingRIFEModel()

    adapter.load_checkpoint()

    assert sorted(adapter._model.flownet.loaded_state_dict) == ["teacher.weight", "weight"]
    assert adapter._model.flownet.strict is False


def test_rife_state_dict_normalisation_strips_module_prefix() -> None:
    state_dict = _normalise_rife_state_dict({"module.block.weight": torch.ones(1)})

    assert list(state_dict) == ["block.weight"]


class _AverageRIFEModel:
    def __init__(self) -> None:
        self.inference_calls = []
        self.inference_call_count = 0

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        self.inference_call_count += 1
        self.inference_calls.append(
            {
                "left_shape": tuple(img0.shape),
                "right_shape": tuple(img1.shape),
                "timestep": timestep,
                "scale": scale,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return torch.clamp(img0 * (1.0 - timestep) + img1 * timestep, 0.0, 1.0)


def _timestep_values(value) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().reshape(-1)]
    return [float(value)]


class _LoadRecordingRIFEModel:
    def __init__(self) -> None:
        self.flownet = _LoadRecordingFlowNet()

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False


class _LoadRecordingFlowNet:
    loaded_state_dict = None
    strict = None

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = dict(state_dict)
        self.strict = strict
        return [], ["teacher.weight"]
