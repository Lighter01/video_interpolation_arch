import pytest
import torch

from video_interpolation.adapters.ema_vfi import EMAVFIAdapter, EMAVFIAdapterConfig
from video_interpolation.inference_runtime import FramePairRequest, InferenceMode
from video_interpolation.inference_runtime.ema import EMAVFIPyTorchRuntime, EMAVFIPyTorchRuntimeConfig


def test_ema_runtime_predicts_fixed_2x_through_request_result_api() -> None:
    model = _AverageEMAModel()
    runtime = EMAVFIPyTorchRuntime(
        model,
        _NoOpPadder,
        EMAVFIPyTorchRuntimeConfig(model_name="ema_unit", device="cpu", divisor=8, tta=True),
    )
    runtime.load()
    left = torch.full((3, 5, 7), 0.25)
    right = torch.full((3, 5, 7), 0.75)

    result = runtime.predict(FramePairRequest(left=left, right=right))

    assert result.model_name == "ema_unit"
    assert result.timesteps == (0.5,)
    assert result.original_shape == (3, 5, 7)
    assert result.padded_shape == (1, 3, 5, 7)
    assert torch.allclose(result.middle_frame, torch.full_like(left, 0.5))
    assert model.inference_calls == [
        {
            "left_shape": (1, 3, 5, 7),
            "right_shape": (1, 3, 5, 7),
            "tta": True,
            "fast_tta": False,
            "timestep": 0.5,
            "grad_enabled": False,
        }
    ]


def test_ema_runtime_rejects_non_torch_backend() -> None:
    runtime = EMAVFIPyTorchRuntime(_AverageEMAModel(), _NoOpPadder, EMAVFIPyTorchRuntimeConfig(device="cpu"))
    runtime.load()

    with pytest.raises(ValueError, match="only supports the torch backend"):
        runtime.predict(FramePairRequest(left=torch.zeros(3, 5, 7), right=torch.ones(3, 5, 7), backend_kind="onnx"))


def test_ema_runtime_predicts_arbitrary_nx_frames_from_request_factor() -> None:
    model = _AverageEMAModel()
    runtime = EMAVFIPyTorchRuntime(model, _NoOpPadder, EMAVFIPyTorchRuntimeConfig(device="cpu", divisor=8))
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


def test_ema_adapter_prediction_methods_remain_wrappers_over_runtime() -> None:
    adapter = EMAVFIAdapter(EMAVFIAdapterConfig(device="cpu", divisor=8))
    adapter._model = _AverageEMAModel()
    adapter._input_padder_cls = _NoOpPadder
    left = torch.full((3, 5, 7), 0.25)
    right = torch.full((3, 5, 7), 0.75)

    pair_prediction = adapter.predict_pair(left, right)
    batch_prediction = adapter.predict_batch([(left, right)])[0]
    predict_prediction = adapter.predict(left, right)
    call_prediction = adapter(left, right)

    assert torch.allclose(pair_prediction, torch.full_like(left, 0.5))
    assert torch.allclose(batch_prediction, pair_prediction)
    assert torch.allclose(predict_prediction, pair_prediction)
    assert torch.allclose(call_prediction, pair_prediction)
    assert adapter._runtime is not None
    assert adapter._model.inference_call_count == 4


def test_ema_adapter_predict_intermediate_frames_uses_arbitrary_nx_request() -> None:
    adapter = EMAVFIAdapter(EMAVFIAdapterConfig(device="cpu", divisor=8))
    adapter._model = _AverageEMAModel()
    adapter._input_padder_cls = _NoOpPadder
    left = torch.zeros(3, 5, 7)
    right = torch.ones(3, 5, 7)

    frames = adapter.predict_intermediate_frames(left, right, interpolation_factor=8)

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


def test_ema_config_keeps_inference_and_training_checkpoint_policy_separate() -> None:
    config = EMAVFIAdapterConfig.from_mapping(
        {
            "checkpoint_path": "EMA-VFI/ours_small_t.pkl",
            "inference_checkpoint_path": "EMA-VFI/ours_small_t.pkl",
            "training_checkpoint_path": "EMA-VFI/ours_small.pkl",
            "supported_modes": ["fixed_2x", "arbitrary_nx"],
        }
    )

    assert config.checkpoint_path.name == "ours_small_t.pkl"
    assert config.inference_checkpoint_path.name == "ours_small_t.pkl"
    assert config.training_checkpoint_path.name == "ours_small.pkl"
    assert config.supported_modes == ("fixed_2x", "arbitrary_nx")


def test_ema_adapter_train_and_eval_steps_stay_on_training_model_path() -> None:
    adapter = EMAVFIAdapter(EMAVFIAdapterConfig(device="cpu", divisor=8))
    adapter._model = _TrainingEMAModel()
    adapter._input_padder_cls = _NoOpPadder
    left = torch.full((3, 5, 7), 0.25)
    middle = torch.full((3, 5, 7), 0.5)
    right = torch.full((3, 5, 7), 0.75)

    prediction, loss = adapter.train_step(left, middle, right, learning_rate=1e-4)
    eval_prediction = adapter.eval_step(left, middle, right)

    assert torch.allclose(prediction, torch.full((1, 3, 5, 7), 0.5))
    assert loss == 0.125
    assert torch.allclose(eval_prediction, torch.full((1, 3, 5, 7), 0.5))
    assert adapter._runtime is None
    assert adapter._model.update_calls == [
        {"training": True, "learning_rate": 1e-4},
        {"training": False, "learning_rate": 0},
    ]


def test_ema_checkpoint_normalisation_accepts_state_dict_wrapper(tmp_path) -> None:
    checkpoint_path = tmp_path / "ema.pkl"
    torch.save({"state_dict": {"module.weight": torch.ones(1), "attn_mask": torch.ones(1)}}, checkpoint_path)
    adapter = EMAVFIAdapter(EMAVFIAdapterConfig(device="cpu", strict_checkpoint=False))
    adapter._model = _LoadRecordingEMAModel()
    adapter._input_padder_cls = _NoOpPadder

    adapter.load_checkpoint(checkpoint_path)

    assert list(adapter._model.net.loaded_state_dict) == ["weight"]
    assert torch.allclose(adapter._model.net.loaded_state_dict["weight"], torch.ones(1))
    assert adapter._model.net.strict is False


class _AverageEMAModel:
    def __init__(self) -> None:
        self.inference_calls = []
        self.inference_call_count = 0

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def inference(self, img0, img1, TTA=False, timestep=0.5, fast_TTA=False):
        self.inference_call_count += 1
        self.inference_calls.append(
            {
                "left_shape": tuple(img0.shape),
                "right_shape": tuple(img1.shape),
                "tta": TTA,
                "fast_tta": fast_TTA,
                "timestep": timestep,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return torch.clamp(img0 * (1.0 - timestep) + img1 * timestep, 0.0, 1.0)


class _TrainingEMAModel:
    def __init__(self) -> None:
        self.update_calls = []

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def update(self, imgs, gt, learning_rate=0, training=True):
        del gt
        self.update_calls.append({"training": training, "learning_rate": learning_rate})
        prediction = torch.clamp((imgs[:, :3] + imgs[:, 3:6]) / 2.0, 0.0, 1.0)
        return prediction, torch.tensor(0.125)


class _LoadRecordingEMAModel:
    def __init__(self) -> None:
        self.net = _LoadRecordingNet()

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False


class _LoadRecordingNet:
    loaded_state_dict = None
    strict = None

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = dict(state_dict)
        self.strict = strict
        return [], []


class _NoOpPadder:
    def __init__(self, dims, divisor=16):
        self.dims = dims
        self.divisor = divisor

    def pad(self, *inputs):
        return list(inputs)

    def unpad(self, *inputs):
        if len(inputs) == 1:
            return inputs[0]
        return list(inputs)
