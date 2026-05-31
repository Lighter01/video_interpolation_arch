from pathlib import Path

import torch

from video_interpolation.adapters.rife import (
    PracticalRIFEAdapter,
    PracticalRIFEAdapterConfig,
    _normalise_rife_state_dict,
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


def test_rife_configs_can_be_used_by_shared_inference_and_validation_configs() -> None:
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
    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def inference(self, img0, img1, timestep=0.5, scale=1.0):
        return torch.clamp((img0 + img1) / 2.0, 0.0, 1.0)


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
