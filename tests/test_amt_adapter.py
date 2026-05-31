from pathlib import Path

import torch

from video_interpolation.adapters.amt import AMTAdapter, AMTAdapterConfig
from video_interpolation.inference import VideoInferenceConfig
from video_interpolation.mlflow import MlflowRunConfig
from video_interpolation.validation import CandidateValidationConfig


def test_amt_adapter_predict_pair_returns_unbatched_middle_frame() -> None:
    adapter = AMTAdapter(AMTAdapterConfig(device="cpu"))
    adapter._model = _AverageAMTModel()
    adapter._input_padder_cls = _NoOpPadder

    left = torch.full((3, 5, 7), 0.25)
    right = torch.full((3, 5, 7), 0.75)

    prediction = adapter.predict_pair(left, right)

    assert prediction.shape == left.shape
    assert torch.allclose(prediction, torch.full_like(prediction, 0.5))


def test_amt_configs_can_be_used_by_shared_inference_and_validation_configs() -> None:
    inference = VideoInferenceConfig.from_mapping(
        {
            "input_path": "raw_data/tmp_test/Dora.mp4",
            "output_path": "outputs/inference/amt_s/dora_2x.mp4",
            "model": {
                "model_name": "amt_s",
                "checkpoint_path": "AMT/amt-s.pth",
                "device": "cpu",
            },
            "mlflow": {"enabled": False},
        },
        model_config_factory=AMTAdapterConfig.from_mapping,
    )
    validation = CandidateValidationConfig.from_mapping(
        {
            "candidate_id": "amt_s_candidate",
            "dataset_version_id": "unit",
            "test_manifest_path": "dataset_versions/unit/test_all.csv",
            "model": {
                "model_name": "amt_s",
                "checkpoint_path": "AMT/amt-s.pth",
                "device": "cpu",
            },
            "mlflow": {"enabled": False},
        },
        model_config_factory=AMTAdapterConfig.from_mapping,
    )

    assert inference.model.checkpoint_path == Path("AMT/amt-s.pth")
    assert validation.model.model_name == "amt_s"
    assert isinstance(validation.mlflow, MlflowRunConfig)


class _AverageAMTModel(torch.nn.Module):
    def forward(self, img0, img1, embt, scale_factor=1.0, eval=False):
        return {"imgt_pred": torch.clamp((img0 + img1) / 2.0, 0.0, 1.0)}


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
