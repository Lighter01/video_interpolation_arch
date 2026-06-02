from pathlib import Path

import av
import numpy as np
from PIL import Image
import pytest
import torch

from video_interpolation.adapters.base import AdapterEnvironmentReport, ModelAdapter
from video_interpolation.inference_runtime import FramePairRequest, FramePairResult, InferenceMode, RuntimeBackendKind
from video_interpolation.video_quality import (
    VideoQualityEvaluationConfig,
    evaluate_video_quality,
    sample_triplet_starts,
    selected_triplets_from_video,
)


def test_quality_config_validates_arguments() -> None:
    assert VideoQualityEvaluationConfig(enabled=True, sample_count=0).sample_count == 0

    with pytest.raises(ValueError, match="sample_count"):
        VideoQualityEvaluationConfig(sample_count=-1)
    with pytest.raises(ValueError, match="scene_cut"):
        VideoQualityEvaluationConfig(scene_cut_ssim_threshold=2.0)
    with pytest.raises(ValueError, match="fail_policy"):
        VideoQualityEvaluationConfig(fail_policy="skip")  # type: ignore[arg-type]


def test_sample_triplet_starts_is_deterministic_and_sorted() -> None:
    first = sample_triplet_starts(frame_count=20, sample_count=5, random_seed=7)
    second = sample_triplet_starts(frame_count=20, sample_count=5, random_seed=7)

    assert first == second
    assert first == tuple(sorted(first))
    assert len(first) == 5
    assert sample_triplet_starts(frame_count=2, sample_count=5, random_seed=7) == ()


def test_selected_triplets_from_short_video_returns_fewer_than_requested(tmp_path: Path) -> None:
    input_path = tmp_path / "short.mp4"
    _write_synthetic_input_video(input_path, frame_values=(0, 96, 192))

    triplets = selected_triplets_from_video(input_path, sample_count=16, random_seed=1)

    assert len(triplets) == 1
    assert triplets[0].start_index == 0


def test_quality_evaluation_writes_vimeo_triplets_and_metrics(tmp_path: Path) -> None:
    input_path = tmp_path / "quality.mp4"
    output_path = tmp_path / "output.mp4"
    triplet_root = tmp_path / "triplets"
    _write_synthetic_input_video(input_path, frame_values=(0, 96, 192))
    adapter = _PracticalQualityAdapter()

    result = evaluate_video_quality(
        input_path=input_path,
        output_path=output_path,
        adapter=adapter,
        mode=InferenceMode.ARBITRARY_NX,
        runtime_options={"scale": 1.0},
        config=VideoQualityEvaluationConfig(
            enabled=True,
            triplet_output_dir=triplet_root,
            sample_count=1,
            random_seed=0,
            scene_cut_ssim_threshold=None,
            source_video_id="source123",
        ),
    )

    assert result.triplets_written == 1
    assert result.psnr_mean is not None
    assert result.ssim_mean is not None
    assert result.triplet_output_dir == triplet_root / "source123"
    assert (triplet_root / "source123" / "000000" / "im1.png").is_file()
    assert (triplet_root / "source123" / "000000" / "im2.png").is_file()
    assert (triplet_root / "source123" / "000000" / "im3.png").is_file()
    assert len(adapter.requests) == 1
    assert adapter.requests[0].mode is InferenceMode.ARBITRARY_NX
    assert adapter.requests[0].interpolation_factor == 2
    assert adapter.requests[0].backend_options["scale"] == 1.0


def test_quality_evaluation_scene_cut_filter_can_leave_no_valid_triplets(tmp_path: Path) -> None:
    input_path = tmp_path / "scene_cut.mp4"
    output_path = tmp_path / "output.mp4"
    _write_synthetic_input_video(input_path, frame_values=(0, 255, 0))
    adapter = _PracticalQualityAdapter()

    result = evaluate_video_quality(
        input_path=input_path,
        output_path=output_path,
        adapter=adapter,
        mode=InferenceMode.FIXED_2X,
        runtime_options={},
        config=VideoQualityEvaluationConfig(
            enabled=True,
            sample_count=1,
            random_seed=0,
            scene_cut_ssim_threshold=0.99,
            source_video_id="scene",
        ),
    )

    assert result.psnr_mean is None
    assert result.ssim_mean is None
    assert result.triplets_written == 0
    assert adapter.requests == []


class _PracticalQualityAdapter(ModelAdapter):
    model_name = "practical_rife_v4_26"

    def __init__(self) -> None:
        self.requests: list[FramePairRequest] = []

    def validate_environment(self) -> AdapterEnvironmentReport:
        return AdapterEnvironmentReport(status="ok")

    def build_model(self) -> None:
        return None

    def load_checkpoint(self, checkpoint_path: Path | None = None) -> None:
        del checkpoint_path

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        del checkpoint_path

    def train(self) -> None:
        return None

    def eval(self) -> None:
        return None

    def predict_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left + right) / 2.0

    def predict_frame_pair(self, request: FramePairRequest) -> FramePairResult:
        self.requests.append(request)
        return FramePairResult(
            intermediate_frames=tuple(
                torch.clamp((1.0 - timestep) * request.left + timestep * request.right, 0.0, 1.0)
                for timestep in request.timesteps
            ),
            timesteps=request.timesteps,
            mode=request.mode,
            interpolation_factor=request.interpolation_factor,
            backend_kind=RuntimeBackendKind.TORCH,
            model_name=self.model_name,
        )

    def close(self) -> None:
        return None


def _write_synthetic_input_video(path: Path, *, frame_values: tuple[int, ...]) -> None:
    with av.open(str(path), mode="w") as container:
        video_stream = container.add_stream("libx264", rate=24, options={"crf": "0", "preset": "ultrafast"})
        video_stream.width = 32
        video_stream.height = 32
        video_stream.pix_fmt = "yuv420p"
        for value in frame_values:
            frame = av.VideoFrame.from_image(Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)))
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)
