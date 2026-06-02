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
    first_triplet_starts,
    first_triplets_from_video,
)


def test_quality_config_validates_arguments() -> None:
    assert VideoQualityEvaluationConfig(enabled=True, sample_count=0).sample_count == 0

    with pytest.raises(ValueError, match="sample_count"):
        VideoQualityEvaluationConfig(sample_count=-1)
    with pytest.raises(ValueError, match="max_image_side"):
        VideoQualityEvaluationConfig(max_image_side=0)
    with pytest.raises(ValueError, match="write_triplets"):
        VideoQualityEvaluationConfig(write_triplets=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fail_policy"):
        VideoQualityEvaluationConfig(fail_policy="skip")  # type: ignore[arg-type]


def test_first_triplet_starts_returns_leading_overlapping_windows() -> None:
    starts = first_triplet_starts(frame_count=20, sample_count=5)

    assert starts == (0, 1, 2, 3, 4)
    assert first_triplet_starts(frame_count=4, sample_count=5) == (0, 1)
    assert first_triplet_starts(frame_count=2, sample_count=5) == ()


def test_first_triplets_from_short_video_returns_fewer_than_requested(tmp_path: Path) -> None:
    input_path = tmp_path / "short.mp4"
    _write_synthetic_input_video(input_path, frame_values=(0, 96, 192))

    triplets = first_triplets_from_video(input_path, sample_count=16)

    assert len(triplets) == 1
    assert triplets[0].start_index == 0


def test_first_triplets_from_video_uses_first_frames_only(tmp_path: Path) -> None:
    input_path = tmp_path / "leading.mp4"
    _write_synthetic_input_video(input_path, frame_values=(0, 32, 64, 96, 128, 160))

    triplets = first_triplets_from_video(input_path, sample_count=2)

    assert [triplet.start_index for triplet in triplets] == [0, 1]
    assert _pixel_value(triplets[0].middle) == pytest.approx(32, abs=2)
    assert _pixel_value(triplets[1].middle) == pytest.approx(64, abs=2)


def test_quality_evaluation_computes_metrics_without_writing_triplets_by_default(tmp_path: Path) -> None:
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
            source_video_id="source123",
        ),
    )

    assert result.triplets_written == 0
    assert result.psnr_mean is not None
    assert result.ssim_mean is not None
    assert result.triplet_output_dir is None
    assert not triplet_root.exists()
    assert len(adapter.requests) == 1
    assert adapter.requests[0].mode is InferenceMode.ARBITRARY_NX
    assert adapter.requests[0].interpolation_factor == 2
    assert adapter.requests[0].backend_options["scale"] == 1.0


def test_quality_evaluation_resizes_frames_before_prediction_and_metrics(tmp_path: Path) -> None:
    input_path = tmp_path / "quality_large.mp4"
    output_path = tmp_path / "output.mp4"
    _write_synthetic_input_video(input_path, frame_values=(0, 96, 192), width=640, height=320)
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
            max_image_side=360,
            source_video_id="source123",
        ),
    )

    assert result.psnr_mean is not None
    assert len(adapter.requests) == 1
    assert tuple(adapter.requests[0].left.shape[-2:]) == (180, 360)


def test_quality_evaluation_can_disable_frame_resize(tmp_path: Path) -> None:
    input_path = tmp_path / "quality_large.mp4"
    output_path = tmp_path / "output.mp4"
    _write_synthetic_input_video(input_path, frame_values=(0, 96, 192), width=640, height=320)
    adapter = _PracticalQualityAdapter()

    evaluate_video_quality(
        input_path=input_path,
        output_path=output_path,
        adapter=adapter,
        mode=InferenceMode.FIXED_2X,
        runtime_options={},
        config=VideoQualityEvaluationConfig(
            enabled=True,
            sample_count=1,
            max_image_side=None,
            source_video_id="source123",
        ),
    )

    assert tuple(adapter.requests[0].left.shape[-2:]) == (320, 640)


def test_quality_evaluation_writes_vimeo_triplets_when_enabled(tmp_path: Path) -> None:
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
            write_triplets=True,
            source_video_id="source123",
        ),
    )

    assert result.triplets_written == 1
    assert result.triplet_output_dir == triplet_root / "source123"
    assert (triplet_root / "source123" / "000000" / "im1.png").is_file()
    assert (triplet_root / "source123" / "000000" / "im2.png").is_file()
    assert (triplet_root / "source123" / "000000" / "im3.png").is_file()
    assert len(adapter.requests) == 1
    assert adapter.requests[0].mode is InferenceMode.ARBITRARY_NX
    assert adapter.requests[0].interpolation_factor == 2
    assert adapter.requests[0].backend_options["scale"] == 1.0


def test_quality_evaluation_assumes_single_scene_and_keeps_high_change_triplet(tmp_path: Path) -> None:
    input_path = tmp_path / "single_scene.mp4"
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
            source_video_id="scene",
        ),
    )

    assert result.psnr_mean is not None
    assert result.ssim_mean is not None
    assert result.triplets_written == 0
    assert len(adapter.requests) == 1


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


def _write_synthetic_input_video(
    path: Path,
    *,
    frame_values: tuple[int, ...],
    width: int = 32,
    height: int = 32,
) -> None:
    with av.open(str(path), mode="w") as container:
        video_stream = container.add_stream("libx264", rate=24, options={"crf": "0", "preset": "ultrafast"})
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"
        for value in frame_values:
            frame = av.VideoFrame.from_image(Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8)))
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)


def _pixel_value(image: Image.Image) -> int:
    return int(np.asarray(image.convert("RGB"))[0, 0, 0])
