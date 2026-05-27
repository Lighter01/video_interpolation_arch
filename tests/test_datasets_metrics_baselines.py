import csv
import math
from pathlib import Path

from PIL import Image
import pytest
import torch

from video_interpolation.baselines import BaselineEvaluationConfig, evaluate_baselines
from video_interpolation.contracts import TRIPLET_MANIFEST_COLUMNS
from video_interpolation.data.datasets import TripletFrames, UniversalTripletDataset
from video_interpolation.metrics import compute_psnr, compute_ssim
from video_interpolation.mlflow import MlflowRunConfig
from video_interpolation.settings import Settings


def test_universal_triplet_dataset_loads_frames_and_applies_synchronized_transform(tmp_path) -> None:
    dataset_root = tmp_path / "datasets"
    manifest_path = tmp_path / "manifest.csv"
    _write_triplet_sample(dataset_root, manifest_path, "sample_001", "video_001")

    plain_dataset = UniversalTripletDataset(
        manifest_path=manifest_path,
        settings=Settings(DATASET_ROOT=dataset_root),
    )

    def flip_triplet(frames: TripletFrames) -> TripletFrames:
        return TripletFrames(
            left=torch.flip(frames.left, dims=(2,)),
            middle=torch.flip(frames.middle, dims=(2,)),
            right=torch.flip(frames.right, dims=(2,)),
        )

    flipped_dataset = UniversalTripletDataset(
        manifest_path=manifest_path,
        transform=flip_triplet,
        settings=Settings(DATASET_ROOT=dataset_root),
    )

    plain_sample = plain_dataset[0]
    flipped_sample = flipped_dataset[0]

    assert plain_sample["left"].shape == (3, 16, 16)
    assert plain_sample["metadata"]["sample_id"] == "sample_001"
    assert torch.equal(flipped_sample["left"], torch.flip(plain_sample["left"], dims=(2,)))
    assert torch.equal(flipped_sample["middle"], torch.flip(plain_sample["middle"], dims=(2,)))
    assert torch.equal(flipped_sample["right"], torch.flip(plain_sample["right"], dims=(2,)))


def test_metric_functions_are_sane_for_identical_and_different_images() -> None:
    zeros = torch.zeros((3, 16, 16), dtype=torch.float32)
    ones = torch.ones((3, 16, 16), dtype=torch.float32)

    assert math.isinf(compute_psnr(zeros, zeros))
    assert compute_ssim(zeros, zeros) == pytest.approx(1.0)
    assert compute_psnr(zeros, ones) == pytest.approx(0.0)
    assert compute_ssim(zeros, ones) < 0.1


def test_baseline_evaluation_writes_metrics_and_sample_predictions(tmp_path) -> None:
    dataset_root = tmp_path / "datasets"
    manifest_path = tmp_path / "test_all.csv"
    _write_triplet_sample(dataset_root, manifest_path, "sample_001", "video_001")
    _append_triplet_sample(dataset_root, manifest_path, "sample_002", "video_002")
    output_dir = tmp_path / "baseline_output"

    result = evaluate_baselines(
        BaselineEvaluationConfig(
            manifest_path=manifest_path,
            output_dir=output_dir,
            baselines=("duplicate_left", "blend", "farneback"),
            compute_lpips=False,
            save_predictions=True,
            prediction_limit=1,
            mlflow=MlflowRunConfig(enabled=False),
        ),
        settings=Settings(DATASET_ROOT=dataset_root),
    )

    assert result.samples_evaluated == 2
    assert result.predictions_evaluated == 6
    assert result.sample_predictions_written == 3
    assert result.mlflow_run_id is None
    assert result.metrics_csv_path.is_file()
    assert result.summary_csv_path.is_file()

    with result.metrics_csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 6
    assert {row["prediction_name"] for row in rows} == {"duplicate_left", "blend", "farneback"}
    assert all(not Path(row["prediction_path"]).is_absolute() for row in rows if row["prediction_path"])
    saved_prediction_dirs = [output_dir / row["prediction_path"] for row in rows if row["prediction_path"]]
    assert len(saved_prediction_dirs) == 3
    for prediction_dir in saved_prediction_dirs:
        assert prediction_dir.is_dir()
        assert sorted(path.name for path in prediction_dir.iterdir()) == [
            "im1.png",
            "im2_generated.png",
            "im2_gt.png",
            "im3.png",
        ]


def _write_triplet_sample(
    dataset_root: Path,
    manifest_path: Path,
    sample_id: str,
    source_video_id: str,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRIPLET_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(_triplet_row(dataset_root, sample_id, source_video_id))


def _append_triplet_sample(
    dataset_root: Path,
    manifest_path: Path,
    sample_id: str,
    source_video_id: str,
) -> None:
    with manifest_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRIPLET_MANIFEST_COLUMNS)
        writer.writerow(_triplet_row(dataset_root, sample_id, source_video_id))


def _triplet_row(dataset_root: Path, sample_id: str, source_video_id: str) -> dict[str, str]:
    sequence_dir = f"sources/unit/sequences/{source_video_id}/000000"
    frame_dir = dataset_root / sequence_dir
    frame_dir.mkdir(parents=True, exist_ok=True)
    _gradient_image((16, 16), 0).save(frame_dir / "frame_000.png")
    _gradient_image((16, 16), 64).save(frame_dir / "frame_001.png")
    _gradient_image((16, 16), 128).save(frame_dir / "frame_002.png")
    return {
        "sample_id": sample_id,
        "source_group": "unit",
        "source_dataset": "unit_dataset",
        "source_video_id": source_video_id,
        "sequence_id": f"seq_{source_video_id}",
        "left_frame_path": f"{sequence_dir}/frame_000.png",
        "mid_frame_path": f"{sequence_dir}/frame_001.png",
        "right_frame_path": f"{sequence_dir}/frame_002.png",
        "t_value": "0.5",
        "width": "16",
        "height": "16",
        "fps": "24",
        "frame_step": "0",
    }


def _gradient_image(size: tuple[int, int], offset: int) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size)
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            value = (x * 8 + y * 4 + offset) % 256
            pixels[x, y] = (value, value, value)
    return image
