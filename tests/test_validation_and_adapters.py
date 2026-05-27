import csv
import json
from pathlib import Path

from PIL import Image
import torch

from video_interpolation.adapters.base import ModelAdapter
from video_interpolation.contracts import TRIPLET_MANIFEST_COLUMNS
from video_interpolation.mlflow import MlflowRunConfig
from video_interpolation.settings import Settings
from video_interpolation.validation import (
    CandidateValidationConfig,
    ValidationThresholds,
    decide_candidate,
    validate_candidate,
)


def test_candidate_validation_decision_logic_applies_thresholds() -> None:
    approved, reasons = decide_candidate(
        {"psnr_mean": 30.0, "ssim_mean": 0.9, "lpips_mean": 0.1},
        ValidationThresholds(min_psnr_mean=25.0, min_ssim_mean=0.8, max_lpips_mean=0.2),
    )
    assert approved
    assert reasons == ["all configured thresholds passed"]

    rejected, reasons = decide_candidate(
        {"psnr_mean": 24.0, "ssim_mean": 0.9, "lpips_mean": 0.1},
        ValidationThresholds(min_psnr_mean=25.0, min_ssim_mean=0.8, max_lpips_mean=0.2),
    )
    assert not rejected
    assert any("psnr_mean below threshold" in reason for reason in reasons)


def test_candidate_validation_writes_report_metrics_and_triplet_prediction_set(tmp_path) -> None:
    dataset_root = tmp_path / "datasets"
    manifest_path = tmp_path / "test_all.csv"
    _write_triplet_sample(dataset_root, manifest_path)
    output_dir = tmp_path / "candidate_output"

    result = validate_candidate(
        CandidateValidationConfig(
            candidate_id="fake_candidate",
            dataset_version_id="unit_version",
            test_manifest_path=manifest_path,
            output_dir=output_dir,
            thresholds=ValidationThresholds(min_psnr_mean=1.0, min_ssim_mean=0.1),
            limit_samples=1,
            compute_lpips=False,
            save_predictions=True,
            prediction_limit=1,
            mlflow=MlflowRunConfig(enabled=False),
        ),
        settings=Settings(DATASET_ROOT=dataset_root),
        adapter=_BlendAdapter(),
    )

    assert result.samples_evaluated == 1
    assert result.sample_predictions_written == 1
    assert result.approved
    assert result.metrics_csv_path.is_file()
    assert result.summary_csv_path.is_file()
    assert result.report_path.is_file()

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["approval_decision"] == "approved"
    assert report["candidate_id"] == "fake_candidate"

    prediction_dir = output_dir / "sample_predictions" / "fake_candidate" / "sample_001"
    assert sorted(path.name for path in prediction_dir.iterdir()) == [
        "im1.png",
        "im2_generated.png",
        "im2_gt.png",
        "im3.png",
    ]


class _BlendAdapter(ModelAdapter):
    model_name = "blend_adapter"

    def validate_environment(self):  # pragma: no cover - not used by this focused test.
        raise NotImplementedError

    def build_model(self) -> None:  # pragma: no cover - not used by this focused test.
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path: Path | None = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def save_checkpoint(self, checkpoint_path: Path) -> None:  # pragma: no cover
        raise NotImplementedError

    def train(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def eval(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def predict_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.clamp((left + right) / 2.0, 0.0, 1.0)


def _write_triplet_sample(dataset_root: Path, manifest_path: Path) -> None:
    sequence_dir = "sources/unit/sequences/video_001/000000"
    frame_dir = dataset_root / sequence_dir
    frame_dir.mkdir(parents=True, exist_ok=True)
    _solid_image((16, 16), 32).save(frame_dir / "im1.png")
    _solid_image((16, 16), 64).save(frame_dir / "im2.png")
    _solid_image((16, 16), 96).save(frame_dir / "im3.png")

    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRIPLET_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample_001",
                "source_group": "unit",
                "source_dataset": "unit_dataset",
                "source_video_id": "video_001",
                "sequence_id": "seq_video_001",
                "left_frame_path": f"{sequence_dir}/im1.png",
                "mid_frame_path": f"{sequence_dir}/im2.png",
                "right_frame_path": f"{sequence_dir}/im3.png",
                "t_value": "0.5",
                "width": "16",
                "height": "16",
                "fps": "24",
                "frame_step": "0",
            }
        )


def _solid_image(size: tuple[int, int], value: int) -> Image.Image:
    return Image.new("RGB", size, (value, value, value))
