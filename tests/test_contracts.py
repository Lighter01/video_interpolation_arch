import csv

import pytest

from video_interpolation.contracts import (
    ContractValidationError,
    SEQUENCE_INDEX_CONTRACT,
    TRIPLET_MANIFEST_CONTRACT,
    validate_csv_file,
    validate_relative_artifact_path,
)


def test_relative_artifact_paths_reject_absolute_and_uri_values() -> None:
    validate_relative_artifact_path("datasets/sources/anime/seq001", "relative_sequence_dir")

    for value in ("/tmp/frame.png", "s3://bucket/frame.png", "../outside/frame.png"):
        with pytest.raises(ContractValidationError):
            validate_relative_artifact_path(value, "frame_path")


def test_triplet_manifest_contract_validates_relative_frame_paths(tmp_path) -> None:
    manifest_path = tmp_path / "train_all.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRIPLET_MANIFEST_CONTRACT.required_columns)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample-001",
                "source_group": "vimeo",
                "source_dataset": "vimeo_triplet",
                "source_video_id": "00001",
                "sequence_id": "00001_0001",
                "left_frame_path": "sources/vimeo_triplet/sequences/00001/0001/im1.png",
                "mid_frame_path": "sources/vimeo_triplet/sequences/00001/0001/im2.png",
                "right_frame_path": "sources/vimeo_triplet/sequences/00001/0001/im3.png",
                "t_value": "0.5",
                "width": "448",
                "height": "256",
                "fps": "30",
                "frame_step": "0",
            }
        )

    validate_csv_file(str(manifest_path), TRIPLET_MANIFEST_CONTRACT)


def test_sequence_index_contract_rejects_absolute_sequence_dir(tmp_path) -> None:
    index_path = tmp_path / "sequence_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SEQUENCE_INDEX_CONTRACT.required_columns)
        writer.writeheader()
        writer.writerow(
            {
                "sequence_id": "seq-001",
                "source_group": "anime",
                "source_dataset": "anime",
                "source_video_id": "video-001",
                "relative_sequence_dir": "/absolute/path",
                "sequence_length": "3",
                "frame_step": "0",
                "width": "1920",
                "height": "1080",
                "fps": "24",
                "frame_start": "10",
                "frame_end": "12",
                "scene_id": "scene-001",
                "resize": "",
                "created_at": "2026-05-22T00:00:00Z",
            }
        )

    with pytest.raises(ContractValidationError):
        validate_csv_file(str(index_path), SEQUENCE_INDEX_CONTRACT)
