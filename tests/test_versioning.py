from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from video_interpolation.contracts import (
    SEQUENCE_INDEX_COLUMNS,
    TRIPLET_MANIFEST_CONTRACT,
    validate_csv_file,
    validate_relative_artifact_path,
)
from video_interpolation.data.indexing import GlobalIndexConfig, build_global_sequence_index
from video_interpolation.data.versioning import DatasetVersionConfig, build_dataset_version
from video_interpolation.settings import Settings


def test_global_sequence_index_combines_sources_with_relative_paths(tmp_path) -> None:
    dataset_root = tmp_path / "datasets"
    first_index = dataset_root / "sources" / "z_source" / "sequence_index.csv"
    second_index = dataset_root / "sources" / "a_source" / "sequence_index.csv"
    _write_sequence_index(first_index, [_sequence_row("z", "z_dataset", "video_b", "seq_b")])
    _write_sequence_index(second_index, [_sequence_row("a", "a_dataset", "video_a", "seq_a")])

    result = build_global_sequence_index(
        GlobalIndexConfig(output_path=Path("global_sequence_index.csv")),
        settings=Settings(DATASET_ROOT=dataset_root),
    )

    with result.index_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert result.rows_written == 2
    assert [row["source_group"] for row in rows] == ["a", "z"]
    for row in rows:
        validate_relative_artifact_path(row["relative_sequence_dir"], "relative_sequence_dir")


def test_dataset_version_split_by_source_video_id_without_leakage(tmp_path) -> None:
    dataset_root = tmp_path / "datasets"
    global_index_path = dataset_root / "global_sequence_index.csv"
    rows = []
    for index in range(4):
        source_video_id = f"video_{index}"
        sequence_id = f"seq_{index}"
        relative_sequence_dir = f"sources/unit/sequences/{source_video_id}/000000"
        _write_triplet_frames(dataset_root / relative_sequence_dir)
        rows.append(
            _sequence_row(
                "unit",
                "unit_dataset",
                source_video_id,
                sequence_id,
                relative_sequence_dir=relative_sequence_dir,
            )
        )
    _write_sequence_index(global_index_path, rows)

    result = build_dataset_version(
        DatasetVersionConfig(
            dataset_version_id="unit_version",
            output_root=tmp_path / "dataset_versions",
            split_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
        ),
        settings=Settings(DATASET_ROOT=dataset_root),
    )

    split_sources = {
        "train": _source_video_ids(result.train_manifest_path),
        "val": _source_video_ids(result.val_manifest_path),
        "test": _source_video_ids(result.test_manifest_path),
    }

    assert split_sources["train"].isdisjoint(split_sources["val"])
    assert split_sources["train"].isdisjoint(split_sources["test"])
    assert split_sources["val"].isdisjoint(split_sources["test"])
    assert result.train_samples + result.val_samples + result.test_samples == 4


def test_dataset_version_generates_triplet_manifest_with_relative_paths(tmp_path) -> None:
    dataset_root = tmp_path / "datasets"
    global_index_path = dataset_root / "global_sequence_index.csv"
    relative_sequence_dir = "sources/unit/sequences/video_001/000000"
    _write_triplet_frames(dataset_root / relative_sequence_dir)
    _write_sequence_index(
        global_index_path,
        [
            _sequence_row(
                "unit",
                "unit_dataset",
                "video_001",
                "seq_001",
                relative_sequence_dir=relative_sequence_dir,
            )
        ],
    )

    result = build_dataset_version(
        DatasetVersionConfig(
            dataset_version_id="unit_version",
            output_root=tmp_path / "dataset_versions",
            split_ratios={"train": 1.0, "val": 0.0, "test": 0.0},
        ),
        settings=Settings(DATASET_ROOT=dataset_root),
    )

    validate_csv_file(str(result.train_manifest_path), TRIPLET_MANIFEST_CONTRACT)
    with result.train_manifest_path.open(newline="", encoding="utf-8") as csv_file:
        row = next(csv.DictReader(csv_file))

    assert row["left_frame_path"] == f"{relative_sequence_dir}/frame_000.png"
    assert row["mid_frame_path"] == f"{relative_sequence_dir}/frame_001.png"
    assert row["right_frame_path"] == f"{relative_sequence_dir}/frame_002.png"
    for column in ("left_frame_path", "mid_frame_path", "right_frame_path"):
        validate_relative_artifact_path(row[column], column)


def _write_triplet_frames(sequence_dir: Path) -> None:
    sequence_dir.mkdir(parents=True)
    for index, color in enumerate(((0, 0, 0), (32, 32, 32), (64, 64, 64))):
        Image.new("RGB", (8, 6), color=color).save(sequence_dir / f"frame_{index:03d}.png")


def _write_sequence_index(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SEQUENCE_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _sequence_row(
    source_group: str,
    source_dataset: str,
    source_video_id: str,
    sequence_id: str,
    *,
    relative_sequence_dir: str | None = None,
) -> dict[str, str]:
    relative_sequence_dir = relative_sequence_dir or f"sources/{source_dataset}/sequences/{source_video_id}/000000"
    return {
        "sequence_id": sequence_id,
        "source_group": source_group,
        "source_dataset": source_dataset,
        "source_video_id": source_video_id,
        "relative_sequence_dir": relative_sequence_dir,
        "sequence_length": "3",
        "frame_step": "0",
        "width": "8",
        "height": "6",
        "fps": "24",
        "frame_start": "0",
        "frame_end": "2",
        "scene_id": "scene_0000",
        "resize": "",
        "created_at": "2026-05-22T00:00:00+00:00",
    }


def _source_video_ids(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return {row["source_video_id"] for row in csv.DictReader(csv_file)}
