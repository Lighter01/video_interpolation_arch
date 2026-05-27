from __future__ import annotations

import csv

from PIL import Image

from video_interpolation.contracts import SEQUENCE_INDEX_CONTRACT, validate_csv_file
from video_interpolation.data.indexing import VimeoTripletIndexConfig, build_vimeo_triplet_index
from video_interpolation.settings import Settings


def test_vimeo_triplet_index_generation_uses_relative_sequence_paths(tmp_path) -> None:
    dataset_root = tmp_path / "datasets"
    source_root = dataset_root / "sources" / "vimeo_triplet"
    sequence_dir = source_root / "sequences" / "00001" / "0001"
    sequence_dir.mkdir(parents=True)
    for frame_name in ("im1.png", "im2.png", "im3.png"):
        Image.new("RGB", (8, 6), color=(12, 34, 56)).save(sequence_dir / frame_name)

    (source_root / "tri_trainlist.txt").write_text("00001/0001\n", encoding="utf-8")
    (source_root / "tri_testlist.txt").write_text("", encoding="utf-8")

    output_path = tmp_path / "vimeo_sequence_index.csv"
    result = build_vimeo_triplet_index(
        VimeoTripletIndexConfig(output_path=output_path),
        settings=Settings(DATASET_ROOT=dataset_root),
    )

    assert result.rows_written == 1
    validate_csv_file(str(output_path), SEQUENCE_INDEX_CONTRACT)
    with output_path.open(newline="", encoding="utf-8") as csv_file:
        row = next(csv.DictReader(csv_file))

    assert row["relative_sequence_dir"] == "sources/vimeo_triplet/sequences/00001/0001"
    assert row["width"] == "8"
    assert row["height"] == "6"
    assert row["original_split"] == "train"
