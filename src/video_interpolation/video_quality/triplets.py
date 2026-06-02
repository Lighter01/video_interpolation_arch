from __future__ import annotations

from pathlib import Path

from PIL import Image


def write_vimeo_triplet(
    root_dir: Path,
    *,
    source_video_id: str,
    triplet_id: str,
    left: Image.Image,
    middle: Image.Image,
    right: Image.Image,
) -> Path:
    triplet_dir = root_dir / source_video_id / triplet_id
    triplet_dir.mkdir(parents=True, exist_ok=True)
    left.convert("RGB").save(triplet_dir / "im1.png")
    middle.convert("RGB").save(triplet_dir / "im2.png")
    right.convert("RGB").save(triplet_dir / "im3.png")
    return triplet_dir
