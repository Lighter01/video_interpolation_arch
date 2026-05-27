import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from video_interpolation.contracts import TRIPLET_MANIFEST_CONTRACT, validate_csv_file
from video_interpolation.settings import Settings, load_settings


@dataclass(frozen=True)
class TripletFrames:
    left: torch.Tensor
    middle: torch.Tensor
    right: torch.Tensor


SynchronizedTripletTransform = Callable[[TripletFrames], TripletFrames | tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class UniversalTripletDataset(Dataset):
    """PyTorch dataset for Stage 1 triplet manifests."""

    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path | None = None,
        transform: SynchronizedTripletTransform | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.settings = settings or load_settings()
        self.dataset_root = (
            dataset_root
            if dataset_root is not None and dataset_root.is_absolute()
            else self.settings.resolve_path(dataset_root or self.settings.dataset_root)
        )
        self.transform = transform
        validate_csv_file(str(self.manifest_path), TRIPLET_MANIFEST_CONTRACT)
        self.rows = _read_manifest_rows(self.manifest_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        frames = TripletFrames(
            left=_load_frame_tensor(self.dataset_root / row["left_frame_path"]),
            middle=_load_frame_tensor(self.dataset_root / row["mid_frame_path"]),
            right=_load_frame_tensor(self.dataset_root / row["right_frame_path"]),
        )
        if self.transform is not None:
            transformed = self.transform(frames)
            if isinstance(transformed, TripletFrames):
                frames = transformed
            else:
                frames = TripletFrames(*transformed)

        return {
            "left": frames.left,
            "middle": frames.middle,
            "right": frames.right,
            "metadata": _metadata_from_row(row),
        }


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return [dict(row) for row in csv.DictReader(csv_file)]


def _load_frame_tensor(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Triplet frame does not exist: {path}")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _metadata_from_row(row: Mapping[str, str]) -> dict[str, str]:
    return {
        "sample_id": row["sample_id"],
        "source_group": row["source_group"],
        "source_dataset": row["source_dataset"],
        "source_video_id": row["source_video_id"],
        "sequence_id": row["sequence_id"],
        "left_frame_path": row["left_frame_path"],
        "mid_frame_path": row["mid_frame_path"],
        "right_frame_path": row["right_frame_path"],
        "t_value": row["t_value"],
        "width": row["width"],
        "height": row["height"],
        "fps": row["fps"],
        "frame_step": row["frame_step"],
    }


def sample_metadata(rows: Sequence[Mapping[str, str]], limit: int = 3) -> list[dict[str, str]]:
    return [dict(row) for row in rows[:limit]]
