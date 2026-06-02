from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image

from video_interpolation.data.preprocessing import (
    decode_selected_video_frames,
    read_video_metadata,
)


@dataclass(frozen=True)
class DecodedTriplet:
    start_index: int
    left: Image.Image
    middle: Image.Image
    right: Image.Image


def sample_triplet_starts(
    *,
    frame_count: int,
    sample_count: int,
    random_seed: int | None = None,
) -> tuple[int, ...]:
    if frame_count < 3 or sample_count <= 0:
        return ()
    valid_starts = tuple(range(frame_count - 2))
    if sample_count >= len(valid_starts):
        return valid_starts
    rng = random.Random(random_seed)
    return tuple(sorted(rng.sample(valid_starts, sample_count)))


def selected_triplets_from_video(
    video_path: Path,
    *,
    sample_count: int,
    random_seed: int | None = None,
    decode_strategy: str = "auto",
) -> tuple[DecodedTriplet, ...]:
    metadata = read_video_metadata(video_path)
    starts = sample_triplet_starts(
        frame_count=metadata.frame_count,
        sample_count=sample_count,
        random_seed=random_seed,
    )
    if not starts:
        return ()

    frame_indices = sorted({index for start in starts for index in (start, start + 1, start + 2)})
    decode_result = decode_selected_video_frames(
        video_path,
        frame_indices,
        fps=metadata.fps,
        strategy=decode_strategy,
    )
    triplets: list[DecodedTriplet] = []
    for start in starts:
        frames = [decode_result.frames.get(index) for index in (start, start + 1, start + 2)]
        if any(frame is None for frame in frames):
            continue
        left, middle, right = frames
        triplets.append(
            DecodedTriplet(
                start_index=start,
                left=_copy_rgb(left),
                middle=_copy_rgb(middle),
                right=_copy_rgb(right),
            )
        )
    return tuple(triplets)


def _copy_rgb(image: Image.Image | None) -> Image.Image:
    if image is None:  # pragma: no cover - guarded before this helper is called.
        raise ValueError("Cannot copy a missing image frame")
    return image.convert("RGB").copy()
