from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image


@dataclass(frozen=True)
class DecodedTriplet:
    start_index: int
    left: Image.Image
    middle: Image.Image
    right: Image.Image


def first_triplet_starts(
    *,
    frame_count: int,
    sample_count: int,
) -> tuple[int, ...]:
    if frame_count < 3 or sample_count <= 0:
        return ()
    return tuple(range(min(sample_count, frame_count - 2)))


def first_triplets_from_video(
    video_path: Path,
    *,
    sample_count: int,
) -> tuple[DecodedTriplet, ...]:
    if sample_count <= 0:
        return ()

    decoded_frames: list[Image.Image] = []
    max_frames = sample_count + 2
    with av.open(str(video_path), mode="r") as container:
        try:
            stream = container.streams.video[0]
        except IndexError as exc:
            raise ValueError(f"{video_path} does not contain a video stream.") from exc
        for frame in container.decode(stream):
            decoded_frames.append(frame.to_image().convert("RGB").copy())
            if len(decoded_frames) >= max_frames:
                break

    triplets: list[DecodedTriplet] = []
    for start in first_triplet_starts(frame_count=len(decoded_frames), sample_count=sample_count):
        left, middle, right = decoded_frames[start : start + 3]
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
