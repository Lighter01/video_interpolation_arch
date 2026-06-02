from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


VideoQualityFailPolicy = Literal["raise", "warn"]


@dataclass(frozen=True)
class VideoQualityEvaluationConfig:
    enabled: bool = False
    triplet_output_dir: Path | None = None
    sample_count: int = 16
    max_image_side: int | None = 360
    write_triplets: bool = False
    fail_policy: VideoQualityFailPolicy = "raise"
    source_video_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("quality evaluation enabled must be a boolean")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 0:
            raise ValueError("quality evaluation sample_count must be a non-negative integer")
        if self.max_image_side is not None:
            if (
                isinstance(self.max_image_side, bool)
                or not isinstance(self.max_image_side, int)
                or self.max_image_side <= 0
            ):
                raise ValueError("quality evaluation max_image_side must be a positive integer or None")
        if not isinstance(self.write_triplets, bool):
            raise ValueError("quality evaluation write_triplets must be a boolean")
        if self.fail_policy not in ("raise", "warn"):
            raise ValueError("quality evaluation fail_policy must be 'raise' or 'warn'")
        if self.triplet_output_dir is not None:
            object.__setattr__(self, "triplet_output_dir", Path(self.triplet_output_dir))
        if self.source_video_id is not None and not self.source_video_id:
            raise ValueError("quality evaluation source_video_id must be non-empty when set")

    @classmethod
    def disabled(cls) -> "VideoQualityEvaluationConfig":
        return cls(enabled=False)


@dataclass(frozen=True)
class VideoQualityEvaluationResult:
    psnr_mean: float | None = None
    ssim_mean: float | None = None
    triplets_written: int = 0
    triplet_output_dir: Path | None = None
    error: str | None = None
