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
    random_seed: int | None = None
    scene_cut_ssim_threshold: float | None = 0.75
    fail_policy: VideoQualityFailPolicy = "raise"
    source_video_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("quality evaluation enabled must be a boolean")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 0:
            raise ValueError("quality evaluation sample_count must be a non-negative integer")
        if self.random_seed is not None and (isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int)):
            raise ValueError("quality evaluation random_seed must be an integer when set")
        if self.scene_cut_ssim_threshold is not None:
            threshold = float(self.scene_cut_ssim_threshold)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("quality evaluation scene_cut_ssim_threshold must be in [0, 1]")
            object.__setattr__(self, "scene_cut_ssim_threshold", threshold)
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
