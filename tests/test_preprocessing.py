from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from video_interpolation.data.preprocessing import (
    SceneSpan,
    SequenceSamplingConfig,
    is_static_triplet,
    sample_sequence_candidates,
)


def test_scene_safe_sampling_does_not_cross_scene_boundaries() -> None:
    scenes = [
        SceneSpan(scene_id="scene_0000", start_frame=0, end_frame=4),
        SceneSpan(scene_id="scene_0001", start_frame=5, end_frame=9),
    ]
    config = SequenceSamplingConfig(
        sequence_length=3,
        max_frame_step=2,
        min_sequences_per_video=1,
        max_sequences_per_video=100,
        quota_scale=1.0,
        random_seed=1,
    )

    candidates = sample_sequence_candidates(frame_count=10, scene_spans=scenes, config=config)

    assert candidates
    for candidate in candidates:
        scene = next(scene for scene in scenes if scene.scene_id == candidate.scene_id)
        assert scene.start_frame <= min(candidate.frame_indices)
        assert max(candidate.frame_indices) <= scene.end_frame


def test_sampler_terminates_when_quota_cannot_be_reached() -> None:
    config = SequenceSamplingConfig(
        sequence_length=3,
        max_frame_step=0,
        min_sequences_per_video=10,
        max_sequences_per_video=20,
        quota_scale=1.0,
        random_seed=1,
    )

    candidates = sample_sequence_candidates(
        frame_count=3,
        scene_spans=[SceneSpan(scene_id="scene_0000", start_frame=0, end_frame=2)],
        config=config,
    )

    assert len(candidates) == 1
    assert candidates[0].frame_indices == (0, 1, 2)


def test_frame_step_config_is_rejected_for_longer_sequences() -> None:
    config = SequenceSamplingConfig(sequence_length=4, max_frame_step=1)

    with pytest.raises(ValueError, match="frame-step sampling"):
        sample_sequence_candidates(frame_count=12, scene_spans=[], config=config)


def test_static_triplet_filter_rejects_identical_frames() -> None:
    identical = Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8))
    changed = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))

    assert is_static_triplet(identical, identical, identical, threshold=0.95)
    assert is_static_triplet(identical, identical, changed, threshold=0.95)
    assert is_static_triplet(changed, identical, identical, threshold=0.95)
    assert not is_static_triplet(identical, changed, identical, threshold=0.95)
