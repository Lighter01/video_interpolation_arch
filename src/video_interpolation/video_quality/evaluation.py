from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import uuid

import numpy as np
import torch

from video_interpolation.adapters.base import ModelAdapter
from video_interpolation.image_io import tensor_to_uint8_hwc, uint8_hwc_to_tensor
from video_interpolation.inference_runtime import (
    FramePairRequest,
    FramePairResult,
    InferenceMode,
)
from video_interpolation.metrics import compute_psnr, compute_ssim
from video_interpolation.video_quality.config import (
    VideoQualityEvaluationConfig,
    VideoQualityEvaluationResult,
)
from video_interpolation.video_quality.sampling import DecodedTriplet, selected_triplets_from_video
from video_interpolation.video_quality.triplets import write_vimeo_triplet


QUALITY_INTERPOLATION_FACTOR = 2


def evaluate_video_quality(
    *,
    input_path: Path,
    output_path: Path,
    adapter: ModelAdapter,
    mode: InferenceMode,
    runtime_options: Mapping[str, object],
    config: VideoQualityEvaluationConfig,
) -> VideoQualityEvaluationResult:
    if not config.enabled:
        return VideoQualityEvaluationResult()
    _validate_practical_rife_adapter(adapter)

    source_video_id = config.source_video_id or uuid.uuid4().hex[:12]
    root_dir = config.triplet_output_dir or output_path.parent
    triplets = selected_triplets_from_video(
        input_path,
        sample_count=config.sample_count,
        random_seed=config.random_seed,
    )

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    written_count = 0
    for accepted_index, triplet in enumerate(_accepted_triplets(triplets, config.scene_cut_ssim_threshold)):
        triplet_id = f"{accepted_index:06d}"
        write_vimeo_triplet(
            root_dir,
            source_video_id=source_video_id,
            triplet_id=triplet_id,
            left=triplet.left,
            middle=triplet.middle,
            right=triplet.right,
        )
        written_count += 1
        prediction = _predict_quality_middle(
            adapter,
            triplet,
            mode=mode,
            runtime_options=runtime_options,
        )
        target = np.asarray(triplet.middle.convert("RGB"), dtype=np.uint8)
        psnr_values.append(compute_psnr(prediction, target))
        ssim_values.append(compute_ssim(prediction, target))

    return VideoQualityEvaluationResult(
        psnr_mean=_mean(psnr_values),
        ssim_mean=_mean(ssim_values),
        triplets_written=written_count,
        triplet_output_dir=root_dir / source_video_id if written_count > 0 else None,
    )


def _validate_practical_rife_adapter(adapter: ModelAdapter) -> None:
    model_name = str(getattr(adapter, "model_name", ""))
    if not model_name.startswith("practical_rife"):
        raise ValueError("Video quality evaluation is currently supported only for Practical-RIFE adapters.")


def _accepted_triplets(
    triplets: tuple[DecodedTriplet, ...],
    scene_cut_ssim_threshold: float | None,
) -> tuple[DecodedTriplet, ...]:
    if scene_cut_ssim_threshold is None:
        return triplets
    accepted: list[DecodedTriplet] = []
    for triplet in triplets:
        left_middle = compute_ssim(np.asarray(triplet.left.convert("RGB")), np.asarray(triplet.middle.convert("RGB")))
        middle_right = compute_ssim(np.asarray(triplet.middle.convert("RGB")), np.asarray(triplet.right.convert("RGB")))
        if left_middle < scene_cut_ssim_threshold or middle_right < scene_cut_ssim_threshold:
            continue
        accepted.append(triplet)
    return tuple(accepted)


def _predict_quality_middle(
    adapter: ModelAdapter,
    triplet: DecodedTriplet,
    *,
    mode: InferenceMode,
    runtime_options: Mapping[str, object],
) -> np.ndarray:
    predict_frame_pair = getattr(adapter, "predict_frame_pair", None)
    if not callable(predict_frame_pair):
        raise ValueError(f"{adapter.__class__.__name__} does not support quality FramePairRequest evaluation.")

    result = predict_frame_pair(
        FramePairRequest(
            left=_image_to_tensor(triplet.left),
            right=_image_to_tensor(triplet.right),
            mode=mode,
            interpolation_factor=QUALITY_INTERPOLATION_FACTOR,
            backend_options=dict(runtime_options),
        )
    )
    if not isinstance(result, FramePairResult):
        raise ValueError(
            f"{adapter.__class__.__name__}.predict_frame_pair returned {type(result).__name__}, "
            "expected FramePairResult."
        )
    return tensor_to_uint8_hwc(result.middle_frame)


def _image_to_tensor(image) -> torch.Tensor:
    return uint8_hwc_to_tensor(np.asarray(image.convert("RGB"), dtype=np.uint8))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))
