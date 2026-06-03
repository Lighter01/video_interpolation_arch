from __future__ import annotations

from pathlib import Path

import bentoml

from video_interpolation.serving import (
    DEFAULT_SERVING_CODEC,
    DEFAULT_SERVING_ONNX_PROVIDER,
    PracticalRIFEServingConfig,
    PracticalRIFEVideoInferenceRunner,
)

SERVICE_CONFIG = PracticalRIFEServingConfig(
    backend="onnx",
    provider=DEFAULT_SERVING_ONNX_PROVIDER,
    codec=DEFAULT_SERVING_CODEC,
)


def create_runner() -> PracticalRIFEVideoInferenceRunner:
    return PracticalRIFEVideoInferenceRunner(SERVICE_CONFIG)


@bentoml.service(name="practical_rife_v4_26_onnx_example")
class PracticalRIFEOnnxService:
    def __init__(self) -> None:
        self.runner = create_runner()

    @bentoml.api
    def interpolate_video(
        self,
        input_path: str,
        output_path: str,
        interpolation_factor: int = 2,
        scale: float = 1.0,
        output_playback_mode: str = "real_time",
        enable_quality_evaluation: bool = True,
    ) -> dict[str, object]:
        result = self.runner.run(
            input_path=Path(input_path),
            output_path=Path(output_path),
            interpolation_factor=interpolation_factor,
            scale=scale,
            output_playback_mode=output_playback_mode,
            enable_quality_evaluation=enable_quality_evaluation,
        )
        return {
            "output_path": str(result.output_path),
            "interpolation_factor": result.interpolation_factor,
            "scale": scale,
            "input_fps": result.input_fps,
            "output_fps": result.output_fps,
            "output_playback_mode": result.output_playback_mode,
            "backend": result.runtime_backend,
            "execution_mode": result.execution_mode,
            "pairs_processed": result.pairs_processed,
            "frames_written": result.frames_written,
            "psnr_mean": result.quality_psnr_mean,
            "ssim_mean": result.quality_ssim_mean,
            "quality_triplets_written": result.quality_triplets_written,
            "quality_triplet_output_dir": str(result.quality_triplet_output_dir)
            if result.quality_triplet_output_dir is not None
            else None,
            "quality_error": result.quality_error,
        }
