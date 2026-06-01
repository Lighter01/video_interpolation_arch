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
    ) -> dict[str, object]:
        result = self.runner.run(
            input_path=Path(input_path),
            output_path=Path(output_path),
            interpolation_factor=interpolation_factor,
            scale=scale,
        )
        return {
            "output_path": str(result.output_path),
            "interpolation_factor": result.interpolation_factor,
            "scale": scale,
            "backend": result.runtime_backend,
            "execution_mode": result.execution_mode,
            "pairs_processed": result.pairs_processed,
            "frames_written": result.frames_written,
        }
