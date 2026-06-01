from pathlib import Path
import json

import av
import numpy as np
import pytest
import torch

from video_interpolation.adapters.base import ModelAdapter
from video_interpolation.inference_benchmark import (
    DEFAULT_VIDEO_BENCHMARK_INPUT,
    BenchmarkExecutionMode,
    VideoBenchmarkConfig,
    VideoBenchmarkRecord,
    benchmark_metrics,
    benchmark_profile_slug,
    discover_benchmark_videos,
    run_video_benchmark,
    video_benchmark_output_path,
    write_video_benchmark_report,
)
from video_interpolation.inference_runtime import (
    FramePairRequest,
    FramePairResult,
    InferenceMode,
    ModelBatchRequest,
    ModelBatchResult,
    RuntimeBackendKind,
)
from video_interpolation.mlflow import MlflowRunConfig
from video_interpolation.settings import Settings


def test_video_benchmark_config_defaults_to_video_input_and_validates_arguments() -> None:
    config = VideoBenchmarkConfig(model_name="ema_vfi_small", backend="onnx", execution_mode="batched")

    assert config.input_path == DEFAULT_VIDEO_BENCHMARK_INPUT
    assert config.backend is RuntimeBackendKind.ONNX
    assert config.execution_mode is BenchmarkExecutionMode.BATCHED

    with pytest.raises(ValueError, match="Unsupported benchmark model"):
        VideoBenchmarkConfig(model_name="amt_s")

    with pytest.raises(ValueError, match="inference_batch_size"):
        VideoBenchmarkConfig(model_name="ema_vfi_small", inference_batch_size=0)

    with pytest.raises(ValueError, match="fixed_2x"):
        VideoBenchmarkConfig(model_name="ema_vfi_small", interpolation_factor=4)

    with pytest.raises(ValueError, match="limit_videos"):
        VideoBenchmarkConfig(model_name="ema_vfi_small", limit_videos=1)

    with pytest.raises(ValueError, match="mutually exclusive"):
        VideoBenchmarkConfig(
            model_name="ema_vfi_small",
            input_path=Path("input.mp4"),
            input_dir=Path("videos"),
        )


def test_discover_benchmark_videos_supports_single_input_and_directory_limit(tmp_path: Path) -> None:
    single_input = tmp_path / "single.mp4"
    single_input.write_bytes(b"video")
    directory = tmp_path / "videos"
    nested = directory / "nested"
    nested.mkdir(parents=True)
    (directory / "a.mp4").write_bytes(b"video")
    (nested / "b.mkv").write_bytes(b"video")

    single = discover_benchmark_videos(
        VideoBenchmarkConfig(
            model_name="ema_vfi_small",
            input_path=single_input,
            mlflow=MlflowRunConfig(enabled=False),
        ),
        settings=Settings(),
    )
    directory_videos = discover_benchmark_videos(
        VideoBenchmarkConfig(
            model_name="ema_vfi_small",
            input_path=None,
            input_dir=directory,
            limit_videos=1,
            mlflow=MlflowRunConfig(enabled=False),
        ),
        settings=Settings(),
    )

    assert [video.relative_path for video in single] == [Path("single.mp4")]
    assert [video.relative_path for video in directory_videos] == [Path("a.mp4")]


def test_video_benchmark_output_path_is_stable_and_preserves_relative_layout() -> None:
    path = video_benchmark_output_path(
        output_dir=Path("outputs/benchmarks/video"),
        profile="rife_onnx_batched",
        relative_input_path=Path("nested/source.mov"),
        repeat_index=3,
        interpolation_factor=4,
    )

    assert path == Path("outputs/benchmarks/video/videos/rife_onnx_batched/nested/source_repeat03_4x.mov")


def test_run_video_benchmark_sequential_records_pipeline_timings(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    _write_synthetic_input_video(input_path, frame_values=(0, 96))
    adapters: list[_RuntimeAdapter] = []

    result = run_video_benchmark(
        VideoBenchmarkConfig(
            model_name="ema_vfi_small",
            backend="torch",
            execution_mode="sequential",
            input_path=input_path,
            limit_pairs=1,
            codec="libx264",
            encoder_options={"crf": "28"},
            output_dir=tmp_path / "benchmark",
            mlflow=MlflowRunConfig(enabled=False),
        ),
        settings=Settings(),
        adapter_factory=_adapter_factory(adapters),
    )

    record = result.records[0]
    assert result.mlflow_run_id is None
    assert result.success
    assert record.status == "ok"
    assert record.pairs_processed == 1
    assert record.frames_written == 3
    assert record.generated_frames == 1
    assert record.decode_sec >= 0
    assert record.preprocessing_sec >= 0
    assert record.model_inference_sec >= 0
    assert record.postprocessing_sec >= 0
    assert record.encode_sec >= 0
    assert record.audio_remux_sec >= 0
    assert record.total_sec >= record.model_inference_sec
    assert record.output_video is not None
    assert record.output_video.is_file()
    assert len(adapters) == 1
    assert len(adapters[0].requests) == 1
    assert result.artifacts.report_path.is_file()
    assert result.artifacts.csv_path.is_file()


def test_run_video_benchmark_batched_nx_records_batch_chunks(tmp_path: Path) -> None:
    input_path = tmp_path / "input_batch.mp4"
    _write_synthetic_input_video(input_path, frame_values=(0, 80, 160))
    adapters: list[_RuntimeAdapter] = []

    result = run_video_benchmark(
        VideoBenchmarkConfig(
            model_name="practical_rife_v4_26",
            backend="torch",
            execution_mode="batched",
            input_path=input_path,
            interpolation_mode=InferenceMode.ARBITRARY_NX,
            interpolation_factor=4,
            inference_batch_size=3,
            limit_pairs=2,
            codec="libx264",
            encoder_options={"crf": "28"},
            output_dir=tmp_path / "benchmark_batch",
            mlflow=MlflowRunConfig(enabled=False),
        ),
        settings=Settings(),
        adapter_factory=_adapter_factory(adapters),
    )

    record = result.records[0]
    assert result.success
    assert record.pairs_processed == 2
    assert record.generated_frames == 6
    assert record.frames_written == 9
    assert record.batch_chunks_processed == 2
    assert record.model_batch_requests == 2
    assert record.inference_batch_size == 3
    assert record.runtime_backend == RuntimeBackendKind.TORCH.value
    assert [request.flattened_size for request in adapters[0].batch_requests] == [3, 3]


def test_run_video_benchmark_records_failed_video_and_writes_reports(tmp_path: Path) -> None:
    input_path = tmp_path / "broken.mp4"
    input_path.write_bytes(b"not a video")

    result = run_video_benchmark(
        VideoBenchmarkConfig(
            model_name="ema_vfi_small",
            backend="torch",
            input_path=input_path,
            output_dir=tmp_path / "failed",
            mlflow=MlflowRunConfig(enabled=False),
        ),
        settings=Settings(),
        adapter_factory=_adapter_factory([]),
    )

    assert not result.success
    assert result.records[0].status == "failed"
    assert result.records[0].error
    assert result.artifacts.report_path.is_file()
    assert result.artifacts.csv_path.is_file()


def test_benchmark_metrics_aggregate_video_timing_and_throughput() -> None:
    records = (
        _record(pairs_processed=2, generated_frames=6, total_sec=2.0, model_inference_sec=1.0),
        _record(pairs_processed=1, generated_frames=3, total_sec=1.0, model_inference_sec=0.5),
    )

    metrics = benchmark_metrics(records)

    assert metrics["benchmark.records"] == 2.0
    assert metrics["benchmark.successful_records"] == 2.0
    assert metrics["benchmark.pairs_processed"] == 3.0
    assert metrics["benchmark.generated_frames"] == 9.0
    assert metrics["benchmark.total_pairs_per_sec"] == pytest.approx(1.0)
    assert metrics["benchmark.model_generated_frames_per_sec"] == pytest.approx(6.0)


def test_write_video_benchmark_report_writes_json_and_csv(tmp_path: Path) -> None:
    config = VideoBenchmarkConfig(
        model_name="ema_vfi_small",
        input_path=tmp_path / "input.mp4",
        output_dir=tmp_path,
        mlflow=MlflowRunConfig(enabled=False),
    )
    artifacts = write_video_benchmark_report(tmp_path, config=config, records=(_record(),))

    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    csv_text = artifacts.csv_path.read_text(encoding="utf-8")

    assert report["config"]["model_name"] == "ema_vfi_small"
    assert report["records"][0]["model_name"] == "ema_vfi_small"
    assert "benchmark.total_sec" in report["summary"]
    assert "model_name,backend,execution_mode" in csv_text
    assert "decode_sec" in csv_text
    assert "audio_remux_sec" in csv_text


def test_benchmark_profile_slug_includes_video_runtime_profile() -> None:
    config = VideoBenchmarkConfig(
        model_name="practical_rife_v4_26",
        backend="onnx",
        execution_mode="batched",
        interpolation_mode="arbitrary_nx",
        interpolation_factor=4,
        inference_batch_size=6,
        providers=("CPUExecutionProvider",),
        input_path=Path("input.mp4"),
        mlflow=MlflowRunConfig(enabled=False),
    )

    assert benchmark_profile_slug(config) == "practical_rife_v4_26_onnx_cpu_batched_arbitrary_nx_4x_b6"


class _RuntimeAdapter(ModelAdapter):
    model_name = "runtime_adapter"

    def __init__(self) -> None:
        self.requests: list[FramePairRequest] = []
        self.batch_requests: list[ModelBatchRequest] = []
        self.closed = False
        self.loaded = False

    def validate_environment(self):  # pragma: no cover - not used by these tests.
        raise NotImplementedError

    def build_model(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path: Path | None = None) -> None:
        del checkpoint_path
        self.loaded = True

    def save_checkpoint(self, checkpoint_path: Path) -> None:  # pragma: no cover
        raise NotImplementedError

    def train(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def eval(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def predict_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise AssertionError("benchmark video inference should use runtime request methods")

    def predict_frame_pair(self, request: FramePairRequest) -> FramePairResult:
        self.requests.append(request)
        frames = tuple(_interpolate(request.left, request.right, timestep) for timestep in request.timesteps)
        return FramePairResult(
            intermediate_frames=frames,
            timesteps=request.timesteps,
            mode=request.mode,
            interpolation_factor=request.interpolation_factor,
            backend_kind=RuntimeBackendKind.TORCH,
            model_name=self.model_name,
            original_shape=request.original_shape,
        )

    def predict_frame_pairs_batch(self, request: ModelBatchRequest) -> ModelBatchResult:
        self.batch_requests.append(request)
        outputs = tuple(
            tuple(_interpolate(request.left[pair_index], request.right[pair_index], timestep) for timestep in request.timesteps)
            for pair_index in range(request.pair_count)
        )
        return ModelBatchResult(
            outputs=outputs,
            timesteps=request.timesteps,
            mode=request.mode,
            interpolation_factor=request.interpolation_factor,
            backend_kind=RuntimeBackendKind.TORCH,
            model_name=self.model_name,
            original_shape=request.original_shape,
            metadata={
                "flattening_order": request.flattening_order,
                "flattened_size": request.flattened_size,
                "model_call_count": 1,
            },
        )

    def close(self) -> None:
        self.closed = True


def _adapter_factory(adapters: list[_RuntimeAdapter]):
    def benchmark_factory(_config: VideoBenchmarkConfig, _settings: Settings):
        def factory(_model_config, _run_settings: Settings) -> ModelAdapter:
            adapter = _RuntimeAdapter()
            adapters.append(adapter)
            return adapter

        return factory

    return benchmark_factory


def _interpolate(left: torch.Tensor, right: torch.Tensor, timestep: float) -> torch.Tensor:
    return torch.clamp((1.0 - timestep) * left + timestep * right, 0.0, 1.0)


def _write_synthetic_input_video(path: Path, *, frame_values: tuple[int, ...] = (32, 96)) -> None:
    with av.open(str(path), mode="w") as container:
        video_stream = container.add_stream("libx264", rate=24)
        video_stream.width = 32
        video_stream.height = 32
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=48_000)
        audio_stream.layout = "mono"

        for offset in frame_values:
            frame = av.VideoFrame.from_ndarray(_solid_rgb_frame(offset), format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

        samples = (np.sin(2 * np.pi * 440 * np.arange(4_800) / 48_000) * 1000).astype(np.int16)
        audio_frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        audio_frame.sample_rate = 48_000
        for packet in audio_stream.encode(audio_frame):
            container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)


def _solid_rgb_frame(value: int) -> np.ndarray:
    return np.full((32, 32, 3), value, dtype=np.uint8)


def _record(
    *,
    pairs_processed: int = 1,
    generated_frames: int = 1,
    total_sec: float = 1.0,
    model_inference_sec: float = 0.5,
    status: str = "ok",
) -> VideoBenchmarkRecord:
    return VideoBenchmarkRecord(
        model_name="ema_vfi_small",
        backend="torch",
        execution_mode="sequential",
        interpolation_mode="fixed_2x",
        interpolation_factor=2,
        inference_batch_size=1,
        repeat_index=0,
        input_video=Path("input.mp4"),
        relative_input_video=Path("input.mp4"),
        output_video=Path("output.mp4"),
        status=status,
        error=None,
        provider=None,
        artifact_path=None,
        rife_scale=None,
        source_frames=pairs_processed + 1,
        pairs_processed=pairs_processed,
        generated_frames=generated_frames,
        frames_written=pairs_processed + 1 + generated_frames,
        input_fps=24.0,
        output_fps=48.0,
        decode_sec=0.01,
        preprocessing_sec=0.02,
        model_inference_sec=model_inference_sec,
        postprocessing_sec=0.03,
        encode_sec=0.04,
        audio_remux_sec=0.01,
        total_sec=total_sec,
        model_pairs_per_sec=pairs_processed / model_inference_sec,
        total_pairs_per_sec=pairs_processed / total_sec,
        generated_frames_per_sec=generated_frames / model_inference_sec,
        total_generated_frames_per_sec=generated_frames / total_sec,
        batch_chunks_processed=0,
        model_batch_requests=0,
        runtime_backend="torch",
        peak_vram_mb=None,
        runtime_options={},
    )
