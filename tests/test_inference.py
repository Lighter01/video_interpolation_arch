from pathlib import Path

import av
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from typer.testing import CliRunner

from video_interpolation.adapters.base import ModelAdapter
from video_interpolation.adapters.baseline import BaselineAdapter, BaselineAdapterConfig
from video_interpolation.batch_inference import (
    batch_measurements_path,
    batch_output_path,
    batch_run_name,
    discover_inference_videos,
    resolve_batch_target_names,
    write_batch_measurements_csv,
)
from video_interpolation.cli import app
from video_interpolation.inference import (
    VideoInferenceConfig,
    resolve_encoder_options,
    run_ema_video_inference,
    run_video_inference,
)
from video_interpolation.inference_runtime import FramePairRequest, FramePairResult, InferenceMode, RuntimeBackendKind
from video_interpolation.inference_runtime.api import ModelBatchRequest, ModelBatchResult
from video_interpolation.mlflow import MlflowRunConfig
from video_interpolation.settings import Settings


def test_video_inference_config_parses_pyav_output_fields() -> None:
    config = VideoInferenceConfig.from_mapping(
        {
            "input_path": "raw_data/tmp_test/Dora.mp4",
            "output_path": "outputs/inference/test.mp4",
            "codec": "libx264",
            "container": "mp4",
            "pix_fmt": "yuv420p",
            "frame_format": "rgb24",
            "encoder_options_by_codec": {"libx264": {"preset": "slow"}},
            "encoder_options": {"crf": "21"},
            "mlflow": {"enabled": False},
        }
    )

    config.validate()
    assert config.codec == "libx264"
    assert config.container == "mp4"
    assert config.encoder_options_by_codec["libx264"]["preset"] == "slow"
    assert config.encoder_options["crf"] == "21"


def test_encoder_option_resolution_uses_codec_specific_defaults_without_leaking_options() -> None:
    nvenc_options = resolve_encoder_options("h264_nvenc", 60.0)
    libx_options = resolve_encoder_options("libx264", 60.0)

    assert nvenc_options["rc"] == "vbr"
    assert nvenc_options["cq"] == "23"
    assert "crf" not in nvenc_options
    assert libx_options["crf"] == "22"
    assert "rc" not in libx_options
    assert "cq" not in libx_options


def test_encoder_option_resolution_applies_per_codec_and_top_level_overrides() -> None:
    options = resolve_encoder_options(
        "libx264",
        59.94,
        encoder_options={"crf": 20},
        encoder_options_by_codec={"libx264": {"preset": "slow"}},
    )

    assert options["preset"] == "slow"
    assert options["crf"] == "20"
    assert options["bf"] == "0"
    assert options["g"] == "119"


def test_mp4v_is_rejected_as_legacy_opencv_fourcc() -> None:
    with pytest.raises(ValueError, match="mp4v"):
        resolve_encoder_options("mp4v", 60.0)

    config = VideoInferenceConfig(input_path=Path("in.mp4"), output_path=Path("out.mp4"), codec="mp4v")
    with pytest.raises(ValueError, match="mp4v"):
        config.validate()


def test_pyav_inference_writer_writes_readable_video_and_preserves_audio(tmp_path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _write_synthetic_input_video(input_path)

    result = run_ema_video_inference(
        VideoInferenceConfig(
            input_path=input_path,
            output_path=output_path,
            codec="libx264",
            encoder_options={"crf": "28"},
            limit_pairs=1,
            mlflow=MlflowRunConfig(enabled=False),
        ),
        settings=Settings(),
        adapter=_BlendAdapter(),
    )

    assert result.frames_written == 3
    assert result.pairs_processed == 1
    assert result.interpolation_mode == "fixed_2x"
    assert result.interpolation_factor == 2
    assert result.runtime_backend == "adapter"
    assert dict(result.runtime_options) == {}
    assert result.audio_streams_available == 1
    assert result.audio_streams_preserved == 1
    assert result.timing.decode_sec >= 0
    assert result.timing.preprocessing_sec >= 0
    assert result.timing.model_inference_sec == result.model_inference_elapsed_sec
    assert result.timing.postprocessing_sec >= 0
    assert result.timing.encode_sec >= 0
    assert result.timing.audio_remux_sec >= 0
    assert result.timing.total_sec == result.total_elapsed_sec
    assert result.timing.total_sec >= result.timing.model_inference_sec
    assert output_path.is_file()

    with av.open(str(output_path)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert len(list(container.decode(video=0))) == 3


@pytest.mark.parametrize(
    ("factor", "expected_frames", "expected_fps"),
    [
        (2, 3, 48.0),
        (4, 5, 96.0),
        (8, 9, 192.0),
    ],
)
def test_video_inference_uses_runtime_api_for_fixed_2x_and_arbitrary_nx(
    tmp_path,
    factor: int,
    expected_frames: int,
    expected_fps: float,
) -> None:
    input_path = tmp_path / f"input_{factor}x.mp4"
    output_path = tmp_path / f"output_{factor}x.mp4"
    _write_synthetic_input_video(input_path)
    mode = InferenceMode.FIXED_2X if factor == 2 else InferenceMode.ARBITRARY_NX
    adapter = _RuntimeAdapter()

    result = run_video_inference(
        VideoInferenceConfig(
            input_path=input_path,
            output_path=output_path,
            model=_runtime_model_config(),
            interpolation_mode=mode,
            interpolation_factor=factor,
            codec="libx264",
            encoder_options={"crf": "28"},
            limit_pairs=1,
            mlflow=MlflowRunConfig(enabled=False),
        ),
        adapter_factory=lambda _model_config, _settings: adapter,
        settings=Settings(),
        adapter=adapter,
    )

    assert result.pairs_processed == 1
    assert result.frames_written == expected_frames
    assert result.output_fps == expected_fps
    assert result.interpolation_mode == mode.value
    assert result.interpolation_factor == factor
    assert result.runtime_backend == RuntimeBackendKind.TORCH.value
    assert dict(result.runtime_options) == {}
    assert [request.interpolation_factor for request in adapter.requests] == [factor]

    with av.open(str(output_path)) as container:
        assert len(list(container.decode(video=0))) == expected_frames


@pytest.mark.parametrize(
    ("factor", "expected_frames", "expected_fps"),
    [
        (2, 5, 48.0),
        (4, 9, 96.0),
        (8, 17, 192.0),
    ],
)
def test_video_batched_inference_counts_frames_and_multiplies_output_fps(
    tmp_path,
    factor: int,
    expected_frames: int,
    expected_fps: float,
) -> None:
    input_path = tmp_path / f"batch_count_{factor}x.mp4"
    output_path = tmp_path / f"batch_count_{factor}x_output.mp4"
    _write_synthetic_input_video(input_path, frame_values=(0, 80, 160))
    adapter = _BatchRuntimeAdapter()
    mode = InferenceMode.FIXED_2X if factor == 2 else InferenceMode.ARBITRARY_NX

    result = run_video_inference(
        VideoInferenceConfig(
            input_path=input_path,
            output_path=output_path,
            model=_runtime_model_config(),
            interpolation_mode=mode,
            interpolation_factor=factor,
            execution_mode="batched",
            inference_batch_size=32,
            codec="libx264",
            encoder_options={"crf": "28"},
            limit_pairs=2,
            mlflow=MlflowRunConfig(enabled=False),
        ),
        adapter_factory=lambda _model_config, _settings: adapter,
        settings=Settings(),
        adapter=adapter,
    )

    assert result.frames_written == expected_frames
    assert result.output_fps == expected_fps
    assert result.execution_mode == "batched"
    assert result.requested_execution_mode == "batched"
    assert result.inference_batch_size == 32
    assert result.batch_chunks_processed == 1
    assert result.model_batch_requests == 1
    assert len(adapter.batch_requests) == 1

    with av.open(str(output_path)) as container:
        assert len(list(container.decode(video=0))) == expected_frames


def test_video_batched_inference_uses_one_frame_chunk_overlap(tmp_path) -> None:
    input_path = tmp_path / "batch_overlap_input.mp4"
    output_path = tmp_path / "batch_overlap_output.mp4"
    _write_synthetic_input_video(input_path, frame_values=(0, 40, 80, 120, 160))
    decoded_values = [int(round(value)) for value in _decode_video_frame_means(input_path)]
    adapter = _BatchRuntimeAdapter()

    result = run_video_inference(
        VideoInferenceConfig(
            input_path=input_path,
            output_path=output_path,
            model=_runtime_model_config(),
            interpolation_mode=InferenceMode.FIXED_2X,
            interpolation_factor=2,
            execution_mode="batched",
            inference_batch_size=2,
            codec="libx264",
            encoder_options={"crf": "28"},
            mlflow=MlflowRunConfig(enabled=False),
        ),
        adapter_factory=lambda _model_config, _settings: adapter,
        settings=Settings(),
        adapter=adapter,
    )

    assert result.pairs_processed == 4
    assert result.frames_written == 9
    assert result.batch_chunks_processed == 2
    assert [request.pair_count for request in adapter.batch_requests] == [2, 2]
    assert adapter.batch_pair_values == [
        [(decoded_values[0], decoded_values[1]), (decoded_values[1], decoded_values[2])],
        [(decoded_values[2], decoded_values[3]), (decoded_values[3], decoded_values[4])],
    ]


def test_video_inference_orders_nx_frames_by_timestep(tmp_path) -> None:
    input_path = tmp_path / "input_order.mp4"
    output_path = tmp_path / "output_order.mp4"
    _write_synthetic_input_video(input_path, frame_values=(0, 240))
    adapter = _RuntimeAdapter()

    run_video_inference(
        VideoInferenceConfig(
            input_path=input_path,
            output_path=output_path,
            model=_runtime_model_config(),
            interpolation_mode=InferenceMode.ARBITRARY_NX,
            interpolation_factor=4,
            codec="libx264",
            encoder_options={"crf": "0", "preset": "ultrafast"},
            limit_pairs=1,
            mlflow=MlflowRunConfig(enabled=False),
        ),
        adapter_factory=lambda _model_config, _settings: adapter,
        settings=Settings(),
        adapter=adapter,
    )

    with av.open(str(output_path)) as container:
        means = [float(frame.to_ndarray(format="rgb24").mean()) for frame in container.decode(video=0)]

    assert len(means) == 5
    assert means == sorted(means)
    assert adapter.requests[0].timesteps == (0.25, 0.5, 0.75)


def test_video_batched_inference_orders_nx_frames_by_timestep(tmp_path) -> None:
    input_path = tmp_path / "batch_order_input.mp4"
    output_path = tmp_path / "batch_order_output.mp4"
    _write_synthetic_input_video(input_path, frame_values=(0, 240))
    adapter = _BatchRuntimeAdapter()

    result = run_video_inference(
        VideoInferenceConfig(
            input_path=input_path,
            output_path=output_path,
            model=_runtime_model_config(),
            interpolation_mode=InferenceMode.ARBITRARY_NX,
            interpolation_factor=4,
            execution_mode="batched",
            inference_batch_size=3,
            codec="libx264",
            encoder_options={"crf": "0", "preset": "ultrafast"},
            limit_pairs=1,
            mlflow=MlflowRunConfig(enabled=False),
        ),
        adapter_factory=lambda _model_config, _settings: adapter,
        settings=Settings(),
        adapter=adapter,
    )

    assert result.execution_mode == "batched"
    assert adapter.batch_requests[0].timesteps == (0.25, 0.5, 0.75)
    with av.open(str(output_path)) as container:
        means = [float(frame.to_ndarray(format="rgb24").mean()) for frame in container.decode(video=0)]

    assert len(means) == 5
    assert means == sorted(means)


def test_video_inference_passes_runtime_options_to_request_and_result(tmp_path) -> None:
    input_path = tmp_path / "input_scale.mp4"
    output_path = tmp_path / "output_scale.mp4"
    _write_synthetic_input_video(input_path)
    adapter = _RuntimeAdapter()

    result = run_video_inference(
        VideoInferenceConfig(
            input_path=input_path,
            output_path=output_path,
            model=_runtime_model_config(),
            interpolation_mode=InferenceMode.ARBITRARY_NX,
            interpolation_factor=4,
            runtime_options={"scale": 0.5},
            codec="libx264",
            encoder_options={"crf": "28"},
            limit_pairs=1,
            mlflow=MlflowRunConfig(enabled=False),
        ),
        adapter_factory=lambda _model_config, _settings: adapter,
        settings=Settings(),
        adapter=adapter,
    )

    assert dict(result.runtime_options) == {"scale": 0.5}
    assert adapter.requests[0].backend_options["scale"] == 0.5


def test_video_batched_inference_passes_batch_size_to_model_request(tmp_path) -> None:
    input_path = tmp_path / "batch_options_input.mp4"
    output_path = tmp_path / "batch_options_output.mp4"
    _write_synthetic_input_video(input_path, frame_values=(0, 96, 192))
    adapter = _BatchRuntimeAdapter()

    result = run_video_inference(
        VideoInferenceConfig(
            input_path=input_path,
            output_path=output_path,
            model=_runtime_model_config(),
            interpolation_mode=InferenceMode.ARBITRARY_NX,
            interpolation_factor=4,
            execution_mode="batched",
            inference_batch_size=4,
            runtime_options={"scale": 0.5},
            codec="libx264",
            encoder_options={"crf": "28"},
            limit_pairs=2,
            mlflow=MlflowRunConfig(enabled=False),
        ),
        adapter_factory=lambda _model_config, _settings: adapter,
        settings=Settings(),
        adapter=adapter,
    )

    assert result.inference_batch_size == 4
    assert result.batch_chunks_processed == 2
    assert [request.flattened_size for request in adapter.batch_requests] == [3, 3]
    assert all(request.backend_options["inference_batch_size"] == 4 for request in adapter.batch_requests)
    assert all(request.backend_options["scale"] == 0.5 for request in adapter.batch_requests)


def test_video_inference_rejects_invalid_inference_batch_size() -> None:
    config = VideoInferenceConfig(
        input_path=Path("in.mp4"),
        output_path=Path("out.mp4"),
        model=_runtime_model_config(),
        inference_batch_size=0,
    )

    with pytest.raises(ValueError, match="inference_batch_size"):
        config.validate()


def test_video_inference_sequential_mode_bypasses_batch_adapter_path(tmp_path) -> None:
    input_path = tmp_path / "sequential_mode_input.mp4"
    output_path = tmp_path / "sequential_mode_output.mp4"
    _write_synthetic_input_video(input_path, frame_values=(0, 100, 200))
    adapter = _BatchRuntimeAdapter()

    result = run_video_inference(
        VideoInferenceConfig(
            input_path=input_path,
            output_path=output_path,
            model=_runtime_model_config(),
            interpolation_mode=InferenceMode.FIXED_2X,
            interpolation_factor=2,
            execution_mode="sequential",
            inference_batch_size=8,
            codec="libx264",
            encoder_options={"crf": "28"},
            limit_pairs=2,
            mlflow=MlflowRunConfig(enabled=False),
        ),
        adapter_factory=lambda _model_config, _settings: adapter,
        settings=Settings(),
        adapter=adapter,
    )

    assert result.execution_mode == "sequential"
    assert result.requested_execution_mode == "sequential"
    assert result.model_batch_requests == 0
    assert adapter.batch_requests == []
    assert len(adapter.requests) == 2


def test_video_inference_batched_mode_falls_back_for_legacy_adapter(tmp_path) -> None:
    input_path = tmp_path / "fallback_input.mp4"
    output_path = tmp_path / "fallback_output.mp4"
    _write_synthetic_input_video(input_path)

    result = run_video_inference(
        VideoInferenceConfig(
            input_path=input_path,
            output_path=output_path,
            codec="libx264",
            encoder_options={"crf": "28"},
            execution_mode="batched",
            inference_batch_size=4,
            limit_pairs=1,
            mlflow=MlflowRunConfig(enabled=False),
        ),
        adapter_factory=lambda _model_config, _settings: _BlendAdapter(),
        settings=Settings(),
        adapter=_BlendAdapter(),
    )

    assert result.execution_mode == "sequential_fallback"
    assert result.requested_execution_mode == "batched"
    assert result.model_batch_requests == 0
    assert result.frames_written == 3


def test_video_sequential_and_batched_paths_preserve_same_frame_order(tmp_path) -> None:
    input_path = tmp_path / "equivalent_input.mp4"
    sequential_output = tmp_path / "equivalent_sequential.mp4"
    batched_output = tmp_path / "equivalent_batched.mp4"
    _write_synthetic_input_video(input_path, frame_values=(0, 80, 160))

    sequential_adapter = _BatchRuntimeAdapter()
    batched_adapter = _BatchRuntimeAdapter()
    common_config = {
        "input_path": input_path,
        "model": _runtime_model_config(),
        "interpolation_mode": InferenceMode.ARBITRARY_NX,
        "interpolation_factor": 4,
        "inference_batch_size": 8,
        "codec": "libx264",
        "encoder_options": {"crf": "0", "preset": "ultrafast"},
        "limit_pairs": 2,
        "mlflow": MlflowRunConfig(enabled=False),
    }

    run_video_inference(
        VideoInferenceConfig(
            **common_config,
            output_path=sequential_output,
            execution_mode="sequential",
        ),
        adapter_factory=lambda _model_config, _settings: sequential_adapter,
        settings=Settings(),
        adapter=sequential_adapter,
    )
    run_video_inference(
        VideoInferenceConfig(
            **common_config,
            output_path=batched_output,
            execution_mode="batched",
        ),
        adapter_factory=lambda _model_config, _settings: batched_adapter,
        settings=Settings(),
        adapter=batched_adapter,
    )

    assert _decode_video_frame_means(sequential_output) == pytest.approx(_decode_video_frame_means(batched_output), abs=3)


def test_video_inference_rejects_invalid_interpolation_factor_before_model_execution(tmp_path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _write_synthetic_input_video(input_path)
    adapter = _RuntimeAdapter()

    with pytest.raises(ValueError, match="interpolation_factor"):
        run_video_inference(
            VideoInferenceConfig(
                input_path=input_path,
                output_path=output_path,
                model=_runtime_model_config(),
                interpolation_mode=InferenceMode.ARBITRARY_NX,
                interpolation_factor=9,
                codec="libx264",
                mlflow=MlflowRunConfig(enabled=False),
            ),
            adapter_factory=lambda _model_config, _settings: adapter,
            settings=Settings(),
            adapter=adapter,
        )

    assert adapter.requests == []


def test_cli_rejects_invalid_video_interpolation_factor() -> None:
    result = CliRunner().invoke(
        app,
        [
            "ema",
            "infer-video",
            "--config",
            "configs/inference/ema_vfi_small_2x.yaml",
            "--interpolation-factor",
            "9",
            "--disable-mlflow",
        ],
    )

    assert result.exit_code != 0
    assert "8" in result.output


def test_rife_cli_rejects_invalid_scale_before_model_execution() -> None:
    result = CliRunner().invoke(
        app,
        [
            "rife",
            "infer-video",
            "--config",
            "configs/inference/practical_rife_v4_26_2x.yaml",
            "--scale",
            "0.75",
            "--disable-mlflow",
        ],
    )

    assert result.exit_code != 0
    assert "scale" in result.output


def test_existing_ema_fixed_2x_config_remains_compatible() -> None:
    config = VideoInferenceConfig.from_mapping(OmegaConf.to_container(OmegaConf.load("configs/inference/ema_vfi_small_2x.yaml")))

    config.validate()

    assert config.resolved_interpolation_mode() is InferenceMode.FIXED_2X
    assert config.resolved_interpolation_factor() == 2
    assert config.resolved_interpolation_timesteps() == (0.5,)


def test_baseline_adapter_reuses_video_inference_workflow(tmp_path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "baseline_output.mp4"
    _write_synthetic_input_video(input_path)

    result = run_video_inference(
        VideoInferenceConfig(
            input_path=input_path,
            output_path=output_path,
            model=BaselineAdapterConfig(model_name="baseline_blend", baseline_name="blend"),
            codec="libx264",
            encoder_options={"crf": "28"},
            limit_pairs=1,
            mlflow=MlflowRunConfig(enabled=False),
        ),
        adapter_factory=lambda model_config, _settings: BaselineAdapter(model_config),
    )

    assert result.pairs_processed == 1
    assert result.frames_written == 3
    assert result.interpolation_mode == "fixed_2x"
    assert result.interpolation_factor == 2
    assert result.runtime_backend == "adapter"
    assert dict(result.runtime_options) == {}
    assert result.audio_streams_preserved == 1
    assert output_path.is_file()

    with av.open(str(output_path)) as container:
        assert len(list(container.decode(video=0))) == 3


def test_batch_inference_discovers_videos_and_preserves_relative_output_layout(tmp_path) -> None:
    input_dir = tmp_path / "inputs"
    nested_dir = input_dir / "nested"
    nested_dir.mkdir(parents=True)
    (input_dir / "a.mp4").write_bytes(b"video")
    (nested_dir / "b.mkv").write_bytes(b"video")
    (nested_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    videos = discover_inference_videos(input_dir)

    assert [video.relative_path.as_posix() for video in videos] == ["a.mp4", "nested/b.mkv"]
    assert batch_output_path(
        output_root=tmp_path / "outputs",
        output_group=Path("baselines") / "blend",
        relative_input_path=videos[1].relative_path,
        output_extension=".mp4",
    ) == tmp_path / "outputs" / "baselines" / "blend" / "nested" / "b_2x.mp4"
    assert batch_run_name(target_name="baseline_blend", relative_input_path=videos[1].relative_path) == (
        "baseline_blend_nested_b_2x"
    )

    measurements_path = batch_measurements_path(
        output_root=tmp_path / "outputs",
        output_group=Path("baselines") / "blend",
    )
    write_batch_measurements_csv(
        measurements_path,
        [
            {
                "target_name": "baseline_blend",
                "relative_input_video": "nested/b.mkv",
                "output_video": "outputs/baselines/blend/nested/b_2x.mp4",
                "status": "ok",
                "pairs_processed": 1,
                "interpolation_mode": "fixed_2x",
                "interpolation_factor": 2,
                "runtime_backend": "adapter",
                "runtime_options": {"scale": 0.5},
                "execution_mode": "batched",
                "requested_execution_mode": "batched",
                "inference_batch_size": 4,
                "batch_chunks_processed": 1,
                "model_batch_requests": 1,
                "decode_sec": "0.01000000",
                "preprocessing_sec": "0.02000000",
                "model_inference_elapsed_sec": "0.10000000",
                "postprocessing_sec": "0.03000000",
                "encode_sec": "0.04000000",
                "audio_remux_sec": "0.05000000",
                "total_elapsed_sec": "0.20000000",
            }
        ],
    )

    csv_text = measurements_path.read_text(encoding="utf-8")
    assert "model_inference_elapsed_sec" in csv_text
    assert "decode_sec" in csv_text
    assert "audio_remux_sec" in csv_text
    assert "interpolation_mode" in csv_text
    assert "fixed_2x" in csv_text
    assert "runtime_options" in csv_text
    assert "scale" in csv_text
    assert "execution_mode" in csv_text
    assert "inference_batch_size" in csv_text
    assert "model_batch_requests" in csv_text
    assert "baseline_blend" in csv_text
    assert "nested/b.mkv" in csv_text


def test_batch_target_selection_supports_aliases_and_groups() -> None:
    available = (
        "practical_rife_v4_26",
        "practical_rife_v4_25",
        "amt_s",
        "ema_vfi_small",
        "baseline_duplicate_left",
        "baseline_blend",
        "baseline_farneback",
    )
    aliases = {
        "models": ("practical_rife_v4_26", "ema_vfi_small"),
        "rife_v4_25": ("practical_rife_v4_25",),
        "ema": ("ema_vfi_small",),
        "blend": ("baseline_blend",),
    }

    assert resolve_batch_target_names(available, ["ema", "blend"], aliases=aliases) == [
        "ema_vfi_small",
        "baseline_blend",
    ]
    assert resolve_batch_target_names(available, ["models"], aliases=aliases) == [
        "practical_rife_v4_26",
        "ema_vfi_small",
    ]
    assert resolve_batch_target_names(available, ["rife_v4_25"], aliases=aliases) == ["practical_rife_v4_25"]

    with pytest.raises(ValueError, match="Unknown target"):
        resolve_batch_target_names(available, ["unknown"], aliases=aliases)


class _BlendAdapter(ModelAdapter):
    model_name = "blend_adapter"

    def validate_environment(self):  # pragma: no cover - not used by this focused test.
        raise NotImplementedError

    def build_model(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path: Path | None = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def save_checkpoint(self, checkpoint_path: Path) -> None:  # pragma: no cover
        raise NotImplementedError

    def train(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def eval(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def predict_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.clamp((left + right) / 2.0, 0.0, 1.0)


class _RuntimeAdapter(ModelAdapter):
    model_name = "runtime_adapter"

    def __init__(self) -> None:
        self.requests: list[FramePairRequest] = []

    def validate_environment(self):  # pragma: no cover - not used by this focused test.
        raise NotImplementedError

    def build_model(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path: Path | None = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def save_checkpoint(self, checkpoint_path: Path) -> None:  # pragma: no cover
        raise NotImplementedError

    def train(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def eval(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def predict_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise AssertionError("local video inference should call predict_frame_pair when available")

    def predict_frame_pair(self, request: FramePairRequest) -> FramePairResult:
        self.requests.append(request)
        frames = tuple(
            torch.clamp((1.0 - timestep) * request.left + timestep * request.right, 0.0, 1.0)
            for timestep in request.timesteps
        )
        return FramePairResult(
            intermediate_frames=frames,
            timesteps=request.timesteps,
            mode=request.mode,
            interpolation_factor=request.interpolation_factor,
            backend_kind=RuntimeBackendKind.TORCH,
            model_name=self.model_name,
            original_shape=request.original_shape,
        )


class _BatchRuntimeAdapter(_RuntimeAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.batch_requests: list[ModelBatchRequest] = []
        self.batch_pair_values: list[list[tuple[int, int]]] = []

    def predict_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise AssertionError("local video inference should call a runtime request method")

    def predict_frame_pairs_batch(self, request: ModelBatchRequest) -> ModelBatchResult:
        self.batch_requests.append(request)
        self.batch_pair_values.append(
            [
                (int(round(float(request.left[index].mean()) * 255)), int(round(float(request.right[index].mean()) * 255)))
                for index in range(request.pair_count)
            ]
        )
        outputs = tuple(
            tuple(
                torch.clamp((1.0 - timestep) * request.left[pair_index] + timestep * request.right[pair_index], 0.0, 1.0)
                for timestep in request.timesteps
            )
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


def _runtime_model_config() -> dict[str, object]:
    return {
        "model_name": "runtime_adapter",
        "checkpoint_path": "",
        "supported_modes": ("fixed_2x", "arbitrary_nx"),
        "default_interpolation_factor": 2,
        "min_interpolation_factor": 2,
        "max_interpolation_factor": 8,
        "inference_batch_size": None,
    }


def _write_synthetic_input_video(path: Path, *, frame_values: tuple[int, ...] = (32, 96)) -> None:
    with av.open(str(path), mode="w") as container:
        video_stream = container.add_stream("libx264", rate=24)
        video_stream.width = 64
        video_stream.height = 64
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


def _decode_video_frame_means(path: Path) -> list[float]:
    with av.open(str(path)) as container:
        return [float(frame.to_ndarray(format="rgb24").mean()) for frame in container.decode(video=0)]


def _solid_rgb_frame(value: int) -> np.ndarray:
    return np.full((64, 64, 3), value, dtype=np.uint8)
