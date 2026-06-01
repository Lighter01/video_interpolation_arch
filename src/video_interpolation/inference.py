from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import av
import cv2
import numpy as np

from video_interpolation.adapters.base import ModelAdapter
from video_interpolation.adapters.ema_vfi import EMAVFIAdapter, EMAVFIAdapterConfig
from video_interpolation.image_io import tensor_to_uint8_hwc, uint8_hwc_to_tensor
from video_interpolation.inference_runtime import (
    FramePairRequest,
    FramePairResult,
    InferenceMode,
    resolve_interpolation_timesteps,
    validate_interpolation_factor,
)
from video_interpolation.inference_runtime.rife import validate_rife_scale
from video_interpolation.mlflow import MlflowRunConfig, log_stage1_run
from video_interpolation.settings import Settings, load_settings

ProgressCallback = Callable[[str, Mapping[str, object]], None]
AdapterFactory = Callable[[Any, Settings], ModelAdapter]

DEFAULT_ENCODER_OPTIONS: dict[str, dict[str, str]] = {
    "h264_nvenc": {"preset": "p3", "rc": "vbr", "cq": "23", "bf": "0"},
    "hevc_nvenc": {"preset": "p3", "rc": "vbr", "cq": "26", "bf": "0"},
    "libx264": {"preset": "veryfast", "crf": "22", "bf": "0"},
    "libx265": {"preset": "veryfast", "crf": "28", "bf": "0"},
}

SUPPORTED_FRAME_FORMATS = {"rgb24", "bgr24"}


@dataclass(frozen=True)
class VideoInferenceConfig:
    input_path: Path
    output_path: Path
    model: Any = field(default_factory=EMAVFIAdapterConfig)
    limit_pairs: int | None = None
    interpolation_mode: InferenceMode | str = InferenceMode.FIXED_2X
    interpolation_factor: int | None = None
    runtime_options: Mapping[str, Any] = field(default_factory=dict)
    output_fps_multiplier: float = 2.0
    codec: str = "h264_nvenc"
    container: str | None = None
    pix_fmt: str = "yuv420p"
    frame_format: str = "rgb24"
    encoder_options: Mapping[str, Any] = field(default_factory=dict)
    encoder_options_by_codec: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    mlflow: MlflowRunConfig = field(default_factory=lambda: MlflowRunConfig(experiment_name="stage1-ema-inference"))
    config_path: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        model_config_factory: Callable[[Mapping[str, Any] | None], Any] = EMAVFIAdapterConfig.from_mapping,
    ) -> "VideoInferenceConfig":
        values = dict(data)
        values["input_path"] = Path(values["input_path"])
        values["output_path"] = Path(values["output_path"])
        values["model"] = model_config_factory(values.get("model"))
        values["runtime_options"] = dict(values.get("runtime_options") or {})
        values["mlflow"] = MlflowRunConfig.from_mapping(values.get("mlflow"))
        if values.get("config_path") is not None:
            values["config_path"] = Path(values["config_path"])
        return cls(**values)

    def validate(self) -> None:
        if self.limit_pairs is not None and self.limit_pairs <= 0:
            raise ValueError("limit_pairs must be positive when set")
        mode = self.resolved_interpolation_mode()
        factor = self.resolved_interpolation_factor()
        resolve_interpolation_timesteps(mode, factor)
        _validate_model_interpolation_support(self.model, mode, factor)
        _validate_runtime_options(self.runtime_options)
        if self.output_fps_multiplier <= 0:
            raise ValueError("output_fps_multiplier must be positive")
        if self.codec == "mp4v":
            raise ValueError(
                "codec=mp4v is an OpenCV fourcc and is no longer supported. "
                "Choose an FFmpeg encoder such as libx264, h264_nvenc, libx265, or hevc_nvenc."
            )
        if self.frame_format not in SUPPORTED_FRAME_FORMATS:
            allowed = ", ".join(sorted(SUPPORTED_FRAME_FORMATS))
            raise ValueError(f"frame_format must be one of: {allowed}")

    def resolved_interpolation_mode(self) -> InferenceMode:
        try:
            return InferenceMode(str(self.interpolation_mode))
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in InferenceMode)
            raise ValueError(f"Unsupported interpolation_mode: {self.interpolation_mode!r}. Allowed: {allowed}.") from exc

    def resolved_interpolation_factor(self) -> int:
        factor = self.interpolation_factor
        if factor is None:
            factor = _model_config_value(self.model, "default_interpolation_factor", default=None)
        if factor is None:
            factor = 2
        return validate_interpolation_factor(factor)

    def resolved_interpolation_timesteps(self) -> tuple[float, ...]:
        return resolve_interpolation_timesteps(
            self.resolved_interpolation_mode(),
            self.resolved_interpolation_factor(),
        )


@dataclass(frozen=True)
class VideoInferenceResult:
    input_path: Path
    output_path: Path
    input_fps: float
    output_fps: float
    interpolation_mode: str
    interpolation_factor: int
    interpolation_timesteps: tuple[float, ...]
    runtime_backend: str
    runtime_options: Mapping[str, object]
    pairs_processed: int
    frames_written: int
    model_inference_elapsed_sec: float
    total_elapsed_sec: float
    model_pairs_per_sec: float
    total_pairs_per_sec: float
    codec: str
    container: str | None
    pix_fmt: str
    frame_format: str
    encoder_options: Mapping[str, str]
    audio_streams_available: int
    audio_streams_preserved: int
    mlflow_run_id: str | None


@dataclass(frozen=True)
class _FramePairVideoPrediction:
    intermediate_frames: tuple[Any, ...]
    timesteps: tuple[float, ...]
    runtime_backend: str


def run_ema_video_inference(
    config: VideoInferenceConfig,
    settings: Settings | None = None,
    adapter: ModelAdapter | None = None,
    progress_callback: ProgressCallback | None = None,
) -> VideoInferenceResult:
    return run_video_inference(
        config,
        adapter_factory=lambda model_config, run_settings: EMAVFIAdapter(model_config, settings=run_settings),
        settings=settings,
        adapter=adapter,
        progress_callback=progress_callback,
        mlflow_mode="ema_video_inference",
    )


def run_video_inference(
    config: VideoInferenceConfig,
    *,
    adapter_factory: AdapterFactory,
    settings: Settings | None = None,
    adapter: ModelAdapter | None = None,
    progress_callback: ProgressCallback | None = None,
    mlflow_mode: str = "video_inference",
) -> VideoInferenceResult:
    config.validate()
    interpolation_mode = config.resolved_interpolation_mode()
    interpolation_factor = config.resolved_interpolation_factor()
    interpolation_timesteps = config.resolved_interpolation_timesteps()
    runtime_options = dict(config.runtime_options)
    settings = settings or load_settings()
    input_path = _resolve_project_path(settings, config.input_path)
    output_path = _resolve_project_path(settings, config.output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_pairs = max(frame_count - 1, 0) if frame_count else None
    if config.limit_pairs is not None and total_pairs is not None:
        total_pairs = min(total_pairs, config.limit_pairs)
    output_fps = fps * interpolation_factor
    output_rate = _fps_to_fraction(output_fps)
    encoder_options = resolve_encoder_options(
        config.codec,
        output_fps,
        encoder_options=config.encoder_options,
        encoder_options_by_codec=config.encoder_options_by_codec,
    )

    owns_adapter = adapter is None
    model_adapter = adapter or adapter_factory(config.model, settings)
    input_audio_container: av.container.InputContainer | None = None
    output_container: av.container.OutputContainer | None = None
    video_stream: av.video.stream.VideoStream | None = None
    audio_stream_pairs: list[tuple[av.audio.stream.AudioStream, av.audio.stream.AudioStream]] = []
    pairs_processed = 0
    frames_written = 0
    model_inference_elapsed_sec = 0.0
    runtime_backend = "adapter"
    audio_streams_available = 0
    audio_streams_preserved = 0
    total_start = perf_counter()
    _emit_progress(progress_callback, "start", total=total_pairs or 0)

    try:
        if owns_adapter:
            model_adapter.load_checkpoint()
        ok, previous_bgr = capture.read()
        if not ok:
            raise ValueError(f"Input video contains no readable frames: {input_path}")
        first_frame = _bgr_to_writer_frame(previous_bgr, config.frame_format)
        height, width = first_frame.shape[:2]
        input_audio_container = av.open(str(input_path))
        audio_streams_available = len(input_audio_container.streams.audio)
        output_container = _open_output_container(output_path, config.container)
        video_stream = _add_video_stream(
            output_container,
            codec=config.codec,
            output_rate=output_rate,
            width=width,
            height=height,
            pix_fmt=config.pix_fmt,
            encoder_options=encoder_options,
        )
        audio_stream_pairs = _add_audio_streams(
            input_audio_container,
            output_container,
            codec=config.codec,
        )
        _emit_progress(
            progress_callback,
            "encoding_start",
            output_path=str(output_path),
            container=config.container or "inferred",
            codec=config.codec,
            output_fps=output_fps,
            interpolation_mode=interpolation_mode.value,
            interpolation_factor=interpolation_factor,
            interpolation_timesteps=interpolation_timesteps,
            runtime_options=runtime_options,
            width=width,
            height=height,
            pix_fmt=config.pix_fmt,
            encoder_options=encoder_options,
            frame_format=config.frame_format,
            audio_streams_available=audio_streams_available,
            audio_streams_to_preserve=len(audio_stream_pairs),
        )
        _encode_video_frame(output_container, video_stream, first_frame, config.frame_format, config.codec)
        frames_written += 1

        while config.limit_pairs is None or pairs_processed < config.limit_pairs:
            ok, current_bgr = capture.read()
            if not ok:
                break
            left = _bgr_to_tensor(previous_bgr)
            right = _bgr_to_tensor(current_bgr)
            inference_start = perf_counter()
            prediction = _predict_video_pair(
                model_adapter,
                left,
                right,
                mode=interpolation_mode,
                interpolation_factor=interpolation_factor,
                runtime_options=runtime_options,
            )
            model_inference_elapsed_sec += perf_counter() - inference_start
            runtime_backend = prediction.runtime_backend
            for intermediate_frame in prediction.intermediate_frames:
                _encode_video_frame(
                    output_container,
                    video_stream,
                    _tensor_to_writer_frame(intermediate_frame, config.frame_format),
                    config.frame_format,
                    config.codec,
                )
            _encode_video_frame(
                output_container,
                video_stream,
                _bgr_to_writer_frame(current_bgr, config.frame_format),
                config.frame_format,
                config.codec,
            )
            frames_written += len(prediction.intermediate_frames) + 1
            pairs_processed += 1
            previous_bgr = current_bgr
            _emit_progress(
                progress_callback,
                "pair_advanced",
                generated_frames=len(prediction.intermediate_frames),
                runtime_backend=runtime_backend,
            )
        _flush_video_stream(output_container, video_stream, config.codec)
        audio_streams_preserved = _copy_audio_streams(
            input_audio_container,
            output_container,
            audio_stream_pairs,
            max_duration_sec=_audio_duration_limit(config, frames_written, output_fps),
        )
    finally:
        capture.release()
        if output_container is not None:
            output_container.close()
        if input_audio_container is not None:
            input_audio_container.close()
        if owns_adapter:
            model_adapter.close()

    total_elapsed_sec = perf_counter() - total_start
    model_pairs_per_sec = _safe_rate(pairs_processed, model_inference_elapsed_sec)
    total_pairs_per_sec = _safe_rate(pairs_processed, total_elapsed_sec)

    artifact_paths = [output_path]
    if config.config_path is not None:
        artifact_paths.append(_resolve_project_path(settings, config.config_path))

    mlflow_run_id = log_stage1_run(
        config.mlflow,
        params={
            "mode": mlflow_mode,
            "model_name": _model_config_value(config.model, "model_name"),
            "checkpoint_path": str(_model_config_value(config.model, "checkpoint_path")),
            "input_path": str(config.input_path),
            "output_path": str(config.output_path),
            "limit_pairs": config.limit_pairs,
            "interpolation_mode": interpolation_mode.value,
            "interpolation_factor": interpolation_factor,
            "interpolation_timesteps": interpolation_timesteps,
            "runtime_backend": runtime_backend,
            "runtime_options": runtime_options,
            "output_fps_multiplier": interpolation_factor,
            "legacy_output_fps_multiplier_config": config.output_fps_multiplier,
            "output_fps": output_fps,
            "codec": config.codec,
            "container": config.container,
            "pix_fmt": config.pix_fmt,
            "frame_format": config.frame_format,
            "encoder_options": encoder_options,
            "audio_streams_available": audio_streams_available,
            "audio_streams_preserved": audio_streams_preserved,
        },
        metrics={
            "inference.pairs_processed": float(pairs_processed),
            "inference.frames_written": float(frames_written),
            "inference.input_fps": fps,
            "inference.output_fps": output_fps,
            "inference.interpolation_factor": float(interpolation_factor),
            "inference.audio_streams_available": float(audio_streams_available),
            "inference.audio_streams_preserved": float(audio_streams_preserved),
            "inference.model_elapsed_sec": model_inference_elapsed_sec,
            "inference.total_elapsed_sec": total_elapsed_sec,
            "inference.model_pairs_per_sec": model_pairs_per_sec,
            "inference.total_pairs_per_sec": total_pairs_per_sec,
        },
        artifact_paths=artifact_paths,
        settings=settings,
    )

    return VideoInferenceResult(
        input_path=input_path,
        output_path=output_path,
        input_fps=fps,
        output_fps=output_fps,
        interpolation_mode=interpolation_mode.value,
        interpolation_factor=interpolation_factor,
        interpolation_timesteps=interpolation_timesteps,
        runtime_backend=runtime_backend,
        runtime_options=runtime_options,
        pairs_processed=pairs_processed,
        frames_written=frames_written,
        model_inference_elapsed_sec=model_inference_elapsed_sec,
        total_elapsed_sec=total_elapsed_sec,
        model_pairs_per_sec=model_pairs_per_sec,
        total_pairs_per_sec=total_pairs_per_sec,
        codec=config.codec,
        container=config.container,
        pix_fmt=config.pix_fmt,
        frame_format=config.frame_format,
        encoder_options=encoder_options,
        audio_streams_available=audio_streams_available,
        audio_streams_preserved=audio_streams_preserved,
        mlflow_run_id=mlflow_run_id,
    )


def with_inference_input(
    config: VideoInferenceConfig,
    input_path: Path | None,
) -> VideoInferenceConfig:
    if input_path is None:
        return config
    return replace(config, input_path=input_path)


def with_inference_output(
    config: VideoInferenceConfig,
    output_path: Path | None,
) -> VideoInferenceConfig:
    if output_path is None:
        return config
    return replace(config, output_path=output_path)


def with_inference_limit(
    config: VideoInferenceConfig,
    limit_pairs: int | None,
) -> VideoInferenceConfig:
    if limit_pairs is None:
        return config
    return replace(config, limit_pairs=limit_pairs)


def with_inference_interpolation_mode(
    config: VideoInferenceConfig,
    interpolation_mode: str | InferenceMode | None,
) -> VideoInferenceConfig:
    if interpolation_mode is None:
        return config
    return replace(config, interpolation_mode=interpolation_mode)


def with_inference_interpolation_factor(
    config: VideoInferenceConfig,
    interpolation_factor: int | None,
) -> VideoInferenceConfig:
    if interpolation_factor is None:
        return config
    return replace(config, interpolation_factor=interpolation_factor)


def with_inference_runtime_option(
    config: VideoInferenceConfig,
    name: str,
    value: object | None,
) -> VideoInferenceConfig:
    if value is None:
        return config
    runtime_options = dict(config.runtime_options)
    runtime_options[name] = value
    return replace(config, runtime_options=runtime_options)


def with_inference_checkpoint(
    config: VideoInferenceConfig,
    checkpoint_path: Path | None,
) -> VideoInferenceConfig:
    if checkpoint_path is None:
        return config
    return replace(config, model=replace(config.model, checkpoint_path=checkpoint_path))


def with_inference_codec(
    config: VideoInferenceConfig,
    codec: str | None,
) -> VideoInferenceConfig:
    if codec is None:
        return config
    return replace(config, codec=codec)


def with_inference_config_path(
    config: VideoInferenceConfig,
    config_path: Path,
) -> VideoInferenceConfig:
    return replace(config, config_path=config_path)


def with_inference_mlflow_disabled(
    config: VideoInferenceConfig,
    disabled: bool,
) -> VideoInferenceConfig:
    if not disabled:
        return config
    return replace(config, mlflow=replace(config.mlflow, enabled=False))


def resolve_encoder_options(
    codec: str,
    output_fps: float,
    *,
    encoder_options: Mapping[str, Any] | None = None,
    encoder_options_by_codec: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    if codec == "mp4v":
        raise ValueError(
            "codec=mp4v is an OpenCV fourcc and is no longer supported. "
            "Choose an FFmpeg encoder such as libx264, h264_nvenc, libx265, or hevc_nvenc."
        )
    resolved: dict[str, str] = dict(DEFAULT_ENCODER_OPTIONS.get(codec, {}))
    codec_overrides = (encoder_options_by_codec or {}).get(codec, {})
    resolved.update(_stringify_options(codec_overrides))
    resolved.update(_stringify_options(encoder_options or {}))
    resolved.setdefault("g", str(int(output_fps * 2)))
    return resolved


def _stringify_options(options: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in options.items() if value is not None}


def _open_output_container(output_path: Path, container: str | None) -> av.container.OutputContainer:
    try:
        if container:
            return av.open(str(output_path), mode="w", format=container)
        return av.open(str(output_path), mode="w")
    except Exception as exc:
        raise ValueError(f"Could not open PyAV output container for writing: {output_path}. {exc}") from exc


def _add_video_stream(
    output_container: av.container.OutputContainer,
    *,
    codec: str,
    output_rate: Fraction,
    width: int,
    height: int,
    pix_fmt: str,
    encoder_options: Mapping[str, str],
) -> av.video.stream.VideoStream:
    try:
        stream = output_container.add_stream(codec, rate=output_rate, options=dict(encoder_options))
    except Exception as exc:
        raise ValueError(
            f"Could not create PyAV video stream with codec '{codec}'. "
            "Ensure this FFmpeg/PyAV build supports it, or try codec 'libx264'. "
            f"Original error: {exc}"
        ) from exc
    stream.width = width
    stream.height = height
    stream.pix_fmt = pix_fmt
    return stream


def _add_audio_streams(
    input_container: av.container.InputContainer,
    output_container: av.container.OutputContainer,
    *,
    codec: str,
) -> list[tuple[av.audio.stream.AudioStream, av.audio.stream.AudioStream]]:
    pairs: list[tuple[av.audio.stream.AudioStream, av.audio.stream.AudioStream]] = []
    for input_stream in input_container.streams.audio:
        try:
            output_stream = output_container.add_stream_from_template(input_stream)
            output_stream.time_base = input_stream.time_base
        except Exception as exc:
            raise ValueError(
                "Could not prepare audio stream remuxing for the output container. "
                f"Video codec is '{codec}'. If the input audio codec is incompatible with the selected "
                "container, choose a compatible container such as matroska or use a source with compatible audio. "
                f"Original error: {exc}"
            ) from exc
        pairs.append((input_stream, output_stream))
    return pairs


def _encode_video_frame(
    output_container: av.container.OutputContainer,
    stream: av.video.stream.VideoStream,
    frame_array: np.ndarray,
    frame_format: str,
    codec: str,
) -> None:
    try:
        frame = av.VideoFrame.from_ndarray(frame_array, format=frame_format)
        for packet in stream.encode(frame):
            output_container.mux(packet)
    except Exception as exc:
        raise ValueError(
            f"Could not encode video frame with codec '{codec}'. "
            "Ensure this FFmpeg/PyAV build supports the encoder and options, or try codec 'libx264'. "
            f"Original error: {exc}"
        ) from exc


def _flush_video_stream(
    output_container: av.container.OutputContainer,
    stream: av.video.stream.VideoStream,
    codec: str,
) -> None:
    try:
        for packet in stream.encode():
            output_container.mux(packet)
    except Exception as exc:
        raise ValueError(
            f"Could not flush video encoder '{codec}'. Try codec 'libx264' or adjust encoder options. "
            f"Original error: {exc}"
        ) from exc


def _copy_audio_streams(
    input_container: av.container.InputContainer,
    output_container: av.container.OutputContainer,
    audio_stream_pairs: list[tuple[av.audio.stream.AudioStream, av.audio.stream.AudioStream]],
    *,
    max_duration_sec: float | None,
) -> int:
    streams_preserved = 0
    for input_stream, output_stream in audio_stream_pairs:
        packet_count = 0
        try:
            input_container.seek(0)
            for packet in input_container.demux(input_stream):
                if packet.dts is None:
                    continue
                packet_time = _packet_time_sec(packet, input_stream)
                if max_duration_sec is not None and packet_time is not None and packet_time > max_duration_sec:
                    break
                packet.stream = output_stream
                output_container.mux(packet)
                packet_count += 1
        except Exception as exc:
            raise ValueError(
                "Could not remux input audio into the inference output. "
                "The input audio codec may be incompatible with the selected output container. "
                "Try a compatible container such as matroska, or inspect the source audio codec. "
                f"Original error: {exc}"
            ) from exc
        if packet_count > 0:
            streams_preserved += 1
    return streams_preserved


def _packet_time_sec(packet: av.Packet, stream: av.audio.stream.AudioStream) -> float | None:
    timestamp = packet.pts if packet.pts is not None else packet.dts
    if timestamp is None or stream.time_base is None:
        return None
    return float(timestamp * stream.time_base)


def _audio_duration_limit(
    config: VideoInferenceConfig,
    frames_written: int,
    output_fps: float,
) -> float | None:
    if config.limit_pairs is None:
        return None
    return frames_written / output_fps


def _fps_to_fraction(fps: float) -> Fraction:
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"output_fps must be positive and finite, got {fps}")
    return Fraction(fps).limit_denominator(100_000)


def _bgr_to_tensor(frame: np.ndarray):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return uint8_hwc_to_tensor(rgb)


def _bgr_to_writer_frame(frame: np.ndarray, frame_format: str) -> np.ndarray:
    if frame_format == "bgr24":
        return np.ascontiguousarray(frame)
    if frame_format == "rgb24":
        return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    raise ValueError(f"Unsupported frame_format: {frame_format}")


def _tensor_to_writer_frame(tensor, frame_format: str) -> np.ndarray:
    rgb = tensor_to_uint8_hwc(tensor)
    if frame_format == "rgb24":
        return np.ascontiguousarray(rgb)
    if frame_format == "bgr24":
        return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    raise ValueError(f"Unsupported frame_format: {frame_format}")


def _predict_video_pair(
    adapter: ModelAdapter,
    left,
    right,
    *,
    mode: InferenceMode,
    interpolation_factor: int,
    runtime_options: Mapping[str, object],
) -> _FramePairVideoPrediction:
    predict_frame_pair = getattr(adapter, "predict_frame_pair", None)
    if callable(predict_frame_pair):
        result = predict_frame_pair(
            FramePairRequest(
                left=left,
                right=right,
                mode=mode,
                interpolation_factor=interpolation_factor,
                backend_options=runtime_options,
            )
        )
        if not isinstance(result, FramePairResult):
            raise ValueError(
                f"{adapter.__class__.__name__}.predict_frame_pair returned {type(result).__name__}, "
                "expected FramePairResult."
            )
        return _FramePairVideoPrediction(
            intermediate_frames=tuple(result.intermediate_frames),
            timesteps=tuple(result.timesteps),
            runtime_backend=result.backend_kind.value,
        )

    if mode is not InferenceMode.FIXED_2X or interpolation_factor != 2:
        raise ValueError(
            f"{adapter.__class__.__name__} only supports fixed_2x video inference through the legacy adapter API."
        )
    return _FramePairVideoPrediction(
        intermediate_frames=(adapter.predict_pair(left, right),),
        timesteps=(0.5,),
        runtime_backend="adapter",
    )


def _resolve_project_path(settings: Settings, path: Path) -> Path:
    if path.is_absolute():
        return path
    return settings.resolve_path(path)


def _model_config_value(model_config: Any, name: str, *, default: Any = "") -> Any:
    if isinstance(model_config, Mapping):
        return model_config.get(name, default)
    return getattr(model_config, name, default)


def _validate_model_interpolation_support(
    model_config: Any,
    mode: InferenceMode,
    interpolation_factor: int,
) -> None:
    supported_modes = _model_config_value(model_config, "supported_modes", default=None)
    if supported_modes is not None:
        supported = {str(supported_mode) for supported_mode in supported_modes}
        if mode.value not in supported:
            raise ValueError(
                f"Model config does not support interpolation_mode={mode.value!r}. "
                f"Supported modes: {', '.join(sorted(supported))}."
            )
    elif mode is InferenceMode.ARBITRARY_NX:
        raise ValueError("arbitrary_nx video inference requires a model config that declares supported_modes.")

    min_factor = _optional_model_config_int(model_config, "min_interpolation_factor")
    max_factor = _optional_model_config_int(model_config, "max_interpolation_factor")
    if min_factor is not None and interpolation_factor < min_factor:
        raise ValueError(
            f"interpolation_factor={interpolation_factor} is below this model config minimum {min_factor}."
        )
    if max_factor is not None and interpolation_factor > max_factor:
        raise ValueError(
            f"interpolation_factor={interpolation_factor} is above this model config maximum {max_factor}."
        )


def _validate_runtime_options(runtime_options: Mapping[str, object]) -> None:
    if "scale" in runtime_options:
        validate_rife_scale(runtime_options["scale"])


def _optional_model_config_int(model_config: Any, name: str) -> int | None:
    value = _model_config_value(model_config, name, default=None)
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer when set.")
    return value


def _safe_rate(count: int, elapsed_sec: float) -> float:
    if elapsed_sec <= 0.0:
        return 0.0
    return float(count) / elapsed_sec


def _emit_progress(
    callback: ProgressCallback | None,
    event: str,
    **payload: object,
) -> None:
    if callback is not None:
        callback(event, payload)
