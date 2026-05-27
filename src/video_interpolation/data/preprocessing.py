from __future__ import annotations

import csv
import fnmatch
import math
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import av
from scenedetect import ContentDetector, FrameTimecode, SceneManager, open_video

from video_interpolation.contracts import SEQUENCE_INDEX_COLUMNS, SEQUENCE_INDEX_CONTRACT, validate_csv_file
from video_interpolation.settings import Settings, load_settings


VIDEO_EXTENSIONS = (".mkv", ".mov", ".mp4", ".webm")
ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True)
class ResizeSpec:
    width: int
    height: int

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class SceneSpan:
    scene_id: str
    start_frame: int
    end_frame: int

    def contains(self, frame_start: int, frame_end: int) -> bool:
        return self.start_frame <= frame_start and frame_end <= self.end_frame


@dataclass(frozen=True)
class SequenceSamplingConfig:
    sequence_length: int = 3
    max_frame_step: int = 2
    min_sequences_per_video: int = 1
    max_sequences_per_video: int = 50
    quota_scale: float = 1.0
    random_seed: int = 0

    def validate(self) -> None:
        if self.sequence_length < 3:
            raise ValueError("sequence_length must be at least 3")
        if self.max_frame_step not in (0, 1, 2):
            raise ValueError("max_frame_step must be one of 0, 1, or 2")
        if self.sequence_length > 3 and self.max_frame_step != 0:
            raise ValueError("frame-step sampling is only supported when sequence_length is 3")
        if self.min_sequences_per_video < 0:
            raise ValueError("min_sequences_per_video must be non-negative")
        if self.max_sequences_per_video < self.min_sequences_per_video:
            raise ValueError("max_sequences_per_video must be >= min_sequences_per_video")
        if self.quota_scale < 0:
            raise ValueError("quota_scale must be non-negative")


@dataclass(frozen=True)
class SequenceCandidate:
    frame_start: int
    frame_end: int
    frame_step: int
    scene_id: str
    sequence_length: int

    @property
    def frame_indices(self) -> tuple[int, ...]:
        stride = self.frame_step + 1
        return tuple(self.frame_start + offset * stride for offset in range(self.sequence_length))


@dataclass(frozen=True)
class VideoPreprocessConfig:
    raw_input_dir: Path = Path("raw_data/anime")
    output_source_dir: Path = Path("sources/anime")
    source_group: str = "anime"
    source_dataset: str = "anime"
    sequence_length: int = 3
    max_frame_step: int = 2
    min_sequences_per_video: int = 1
    max_sequences_per_video: int = 50
    quota_scale: float = 1.0
    random_seed: int = 0
    resize: str | None = None
    static_ssim_threshold: float = 0.95
    reject_static: bool = True
    limit_videos: int | None = None
    only_video: str | None = None
    video_glob: str | None = None
    max_duration_sec: float | None = None
    max_frames: int | None = None
    decode_strategy: str = "auto"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> VideoPreprocessConfig:
        values = dict(data)
        if "raw_input_dir" in values:
            values["raw_input_dir"] = Path(values["raw_input_dir"])
        if "output_source_dir" in values:
            values["output_source_dir"] = Path(values["output_source_dir"])
        return cls(**values)

    @property
    def sampling(self) -> SequenceSamplingConfig:
        return SequenceSamplingConfig(
            sequence_length=self.sequence_length,
            max_frame_step=self.max_frame_step,
            min_sequences_per_video=self.min_sequences_per_video,
            max_sequences_per_video=self.max_sequences_per_video,
            quota_scale=self.quota_scale,
            random_seed=self.random_seed,
        )


@dataclass(frozen=True)
class PreprocessResult:
    index_path: Path
    output_dir: Path
    videos_discovered: int
    videos_processed: int
    videos_skipped: int
    videos_failed: int
    scenes_detected: int
    sequences_written: int
    static_triplets_rejected: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VideoMetadata:
    frame_count: int
    width: int
    height: int
    fps: float
    container_format: str
    video_codec: str
    audio_codecs: tuple[str, ...]
    average_rate: str
    stream_frames: int
    source_frame_count: int
    frame_count_source: str
    frame_count_exact: bool
    stream_duration_sec: float | None
    container_duration_sec: float | None
    frame_limit_applied: bool


@dataclass(frozen=True)
class SceneDetectionResult:
    spans: tuple[SceneSpan, ...]
    processed_frames: int
    fallback_used: bool
    warning: str | None = None


@dataclass(frozen=True)
class DecodeResult:
    frames: dict[int, Image.Image]
    strategy: str
    requested_count: int
    recovered_count: int
    missing_count: int


def parse_resize(value: str | None) -> ResizeSpec | None:
    if value is None or value == "":
        return None

    match = re.fullmatch(r"(\d+)x(\d+)", value.strip())
    if match is None:
        raise ValueError("resize must be null or WIDTHxHEIGHT")

    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("resize width and height must be positive")
    return ResizeSpec(width=width, height=height)


def enumerate_scene_safe_candidates(
    frame_count: int,
    scene_spans: Sequence[SceneSpan],
    config: SequenceSamplingConfig,
) -> list[SequenceCandidate]:
    config.validate()
    if frame_count <= 0:
        return []

    spans = list(scene_spans) or [SceneSpan("scene_0000", 0, frame_count - 1)]
    steps = (0,) if config.sequence_length > 3 else tuple(range(config.max_frame_step + 1))
    candidates: list[SequenceCandidate] = []

    for scene in spans:
        start = max(0, scene.start_frame)
        end = min(frame_count - 1, scene.end_frame)
        if end < start:
            continue

        for frame_step in steps:
            sequence_span = (config.sequence_length - 1) * (frame_step + 1) + 1
            last_start = end - sequence_span + 1
            if last_start < start:
                continue
            for frame_start in range(start, last_start + 1):
                frame_end = frame_start + sequence_span - 1
                if scene.contains(frame_start, frame_end):
                    candidates.append(
                        SequenceCandidate(
                            frame_start=frame_start,
                            frame_end=frame_end,
                            frame_step=frame_step,
                            scene_id=scene.scene_id,
                            sequence_length=config.sequence_length,
                        )
                    )

    return candidates


def sample_sequence_candidates(
    frame_count: int,
    scene_spans: Sequence[SceneSpan],
    config: SequenceSamplingConfig,
) -> list[SequenceCandidate]:
    candidates = enumerate_scene_safe_candidates(frame_count, scene_spans, config)
    if not candidates:
        return []

    scaled_quota = int(round(len(candidates) * config.quota_scale))
    target = max(config.min_sequences_per_video, scaled_quota)
    target = min(config.max_sequences_per_video, target, len(candidates))

    rng = random.Random(config.random_seed)
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    return sorted(shuffled[:target], key=lambda item: (item.frame_start, item.frame_step))


def is_static_triplet(
    left: Image.Image | np.ndarray,
    middle: Image.Image | np.ndarray,
    right: Image.Image | np.ndarray,
    threshold: float = 0.95,
) -> bool:
    left_score = _ssim(left, middle)
    right_score = _ssim(middle, right)
    return max(left_score, right_score) >= threshold


def discover_video_files(
    root: Path,
    limit: int | None = None,
    only_video: str | None = None,
    video_glob: str | None = None,
) -> list[Path]:
    videos = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if only_video:
        videos = [path for path in videos if _video_matches(path, root, only_video)]
    if video_glob:
        videos = [path for path in videos if _video_matches_glob(path, root, video_glob)]
    if limit is not None:
        return videos[:limit]
    return videos


def read_video_metadata(
    video_path: Path,
    max_duration_sec: float | None = None,
    max_frames: int | None = None,
) -> VideoMetadata:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
        width = stream.codec_context.width or stream.width
        height = stream.codec_context.height or stream.height
        container_format = container.format.name if container.format is not None else "unknown"
        video_codec = stream.codec_context.name or "unknown"
        audio_codecs = tuple(
            audio.codec_context.name or "unknown"
            for audio in container.streams.audio
        )
        average_rate = str(stream.average_rate or "")
        source_frame_count = int(stream.frames or 0)
        stream_duration_sec = _stream_duration_seconds(stream)
        container_duration_sec = _container_duration_seconds(container)
        frame_count = source_frame_count
        frame_count_source = "metadata" if frame_count > 0 else "unknown"
        frame_count_exact = frame_count > 0

        if frame_count <= 0:
            estimated_seconds = stream_duration_sec or container_duration_sec
            if estimated_seconds is not None and fps > 0:
                frame_count = max(1, int(math.ceil(estimated_seconds * fps)))
                frame_count_source = "duration_estimate"
            elif max_frames is not None:
                frame_count = max_frames
                frame_count_source = "configured_max_frames"
            else:
                frame_count = sum(1 for _ in container.decode(stream))
                frame_count_source = "full_decode"
                frame_count_exact = True

        limited_frame_count = _apply_frame_limits(
            frame_count,
            fps=fps,
            max_duration_sec=max_duration_sec,
            max_frames=max_frames,
        )

    return VideoMetadata(
        frame_count=limited_frame_count,
        width=width,
        height=height,
        fps=fps,
        container_format=container_format,
        video_codec=video_codec,
        audio_codecs=audio_codecs,
        average_rate=average_rate,
        stream_frames=source_frame_count,
        source_frame_count=frame_count,
        frame_count_source=frame_count_source,
        frame_count_exact=frame_count_exact and limited_frame_count == frame_count,
        stream_duration_sec=stream_duration_sec,
        container_duration_sec=container_duration_sec,
        frame_limit_applied=limited_frame_count != frame_count,
    )


def detect_scene_spans(
    video_path: Path,
    frame_count: int,
    fps: float,
) -> SceneDetectionResult:
    if frame_count <= 0:
        return SceneDetectionResult(spans=(), processed_frames=0, fallback_used=True)

    try:
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        end_time = FrameTimecode(frame_count, fps or video.frame_rate)
        processed_frames = scene_manager.detect_scenes(video=video, end_time=end_time)
        scene_list = scene_manager.get_scene_list()
    except Exception as exc:
        return SceneDetectionResult(
            spans=(SceneSpan("scene_0000", 0, frame_count - 1),),
            processed_frames=0,
            fallback_used=True,
            warning=f"{type(exc).__name__}: {exc}",
        )

    spans = [
        SceneSpan(
            scene_id=f"scene_{index:04d}",
            start_frame=start.get_frames(),
            end_frame=max(start.get_frames(), end.get_frames() - 1),
        )
        for index, (start, end) in enumerate(scene_list)
    ]
    if not spans:
        spans = [SceneSpan("scene_0000", 0, frame_count - 1)]

    return SceneDetectionResult(
        spans=tuple(spans),
        processed_frames=processed_frames,
        fallback_used=False,
    )


def preprocess_videos(
    config: VideoPreprocessConfig,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PreprocessResult:
    settings = settings or load_settings()
    config.sampling.validate()
    if config.decode_strategy not in ("auto", "sequential", "segment_seek"):
        raise ValueError("decode_strategy must be one of auto, sequential, or segment_seek")
    resize = parse_resize(config.resize)

    raw_root = settings.resolve_path(config.raw_input_dir)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw input directory does not exist: {raw_root}")

    source_root = settings.dataset_root_abs / config.output_source_dir
    sequences_root = source_root / "sequences"
    sequences_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    created_at = datetime.now(UTC).isoformat()
    videos = discover_video_files(
        raw_root,
        limit=config.limit_videos,
        only_video=config.only_video,
        video_glob=config.video_glob,
    )
    videos_processed = 0
    videos_skipped = 0
    videos_failed = 0
    scenes_detected = 0
    static_triplets_rejected = 0
    warnings: list[str] = []

    _emit_progress(progress_callback, "start", total=len(videos))
    for video_index, video_path in enumerate(videos, start=1):
        video_started_at = perf_counter()
        _emit_progress(
            progress_callback,
            "video_start",
            video_index=video_index,
            video_path=str(video_path),
            relative_video_path=video_path.relative_to(raw_root).as_posix(),
        )

        try:
            metadata_started_at = perf_counter()
            _emit_progress(progress_callback, "debug_step_start", step="metadata")
            metadata = read_video_metadata(
                video_path,
                max_duration_sec=config.max_duration_sec,
                max_frames=config.max_frames,
            )
            _emit_progress(
                progress_callback,
                "metadata_done",
                elapsed_sec=perf_counter() - metadata_started_at,
                **_metadata_payload(metadata),
            )

            scene_started_at = perf_counter()
            _emit_progress(progress_callback, "debug_step_start", step="scene detection")
            scene_result = detect_scene_spans(video_path, metadata.frame_count, metadata.fps)
            scenes = scene_result.spans
            scenes_detected += len(scenes)
            if scene_result.warning is not None:
                warnings.append(f"{video_path}: scene detection fallback: {scene_result.warning}")
            _emit_progress(
                progress_callback,
                "scene_detection_done",
                elapsed_sec=perf_counter() - scene_started_at,
                scenes=len(scenes),
                processed_frames=scene_result.processed_frames,
                fallback_used=scene_result.fallback_used,
                warning=scene_result.warning or "",
            )

            sampling_started_at = perf_counter()
            _emit_progress(progress_callback, "debug_step_start", step="candidate sampling")
            candidates = sample_sequence_candidates(metadata.frame_count, scenes, config.sampling)
            selected_indices = sorted({index for candidate in candidates for index in candidate.frame_indices})
            _emit_progress(
                progress_callback,
                "candidate_sampling_done",
                elapsed_sec=perf_counter() - sampling_started_at,
                candidates=len(candidates),
                unique_frame_indices=len(selected_indices),
            )
            _emit_progress(
                progress_callback,
                "video_candidates",
                candidates=len(candidates),
                scenes=len(scenes),
            )
            if not candidates:
                videos_skipped += 1
                _emit_progress(
                    progress_callback,
                    "video_done",
                    sequences_written=0,
                    static_triplets_rejected=0,
                    status="skipped",
                )
                continue

            decode_started_at = perf_counter()
            _emit_progress(progress_callback, "debug_step_start", step="selected frame decoding")
            decode_result = _decode_selected_frames(
                video_path,
                selected_indices,
                resize,
                fps=metadata.fps,
                strategy=config.decode_strategy,
            )
            decoded_frames = decode_result.frames
            _emit_progress(
                progress_callback,
                "decode_done",
                elapsed_sec=perf_counter() - decode_started_at,
                strategy=decode_result.strategy,
                requested_count=decode_result.requested_count,
                recovered_count=decode_result.recovered_count,
                missing_count=decode_result.missing_count,
            )
            source_video_id = _source_video_id(raw_root, video_path)

            write_started_at = perf_counter()
            _emit_progress(progress_callback, "debug_step_start", step="SSIM filtering and PNG writing")
            written_for_video = 0
            rejected_for_video = 0
            for candidate in candidates:
                frames = [decoded_frames.get(index) for index in candidate.frame_indices]
                if any(frame is None for frame in frames):
                    _emit_progress(progress_callback, "sequence_advanced")
                    continue

                if config.reject_static and is_static_triplet(
                    frames[0],
                    frames[len(frames) // 2],
                    frames[-1],
                    threshold=config.static_ssim_threshold,
                ):
                    rejected_for_video += 1
                    static_triplets_rejected += 1
                    _emit_progress(progress_callback, "sequence_advanced")
                    continue

                sequence_id = f"{config.source_dataset}__{source_video_id}__{written_for_video:06d}"
                sequence_dir = sequences_root / source_video_id / f"{written_for_video:06d}"
                sequence_dir.mkdir(parents=True, exist_ok=True)
                for frame_offset, frame in enumerate(frames):
                    frame.save(sequence_dir / f"frame_{frame_offset:03d}.png")

                first_frame = frames[0]
                rows.append(
                    {
                        "sequence_id": sequence_id,
                        "source_group": config.source_group,
                        "source_dataset": config.source_dataset,
                        "source_video_id": source_video_id,
                        "relative_sequence_dir": sequence_dir.relative_to(settings.dataset_root_abs).as_posix(),
                        "sequence_length": str(config.sequence_length),
                        "frame_step": str(candidate.frame_step),
                        "width": str(first_frame.width),
                        "height": str(first_frame.height),
                        "fps": f"{metadata.fps:g}",
                        "frame_start": str(candidate.frame_start),
                        "frame_end": str(candidate.frame_end),
                        "scene_id": candidate.scene_id,
                        "resize": resize.label if resize else "",
                        "created_at": created_at,
                    }
                )
                written_for_video += 1
                _emit_progress(progress_callback, "sequence_advanced")

            _emit_progress(
                progress_callback,
                "write_done",
                elapsed_sec=perf_counter() - write_started_at,
                sequences_written=written_for_video,
                static_triplets_rejected=rejected_for_video,
            )

            if written_for_video == 0:
                videos_skipped += 1
                status = "skipped"
            else:
                videos_processed += 1
                status = "processed"

            _emit_progress(
                progress_callback,
                "video_done",
                sequences_written=written_for_video,
                static_triplets_rejected=rejected_for_video,
                status=status,
            )
            _emit_progress(
                progress_callback,
                "video_summary",
                elapsed_sec=perf_counter() - video_started_at,
                status=status,
                sequences_written=written_for_video,
                static_triplets_rejected=rejected_for_video,
            )
        except Exception as exc:
            videos_failed += 1
            message = f"{video_path}: {type(exc).__name__}: {exc}"
            warnings.append(message)
            _emit_progress(progress_callback, "video_failed", error=message)
            _emit_progress(
                progress_callback,
                "video_summary",
                elapsed_sec=perf_counter() - video_started_at,
                status="failed",
                sequences_written=0,
                static_triplets_rejected=0,
            )
        finally:
            _emit_progress(progress_callback, "video_advanced")

    index_path = source_root / "sequence_index.csv"
    _write_sequence_index(index_path, rows)
    return PreprocessResult(
        index_path=index_path,
        output_dir=source_root,
        videos_discovered=len(videos),
        videos_processed=videos_processed,
        videos_skipped=videos_skipped,
        videos_failed=videos_failed,
        scenes_detected=scenes_detected,
        sequences_written=len(rows),
        static_triplets_rejected=static_triplets_rejected,
        warnings=tuple(warnings),
    )


def with_limit(config: VideoPreprocessConfig, limit_videos: int | None) -> VideoPreprocessConfig:
    if limit_videos is None:
        return config
    return replace(config, limit_videos=limit_videos)


def with_debug_overrides(
    config: VideoPreprocessConfig,
    *,
    limit_videos: int | None = None,
    only_video: str | None = None,
    video_glob: str | None = None,
    max_duration_sec: float | None = None,
    max_frames: int | None = None,
) -> VideoPreprocessConfig:
    values: dict[str, object] = {}
    if limit_videos is not None:
        values["limit_videos"] = limit_videos
    if only_video is not None:
        values["only_video"] = only_video
    if video_glob is not None:
        values["video_glob"] = video_glob
    if max_duration_sec is not None:
        values["max_duration_sec"] = max_duration_sec
    if max_frames is not None:
        values["max_frames"] = max_frames
    if not values:
        return config
    return replace(config, **values)


def _decode_selected_frames(
    video_path: Path,
    frame_indices: Sequence[int],
    resize: ResizeSpec | None,
    *,
    fps: float,
    strategy: str,
) -> DecodeResult:
    wanted = set(frame_indices)
    frames: dict[int, Image.Image] = {}
    if not wanted:
        return DecodeResult(
            frames=frames,
            strategy="sequential",
            requested_count=0,
            recovered_count=0,
            missing_count=0,
        )

    use_segment_seek = strategy == "segment_seek" or (
        strategy == "auto" and fps > 0 and max(wanted) >= 300
    )
    if use_segment_seek:
        segment_result = _decode_selected_frames_by_segments(video_path, wanted, resize, fps=fps)
        if segment_result.missing_count == 0 or strategy == "segment_seek":
            return segment_result

        sequential_result = _decode_selected_frames_sequential(video_path, wanted, resize)
        return DecodeResult(
            frames=sequential_result.frames,
            strategy=f"{segment_result.strategy}+sequential_fallback",
            requested_count=sequential_result.requested_count,
            recovered_count=sequential_result.recovered_count,
            missing_count=sequential_result.missing_count,
        )

    return _decode_selected_frames_sequential(video_path, wanted, resize)


def _decode_selected_frames_sequential(
    video_path: Path,
    wanted: set[int],
    resize: ResizeSpec | None,
) -> DecodeResult:
    frames: dict[int, Image.Image] = {}

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index not in wanted:
                continue

            image = frame.to_image()
            if resize is not None:
                image = image.resize((resize.width, resize.height), Image.Resampling.BICUBIC)
            frames[frame_index] = image

            if len(frames) == len(wanted):
                break

    return DecodeResult(
        frames=frames,
        strategy="sequential",
        requested_count=len(wanted),
        recovered_count=len(frames),
        missing_count=len(wanted) - len(frames),
    )


def _decode_selected_frames_by_segments(
    video_path: Path,
    wanted: set[int],
    resize: ResizeSpec | None,
    *,
    fps: float,
) -> DecodeResult:
    frames: dict[int, Image.Image] = {}
    segments = _frame_segments(sorted(wanted))

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for start_frame, end_frame in segments:
            seek_frame = max(0, start_frame - 12)
            seek_pts = int((seek_frame / fps) / stream.time_base)
            try:
                container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
            except Exception:
                return DecodeResult(
                    frames=frames,
                    strategy="segment_seek_failed",
                    requested_count=len(wanted),
                    recovered_count=len(frames),
                    missing_count=len(wanted) - len(frames),
                )

            for frame in container.decode(stream):
                frame_index = _frame_index_from_pts(frame, stream, fps)
                if frame_index is None:
                    continue
                if frame_index < start_frame - 3:
                    continue
                if frame_index > end_frame + 3:
                    break
                if frame_index not in wanted or frame_index in frames:
                    continue

                image = frame.to_image()
                if resize is not None:
                    image = image.resize((resize.width, resize.height), Image.Resampling.BICUBIC)
                frames[frame_index] = image

                if wanted.issubset(frames.keys()):
                    break

    return DecodeResult(
        frames=frames,
        strategy=f"segment_seek:{len(segments)}",
        requested_count=len(wanted),
        recovered_count=len(frames),
        missing_count=len(wanted) - len(frames),
    )


def _ssim(first: Image.Image | np.ndarray, second: Image.Image | np.ndarray) -> float:
    first_array = _to_grayscale_array(first)
    second_array = _to_grayscale_array(second)
    if first_array.shape != second_array.shape:
        raise ValueError("SSIM inputs must have the same shape")

    min_size = min(first_array.shape)
    if min_size < 3:
        return 1.0 if np.array_equal(first_array, second_array) else 0.0

    win_size = min(7, min_size if min_size % 2 == 1 else min_size - 1)
    return float(structural_similarity(first_array, second_array, data_range=255, win_size=win_size))


def _to_grayscale_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.uint8)

    array = np.asarray(image)
    if array.ndim == 3:
        return np.asarray(Image.fromarray(array.astype(np.uint8)).convert("L"), dtype=np.uint8)
    return array.astype(np.uint8)


def _frame_segments(frame_indices: Sequence[int], max_gap: int = 24) -> list[tuple[int, int]]:
    if not frame_indices:
        return []

    segments: list[tuple[int, int]] = []
    start = frame_indices[0]
    previous = frame_indices[0]
    for frame_index in frame_indices[1:]:
        if frame_index - previous <= max_gap:
            previous = frame_index
            continue
        segments.append((start, previous))
        start = frame_index
        previous = frame_index
    segments.append((start, previous))
    return segments


def _frame_index_from_pts(
    frame: av.VideoFrame,
    stream: av.video.stream.VideoStream,
    fps: float,
) -> int | None:
    if frame.pts is None or fps <= 0:
        return None
    return int(round(float(frame.pts * stream.time_base) * fps))


def _source_video_id(raw_root: Path, video_path: Path) -> str:
    relative = video_path.relative_to(raw_root).with_suffix("").as_posix()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", relative)


def _video_matches(video_path: Path, root: Path, value: str) -> bool:
    relative = video_path.relative_to(root).as_posix()
    normalized = value.replace("\\", "/")
    return relative == normalized or video_path.name == normalized


def _video_matches_glob(video_path: Path, root: Path, pattern: str) -> bool:
    relative = video_path.relative_to(root).as_posix()
    return fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(video_path.name, pattern)


def _stream_duration_seconds(stream: av.video.stream.VideoStream) -> float | None:
    if stream.duration is None:
        return None
    return float(stream.duration * stream.time_base)


def _container_duration_seconds(container: av.container.InputContainer) -> float | None:
    if container.duration is None:
        return None
    return float(container.duration / av.time_base)


def _apply_frame_limits(
    frame_count: int,
    *,
    fps: float,
    max_duration_sec: float | None,
    max_frames: int | None,
) -> int:
    limits = [frame_count]
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        limits.append(max_frames)
    if max_duration_sec is not None:
        if max_duration_sec <= 0:
            raise ValueError("max_duration_sec must be positive")
        if fps <= 0:
            raise ValueError("max_duration_sec requires a positive video FPS")
        limits.append(max(1, int(math.floor(max_duration_sec * fps))))
    return max(0, min(limits))


def _metadata_payload(metadata: VideoMetadata) -> dict[str, object]:
    return {
        "container_format": metadata.container_format,
        "video_codec": metadata.video_codec,
        "audio_codecs": ", ".join(metadata.audio_codecs) if metadata.audio_codecs else "none",
        "width": metadata.width,
        "height": metadata.height,
        "fps": metadata.fps,
        "average_rate": metadata.average_rate,
        "stream_frames": metadata.stream_frames,
        "stream_duration_sec": metadata.stream_duration_sec,
        "container_duration_sec": metadata.container_duration_sec,
        "frame_count": metadata.frame_count,
        "source_frame_count": metadata.source_frame_count,
        "frame_count_source": metadata.frame_count_source,
        "frame_count_exact": metadata.frame_count_exact,
        "frame_limit_applied": metadata.frame_limit_applied,
    }


def _write_sequence_index(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SEQUENCE_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)
    validate_csv_file(str(path), SEQUENCE_INDEX_CONTRACT)


def _emit_progress(
    callback: ProgressCallback | None,
    event: str,
    **payload: object,
) -> None:
    if callback is not None:
        callback(event, payload)
