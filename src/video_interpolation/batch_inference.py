import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v")
MEASUREMENT_FILENAME = "inference_measurements.csv"
MEASUREMENT_COLUMNS = (
    "target_name",
    "input_video",
    "relative_input_video",
    "output_video",
    "status",
    "pairs_processed",
    "frames_written",
    "input_fps",
    "output_fps",
    "interpolation_mode",
    "interpolation_factor",
    "execution_mode",
    "requested_execution_mode",
    "inference_batch_size",
    "batch_chunks_processed",
    "model_batch_requests",
    "runtime_backend",
    "runtime_options",
    "decode_sec",
    "preprocessing_sec",
    "model_inference_elapsed_sec",
    "postprocessing_sec",
    "encode_sec",
    "audio_remux_sec",
    "total_elapsed_sec",
    "model_pairs_per_sec",
    "total_pairs_per_sec",
    "audio_streams_available",
    "audio_streams_preserved",
    "mlflow_run_id",
    "error",
)


@dataclass(frozen=True)
class BatchInferenceVideo:
    input_path: Path
    relative_path: Path


def discover_inference_videos(
    input_dir: Path,
    *,
    extensions: tuple[str, ...] = DEFAULT_VIDEO_EXTENSIONS,
    limit_videos: int | None = None,
) -> list[BatchInferenceVideo]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    normalized_extensions = tuple(extension.lower() for extension in extensions)
    videos = [
        BatchInferenceVideo(input_path=path, relative_path=path.relative_to(input_dir))
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ]
    if limit_videos is not None:
        if limit_videos <= 0:
            raise ValueError("limit_videos must be positive when set")
        videos = videos[:limit_videos]
    return videos


def batch_output_path(
    *,
    output_root: Path,
    output_group: Path,
    relative_input_path: Path,
    output_extension: str,
    interpolation_factor: int = 2,
) -> Path:
    extension = output_extension if output_extension.startswith(".") else f".{output_extension}"
    return (
        output_root
        / output_group
        / relative_input_path.parent
        / f"{relative_input_path.stem}_{interpolation_factor}x{extension}"
    )


def batch_run_name(*, target_name: str, relative_input_path: Path, interpolation_factor: int = 2) -> str:
    relative_stem = "_".join(relative_input_path.with_suffix("").parts)
    return f"{target_name}_{relative_stem}_{interpolation_factor}x"


def resolve_batch_target_names(
    available_names: Sequence[str],
    selections: Sequence[str] | None,
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    if not selections:
        return list(available_names)

    canonical = {_normalize_selector(name): name for name in available_names}
    normalized_aliases = {
        _normalize_selector(alias): tuple(names)
        for alias, names in (aliases or {}).items()
    }
    selected: list[str] = []
    unknown: list[str] = []

    for raw_selection in selections:
        selector = _normalize_selector(raw_selection)
        if selector in normalized_aliases:
            names = normalized_aliases[selector]
        elif selector in canonical:
            names = (canonical[selector],)
        else:
            unknown.append(raw_selection)
            continue

        for name in names:
            if name not in available_names:
                raise ValueError(f"Target selector '{raw_selection}' resolved to unavailable target '{name}'")
            if name not in selected:
                selected.append(name)

    if unknown:
        allowed = ", ".join(sorted([*available_names, *(aliases or {}).keys()]))
        unknown_display = ", ".join(unknown)
        raise ValueError(f"Unknown target selection(s): {unknown_display}. Available selections: {allowed}")
    if not selected:
        raise ValueError("At least one target selection must resolve to a runnable target.")
    return selected


def batch_measurements_path(*, output_root: Path, output_group: Path) -> Path:
    return output_root / output_group / MEASUREMENT_FILENAME


def write_batch_measurements_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MEASUREMENT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in MEASUREMENT_COLUMNS})


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    return value


def _normalize_selector(value: str) -> str:
    return value.strip().lower().replace("-", "_")
