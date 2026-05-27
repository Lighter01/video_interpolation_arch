from __future__ import annotations

import csv
import glob
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from video_interpolation.contracts import (
    GLOBAL_SEQUENCE_INDEX_COLUMNS,
    GLOBAL_SEQUENCE_INDEX_CONTRACT,
    SEQUENCE_INDEX_COLUMNS,
    SEQUENCE_INDEX_CONTRACT,
    validate_csv_file,
    validate_relative_artifact_path,
)
from video_interpolation.settings import Settings, load_settings

ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True)
class VimeoTripletIndexConfig:
    source_dir: Path = Path("sources/vimeo_triplet")
    source_group: str = "vimeo"
    source_dataset: str = "vimeo_triplet"
    train_list: str = "tri_trainlist.txt"
    test_list: str = "tri_testlist.txt"
    output_path: Path | None = None
    limit: int | None = None
    validate_images: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> VimeoTripletIndexConfig:
        values = dict(data)
        if "source_dir" in values:
            values["source_dir"] = Path(values["source_dir"])
        if values.get("output_path") is not None:
            values["output_path"] = Path(values["output_path"])
        return cls(**values)


@dataclass(frozen=True)
class SourceIndexResult:
    index_path: Path
    source_dataset: str
    records_read: int
    rows_written: int
    invalid_records: int = 0


@dataclass(frozen=True)
class GlobalIndexConfig:
    source_index_globs: tuple[str, ...] = ("sources/*/sequence_index.csv",)
    source_index_paths: tuple[Path, ...] = ()
    selected_source_groups: tuple[str, ...] = ()
    output_path: Path = Path("global_sequence_index.csv")
    limit: int | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> GlobalIndexConfig:
        values = dict(data)
        source_index_globs = values.get("source_index_globs", cls.source_index_globs)
        values["source_index_globs"] = tuple(source_index_globs or ())
        values["source_index_paths"] = tuple(Path(path) for path in values.get("source_index_paths") or ())
        values["selected_source_groups"] = tuple(values.get("selected_source_groups") or ())
        if "output_path" in values:
            values["output_path"] = Path(values["output_path"])
        return cls(**values)


@dataclass(frozen=True)
class GlobalIndexResult:
    index_path: Path
    source_indexes_read: int
    records_read: int
    rows_written: int
    filtered_records: int = 0


def build_vimeo_triplet_index(
    config: VimeoTripletIndexConfig,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SourceIndexResult:
    settings = settings or load_settings()
    source_root = _resolve_dataset_child(settings, config.source_dir)
    sequences_root = source_root / "sequences"
    if not sequences_root.is_dir():
        raise FileNotFoundError(f"Vimeo sequences directory does not exist: {sequences_root}")

    output_path = _resolve_output_path(settings, config, source_root)
    created_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, str]] = []
    entries = _iter_vimeo_list_entries(config, source_root)
    if config.limit is not None:
        entries = entries[: config.limit]

    _emit_progress(progress_callback, "start", total=len(entries))
    for relative_sequence, original_split in entries:
        sequence_dir = sequences_root / relative_sequence
        frame_paths = [sequence_dir / f"im{index}.png" for index in range(1, 4)]
        if config.validate_images:
            missing = [path for path in frame_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing Vimeo frame(s): {', '.join(str(path) for path in missing)}")

        width, height = _read_image_size(frame_paths[0])
        source_video_id = relative_sequence.parts[0]
        sequence_id = f"{config.source_dataset}__{relative_sequence.as_posix().replace('/', '_')}"
        rows.append(
            {
                "sequence_id": sequence_id,
                "source_group": config.source_group,
                "source_dataset": config.source_dataset,
                "source_video_id": source_video_id,
                "relative_sequence_dir": sequence_dir.relative_to(settings.dataset_root_abs).as_posix(),
                "sequence_length": "3",
                "frame_step": "0",
                "width": str(width),
                "height": str(height),
                "fps": "0",
                "frame_start": "0",
                "frame_end": "2",
                "scene_id": "vimeo_original",
                "resize": "",
                "created_at": created_at,
                "original_split": original_split,
            }
        )
        _emit_progress(progress_callback, "record_advanced")

    _write_source_index(output_path, rows, extra_columns=("original_split",))
    return SourceIndexResult(
        index_path=output_path,
        source_dataset=config.source_dataset,
        records_read=len(entries),
        rows_written=len(rows),
    )


def build_global_sequence_index(
    config: GlobalIndexConfig,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> GlobalIndexResult:
    settings = settings or load_settings()
    source_index_paths = _resolve_source_index_paths(settings, config)
    rows: list[dict[str, str]] = []
    extra_columns: set[str] = set()
    selected_groups = set(config.selected_source_groups)
    records_read = 0
    filtered_records = 0

    for source_index_path in source_index_paths:
        validate_csv_file(str(source_index_path), SEQUENCE_INDEX_CONTRACT)
        source_rows, source_extra_columns = _read_source_index_rows(source_index_path)
        extra_columns.update(source_extra_columns)
        for row in source_rows:
            records_read += 1
            if selected_groups and row["source_group"] not in selected_groups:
                filtered_records += 1
                continue
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row["source_group"],
            row["source_dataset"],
            row["source_video_id"],
            row["sequence_id"],
            row["relative_sequence_dir"],
        )
    )
    if config.limit is not None:
        rows = rows[: config.limit]

    output_path = _resolve_dataset_output_path(settings, config.output_path)
    _emit_progress(progress_callback, "start", total=len(rows), source_indexes=len(source_index_paths))
    _write_global_index(output_path, rows, sorted(extra_columns), progress_callback=progress_callback)

    return GlobalIndexResult(
        index_path=output_path,
        source_indexes_read=len(source_index_paths),
        records_read=records_read,
        rows_written=len(rows),
        filtered_records=filtered_records,
    )


def with_limit(config: VimeoTripletIndexConfig, limit: int | None) -> VimeoTripletIndexConfig:
    if limit is None:
        return config
    return replace(config, limit=limit)


def with_output_path(config: VimeoTripletIndexConfig, output_path: Path | None) -> VimeoTripletIndexConfig:
    if output_path is None:
        return config
    return replace(config, output_path=output_path)


def with_global_limit(config: GlobalIndexConfig, limit: int | None) -> GlobalIndexConfig:
    if limit is None:
        return config
    return replace(config, limit=limit)


def with_global_output_path(config: GlobalIndexConfig, output_path: Path | None) -> GlobalIndexConfig:
    if output_path is None:
        return config
    return replace(config, output_path=output_path)


def with_source_index_paths(
    config: GlobalIndexConfig,
    source_index_paths: Sequence[Path] | None,
) -> GlobalIndexConfig:
    if not source_index_paths:
        return config
    return replace(config, source_index_paths=tuple(source_index_paths), source_index_globs=())


def _iter_vimeo_list_entries(
    config: VimeoTripletIndexConfig,
    source_root: Path,
) -> Sequence[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    for list_name, split_name in ((config.train_list, "train"), (config.test_list, "test")):
        list_path = source_root / list_name
        if not list_path.is_file():
            raise FileNotFoundError(f"Vimeo split list does not exist: {list_path}")
        with list_path.open(encoding="utf-8") as list_file:
            for line in list_file:
                value = line.strip()
                if not value:
                    continue
                validate_relative_artifact_path(value, "vimeo_list_entry")
                entries.append((Path(value), split_name))
    return entries


def _resolve_dataset_child(settings: Settings, path: Path) -> Path:
    if path.is_absolute():
        return path
    return settings.dataset_root_abs / path


def _resolve_output_path(
    settings: Settings,
    config: VimeoTripletIndexConfig,
    source_root: Path,
) -> Path:
    if config.output_path is None:
        return source_root / "sequence_index.csv"
    if config.output_path.is_absolute():
        return config.output_path
    return settings.resolve_path(config.output_path)


def _resolve_dataset_output_path(settings: Settings, output_path: Path) -> Path:
    if output_path.is_absolute():
        return output_path
    return settings.dataset_root_abs / output_path


def _resolve_source_index_paths(settings: Settings, config: GlobalIndexConfig) -> list[Path]:
    paths: set[Path] = set()
    for path in config.source_index_paths:
        paths.add(path if path.is_absolute() else settings.dataset_root_abs / path)

    for pattern in config.source_index_globs:
        if Path(pattern).is_absolute():
            matches = [Path(match) for match in glob.glob(pattern)]
        else:
            matches = list(settings.dataset_root_abs.glob(pattern))
        paths.update(path for path in matches if path.is_file())

    resolved = sorted(paths)
    if not resolved:
        raise FileNotFoundError("No source sequence_index.csv files matched the global index config")
    return resolved


def _read_source_index_rows(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = [dict(row) for row in reader]
        extra_columns = set(reader.fieldnames or ()) - set(SEQUENCE_INDEX_COLUMNS)
    return rows, extra_columns


def _read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _write_source_index(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    extra_columns: Sequence[str] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = (*SEQUENCE_INDEX_COLUMNS, *extra_columns)
    with tmp_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)
    validate_csv_file(str(path), SEQUENCE_INDEX_CONTRACT)


def _write_global_index(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    extra_columns: Sequence[str] = (),
    progress_callback: ProgressCallback | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = (*GLOBAL_SEQUENCE_INDEX_COLUMNS, *extra_columns)
    with tmp_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
            _emit_progress(progress_callback, "record_advanced")
    tmp_path.replace(path)
    validate_csv_file(str(path), GLOBAL_SEQUENCE_INDEX_CONTRACT)


def _emit_progress(
    callback: ProgressCallback | None,
    event: str,
    **payload: object,
) -> None:
    if callback is not None:
        callback(event, payload)
