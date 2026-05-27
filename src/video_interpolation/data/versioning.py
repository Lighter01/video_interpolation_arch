from __future__ import annotations

import csv
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from video_interpolation.contracts import (
    GLOBAL_SEQUENCE_INDEX_CONTRACT,
    TRIPLET_MANIFEST_COLUMNS,
    TRIPLET_MANIFEST_CONTRACT,
    validate_csv_file,
    validate_dataset_config_mapping,
    validate_relative_artifact_path,
)
from video_interpolation.settings import Settings, load_settings

ProgressCallback = Callable[[str, Mapping[str, object]], None]

TRIPLET_POLICIES = ("first_triplet", "center_triplet", "wide_triplet", "all_local_triplets")
MIXING_MODES = ("none", "data_accumulation", "quality_drift")
FRAME_NAME_PATTERNS = ("auto", "im", "frame")


@dataclass(frozen=True)
class DatasetVersionConfig:
    dataset_version_id: str
    source_index_path: Path = Path("global_sequence_index.csv")
    output_root: Path = Path("dataset_versions")
    split_seed: int = 42
    split_ratios: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    selected_source_groups: tuple[str, ...] = ()
    triplet_policy: str = "wide_triplet"
    mixing_mode: str = "data_accumulation"
    old_source_groups: tuple[str, ...] = ()
    new_source_groups: tuple[str, ...] = ()
    old_min_ratio: float = 0.70
    new_max_ratio: float = 0.30
    new_min_ratio: float = 0.20
    old_max_ratio: float = 0.80
    train_budget: int | None = None
    limit_sequences: int | None = None
    validate_frame_paths: bool = True
    frame_name_pattern: str = "auto"
    preprocessing: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DatasetVersionConfig:
        values = dict(data)
        if "source_index_path" in values:
            values["source_index_path"] = Path(values["source_index_path"])
        if "output_root" in values:
            values["output_root"] = Path(values["output_root"])
        values["selected_source_groups"] = tuple(values.get("selected_source_groups") or ())
        values["old_source_groups"] = tuple(values.get("old_source_groups") or ())
        values["new_source_groups"] = tuple(values.get("new_source_groups") or ())
        values["preprocessing"] = values.get("preprocessing") or {}
        return cls(**values)

    def validate(self) -> None:
        if not self.dataset_version_id:
            raise ValueError("dataset_version_id must not be empty")
        if self.triplet_policy not in TRIPLET_POLICIES:
            raise ValueError(f"triplet_policy must be one of: {', '.join(TRIPLET_POLICIES)}")
        if self.mixing_mode not in MIXING_MODES:
            raise ValueError(f"mixing_mode must be one of: {', '.join(MIXING_MODES)}")
        if self.frame_name_pattern not in FRAME_NAME_PATTERNS:
            raise ValueError(f"frame_name_pattern must be one of: {', '.join(FRAME_NAME_PATTERNS)}")
        _validate_split_ratios(self.split_ratios)
        if self.train_budget is not None and self.train_budget <= 0:
            raise ValueError("train_budget must be positive when set")
        if self.limit_sequences is not None and self.limit_sequences <= 0:
            raise ValueError("limit_sequences must be positive when set")


@dataclass(frozen=True)
class DatasetVersionResult:
    output_dir: Path
    train_manifest_path: Path
    val_manifest_path: Path
    test_manifest_path: Path
    dataset_config_path: Path
    source_records_read: int
    samples_generated: int
    train_samples: int
    val_samples: int
    test_samples: int
    train_source_videos: int
    val_source_videos: int
    test_source_videos: int


def build_dataset_version(
    config: DatasetVersionConfig,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> DatasetVersionResult:
    config.validate()
    settings = settings or load_settings()
    source_index_path = _resolve_source_index_path(settings, config.source_index_path)
    output_root = _resolve_output_root(settings, config.output_root)
    output_dir = output_root / config.dataset_version_id

    source_rows = _read_global_index(source_index_path)
    if config.selected_source_groups:
        selected_groups = set(config.selected_source_groups)
        source_rows = [row for row in source_rows if row["source_group"] in selected_groups]
    if config.limit_sequences is not None:
        source_rows = source_rows[: config.limit_sequences]
    if not source_rows:
        raise ValueError("No source records are available for the dataset version")

    _emit_progress(progress_callback, "start", total=len(source_rows))
    samples: list[dict[str, str]] = []
    for source_row in source_rows:
        samples.extend(_triplets_from_sequence_row(source_row, config, settings))
        _emit_progress(progress_callback, "record_advanced")

    split_by_video = _split_source_videos(
        [sample["source_video_id"] for sample in samples],
        config.split_ratios,
        config.split_seed,
    )
    split_samples = {"train": [], "val": [], "test": []}
    for sample in samples:
        split_samples[split_by_video[sample["source_video_id"]]].append(sample)

    split_samples["train"] = _apply_train_mixing(split_samples["train"], config)
    for split_name in split_samples:
        split_samples[split_name].sort(key=lambda row: row["sample_id"])

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_all.csv"
    val_path = output_dir / "val_all.csv"
    test_path = output_dir / "test_all.csv"
    _write_manifest(train_path, split_samples["train"])
    _write_manifest(val_path, split_samples["val"])
    _write_manifest(test_path, split_samples["test"])

    dataset_config_path = output_dir / "dataset_config.yaml"
    _write_dataset_config(
        dataset_config_path,
        config,
        source_index_path,
        output_root,
        settings,
        created_at=datetime.now(UTC).isoformat(),
        manifest_counts={split: len(rows) for split, rows in split_samples.items()},
        split_video_counts=_split_video_counts(split_by_video),
    )

    return DatasetVersionResult(
        output_dir=output_dir,
        train_manifest_path=train_path,
        val_manifest_path=val_path,
        test_manifest_path=test_path,
        dataset_config_path=dataset_config_path,
        source_records_read=len(source_rows),
        samples_generated=len(samples),
        train_samples=len(split_samples["train"]),
        val_samples=len(split_samples["val"]),
        test_samples=len(split_samples["test"]),
        train_source_videos=_split_video_counts(split_by_video)["train"],
        val_source_videos=_split_video_counts(split_by_video)["val"],
        test_source_videos=_split_video_counts(split_by_video)["test"],
    )


def with_dataset_version_id(
    config: DatasetVersionConfig,
    dataset_version_id: str | None,
) -> DatasetVersionConfig:
    if dataset_version_id is None:
        return config
    return replace(config, dataset_version_id=dataset_version_id)


def with_dataset_output_root(
    config: DatasetVersionConfig,
    output_root: Path | None,
) -> DatasetVersionConfig:
    if output_root is None:
        return config
    return replace(config, output_root=output_root)


def with_dataset_source_index(
    config: DatasetVersionConfig,
    source_index_path: Path | None,
) -> DatasetVersionConfig:
    if source_index_path is None:
        return config
    return replace(config, source_index_path=source_index_path)


def with_dataset_limit_sequences(
    config: DatasetVersionConfig,
    limit_sequences: int | None,
) -> DatasetVersionConfig:
    if limit_sequences is None:
        return config
    return replace(config, limit_sequences=limit_sequences)


def _triplets_from_sequence_row(
    row: Mapping[str, str],
    config: DatasetVersionConfig,
    settings: Settings,
) -> list[dict[str, str]]:
    relative_sequence_dir = row["relative_sequence_dir"]
    validate_relative_artifact_path(relative_sequence_dir, "relative_sequence_dir")
    sequence_length = int(row["sequence_length"])
    offsets = _triplet_offsets(sequence_length, config.triplet_policy)
    samples: list[dict[str, str]] = []
    for index, triplet_offsets in enumerate(offsets):
        left_path = _frame_relative_path(row, triplet_offsets[0], config, settings)
        mid_path = _frame_relative_path(row, triplet_offsets[1], config, settings)
        right_path = _frame_relative_path(row, triplet_offsets[2], config, settings)
        offset_label = "_".join(str(offset) for offset in triplet_offsets)
        sample_id = f"{row['source_dataset']}__{row['sequence_id']}__{offset_label}".replace("/", "_")
        if len(offsets) > 1:
            sample_id = f"{sample_id}__{index:03d}"
        samples.append(
            {
                "sample_id": sample_id,
                "source_group": row["source_group"],
                "source_dataset": row["source_dataset"],
                "source_video_id": row["source_video_id"],
                "sequence_id": row["sequence_id"],
                "left_frame_path": left_path,
                "mid_frame_path": mid_path,
                "right_frame_path": right_path,
                "t_value": "0.5",
                "width": row["width"],
                "height": row["height"],
                "fps": row["fps"],
                "frame_step": row["frame_step"],
            }
        )
    return samples


def _triplet_offsets(sequence_length: int, policy: str) -> list[tuple[int, int, int]]:
    if sequence_length < 3:
        raise ValueError("sequence_length must be at least 3")
    if sequence_length == 3:
        return [(0, 1, 2)]
    if policy == "first_triplet":
        return [(0, 1, 2)]
    if policy == "center_triplet":
        center = sequence_length // 2
        return [(center - 1, center, center + 1)]
    if policy == "wide_triplet":
        if sequence_length % 2 == 0:
            raise ValueError("wide_triplet requires an odd sequence_length")
        return [(0, sequence_length // 2, sequence_length - 1)]
    if policy == "all_local_triplets":
        return [(offset, offset + 1, offset + 2) for offset in range(sequence_length - 2)]
    raise ValueError(f"Unsupported triplet_policy: {policy}")


def _frame_relative_path(
    row: Mapping[str, str],
    offset: int,
    config: DatasetVersionConfig,
    settings: Settings,
) -> str:
    relative_sequence_dir = row["relative_sequence_dir"]
    file_name = _frame_file_name(relative_sequence_dir, offset, config, settings)
    frame_path = f"{relative_sequence_dir}/{file_name}"
    validate_relative_artifact_path(frame_path, "frame_path")
    if config.validate_frame_paths and not (settings.dataset_root_abs / frame_path).is_file():
        raise FileNotFoundError(f"Manifest frame path does not exist under DATASET_ROOT: {frame_path}")
    return frame_path


def _frame_file_name(
    relative_sequence_dir: str,
    offset: int,
    config: DatasetVersionConfig,
    settings: Settings,
) -> str:
    candidates = {
        "im": f"im{offset + 1}.png",
        "frame": f"frame_{offset:03d}.png",
    }
    if config.frame_name_pattern in ("im", "frame"):
        return candidates[config.frame_name_pattern]

    for file_name in (candidates["im"], candidates["frame"]):
        if (settings.dataset_root_abs / relative_sequence_dir / file_name).is_file():
            return file_name
    if config.validate_frame_paths:
        raise FileNotFoundError(
            f"Could not infer frame naming in {relative_sequence_dir}; expected im*.png or frame_*.png"
        )
    return candidates["im"]


def _split_source_videos(
    source_video_ids: Sequence[str],
    split_ratios: Mapping[str, float],
    seed: int,
) -> dict[str, str]:
    source_ids = sorted(set(source_video_ids))
    rng = random.Random(seed)
    rng.shuffle(source_ids)
    counts = _split_counts(len(source_ids), split_ratios)
    split_by_video: dict[str, str] = {}
    cursor = 0
    for split_name in ("train", "val", "test"):
        for source_id in source_ids[cursor : cursor + counts[split_name]]:
            split_by_video[source_id] = split_name
        cursor += counts[split_name]
    return split_by_video


def _split_counts(total: int, split_ratios: Mapping[str, float]) -> dict[str, int]:
    ratios = {name: float(split_ratios.get(name, 0.0)) for name in ("train", "val", "test")}
    positive = [name for name, value in ratios.items() if value > 0]
    counts = {name: 0 for name in ("train", "val", "test")}
    if total <= 0:
        return counts
    if total < len(positive):
        for split_name in sorted(positive, key=lambda name: ratios[name], reverse=True)[:total]:
            counts[split_name] = 1
        return counts

    for split_name in positive:
        counts[split_name] = 1
    remaining = total - len(positive)
    normalized_total = sum(ratios.values())
    ideals = {name: (ratios[name] / normalized_total) * total for name in ratios}
    for _ in range(remaining):
        split_name = max(("train", "val", "test"), key=lambda name: ideals[name] - counts[name])
        counts[split_name] += 1
    return counts


def _apply_train_mixing(
    train_rows: Sequence[dict[str, str]],
    config: DatasetVersionConfig,
) -> list[dict[str, str]]:
    if config.mixing_mode == "none":
        return _take_rows(train_rows, config.train_budget, config.split_seed)

    new_groups = set(config.new_source_groups)
    old_groups = set(config.old_source_groups)
    if not new_groups and not old_groups:
        return _take_rows(train_rows, config.train_budget, config.split_seed)

    old_rows: list[dict[str, str]] = []
    new_rows: list[dict[str, str]] = []
    for row in train_rows:
        if row["source_group"] in new_groups:
            new_rows.append(row)
        elif not old_groups or row["source_group"] in old_groups:
            old_rows.append(row)

    budget = config.train_budget or len(train_rows)
    if not old_rows or not new_rows:
        return _take_rows([*old_rows, *new_rows], budget, config.split_seed)

    rng = random.Random(config.split_seed)
    if config.mixing_mode == "data_accumulation":
        new_limit = math.floor(budget * config.new_max_ratio)
        selected_new = _sample_without_replacement(new_rows, min(len(new_rows), new_limit), rng)
        selected_old = _sample_without_replacement(old_rows, min(len(old_rows), budget - len(selected_new)), rng)
        return [*selected_old, *selected_new]

    old_limit = math.floor(budget * config.old_max_ratio)
    selected_old = _sample_without_replacement(old_rows, min(len(old_rows), old_limit), rng)
    new_needed = max(math.ceil(budget * config.new_min_ratio), budget - len(selected_old))
    selected_new = _sample_with_optional_oversampling(new_rows, min(budget, new_needed), rng)
    return [*selected_old, *selected_new[: max(0, budget - len(selected_old))]]


def _take_rows(
    rows: Sequence[dict[str, str]],
    limit: int | None,
    seed: int,
) -> list[dict[str, str]]:
    if limit is None or len(rows) <= limit:
        return list(rows)
    return _sample_without_replacement(rows, limit, random.Random(seed))


def _sample_without_replacement(
    rows: Sequence[dict[str, str]],
    count: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[:count]


def _sample_with_optional_oversampling(
    rows: Sequence[dict[str, str]],
    count: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    if not rows or count <= 0:
        return []
    if len(rows) >= count:
        return _sample_without_replacement(rows, count, rng)
    selected = list(rows)
    while len(selected) < count:
        duplicate = dict(rng.choice(rows))
        duplicate["sample_id"] = f"{duplicate['sample_id']}__oversample_{len(selected):06d}"
        selected.append(duplicate)
    return selected


def _read_global_index(path: Path) -> list[dict[str, str]]:
    validate_csv_file(str(path), GLOBAL_SEQUENCE_INDEX_CONTRACT)
    with path.open(newline="", encoding="utf-8") as csv_file:
        return [dict(row) for row in csv.DictReader(csv_file)]


def _write_manifest(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRIPLET_MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in TRIPLET_MANIFEST_COLUMNS})
    tmp_path.replace(path)
    validate_csv_file(str(path), TRIPLET_MANIFEST_CONTRACT)


def _write_dataset_config(
    path: Path,
    config: DatasetVersionConfig,
    source_index_path: Path,
    output_root: Path,
    settings: Settings,
    *,
    created_at: str,
    manifest_counts: Mapping[str, int],
    split_video_counts: Mapping[str, int],
) -> None:
    data: dict[str, object] = {
        "dataset_version_id": config.dataset_version_id,
        "source_index_path": _portable_source_index_path(source_index_path, output_root, settings),
        "split_seed": config.split_seed,
        "split_ratios": dict(config.split_ratios),
        "selected_source_groups": list(config.selected_source_groups),
        "triplet_policy": config.triplet_policy,
        "mixing_mode": config.mixing_mode,
        "old_source_groups": list(config.old_source_groups),
        "new_source_groups": list(config.new_source_groups),
        "old_min_ratio": config.old_min_ratio,
        "new_max_ratio": config.new_max_ratio,
        "new_min_ratio": config.new_min_ratio,
        "old_max_ratio": config.old_max_ratio,
        "train_budget": config.train_budget,
        "limit_sequences": config.limit_sequences,
        "validate_frame_paths": config.validate_frame_paths,
        "frame_name_pattern": config.frame_name_pattern,
        "preprocessing": dict(config.preprocessing),
        "manifest_counts": dict(manifest_counts),
        "split_video_counts": dict(split_video_counts),
        "created_at": created_at,
    }
    validate_dataset_config_mapping(data)
    OmegaConf.save(config=OmegaConf.create(data), f=path)


def _portable_source_index_path(source_index_path: Path, output_root: Path, settings: Settings) -> str:
    try:
        return source_index_path.relative_to(output_root).as_posix()
    except ValueError:
        pass
    try:
        return source_index_path.relative_to(settings.dataset_root_abs).as_posix()
    except ValueError:
        pass
    raise ValueError(
        "source_index_path must be inside output_root or DATASET_ROOT "
        "to record a portable dataset_config.yaml reference"
    )


def _resolve_source_index_path(settings: Settings, path: Path) -> Path:
    if path.is_absolute():
        return path
    return settings.dataset_root_abs / path


def _resolve_output_root(settings: Settings, path: Path) -> Path:
    if path.is_absolute():
        return path
    return settings.resolve_path(path)


def _split_video_counts(split_by_video: Mapping[str, str]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for split_name in split_by_video.values():
        counts[split_name] += 1
    return counts


def _validate_split_ratios(split_ratios: Mapping[str, float]) -> None:
    total = 0.0
    for split_name in ("train", "val", "test"):
        value = float(split_ratios.get(split_name, 0.0))
        if value < 0:
            raise ValueError(f"split ratio {split_name} must be non-negative")
        total += value
    if total <= 0:
        raise ValueError("At least one split ratio must be positive")


def _emit_progress(
    callback: ProgressCallback | None,
    event: str,
    **payload: object,
) -> None:
    if callback is not None:
        callback(event, payload)
