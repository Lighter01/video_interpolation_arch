import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath


SEQUENCE_INDEX_COLUMNS: tuple[str, ...] = (
    "sequence_id",
    "source_group",
    "source_dataset",
    "source_video_id",
    "relative_sequence_dir",
    "sequence_length",
    "frame_step",
    "width",
    "height",
    "fps",
    "frame_start",
    "frame_end",
    "scene_id",
    "resize",
    "created_at",
)

GLOBAL_SEQUENCE_INDEX_COLUMNS = SEQUENCE_INDEX_COLUMNS

TRIPLET_MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "source_group",
    "source_dataset",
    "source_video_id",
    "sequence_id",
    "left_frame_path",
    "mid_frame_path",
    "right_frame_path",
    "t_value",
    "width",
    "height",
    "fps",
    "frame_step",
)

DATASET_CONFIG_FIELDS: tuple[str, ...] = (
    "dataset_version_id",
    "source_index_path",
    "split_seed",
    "split_ratios",
    "selected_source_groups",
    "triplet_policy",
    "mixing_mode",
    "preprocessing",
    "created_at",
)
DATASET_CONFIG_RELATIVE_PATH_FIELDS: tuple[str, ...] = ("source_index_path",)

CANDIDATE_VALIDATION_REPORT_FIELDS: tuple[str, ...] = (
    "candidate_id",
    "model_name",
    "model_adapter",
    "checkpoint",
    "dataset_version_id",
    "test_manifest_path",
    "validation_thresholds",
    "aggregate_metrics",
    "approval_decision",
    "decision_reasons",
    "created_at",
)

METRICS_CSV_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "source_group",
    "source_dataset",
    "source_video_id",
    "prediction_name",
    "psnr",
    "ssim",
    "lpips",
)


class ContractValidationError(ValueError):
    """Raised when a Stage 1 artifact does not satisfy its compact contract."""


@dataclass(frozen=True)
class CsvArtifactContract:
    name: str
    required_columns: tuple[str, ...]
    relative_path_columns: tuple[str, ...] = ()


SEQUENCE_INDEX_CONTRACT = CsvArtifactContract(
    name="sequence_index.csv",
    required_columns=SEQUENCE_INDEX_COLUMNS,
    relative_path_columns=("relative_sequence_dir",),
)

GLOBAL_SEQUENCE_INDEX_CONTRACT = CsvArtifactContract(
    name="global_sequence_index.csv",
    required_columns=GLOBAL_SEQUENCE_INDEX_COLUMNS,
    relative_path_columns=("relative_sequence_dir",),
)

TRIPLET_MANIFEST_CONTRACT = CsvArtifactContract(
    name="triplet manifest",
    required_columns=TRIPLET_MANIFEST_COLUMNS,
    relative_path_columns=("left_frame_path", "mid_frame_path", "right_frame_path"),
)

METRICS_CSV_CONTRACT = CsvArtifactContract(
    name="metrics.csv",
    required_columns=METRICS_CSV_COLUMNS,
)


def is_relative_artifact_path(value: str) -> bool:
    if not value:
        return False
    if "://" in value:
        return False

    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute():
        return False
    return ".." not in path.parts


def validate_relative_artifact_path(value: str, field_name: str = "path") -> None:
    if not is_relative_artifact_path(value):
        raise ContractValidationError(f"{field_name} must be a relative artifact path: {value!r}")


def missing_required_fields(required: Sequence[str], available: Iterable[str]) -> list[str]:
    available_set = set(available)
    return [field for field in required if field not in available_set]


def validate_required_fields(
    data: Mapping[str, object],
    required_fields: Sequence[str],
    artifact_name: str,
) -> None:
    missing = missing_required_fields(required_fields, data.keys())
    if missing:
        raise ContractValidationError(
            f"{artifact_name} is missing required field(s): {', '.join(missing)}"
        )


def validate_csv_header(
    fieldnames: Sequence[str] | None,
    contract: CsvArtifactContract,
) -> None:
    if fieldnames is None:
        raise ContractValidationError(f"{contract.name} has no header")
    missing = missing_required_fields(contract.required_columns, fieldnames)
    if missing:
        raise ContractValidationError(
            f"{contract.name} is missing required column(s): {', '.join(missing)}"
        )


def validate_csv_rows(
    rows: Iterable[Mapping[str, str]],
    contract: CsvArtifactContract,
) -> None:
    for row_number, row in enumerate(rows, start=2):
        for column in contract.relative_path_columns:
            validate_relative_artifact_path(
                row.get(column, ""),
                field_name=f"{contract.name}:{row_number}:{column}",
            )


def validate_csv_file(path: str, contract: CsvArtifactContract) -> None:
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_csv_header(reader.fieldnames, contract)
        validate_csv_rows(reader, contract)


def validate_dataset_config_mapping(data: Mapping[str, object]) -> None:
    validate_required_fields(data, DATASET_CONFIG_FIELDS, "dataset_config.yaml")
    for field_name in DATASET_CONFIG_RELATIVE_PATH_FIELDS:
        value = data.get(field_name)
        if not isinstance(value, str):
            raise ContractValidationError(f"dataset_config.yaml:{field_name} must be a string")
        validate_relative_artifact_path(value, f"dataset_config.yaml:{field_name}")
