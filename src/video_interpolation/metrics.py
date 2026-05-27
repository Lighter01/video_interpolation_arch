import csv
import gc
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lpips
import numpy as np
from skimage.metrics import structural_similarity
import torch

from video_interpolation.contracts import METRICS_CSV_COLUMNS, METRICS_CSV_CONTRACT, validate_csv_file


ImageLike = torch.Tensor | np.ndarray

METRICS_OUTPUT_COLUMNS: tuple[str, ...] = (
    *METRICS_CSV_COLUMNS,
    "prediction_path",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "prediction_name",
    "source_group",
    "source_dataset",
    "count",
    "psnr_mean",
    "ssim_mean",
    "lpips_mean",
)


@dataclass(frozen=True)
class SampleMetrics:
    psnr: float
    ssim: float
    lpips: float | None = None


class LPIPSEvaluator:
    """Reusable LPIPS scorer with explicit cleanup for model-backed metrics."""

    def __init__(self, device: str = "cpu", net: str = "alex") -> None:
        self.device = torch.device(device)
        self.model = lpips.LPIPS(net=net, verbose=False).to(self.device)
        self.model.eval()

    def score(self, prediction: ImageLike, target: ImageLike) -> float:
        prediction_tensor = _to_lpips_tensor(prediction, self.device)
        target_tensor = _to_lpips_tensor(target, self.device)
        with torch.no_grad():
            value = self.model(prediction_tensor, target_tensor)
        return float(value.detach().cpu().item())

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def compute_psnr(prediction: ImageLike, target: ImageLike, data_range: float = 1.0) -> float:
    prediction_array = _to_numpy_float(prediction)
    target_array = _to_numpy_float(target)
    mse = float(np.mean((prediction_array - target_array) ** 2))
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def compute_ssim(prediction: ImageLike, target: ImageLike, data_range: float = 1.0) -> float:
    prediction_array = _to_numpy_float(prediction)
    target_array = _to_numpy_float(target)
    minimum_side = min(prediction_array.shape[0], prediction_array.shape[1])
    win_size = min(7, minimum_side if minimum_side % 2 == 1 else minimum_side - 1)
    if win_size < 3:
        raise ValueError("SSIM requires images at least 3 pixels on the shortest side")
    return float(
        structural_similarity(
            target_array,
            prediction_array,
            channel_axis=2,
            data_range=data_range,
            win_size=win_size,
        )
    )


def compute_sample_metrics(
    prediction: ImageLike,
    target: ImageLike,
    lpips_evaluator: LPIPSEvaluator | None = None,
) -> SampleMetrics:
    return SampleMetrics(
        psnr=compute_psnr(prediction, target),
        ssim=compute_ssim(prediction, target),
        lpips=lpips_evaluator.score(prediction, target) if lpips_evaluator is not None else None,
    )


def aggregate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        keys = [
            (str(row["prediction_name"]), "__all__", "__all__"),
            (str(row["prediction_name"]), str(row["source_group"]), "__all__"),
            (str(row["prediction_name"]), str(row["source_group"]), str(row["source_dataset"])),
        ]
        for key in keys:
            grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, str]] = []
    for (prediction_name, source_group, source_dataset), group_rows in sorted(grouped.items()):
        summaries.append(
            {
                "prediction_name": prediction_name,
                "source_group": source_group,
                "source_dataset": source_dataset,
                "count": str(len(group_rows)),
                "psnr_mean": _format_metric(_mean_metric(group_rows, "psnr")),
                "ssim_mean": _format_metric(_mean_metric(group_rows, "ssim")),
                "lpips_mean": _format_metric(_mean_metric(group_rows, "lpips")),
            }
        )
    return summaries


def write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METRICS_OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _stringify_csv_value(row.get(field, "")) for field in METRICS_OUTPUT_COLUMNS})
    validate_csv_file(str(path), METRICS_CSV_CONTRACT)


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_COLUMNS})


def _to_numpy_float(image: ImageLike) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().float().numpy()
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.moveaxis(array, 0, -1)
    else:
        array = np.asarray(image)

    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32)
    if array.ndim == 2:
        array = np.expand_dims(array, axis=2)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    return np.clip(array, 0.0, 1.0)


def _to_lpips_tensor(image: ImageLike, device: torch.device) -> torch.Tensor:
    array = _to_numpy_float(image)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor * 2.0 - 1.0


def _mean_metric(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value in (None, ""):
            continue
        numeric = float(value)
        values.append(numeric)
    if not values:
        return None
    return float(np.mean(values))


def _format_metric(value: float | None) -> str:
    if value is None:
        return ""
    if math.isinf(value):
        return "inf"
    return f"{value:.8f}"


def _stringify_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return _format_metric(value)
    return str(value)
