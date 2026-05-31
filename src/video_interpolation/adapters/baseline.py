from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from video_interpolation.adapters.base import AdapterEnvironmentReport, ModelAdapter, ModelAdapterError
from video_interpolation.baselines import BASELINE_NAMES, predict_middle_frame


@dataclass(frozen=True)
class BaselineAdapterConfig:
    model_name: str = "baseline_blend"
    baseline_name: str = "blend"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "BaselineAdapterConfig":
        return cls(**dict(data or {}))


class BaselineAdapter(ModelAdapter):
    """ModelAdapter-compatible wrapper for non-neural baseline interpolation methods."""

    def __init__(self, config: BaselineAdapterConfig | None = None) -> None:
        self.config = config or BaselineAdapterConfig()
        self.model_name = self.config.model_name

    def validate_environment(self) -> AdapterEnvironmentReport:
        report = AdapterEnvironmentReport()
        status = "ok" if self.config.baseline_name in BASELINE_NAMES else "failed"
        report.add("baseline_name", status, self.config.baseline_name)
        report.add("torch_import", "ok", f"torch {torch.__version__}")
        return report

    def build_model(self) -> None:
        self._validate_baseline_name()

    def load_checkpoint(self, checkpoint_path: Path | None = None) -> None:
        if checkpoint_path is not None:
            raise ModelAdapterError("Baseline adapters do not load checkpoints.")
        self.build_model()

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        raise ModelAdapterError("Baseline adapters do not save checkpoints.")

    def train(self) -> None:
        self._validate_baseline_name()

    def eval(self) -> None:
        self._validate_baseline_name()

    def predict_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        self._validate_baseline_name()
        return predict_middle_frame(self.config.baseline_name, left, right)

    def _validate_baseline_name(self) -> None:
        if self.config.baseline_name not in BASELINE_NAMES:
            raise ModelAdapterError(
                f"Unsupported baseline '{self.config.baseline_name}'. "
                f"Choose one of: {', '.join(BASELINE_NAMES)}."
            )


def with_baseline_adapter_name(
    config: BaselineAdapterConfig,
    baseline_name: str | None,
) -> BaselineAdapterConfig:
    if baseline_name is None:
        return config
    return replace(config, baseline_name=baseline_name, model_name=f"baseline_{baseline_name}")
