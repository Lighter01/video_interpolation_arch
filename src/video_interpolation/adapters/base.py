from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch


class ModelAdapterError(RuntimeError):
    """Raised when a model adapter cannot satisfy a requested operation."""


@dataclass(frozen=True)
class AdapterEnvironmentCheck:
    name: str
    status: str
    detail: str


@dataclass
class AdapterEnvironmentReport:
    status: str = "ok"
    checks: list[AdapterEnvironmentCheck] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(AdapterEnvironmentCheck(name=name, status=status, detail=detail))
        if status == "failed":
            self.status = "failed"
        elif status == "blocked" and self.status == "ok":
            self.status = "blocked"


class ModelAdapter(ABC):
    """Common interface for Stage 1 VFI model adapters."""

    model_name: str

    @abstractmethod
    def validate_environment(self) -> AdapterEnvironmentReport:
        """Check local prerequisites and report actionable compatibility status."""

    @abstractmethod
    def build_model(self) -> None:
        """Instantiate the underlying model implementation."""

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: Path | None = None) -> None:
        """Load model weights from an explicit checkpoint path."""

    @abstractmethod
    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """Save model weights to a checkpoint path."""

    @abstractmethod
    def train(self) -> None:
        """Put the model in training mode."""

    @abstractmethod
    def eval(self) -> None:
        """Put the model in evaluation mode."""

    @abstractmethod
    def predict_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Predict the middle frame for one left/right frame pair."""

    def predict_batch(self, pairs: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> list[torch.Tensor]:
        """Sequential default batch inference; optimized batching can be added per adapter later."""
        return [self.predict_pair(left, right) for left, right in pairs]

    def predict(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.predict_pair(left, right)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.predict_pair(left, right)

    def close(self) -> None:
        """Release adapter-owned resources."""
