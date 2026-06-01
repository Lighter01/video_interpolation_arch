from abc import ABC, abstractmethod

from video_interpolation.inference_runtime.api import RuntimeBackendKind, RuntimeInputs, RuntimeOutputs


class RuntimeBackendError(RuntimeError):
    """Raised when a runtime backend cannot execute a request."""


class RuntimeBackend(ABC):
    """Shared lifecycle and execution boundary for inference backends."""

    kind: RuntimeBackendKind

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the backend has loaded its executable runtime resources."""

    @abstractmethod
    def load(self) -> None:
        """Load runtime resources before the first call."""

    @abstractmethod
    def run(self, inputs: RuntimeInputs) -> RuntimeOutputs:
        """Execute a model-specific runtime call on already formatted tensors."""

    def close(self) -> None:
        """Release runtime resources."""

    def ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeBackendError(f"{self.kind.value} runtime backend is not loaded.")
