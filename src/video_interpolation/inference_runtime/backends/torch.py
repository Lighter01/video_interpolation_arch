from collections.abc import Callable, Mapping, Sequence

import torch

from video_interpolation.inference_runtime.api import RuntimeBackendKind, RuntimeInputs, RuntimeOutputs
from video_interpolation.inference_runtime.backends.base import RuntimeBackend

TorchBackendRunner = Callable[
    [RuntimeInputs],
    RuntimeOutputs | torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
]


class TorchRuntimeBackend(RuntimeBackend):
    """Minimal PyTorch backend around an explicit model-specific runner."""

    kind = RuntimeBackendKind.TORCH

    def __init__(
        self,
        runner: TorchBackendRunner,
        *,
        module: torch.nn.Module | None = None,
        name: str = "torch",
    ) -> None:
        self.runner = runner
        self.module = module
        self.name = name
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        if self.module is not None:
            self.module.eval()
        self._is_loaded = True

    def run(self, inputs: RuntimeInputs) -> RuntimeOutputs:
        self.ensure_loaded()
        with torch.no_grad():
            outputs = self.runner(inputs)
        return _normalise_outputs(outputs)

    def close(self) -> None:
        self._is_loaded = False


def _normalise_outputs(
    outputs: RuntimeOutputs | torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
) -> RuntimeOutputs:
    if isinstance(outputs, RuntimeOutputs):
        return outputs
    if isinstance(outputs, torch.Tensor):
        return RuntimeOutputs(tensors={"output": outputs})
    if isinstance(outputs, Mapping):
        return RuntimeOutputs(tensors=outputs)
    if isinstance(outputs, Sequence) and not isinstance(outputs, str | bytes):
        return RuntimeOutputs(tensors={f"output_{index}": tensor for index, tensor in enumerate(outputs)})
    raise TypeError(f"Unsupported Torch runtime output type: {type(outputs).__name__}.")
