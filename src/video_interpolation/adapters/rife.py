import contextlib
import gc
import importlib
import os
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from video_interpolation.adapters.base import AdapterEnvironmentReport, ModelAdapter, ModelAdapterError
from video_interpolation.settings import Settings, load_settings


@dataclass(frozen=True)
class PracticalRIFEAdapterConfig:
    model_name: str = "practical_rife_v4_25"
    repo_name: str = "Practical-RIFE"
    checkpoint_path: Path = Path("Practical-RIFE/RIFEv4.25/train_log")
    device: str = "cuda"
    timestep: float = 0.5
    scale: float = 1.0
    divisor: int = 128
    strict_checkpoint: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "PracticalRIFEAdapterConfig":
        values = dict(data or {})
        if "checkpoint_path" in values:
            values["checkpoint_path"] = Path(values["checkpoint_path"])
        return cls(**values)


class PracticalRIFEAdapter(ModelAdapter):
    """Stage 1 adapter for Practical-RIFE v4.25 eval/inference from local weights."""

    def __init__(
        self,
        config: PracticalRIFEAdapterConfig | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.config = config or PracticalRIFEAdapterConfig()
        self.settings = settings or load_settings()
        self.model_name = self.config.model_name
        self._model: Any | None = None
        self._repo_context: contextlib.AbstractContextManager[None] | None = None
        self._repo_context_entered = False
        self._device = torch.device(self.config.device)

    @property
    def repo_path(self) -> Path:
        return self.settings.model_repo_path(self.config.repo_name)

    @property
    def checkpoint_path(self) -> Path:
        if self.config.checkpoint_path.is_absolute():
            return self.config.checkpoint_path
        return self.settings.model_weights_root_abs / self.config.checkpoint_path

    @property
    def weight_package_root(self) -> Path:
        return self.checkpoint_path.parent

    @property
    def checkpoint_file_path(self) -> Path:
        return self.checkpoint_path / "flownet.pkl"

    def validate_environment(self) -> AdapterEnvironmentReport:
        report = AdapterEnvironmentReport()
        report.add("rife_repo_path", "ok" if self.repo_path.is_dir() else "failed", str(self.repo_path))
        report.add(
            "rife_checkpoint_dir",
            "ok" if self.checkpoint_path.is_dir() else "failed",
            str(self.checkpoint_path),
        )
        report.add(
            "rife_checkpoint_file",
            "ok" if self.checkpoint_file_path.is_file() else "failed",
            str(self.checkpoint_file_path),
        )
        report.add(
            "rife_model_file",
            "ok" if (self.checkpoint_path / "RIFE_HDv3.py").is_file() else "failed",
            str(self.checkpoint_path / "RIFE_HDv3.py"),
        )
        report.add(
            "rife_ifnet_file",
            "ok" if (self.checkpoint_path / "IFNet_HDv3.py").is_file() else "failed",
            str(self.checkpoint_path / "IFNet_HDv3.py"),
        )
        report.add("torch_import", "ok", f"torch {torch.__version__}")
        if self.config.device == "cuda" and not torch.cuda.is_available():
            report.add("cuda_available", "blocked", "CUDA is unavailable for configured Practical-RIFE device.")
        else:
            report.add("cuda_available", "ok", str(torch.cuda.is_available()))
        report.add(
            "rife_training_manifest_ready",
            "ok",
            "Not required for eval/inference; upstream Practical-RIFE training needs a future manifest/data refactor.",
        )

        if report.status == "failed":
            return report

        try:
            with _rife_import_context(self.repo_path, self.weight_package_root):
                importlib.import_module("model.warplayer")
                importlib.import_module("model.loss")
                importlib.import_module("train_log.IFNet_HDv3")
                importlib.import_module("train_log.RIFE_HDv3")
            report.add("rife_import", "ok", "Imported Practical-RIFE model modules and selected train_log code")
        except Exception as exc:  # pragma: no cover - environment smoke coverage.
            report.add("rife_import", "failed", f"{type(exc).__name__}: {exc}")
        return report

    def build_model(self) -> None:
        if self._model is not None:
            return
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise ModelAdapterError("CUDA is unavailable for Practical-RIFE with device=cuda.")
        try:
            self._enter_repo_context()
            rife_module = importlib.import_module("train_log.RIFE_HDv3")
            _set_rife_module_devices(self._device)
            self._model = rife_module.Model()
            self.eval()
        except Exception:
            self.close()
            raise

    def load_checkpoint(self, checkpoint_path: Path | None = None) -> None:
        if checkpoint_path is not None:
            object.__setattr__(
                self,
                "config",
                PracticalRIFEAdapterConfig(
                    model_name=self.config.model_name,
                    repo_name=self.config.repo_name,
                    checkpoint_path=checkpoint_path,
                    device=self.config.device,
                    timestep=self.config.timestep,
                    scale=self.config.scale,
                    divisor=self.config.divisor,
                    strict_checkpoint=self.config.strict_checkpoint,
                ),
            )
        self.build_model()
        if not self.checkpoint_file_path.is_file():
            raise ModelAdapterError(f"Practical-RIFE checkpoint does not exist: {self.checkpoint_file_path}")
        checkpoint = torch.load(self.checkpoint_file_path, map_location=self._device, weights_only=True)
        state_dict = _normalise_rife_state_dict(checkpoint)
        missing, unexpected = self._model.flownet.load_state_dict(
            state_dict,
            strict=self.config.strict_checkpoint,
        )
        if self.config.strict_checkpoint and (missing or unexpected):
            raise ModelAdapterError(
                "Practical-RIFE strict checkpoint loading reported mismatched keys: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self.eval()

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        self._require_model()
        if checkpoint_path.suffix:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.flownet.state_dict(), checkpoint_path)
        else:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.flownet.state_dict(), checkpoint_path / "flownet.pkl")

    def train(self) -> None:
        self._require_model()
        self._model.train()

    def eval(self) -> None:
        self._require_model()
        self._model.eval()

    def predict_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        self._require_model()
        self.eval()
        left_batch = _prepare_image_tensor(left, self._device)
        right_batch = _prepare_image_tensor(right, self._device)
        if left_batch.shape != right_batch.shape:
            raise ValueError(f"left and right tensors must have matching shape: {left.shape} vs {right.shape}")
        left_padded, padding = _pad_to_divisor(left_batch, self.config.divisor)
        right_padded, _ = _pad_to_divisor(right_batch, self.config.divisor)

        with torch.no_grad():
            prediction = self._model.inference(
                left_padded,
                right_padded,
                self.config.timestep,
                self.config.scale,
            )
            prediction = _unpad(prediction, padding)
        return prediction[0].detach().cpu().clamp(0.0, 1.0).contiguous()

    def close(self) -> None:
        self._model = None
        if self._repo_context is not None and self._repo_context_entered:
            self._repo_context.__exit__(None, None, None)
        self._repo_context = None
        self._repo_context_entered = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _require_model(self) -> None:
        if self._model is None:
            raise ModelAdapterError("Practical-RIFE model is not built. Call build_model/load_checkpoint first.")

    def _enter_repo_context(self) -> None:
        if self._repo_context_entered:
            return
        self._repo_context = _rife_import_context(self.repo_path, self.weight_package_root)
        self._repo_context.__enter__()
        self._repo_context_entered = True


@contextlib.contextmanager
def _rife_import_context(repo_path: Path, weight_package_root: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    tracked_prefixes = ("model", "train_log")
    tracked_modules = {
        name: sys.modules.get(name)
        for name in list(sys.modules)
        if name in tracked_prefixes or name.startswith(tuple(f"{prefix}." for prefix in tracked_prefixes))
    }
    try:
        os.chdir(repo_path)
        sys.path.insert(0, str(weight_package_root))
        sys.path.insert(0, str(repo_path))
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name in tracked_prefixes or name.startswith(tuple(f"{prefix}." for prefix in tracked_prefixes)):
                sys.modules.pop(name, None)
        for name, module in tracked_modules.items():
            if module is not None:
                sys.modules[name] = module


def _set_rife_module_devices(device: torch.device) -> None:
    for module_name in ("model.warplayer", "model.loss", "train_log.IFNet_HDv3", "train_log.RIFE_HDv3"):
        module = importlib.import_module(module_name)
        if hasattr(module, "device"):
            module.device = device


def _normalise_rife_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise ModelAdapterError("Practical-RIFE checkpoint is not a state-dict mapping")
    converted = {
        str(key).replace("module.", ""): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }
    if not converted:
        raise ModelAdapterError("Practical-RIFE checkpoint did not contain loadable tensor weights")
    return converted


def _prepare_image_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"Expected image tensor shape CxHxW or BxCxHxW with 3 channels, got {tuple(tensor.shape)}")
    return tensor.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)


def _pad_to_divisor(tensor: torch.Tensor, divisor: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    height = tensor.shape[-2]
    width = tensor.shape[-1]
    pad_height = ((height - 1) // divisor + 1) * divisor - height
    pad_width = ((width - 1) // divisor + 1) * divisor - width
    padding = (0, pad_width, 0, pad_height)
    if pad_height == 0 and pad_width == 0:
        return tensor, padding
    return F.pad(tensor, padding), padding


def _unpad(tensor: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    _, pad_right, _, pad_bottom = padding
    height = tensor.shape[-2] - pad_bottom
    width = tensor.shape[-1] - pad_right
    return tensor[..., :height, :width]
