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

from video_interpolation.adapters.base import AdapterEnvironmentReport, ModelAdapter, ModelAdapterError
from video_interpolation.settings import Settings, load_settings


@dataclass(frozen=True)
class EMAVFIAdapterConfig:
    model_name: str = "ema_vfi_small"
    repo_name: str = "EMA-VFI"
    checkpoint_path: Path = Path("EMA-VFI/ours_small.pkl")
    device: str = "cuda"
    tta: bool = False
    fast_tta: bool = False
    divisor: int = 32
    strict_checkpoint: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "EMAVFIAdapterConfig":
        values = dict(data or {})
        if "checkpoint_path" in values:
            values["checkpoint_path"] = Path(values["checkpoint_path"])
        return cls(**values)


class EMAVFIAdapter(ModelAdapter):
    """Stage 1 adapter for EMA-VFI-small."""

    def __init__(
        self,
        config: EMAVFIAdapterConfig | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.config = config or EMAVFIAdapterConfig()
        self.settings = settings or load_settings()
        self.model_name = self.config.model_name
        self._model: Any | None = None
        self._input_padder_cls: Any | None = None
        self._repo_context: contextlib.AbstractContextManager[None] | None = None
        self._repo_context_entered = False
        self._device = torch.device(self.config.device)

    @property
    def checkpoint_path(self) -> Path:
        checkpoint_path = self.config.checkpoint_path
        if checkpoint_path.is_absolute():
            return checkpoint_path
        return self.settings.model_weights_root_abs / checkpoint_path

    @property
    def repo_path(self) -> Path:
        return self.settings.model_repo_path(self.config.repo_name)

    def validate_environment(self) -> AdapterEnvironmentReport:
        report = AdapterEnvironmentReport()
        report.add("ema_repo_path", "ok" if self.repo_path.is_dir() else "failed", str(self.repo_path))
        report.add("ema_checkpoint_path", "ok" if self.checkpoint_path.is_file() else "failed", str(self.checkpoint_path))
        report.add("torch_import", "ok", f"torch {torch.__version__}")
        if self.config.device == "cuda" and not torch.cuda.is_available():
            report.add(
                "cuda_available",
                "blocked",
                "CUDA is unavailable; upstream EMA-VFI Trainer.Model hardcodes cuda device setup.",
            )
        else:
            report.add("cuda_available", "ok", str(torch.cuda.is_available()))

        if report.status == "failed":
            return report

        try:
            with _ema_import_context(self.repo_path):
                config_module = importlib.import_module("config")
                _configure_ema_small(config_module)
                importlib.import_module("Trainer")
                importlib.import_module("benchmark.utils.padder")
            report.add("ema_import", "ok", "Imported EMA-VFI config, Trainer, and InputPadder")
        except Exception as exc:  # pragma: no cover - environment smoke coverage.
            report.add("ema_import", "failed", f"{type(exc).__name__}: {exc}")
        return report

    def build_model(self) -> None:
        if self._model is not None:
            return
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise ModelAdapterError(
                "CUDA is unavailable. EMA-VFI upstream Trainer.Model currently hardcodes CUDA setup."
            )
        try:
            self._enter_repo_context()
            config_module = importlib.import_module("config")
            _configure_ema_small(config_module)
            trainer_module = importlib.import_module("Trainer")
            padder_module = importlib.import_module("benchmark.utils.padder")
            self._model = trainer_module.Model(-1)
            self._input_padder_cls = padder_module.InputPadder
        except Exception:
            self.close()
            raise

    def load_checkpoint(self, checkpoint_path: Path | None = None) -> None:
        self.build_model()
        path = checkpoint_path or self.checkpoint_path
        if not path.is_file():
            raise ModelAdapterError(f"EMA-VFI checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location=self.config.device)
        state_dict = _normalise_ema_state_dict(checkpoint)
        self._model.net.load_state_dict(state_dict, strict=self.config.strict_checkpoint)
        self.eval()

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        self._require_model()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.net.state_dict(), checkpoint_path)

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

        with torch.no_grad():
            padder = self._input_padder_cls(left_batch.shape, divisor=self.config.divisor)
            left_padded, right_padded = padder.pad(left_batch, right_batch)
            prediction = self._model.inference(
                left_padded,
                right_padded,
                TTA=self.config.tta,
                fast_TTA=self.config.fast_tta,
            )
            prediction = padder.unpad(prediction)
        return prediction[0].detach().cpu().clamp(0.0, 1.0).contiguous()

    def train_step(
        self,
        left: torch.Tensor,
        middle: torch.Tensor,
        right: torch.Tensor,
        learning_rate: float,
    ) -> tuple[torch.Tensor, float]:
        self._require_model()
        self.train()
        left_device = _prepare_image_tensor(left, self._device)
        middle_device = _prepare_image_tensor(middle, self._device)
        right_device = _prepare_image_tensor(right, self._device)
        padder = self._input_padder_cls(left_device.shape, divisor=self.config.divisor)
        left_padded, middle_padded, right_padded = padder.pad(left_device, middle_device, right_device)
        imgs = torch.cat((left_padded, right_padded), dim=1)
        gt = middle_padded
        prediction, loss = self._model.update(imgs, gt, learning_rate=learning_rate, training=True)
        prediction = padder.unpad(prediction)
        return prediction.detach().cpu().clamp(0.0, 1.0), float(loss.detach().cpu().item())

    def eval_step(self, left: torch.Tensor, middle: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        self._require_model()
        self.eval()
        left_device = _prepare_image_tensor(left, self._device)
        middle_device = _prepare_image_tensor(middle, self._device)
        right_device = _prepare_image_tensor(right, self._device)
        padder = self._input_padder_cls(left_device.shape, divisor=self.config.divisor)
        left_padded, middle_padded, right_padded = padder.pad(left_device, middle_device, right_device)
        imgs = torch.cat((left_padded, right_padded), dim=1)
        gt = middle_padded
        with torch.no_grad():
            prediction, _ = self._model.update(imgs, gt, training=False)
        prediction = padder.unpad(prediction)
        return prediction.detach().cpu().clamp(0.0, 1.0)

    def close(self) -> None:
        self._model = None
        self._input_padder_cls = None
        if self._repo_context is not None and self._repo_context_entered:
            self._repo_context.__exit__(None, None, None)
        self._repo_context = None
        self._repo_context_entered = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _require_model(self) -> None:
        if self._model is None:
            raise ModelAdapterError("EMA-VFI model is not built. Call build_model/load_checkpoint first.")

    def _enter_repo_context(self) -> None:
        if self._repo_context_entered:
            return
        self._repo_context = _ema_import_context(self.repo_path)
        self._repo_context.__enter__()
        self._repo_context_entered = True


@contextlib.contextmanager
def _ema_import_context(repo_path: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    tracked_modules = {
        name: sys.modules.get(name)
        for name in ("config", "Trainer", "model", "benchmark")
        if name in sys.modules
    }
    try:
        os.chdir(repo_path)
        sys.path.insert(0, str(repo_path))
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        for name in ("config", "Trainer", "model", "benchmark"):
            if name in tracked_modules:
                sys.modules[name] = tracked_modules[name]
            else:
                sys.modules.pop(name, None)


def _configure_ema_small(config_module: Any) -> None:
    config_module.MODEL_CONFIG["LOGNAME"] = "ours_small"
    config_module.MODEL_CONFIG["MODEL_ARCH"] = config_module.init_model_config(
        F=16,
        W=7,
        depth=[2, 2, 2, 2, 2],
    )


def _normalise_ema_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise ModelAdapterError("EMA-VFI checkpoint is not a state-dict mapping")

    converted = {
        str(key).replace("module.", ""): value
        for key, value in checkpoint.items()
        if "attn_mask" not in str(key) and "HW" not in str(key)
    }
    if not converted:
        raise ModelAdapterError("EMA-VFI checkpoint did not contain loadable tensor weights")
    return converted


def _prepare_image_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"Expected image tensor shape CxHxW or BxCxHxW with 3 channels, got {tuple(tensor.shape)}")
    return tensor.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
