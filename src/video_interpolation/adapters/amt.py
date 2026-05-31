import contextlib
import gc
import importlib
import os
import sys
import typing
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import torch

from video_interpolation.adapters.base import AdapterEnvironmentReport, ModelAdapter, ModelAdapterError
from video_interpolation.settings import Settings, load_settings


@dataclass(frozen=True)
class AMTAdapterConfig:
    model_name: str = "amt_s"
    repo_name: str = "AMT"
    config_path: Path = Path("cfgs/AMT-S.yaml")
    checkpoint_path: Path = Path("AMT/amt-s.pth")
    device: str = "cuda"
    embt: float = 0.5
    scale_factor: float = 1.0
    divisor: int = 16
    strict_checkpoint: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "AMTAdapterConfig":
        values = dict(data or {})
        for key in ("config_path", "checkpoint_path"):
            if key in values:
                values[key] = Path(values[key])
        return cls(**values)


class AMTAdapter(ModelAdapter):
    """Stage 1 adapter for AMT-S eval/inference from local pretrained weights."""

    def __init__(
        self,
        config: AMTAdapterConfig | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.config = config or AMTAdapterConfig()
        self.settings = settings or load_settings()
        self.model_name = self.config.model_name
        self._model: torch.nn.Module | None = None
        self._input_padder_cls: Any | None = None
        self._repo_context: contextlib.AbstractContextManager[None] | None = None
        self._repo_context_entered = False
        self._device = torch.device(self.config.device)

    @property
    def repo_path(self) -> Path:
        return self.settings.model_repo_path(self.config.repo_name)

    @property
    def model_config_path(self) -> Path:
        if self.config.config_path.is_absolute():
            return self.config.config_path
        return self.repo_path / self.config.config_path

    @property
    def checkpoint_path(self) -> Path:
        if self.config.checkpoint_path.is_absolute():
            return self.config.checkpoint_path
        return self.settings.model_weights_root_abs / self.config.checkpoint_path

    def validate_environment(self) -> AdapterEnvironmentReport:
        report = AdapterEnvironmentReport()
        report.add("amt_repo_path", "ok" if self.repo_path.is_dir() else "failed", str(self.repo_path))
        report.add(
            "amt_config_path",
            "ok" if self.model_config_path.is_file() else "failed",
            str(self.model_config_path),
        )
        report.add(
            "amt_checkpoint_path",
            "ok" if self.checkpoint_path.is_file() else "failed",
            str(self.checkpoint_path),
        )
        report.add("torch_import", "ok", f"torch {torch.__version__}")
        if self.config.device == "cuda" and not torch.cuda.is_available():
            report.add("cuda_available", "blocked", "CUDA is unavailable for the configured AMT device.")
        else:
            report.add("cuda_available", "ok", str(torch.cuda.is_available()))
        flow_dir = self.settings.dataset_root_abs / "sources" / "vimeo_triplet" / "flow"
        flow_detail = (
            f"Found upstream-style flow directory: {flow_dir}"
            if flow_dir.is_dir()
            else f"Not required for eval/inference; missing flow directory blocks upstream AMT fine-tuning: {flow_dir}"
        )
        report.add("amt_training_flow_files", "ok", flow_detail)

        if report.status == "failed":
            return report

        try:
            with _amt_import_context(self.repo_path):
                OmegaConf.load(self.model_config_path)
                importlib.import_module("utils.build_utils")
                importlib.import_module("utils.utils")
                importlib.import_module("networks.AMT-S")
            report.add("amt_import", "ok", "Imported AMT config, build utilities, and AMT-S network")
        except Exception as exc:  # pragma: no cover - environment smoke coverage.
            report.add("amt_import", "failed", f"{type(exc).__name__}: {exc}")
        return report

    def build_model(self) -> None:
        if self._model is not None:
            return
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise ModelAdapterError("CUDA is unavailable for AMT-S with device=cuda.")
        try:
            self._enter_repo_context()
            build_utils = importlib.import_module("utils.build_utils")
            utils_module = importlib.import_module("utils.utils")
            upstream_config = OmegaConf.load(self.model_config_path)
            self._model = build_utils.build_from_cfg(upstream_config.network).to(self._device)
            self._input_padder_cls = utils_module.InputPadder
            self.eval()
        except Exception:
            self.close()
            raise

    def load_checkpoint(self, checkpoint_path: Path | None = None) -> None:
        self.build_model()
        path = checkpoint_path or self.checkpoint_path
        if not path.is_file():
            raise ModelAdapterError(f"AMT-S checkpoint does not exist: {path}")
        checkpoint = _safe_load_amt_checkpoint(path, self._device)
        state_dict = _normalise_amt_state_dict(checkpoint)
        self._model.load_state_dict(state_dict, strict=self.config.strict_checkpoint)
        self.eval()

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        self._require_model()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self._model.state_dict()}, checkpoint_path)

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
            embt = torch.tensor(
                self.config.embt,
                dtype=torch.float32,
                device=self._device,
            ).view(1, 1, 1, 1)
            output = self._model(
                left_padded,
                right_padded,
                embt,
                scale_factor=self.config.scale_factor,
                eval=True,
            )
            prediction = output["imgt_pred"]
            prediction = padder.unpad(prediction)
        return prediction[0].detach().cpu().clamp(0.0, 1.0).contiguous()

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
            raise ModelAdapterError("AMT-S model is not built. Call build_model/load_checkpoint first.")

    def _enter_repo_context(self) -> None:
        if self._repo_context_entered:
            return
        self._repo_context = _amt_import_context(self.repo_path)
        self._repo_context.__enter__()
        self._repo_context_entered = True


@contextlib.contextmanager
def _amt_import_context(repo_path: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    tracked_prefixes = ("networks", "utils", "datasets", "trainers", "losses", "metrics")
    tracked_modules = {
        name: sys.modules.get(name)
        for name in list(sys.modules)
        if name in tracked_prefixes or name.startswith(tuple(f"{prefix}." for prefix in tracked_prefixes))
    }
    try:
        os.chdir(repo_path)
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


def _safe_load_amt_checkpoint(path: Path, device: torch.device) -> Any:
    try:
        with torch.serialization.safe_globals([typing.OrderedDict, OrderedDict]):
            return torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise ModelAdapterError(
            "AMT-S checkpoint could not be loaded with PyTorch safe weights loading. "
            "Do not disable safe loading unless you explicitly trust and approve this checkpoint."
        ) from exc


def _normalise_amt_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise ModelAdapterError("AMT-S checkpoint is not a state-dict mapping")
    converted = {
        str(key).replace("module.", ""): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }
    if not converted:
        raise ModelAdapterError("AMT-S checkpoint did not contain loadable tensor weights")
    return converted


def _prepare_image_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"Expected image tensor shape CxHxW or BxCxHxW with 3 channels, got {tuple(tensor.shape)}")
    return tensor.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
