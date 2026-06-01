import gc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from video_interpolation.adapters.base import AdapterEnvironmentReport, ModelAdapter, ModelAdapterError
from video_interpolation.inference_runtime.api import FramePairRequest, FramePairResult, InferenceMode
from video_interpolation.inference_runtime.rife import (
    PracticalRIFEPyTorchRuntime,
    PracticalRIFEPyTorchRuntimeConfig,
    validate_rife_scale,
)
from video_interpolation.inference_runtime.rife_upstream import Model as RIFEModel
from video_interpolation.settings import Settings, load_settings


@dataclass(frozen=True)
class PracticalRIFEAdapterConfig:
    model_name: str = "practical_rife_v4_26"
    repo_name: str = "Practical-RIFE"
    checkpoint_path: Path = Path("Practical-RIFE/RIFEv4.26/train_log")
    supported_modes: tuple[str, ...] = ("fixed_2x", "arbitrary_nx")
    default_interpolation_factor: int = 2
    min_interpolation_factor: int = 2
    max_interpolation_factor: int = 8
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
        if "supported_modes" in values:
            values["supported_modes"] = tuple(str(mode) for mode in values["supported_modes"])
        return cls(**values)


class PracticalRIFEAdapter(ModelAdapter):
    """Adapter for Practical-RIFE eval/inference from local weights."""

    def __init__(
        self,
        config: PracticalRIFEAdapterConfig | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.config = config or PracticalRIFEAdapterConfig()
        self.settings = settings or load_settings()
        self.model_name = self.config.model_name
        self._model: Any | None = None
        self._runtime: PracticalRIFEPyTorchRuntime | None = None
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
        report.add("rife_runtime_source", "ok", "video_interpolation.inference_runtime.rife_upstream")
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
            RIFEModel(device="cpu")
            report.add("rife_import", "ok", "Imported project-owned Practical-RIFE runtime source")
        except Exception as exc:  # pragma: no cover - environment smoke coverage.
            report.add("rife_import", "failed", f"{type(exc).__name__}: {exc}")
        return report

    def build_model(self) -> None:
        if self._model is not None:
            return
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise ModelAdapterError("CUDA is unavailable for Practical-RIFE with device=cuda.")
        try:
            self._model = RIFEModel(device=self._device)
            self._runtime = None
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
                    supported_modes=self.config.supported_modes,
                    default_interpolation_factor=self.config.default_interpolation_factor,
                    min_interpolation_factor=self.config.min_interpolation_factor,
                    max_interpolation_factor=self.config.max_interpolation_factor,
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
        self._runtime = None
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
        request = FramePairRequest(left=left, right=right)
        return self.predict_frame_pair(request).middle_frame

    def predict_intermediate_frames(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        interpolation_factor: int | None = None,
        scale: float | None = None,
        timesteps: Sequence[float] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        backend_options = {}
        if scale is not None:
            backend_options["scale"] = validate_rife_scale(scale)
        request = FramePairRequest(
            left=left,
            right=right,
            mode=InferenceMode.ARBITRARY_NX,
            interpolation_factor=(
                self.config.default_interpolation_factor if interpolation_factor is None else interpolation_factor
            ),
            timesteps=timesteps,
            backend_options=backend_options,
        )
        return self.predict_frame_pair(request).intermediate_frames

    def predict_frame_pair(self, request: FramePairRequest) -> FramePairResult:
        self._require_model()
        self.eval()
        return self._require_runtime().predict(request)

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
        self._runtime = None
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _require_model(self) -> None:
        if self._model is None:
            raise ModelAdapterError("Practical-RIFE model is not built. Call build_model/load_checkpoint first.")

    def _require_runtime(self) -> PracticalRIFEPyTorchRuntime:
        self._require_model()
        if self._runtime is None:
            self._runtime = PracticalRIFEPyTorchRuntime(
                self._model,
                PracticalRIFEPyTorchRuntimeConfig(
                    model_name=self.model_name,
                    device=self._device,
                    timestep=self.config.timestep,
                    scale=self.config.scale,
                    divisor=self.config.divisor,
                ),
            )
            self._runtime.load()
        return self._runtime


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
