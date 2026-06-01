import contextlib
import gc
import importlib
import os
import sys
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .settings import Settings, load_settings


@dataclass
class PreflightCheck:
    name: str
    status: str
    detail: str


@dataclass
class PreflightReport:
    status: str
    checks: list[PreflightCheck] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(PreflightCheck(name=name, status=status, detail=detail))
        if status == "failed":
            self.status = "failed"
        elif status == "blocked" and self.status == "ok":
            self.status = "blocked"


@contextlib.contextmanager
def _ema_import_context(repo_path: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    tracked_modules = {
        name: sys.modules.get(name)
        for name in ("config", "Trainer", "model")
        if name in sys.modules
    }
    try:
        os.chdir(repo_path)
        sys.path.insert(0, str(repo_path))
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        for name in ("config", "Trainer", "model"):
            if name in tracked_modules:
                sys.modules[name] = tracked_modules[name]
            else:
                sys.modules.pop(name, None)


def _configure_ema_small(config_module: Any) -> None:
    config_module.MODEL_CONFIG["LOGNAME"] = "ours_small_t"
    config_module.MODEL_CONFIG["MODEL_ARCH"] = config_module.init_model_config(
        F=16,
        W=7,
        depth=[2, 2, 2, 2, 2],
    )


def release_model_preflight_resources(torch_module: Any | None) -> None:
    """Release process-local resources after model preflight checks."""
    gc.collect()
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def run_ema_vfi_small_preflight(settings: Settings | None = None) -> PreflightReport:
    settings = settings or load_settings()
    report = PreflightReport(status="ok")

    repo_path = settings.model_repo_path("EMA-VFI")
    checkpoint_path = settings.model_weight_path("EMA-VFI", "ours_small_t.pkl")

    if repo_path.is_dir():
        report.add("ema_repo_path", "ok", str(repo_path))
    else:
        report.add("ema_repo_path", "failed", f"Missing EMA-VFI repo: {repo_path}")
        return report

    if checkpoint_path.is_file():
        report.add("ema_checkpoint_path", "ok", str(checkpoint_path))
    else:
        report.add("ema_checkpoint_path", "failed", f"Missing checkpoint: {checkpoint_path}")
        return report

    torch_module: Any | None = None
    model: Any | None = None
    checkpoint: Any | None = None
    converted: Any | None = None

    try:
        import torch

        torch_module = torch
    except Exception as exc:  # pragma: no cover - exercised by environment smoke checks.
        report.add("torch_import", "failed", repr(exc))
        return report

    report.add("torch_import", "ok", f"torch {torch_module.__version__}")

    try:
        with _ema_import_context(repo_path):
            config_module = importlib.import_module("config")
            _configure_ema_small(config_module)
            trainer_module = importlib.import_module("Trainer")
            report.add("ema_import", "ok", "Imported EMA-VFI config and Trainer")

            if not torch_module.cuda.is_available():
                report.add(
                    "ema_initialize",
                    "blocked",
                    "CUDA is unavailable and the default EMA-VFI preflight initializes the cuda checkpoint path.",
                )
                return report

            model = trainer_module.Model(-1)
            report.add("ema_initialize", "ok", "Initialized EMA-VFI-small model")

            checkpoint = torch_module.load(checkpoint_path, map_location="cuda")
            converted = {
                key.replace("module.", ""): value
                for key, value in checkpoint.items()
                if "module." in key and "attn_mask" not in key and "HW" not in key
            }
            model.net.load_state_dict(converted)
            report.add("ema_checkpoint_load", "ok", "Loaded EMA-VFI-small checkpoint")
    except Exception as exc:
        report.add(
            "ema_preflight_exception",
            "failed",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
        )
    finally:
        model = None
        checkpoint = None
        converted = None
        release_model_preflight_resources(torch_module)

    return report
