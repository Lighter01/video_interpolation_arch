import gc
import traceback
from typing import Any

import torch

from video_interpolation.adapters.rife import PracticalRIFEAdapter, PracticalRIFEAdapterConfig
from video_interpolation.ema_preflight import PreflightReport
from video_interpolation.settings import Settings, load_settings


def run_practical_rife_preflight(
    config: PracticalRIFEAdapterConfig | None = None,
    settings: Settings | None = None,
) -> PreflightReport:
    """Check whether Practical-RIFE can be imported, initialized, and loaded."""
    settings = settings or load_settings()
    adapter = PracticalRIFEAdapter(config or PracticalRIFEAdapterConfig(), settings=settings)
    report = PreflightReport(status="ok")
    model_built = False

    try:
        env_report = adapter.validate_environment()
        for check in env_report.checks:
            report.add(check.name, check.status, check.detail)
        if env_report.status != "ok":
            return report

        adapter.build_model()
        model_built = True
        report.add("rife_initialize", "ok", "Initialized Practical-RIFE model")
        adapter.load_checkpoint()
        report.add("rife_checkpoint_load", "ok", "Loaded Practical-RIFE checkpoint with safe weights loading")
    except Exception as exc:  # pragma: no cover - environment smoke coverage.
        report.add(
            "rife_preflight_exception",
            "failed",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
        )
    finally:
        adapter.close()
        _release_model_preflight_resources(torch)

    if not model_built and report.status == "ok":
        report.add("rife_initialize", "failed", "Practical-RIFE model initialization did not complete")
    return report


def _release_model_preflight_resources(torch_module: Any) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
