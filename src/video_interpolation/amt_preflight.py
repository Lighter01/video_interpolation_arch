import gc
import traceback
from typing import Any

import torch

from video_interpolation.adapters.amt import AMTAdapter, AMTAdapterConfig
from video_interpolation.ema_preflight import PreflightReport
from video_interpolation.settings import Settings, load_settings


def run_amt_s_preflight(
    config: AMTAdapterConfig | None = None,
    settings: Settings | None = None,
) -> PreflightReport:
    """Check whether AMT-S can be imported, initialized, and loaded."""
    settings = settings or load_settings()
    adapter = AMTAdapter(config or AMTAdapterConfig(), settings=settings)
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
        report.add("amt_initialize", "ok", "Initialized AMT-S model")
        adapter.load_checkpoint()
        report.add("amt_checkpoint_load", "ok", "Loaded AMT-S checkpoint with safe weights loading")
    except Exception as exc:  # pragma: no cover - environment smoke coverage.
        report.add(
            "amt_preflight_exception",
            "failed",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
        )
    finally:
        adapter.close()
        _release_model_preflight_resources(torch)

    if not model_built and report.status == "ok":
        report.add("amt_initialize", "failed", "AMT-S model initialization did not complete")
    return report


def _release_model_preflight_resources(torch_module: Any) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
