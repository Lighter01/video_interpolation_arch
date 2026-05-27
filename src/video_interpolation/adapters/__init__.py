"""Model adapter implementations for Stage 1."""

from .base import AdapterEnvironmentReport, ModelAdapter, ModelAdapterError
from .ema_vfi import EMAVFIAdapter, EMAVFIAdapterConfig

__all__ = [
    "AdapterEnvironmentReport",
    "EMAVFIAdapter",
    "EMAVFIAdapterConfig",
    "ModelAdapter",
    "ModelAdapterError",
]
