"""Model adapter implementations for Stage 1."""

from .base import AdapterEnvironmentReport, ModelAdapter, ModelAdapterError
from .amt import AMTAdapter, AMTAdapterConfig
from .baseline import BaselineAdapter, BaselineAdapterConfig
from .ema_vfi import EMAVFIAdapter, EMAVFIAdapterConfig
from .rife import PracticalRIFEAdapter, PracticalRIFEAdapterConfig

__all__ = [
    "AMTAdapter",
    "AMTAdapterConfig",
    "AdapterEnvironmentReport",
    "BaselineAdapter",
    "BaselineAdapterConfig",
    "EMAVFIAdapter",
    "EMAVFIAdapterConfig",
    "ModelAdapter",
    "ModelAdapterError",
    "PracticalRIFEAdapter",
    "PracticalRIFEAdapterConfig",
]
