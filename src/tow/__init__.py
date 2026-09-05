"""Towed-sensor dynamics and sizing-load envelopes.

Import envelope helpers from :mod:`src.tow.envelope`. Keeping this package
initializer lightweight also allows ``python -m src.tow.envelope`` to run
without importing the CLI module twice.
"""

from src.tow.model import TowConfig, TowSimulationResult, simulate_tow

__all__ = [
    "TowConfig",
    "TowSimulationResult",
    "simulate_tow",
]
