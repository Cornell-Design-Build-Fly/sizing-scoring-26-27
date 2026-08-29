"""Airfoil geometry and optimization tools."""

from src.opt.airfoil.geometry import (
    AirfoilKulfanParameters,
    make_airfoil_airplane,
)

__all__ = [
    "AirfoilKulfanParameters",
    "AirfoilOptimizationConfig",
    "AirfoilOptimizationResult",
    "make_airfoil_airplane",
    "run_airfoil_optimization",
]


def __getattr__(name: str):
    if name in {
        "AirfoilOptimizationConfig",
        "AirfoilOptimizationResult",
        "run_airfoil_optimization",
    }:
        from src.opt.airfoil import optimize

        return getattr(optimize, name)
    raise AttributeError(name)
