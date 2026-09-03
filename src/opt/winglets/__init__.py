"""Winglet geometry and optimization tools."""

from src.opt.winglets.geometry import (
    WINGLET_VARIABLES,
    WingletGeometry,
    add_winglets_to_design_vector_airplane,
    make_winglet_airplane,
)

__all__ = [
    "WINGLET_VARIABLES",
    "WingletGeometry",
    "WingletOptimizationConfig",
    "WingletOptimizationResult",
    "add_winglets_to_design_vector_airplane",
    "make_winglet_airplane",
    "run_winglet_optimization",
]


def __getattr__(name: str):
    if name in {
        "WingletOptimizationConfig",
        "WingletOptimizationResult",
        "run_winglet_optimization",
    }:
        from src.opt.winglets import optimize

        return getattr(optimize, name)
    raise AttributeError(name)
