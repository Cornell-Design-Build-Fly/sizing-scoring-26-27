"""Public entry points for the mechanical module.

The implementation is intentionally kept out of this module so the main path
reads as a small caller program: evaluate the complete design, then select the
requested mission output when using the legacy adapter.
"""

from __future__ import annotations

import numpy as np

from src.mech.mechanical_evaluation import evaluate_mechanical_design
from src.mech.models import MechanicalModuleConfig, MechanicalResult
from src.vectors import DesignVector, ParameterVector


def evaluate_mechanical_module(
    design_vector: DesignVector,
    config: MechanicalModuleConfig | None = None,
    parameter_vector: ParameterVector | None = None,
) -> MechanicalResult:
    """Evaluate all mission mass properties for one design/parameter vector."""

    return evaluate_mechanical_design(design_vector, config, parameter_vector)


def mech_main(
    design_vector: DesignVector,
    mission: str = "M1",
    config: MechanicalModuleConfig | None = None,
    parameter_vector: ParameterVector | None = None,
) -> tuple[tuple[float, float, float], np.ndarray, float]:
    """Return ``(CG, inertia tensor, weight)`` for one mission."""

    result = evaluate_mechanical_module(design_vector, config, parameter_vector)
    mission_result = result.for_mission(mission)
    return (
        tuple(float(value) for value in mission_result.cg_m),
        mission_result.inertia_tensor_kg_m2.copy(),
        mission_result.weight_n,
    )


__all__ = ["evaluate_mechanical_module", "mech_main"]
