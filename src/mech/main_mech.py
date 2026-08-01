"""Public entry points for the discrete mechanical module.

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
    disp_res: bool = False,
) -> MechanicalResult:
    """Evaluate mission properties and optionally save M2/M3 placement results."""

    result = evaluate_mechanical_design(design_vector, config, parameter_vector)
    if disp_res:
        # Keep plotting dependencies and file I/O out of normal optimization runs.
        from src.mech.result_display import write_mass_placement_results

        write_mass_placement_results(result)
    return result


def mech_main(
    design_vector: DesignVector,
    mission: str = "M1",
    config: MechanicalModuleConfig | None = None,
    parameter_vector: ParameterVector | None = None,
    disp_res: bool = False,
) -> tuple[tuple[float, float, float], np.ndarray, float]:
    """Return ``(CG, inertia tensor, weight)`` for one mission."""

    result = evaluate_mechanical_module(
        design_vector,
        config,
        parameter_vector,
        disp_res=disp_res,
    )
    mission_result = result.for_mission(mission)
    return (
        mission_result.cg_m,
        mission_result.inertia_tensor_kg_m2.copy(),
        mission_result.weight_n,
    )


__all__ = ["evaluate_mechanical_module", "mech_main"]
