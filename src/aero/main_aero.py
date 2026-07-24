from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import aerosandbox as asb
import numpy as np

from src.aero.custom_classes import AeroOutput, StabilityResult, CruiseCondition, AirplaneAnalysisResult
from src.vectors import ASBDesignVector, DesignVector, ParameterVector
from src.aero.vlm import require_scalar
from src.aero.cruise_analysis import cruise_analysis
from src.aero.aero_analysis import aero_analysis
from src.aero.stability_analysis import stability_analysis
from src.aero.aero_score import AeroScore, aero_score

def aero_main(
        design_vector: DesignVector,
        parameter_vector: ParameterVector,
        thrust_velocity: tuple[float, float, float],
        cg: tuple[float, float, float],
        inertia_matrix: np.ndarray,
        mass: float,
) -> AeroScore:

    """
    Main function for aero analysis of a design vector.

    Args:
        design_vector: The design vector representing the airplane configuration.
        thrust_velocity: Thust vs velocity graph data determined in prop module.
        cg: The center of gravity of the airplane (x, y, z).
        inertia_matrix: The inertia matrix of the airplane.
        mass: The mass of the airplane.
    """

    # Define "mass properties" object for stability analysis.
    mass_props = asb.MassProperties(
    mass=mass,
    x_cg=cg[0],
    y_cg=cg[1],
    z_cg=cg[2],
    Ixx=inertia_matrix[0, 0],
    Iyy=inertia_matrix[1, 1],
    Izz=inertia_matrix[2, 2],
    Ixy=inertia_matrix[0, 1],
    Iyz=inertia_matrix[1, 2],
    Ixz=inertia_matrix[0, 2],
    )

    # Main trim solver. Contains ASB optimization methods and calls to aero_analysis to perform force/moment balance.
    cruise_condition = cruise_analysis(design_vector, parameter_vector, thrust_velocity, cg, mass)

    # If cruise condition doesn't converge for this design, exit early with flagged AeroScore result.
    if not cruise_condition.converged:
        return AeroScore(
            can_fly = False,
        )

    # Final call to stability_analysis to get final stability results for design vector at trim.
    stability_result = stability_analysis(design_vector, cruise_condition, mass_props)

    # Return final score for design vector based on cruise speed, stall speed, and stability numbers.
    return aero_score(cruise_condition, stability_result, parameter_vector)
