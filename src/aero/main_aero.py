from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import aerosandbox as asb
import numpy as np

from src.aero.custom_classes import StabilityResult, CruiseCondition
from src.vectors import ASBDesignVector, DesignVector
from src.aero.vlm import AirplaneAnalysisResult, require_scalar
from src.aero import cruise_analysis, stability_analysis, aero_analysis
# add import statement for to-be-defined CruiseCondition and StabilityResult classes

def aero_main(
        design_vector: DesignVector,
        thrust_velocity: np.ndarray, # not sure what data type will be
        cg: tuple[float, float, float],
        inertia_matrix: np.ndarray, # not sure what data type will be
        weight: float,
) -> tuple[AirplaneAnalysisResult, CruiseCondition, StabilityResult]:

    """
    Main function for aero analysis of a design vector.

    Args:
        design_vector: The design vector representing the airplane configuration.
        thrust_velocity: Thust vs velocity graph data determined in prop module.
        cg: The center of gravity of the airplane (x, y, z).
        inertia_matrix: The inertia matrix of the airplane.
        weight: The weight of the airplane.
    """

    # Main trim solver. Contains ASB optimization methods and calls to aero_analysis to perform force/moment balance.
    cruise_condition = cruise_analysis(design_vector, thrust_velocity, cg, weight)

    # Final call to aero_analysis to get final aerodynamic results for design vector at trim. 
    aero_result = aero_analysis(design_vector, cruise_condition)

    # Final call to stability_analysis to get final stability results for design vector at trim.
    stability_result = stability_analysis(design_vector, cruise_condition, aero_result, inertia_matrix)

    # Return aero, cruise, and stability results. (Might be useful to make a dataclass to contain these three).
    return [aero_result, cruise_condition, stability_result] 