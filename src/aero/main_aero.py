from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import aerosandbox as asb
import numpy as np

from src.aero.custom_classes import AeroOutput, StabilityResult, CruiseCondition, AirplaneAnalysisResult
from src.vectors import ASBDesignVector, DesignVector
from src.aero.vlm import require_scalar
from src.aero import cruise_analysis, stability_analysis, aero_analysis

def aero_main(
        design_vector: DesignVector,
        thrust_velocity: np.ndarray, # array containing a, b, c coefficients of parabola for curve. for now assume throttled thrust curve only
        cg: tuple[float, float, float],
        inertia_matrix: np.ndarray, # not sure what data type will be
        mass: float,
        sm: float,
) -> AeroOutput:

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

    # Define an Airplane object from design vector.
    airplane = design_vector.to_asb_airplane()

    # Main trim solver. Contains ASB optimization methods and calls to aero_analysis to perform force/moment balance.
    cruise_condition = cruise_analysis(design_vector, thrust_velocity, cg, mass)

    # Final call to aero_analysis to get final aerodynamic results for design vector at trim. 
    aero_result = aero_analysis(design_vector, cruise_condition, cg)

    # Final call to stability_analysis to get final stability results for design vector at trim.
    stability_result = stability_analysis(airplane, cruise_condition, mass_props)

    # Return aero, cruise, and stability results in "AeroOutput" object.
    result = AeroOutput(
        aero_result=aero_result,
        cruise_condition=cruise_condition,
        stability_result=stability_result
    )
    return result