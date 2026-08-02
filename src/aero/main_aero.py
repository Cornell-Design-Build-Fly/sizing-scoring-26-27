from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import aerosandbox as asb
import numpy as np

from src.aero.custom_classes import AeroOutput, StabilityResult, CruiseCondition, AirplaneAnalysisResult
from src.vectors import ASBDesignVector, DesignVector, ParameterVector
from src.aero.utils import require_scalar
from src.aero.cruise_analysis import cruise_analysis
from src.aero.cruise_analysis_coarse import cruise_analysis_coarse
from src.aero.aero_analysis import aero_analysis
from src.aero.stability_analysis_coarse import stability_analysis_coarse
from src.aero.stability_analysis import stability_analysis
from src.aero.aero_score import AeroScore, aero_score
from src.aero.plot_aero_result import plot_aero_result

def aero_main(
        design_vector: DesignVector,
        parameter_vector: ParameterVector,
        thrust_velocity: tuple[float, float, float],
        mission: int,
        cg: tuple[float, float, float],
        inertia_matrix: np.ndarray,
        mass: float,
        disp_res: bool = False,
) -> AeroScore:

    """
    Main function for aero analysis of a design vector.

    Args:
        design_vector: The design vector representing the airplane configuration.
        thrust_velocity: Thust vs velocity graph data determined in prop module.
        mission: Mission number being evaluated (1, 2, or 3).
        cg: The center of gravity of the airplane (x, y, z).
        inertia_matrix: The inertia matrix of the airplane.
        mass: The mass of the airplane.
    """

    analysis_start = perf_counter()
    print(
        f"[aero] Starting Mission {mission} aerodynamic evaluation...",
        flush=True,
    )

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

    # Cruise model selection: leave exactly one call active.
    # cruise_condition = cruise_analysis(
    #     design_vector, parameter_vector, thrust_velocity, cg, mass, mission
    # )
    cruise_condition = cruise_analysis_coarse(
        design_vector, parameter_vector, thrust_velocity, cg, mass, mission
    )
    print(
        f"[aero] Cruise analysis complete (converged={cruise_condition.converged}).",
        flush=True,
    )

    # If cruise condition doesn't converge for this design, exit early with flagged AeroScore result.
    if not cruise_condition.converged:
        print("[aero] Stopping evaluation because cruise trim did not converge.", flush=True)
        return AeroScore(
            can_fly = False,
        )

    # Stability model selection: leave exactly one call active.
    print("[aero] Running static and dynamic stability analysis...", flush=True)
    # stability_result = stability_analysis(design_vector, cruise_condition, mass_props)
    stability_result = stability_analysis_coarse(design_vector, cruise_condition, mass_props)

    # Return final score for design vector based on cruise speed, stall speed, and stability numbers.
    score = aero_score(cruise_condition, stability_result, parameter_vector)
    print(
        f"[aero] Aerodynamic evaluation finished in "
        f"{perf_counter() - analysis_start:.2f} s "
        f"(can_fly={score.can_fly}, lap_time={score.lap_time:.2f} s, "
        f"penalty={score.penalty:.2f}).",
        flush=True,
    )

    if disp_res:
        plot_aero_result(
            design_vector,
            cruise_condition,
            thrust_velocity,
            cg,
            parameter_vector,
            mass,
            31,
            mission,
        )

    return score
