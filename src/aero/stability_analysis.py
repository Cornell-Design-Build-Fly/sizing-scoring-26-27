import aerosandbox as asb
import numpy as np
from aerosandbox import OperatingPoint
from aerosandbox import optimization as opti 
from src.aero.custom_classes import CruiseCondition, StabilityResult, AirplaneAnalysisResult
from src.vectors import DesignVector
from aerosandbox.dynamics.flight_dynamics.airplane import get_modes
from aerosandbox.weights.mass_properties import MassProperties

def stability_analysis(
        design_vector: DesignVector,
        cruise_condition: CruiseCondition,
        aero_result: AirplaneAnalysisResult,
        inertia_matrix: np.ndarray, 
        cg: tuple[float, float, float],
        mass: float,
) -> StabilityResult:
    """
    Perform stability analysis for a given design vector, cruise condition, and aerodynamic result.

    Args:
        design_vector: The design vector representing the airplane configuration.
        cruise_condition: The cruise condition containing the operating point and throttle setting.
        aero_result: The aerodynamic result from the aero_analysis function.
        inertia_matrix: The inertia matrix of the airplane.
    """
    
    # Placeholders for stability analysis implementation...
    
    
    # Static stability

    # Dynamic stability

    mass_props = MassProperties(
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


    stability_modes = get_modes(
        airplane=design_vector.to_asb_airplane(),
        op_point=cruise_condition.operating_point,
        mass_props = mass_props,
    )

    


    return StabilityResult(
        phugoid=stability_modes["phugoid"],
        short_period=stability_modes["short_period"],
        dutch_roll=stability_modes["dutch_roll"],
        spiral=stability_modes["spiral"],
        roll_subsidence=stability_modes["roll_subsidence"],
    )
