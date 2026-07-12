import aerosandbox as asb
import numpy as np
from aerosandbox import OperatingPoint
from aerosandbox import optimization as opti 
from src.aero.custom_classes import CruiseCondition, StabilityResult, AirplaneAnalysisResult
from src.vectors import DesignVector
from aerosandbox.dynamics.flight_dynamics.airplane import get_modes
from aerosandbox.weights.mass_properties import MassProperties

_REQUIRED_MODE_AERO_KEYS = {
    "CL",
    "CD",
    "Cma",
    "Cmq",
    "CYb",
    "CYr",
    "Clb",
    "Clp",
    "Clr",
    "Cnb",
    "Cnr",
}

def stability_analysis(
        design_vector: DesignVector,
        airplane: asb.Airplane,
        cruise_condition: CruiseCondition,
        aero_result: AirplaneAnalysisResult,
        mass_props: MassProperties,
) -> StabilityResult:
    """
    Perform stability analysis for a given design vector, cruise condition, and aerodynamic result.

    Args:
        design_vector: The design vector representing the airplane configuration.
        airplane: The Aerosandbox Airplane object created from the design vector.
        cruise_condition: The cruise condition containing the operating point and throttle setting.
        aero_result: The aerodynamic result obtained from aero_analysis.
        mass_props: The mass properties of the airplane.
    """
    # Run AeroBuildup to get stability derivatives
    stability_dict = asb.AeroBuildup(
        airplane=airplane,
        op_point=cruise_condition.operating_point,
    ).run_with_stability_derivatives()

    # Handle missing keys
    missing_keys = _REQUIRED_MODE_AERO_KEYS - stability_dict.keys()
    if missing_keys:
        raise ValueError(f"Missing required stability derivatives: {missing_keys}")

    # Dynamic stability modes
    stability_modes = get_modes(
        airplane=airplane,
        op_point=cruise_condition.operating_point,
        mass_props=mass_props,
        aero=stability_dict
    )

    return StabilityResult(
        phugoid=stability_modes["phugoid"],
        short_period=stability_modes["short_period"],
        dutch_roll=stability_modes["dutch_roll"],
        spiral=stability_modes["spiral"],
        roll_subsidence=stability_modes["roll_subsidence"],
    )
