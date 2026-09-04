import aerosandbox as asb
import numpy as np
from src.aero.custom_classes import CruiseCondition, StabilityResult
from src.aero.utils import dict_to_mode_result, require_scalar
from src.aero.stability_criteria import (
    lateral_modes,
    lateral_state_matrix,
    time_to_double_s,
)
from src.vectors import ASBDesignVector, DesignVector
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
    "x_np",
}

def stability_analysis(
        design_vector: DesignVector,
        cruise_condition: CruiseCondition,
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
    # Define an Airplane object from design vector.
    airplane = ASBDesignVector.from_design_vector(
    design_vector
    ).make_airplane(
        elevator_deflection=cruise_condition.elevator_deflection,
        tail_incidence=cruise_condition.tail_incidence,
    )

    # Run AeroBuildup to get stability derivatives
    stability_dict = asb.AeroBuildup(
        airplane=airplane,
        op_point=cruise_condition.operating_point,
        xyz_ref=(
            require_scalar(mass_props.x_cg),
            require_scalar(mass_props.y_cg),
            require_scalar(mass_props.z_cg),
        ),
        include_wave_drag=False,
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

    # Calculate static margin
    x_np = require_scalar(stability_dict["x_np"]) # neutral point
    x_cg = require_scalar(mass_props.x_cg) # cg
    c_ref = require_scalar(airplane.c_ref) 

    static_margin = (x_np - x_cg) / c_ref

    _lateral = lateral_modes(
        lateral_state_matrix(
            CYb=require_scalar(stability_dict["CYb"]),
            CYr=require_scalar(stability_dict["CYr"]),
            Clb=require_scalar(stability_dict["Clb"]),
            Clp=require_scalar(stability_dict["Clp"]),
            Clr=require_scalar(stability_dict["Clr"]),
            Cnb=require_scalar(stability_dict["Cnb"]),
            Cnr=require_scalar(stability_dict["Cnr"]),
            CL=require_scalar(stability_dict["CL"]),
            dynamic_pressure_pa=require_scalar(
                cruise_condition.operating_point.dynamic_pressure()
            ),
            wing_area_m2=require_scalar(airplane.s_ref),
            wing_span_m=require_scalar(airplane.b_ref),
            mass_kg=require_scalar(mass_props.mass),
            Ixx=require_scalar(mass_props.Ixx),
            Izz=require_scalar(mass_props.Izz),
            velocity_m_s=require_scalar(cruise_condition.operating_point.velocity),
            gravity_m_s2=9.806,
        )
    )
    stability_modes["spiral"] = {
        "eigenvalue_real": _lateral.spiral_eigenvalue_real,
        "eigenvalue_imag": 0.0,
        "damping_ratio": -np.sign(_lateral.spiral_eigenvalue_real),
    }

    return StabilityResult(
        phugoid=dict_to_mode_result(stability_modes["phugoid"]),
        short_period=dict_to_mode_result(stability_modes["short_period"]),
        dutch_roll=dict_to_mode_result(stability_modes["dutch_roll"]),
        spiral=dict_to_mode_result(stability_modes["spiral"]),
        roll_subsidence=dict_to_mode_result(stability_modes["roll_subsidence"]),
        Cma=require_scalar(stability_dict["Cma"]),
        Cnb=require_scalar(stability_dict["Cnb"]),
        static_margin=static_margin,
        neutral_point_x_m=x_np,
        spiral_time_to_double_s=time_to_double_s(_lateral.spiral_eigenvalue_real),
    )
