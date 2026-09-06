from types import SimpleNamespace

import numpy as np
from aerosandbox.dynamics.flight_dynamics.airplane import get_modes
from aerosandbox.weights.mass_properties import MassProperties

from src.aero.custom_classes import CruiseCondition, StabilityResult
from src.aero.static_derivatives import calibrated_static_derivatives
from src.aero.utils import dict_to_mode_result, require_scalar
from src.vectors import DesignVector


def estimate_stability_derivatives(
    design_vector: DesignVector,
    cruise_condition: CruiseCondition,
    mass_props: MassProperties,
) -> dict[str, float]:
    """Return the coarse aerodynamic derivative estimates."""
    if not cruise_condition.converged:
        raise ValueError("Cruise condition must be converged.")

    velocity = require_scalar(cruise_condition.operating_point.velocity)
    if velocity <= 0:
        raise ValueError("Cruise velocity must be positive.")

    s, c, b = design_vector.wing_area, design_vector.wing_chord, design_vector.wing_span
    st, sv = design_vector.hstab_area, design_vector.vstab_area
    xcg = require_scalar(mass_props.x_cg)
    xt = design_vector.tail_arm + 0.25 * design_vector.hstab_chord
    lt = xt - xcg

    # Linear finite-wing lift; assumes 15% downwash and 90% tail efficiency.
    aw = 2 * np.pi / (1 + 2 / (b**2 / s))
    at = 2 * np.pi / (1 + 2 / (design_vector.hstab_span**2 / st))
    av = 2 * np.pi / (1 + 2 / (design_vector.vstab_span**2 / sv))
    eta = 0.90
    cmq_est = -1.6 * eta * at * (st * lt / (s * c)) * lt / c
    cmq = 3.47831 + 1.98806 * cmq_est  # Calibrated linear pitch derivatives.

    # Trim lift and parabolic drag estimates.
    q = require_scalar(cruise_condition.operating_point.dynamic_pressure())
    cl = require_scalar(mass_props.mass) * 9.806 / (q * s)
    cd = 0.02 + cl**2 / (np.pi * 0.85 * b**2 / s)

    # Calibrated tail/body derivatives; assumes small sideslip and yaw rate.
    alpha = require_scalar(cruise_condition.operating_point.alpha)
    cla, cma, cnb = calibrated_static_derivatives(
        design_vector, alpha, require_scalar(cruise_condition.elevator_deflection), xcg,
    )
    cyb_est = -eta * av * sv / s
    cyr_est = -2 * cyb_est * lt / b
    clb_est = -cyb_est * (0.5 * design_vector.vstab_span - require_scalar(mass_props.z_cg)) / b
    cnr_est = 2 * cyb_est * (lt / b) ** 2
    cyb = -0.0434608 + 0.61645 * cyb_est
    cyr = (0.0662136 + 0.708383 * cyr_est + 0.139102 * cyb_est
           - 0.00943366 * b - 0.0723635 * c + 0.00459817 * alpha)
    clb = 0.0147862 + 0.122881 * cyb_est + 0.228923 * clb_est
    cnr = 0.0165917 + 1.17482 * cnr_est

    # Rectangular-wing damping with calibrated buildup bias.
    clp = 0.263378 + 1.46526 * (-aw / 8)
    clr = 0.0341476 + 0.619521 * (cl / 4)

    return {
        "CL": cl, "CD": cd, "Cma": cma, "Cmq": cmq,
        "CYb": cyb, "CYr": cyr, "Clb": clb, "Clp": clp,
        "Clr": clr, "Cnb": cnb, "Cnr": cnr, "CLa": cla,
        "x_np": xcg - cma / cla * c,
    }


def stability_analysis_coarse(
    design_vector: DesignVector,
    cruise_condition: CruiseCondition,
    mass_props: MassProperties,
) -> StabilityResult:
    """Estimate stability from conceptual-design formulas."""
    aero = estimate_stability_derivatives(design_vector, cruise_condition, mass_props)
    s, c, b = design_vector.wing_area, design_vector.wing_chord, design_vector.wing_span
    airplane = SimpleNamespace(s_ref=s, c_ref=c, b_ref=b)
    modes = get_modes(airplane, cruise_condition.operating_point, mass_props, aero)

    return StabilityResult(
        phugoid=dict_to_mode_result(modes["phugoid"]),
        short_period=dict_to_mode_result(modes["short_period"]),
        dutch_roll=dict_to_mode_result(modes["dutch_roll"]),
        spiral=dict_to_mode_result(modes["spiral"]),
        roll_subsidence=dict_to_mode_result(modes["roll_subsidence"]),
        Cma=float(aero["Cma"]),
        Cnb=float(aero["Cnb"]),
        static_margin=float(-aero["Cma"] / aero["CLa"]),
    )
