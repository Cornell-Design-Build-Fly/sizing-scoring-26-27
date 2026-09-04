from types import SimpleNamespace

import numpy as np
from aerosandbox.dynamics.flight_dynamics.airplane import get_modes
from aerosandbox.weights.mass_properties import MassProperties

from src.aero.custom_classes import CruiseCondition, StabilityResult
from src.aero.stability_criteria import (
    lateral_modes,
    lateral_state_matrix,
    time_to_double_s,
)
from src.aero.utils import dict_to_mode_result, require_scalar
# Single source of truth for the neutral point. The mechanical module places the
# CG against this same estimate, so importing it here keeps the quantity the
# optimizer penalizes identical to the quantity the placement solves for.
# (src.mech does not import src.aero, so this introduces no import cycle.)
from src.mech.mass_properties import estimate_aerodynamic_center_x
from src.vectors import DesignVector


# Fuselage yaw-destabilizing coefficient, DATCOM-style: the body contributes
# -FUSELAGE_YAW_COEFFICIENT * volume / (S * b) to Cnb. Literature puts this in
# the 1.3-2.0 band; 1.70 is a least-squares fit of the physics-shaped model
# below against asb.AeroBuildup.run_with_stability_derivatives() over 117 real
# optimizer candidates (see scratchpad/cnb_fit.py in the 2026-09-04 session).
#
# The regression this replaced,
#     cnb = 0.0390501 - 14.3090 * cnb_est - 0.243974 * cyb_est + ...
# multiplied a quantity built to be positive-when-stable by -14.309, and had no
# fuselage term at all. Against the same ground truth it agreed in sign on only
# 87/117 designs with correlation 0.509; the model below achieves 110/117 and
# 0.971. That matters here because the fuselage grows with M2 payload, so the
# body term is the dominant and most design-sensitive contribution.
FUSELAGE_YAW_COEFFICIENT: float = 1.70


def _fuselage_reference_volume_m3(design_vector: DesignVector) -> float:
    """Approximate fuselage bounding volume used for the body yaw term."""

    length = (
        float(design_vector.nose_length)
        + float(design_vector.tail_arm)
        + float(design_vector.hstab_chord)
    )
    return float(
        length
        * float(design_vector.fuselage_width)
        * float(design_vector.fuselage_height)
    )


def _directional_stability(design_vector: DesignVector, x_cg_m: float) -> float:
    """Return Cnb from the vertical tail minus the fuselage body contribution."""

    s = design_vector.wing_area
    b = design_vector.wing_span
    sv = design_vector.vstab_area
    av = 2 * np.pi / (1 + 2 / (design_vector.vstab_span**2 / sv))
    # Vertical-tail arm measured to its own quarter chord, not the h-tail's.
    lv = design_vector.tail_arm + 0.25 * design_vector.vstab_chord - x_cg_m
    tail_term = 0.90 * av * (sv / s) * (lv / b)
    body_term = FUSELAGE_YAW_COEFFICIENT * _fuselage_reference_volume_m3(
        design_vector
    ) / (s * b)
    return float(tail_term - body_term)


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
    xw, xt = 0.25 * c, design_vector.tail_arm + 0.25 * design_vector.hstab_chord
    lt = xt - xcg

    # Linear finite-wing lift; assumes 15% downwash and 90% tail efficiency.
    aw = 2 * np.pi / (1 + 2 / (b**2 / s))
    at = 2 * np.pi / (1 + 2 / (design_vector.hstab_span**2 / st))
    av = 2 * np.pi / (1 + 2 / (design_vector.vstab_span**2 / sv))
    eta, downwash = 0.90, 0.15
    cla_est = aw + eta * st / s * at * (1 - downwash)
    cma_est = aw * (xcg - xw) / c - eta * st / s * at * (1 - downwash) * lt / c
    cmq_est = -1.6 * eta * at * (st * lt / (s * c)) * lt / c
    cla = -1.88073 + 1.37744 * cla_est
    cma = -0.867205 + 0.448798 * cma_est
    cmq = 3.47831 + 1.98806 * cmq_est  # Calibrated linear pitch derivatives.

    # Trim lift and parabolic drag estimates.
    q = require_scalar(cruise_condition.operating_point.dynamic_pressure())
    cl = require_scalar(mass_props.mass) * 9.806 / (q * s)
    cd = 0.02 + cl**2 / (np.pi * 0.85 * b**2 / s)

    # Calibrated tail/body derivatives; assumes small sideslip and yaw rate.
    alpha = require_scalar(cruise_condition.operating_point.alpha)
    cyb_est = -eta * av * sv / s
    cyr_est = -2 * cyb_est * lt / b
    clb_est = -cyb_est * (0.5 * design_vector.vstab_span - require_scalar(mass_props.z_cg)) / b
    cnr_est = 2 * cyb_est * (lt / b) ** 2
    cyb = -0.0434608 + 0.61645 * cyb_est
    cyr = (0.0662136 + 0.708383 * cyr_est + 0.139102 * cyb_est
           - 0.00943366 * b - 0.0723635 * c + 0.00459817 * alpha)
    clb = 0.0147862 + 0.122881 * cyb_est + 0.228923 * clb_est
    cnb = _directional_stability(design_vector, xcg)
    cnr = 0.0165917 + 1.17482 * cnr_est

    # Rectangular-wing damping with calibrated buildup bias.
    clp = 0.263378 + 1.46526 * (-aw / 8)
    clr = 0.0341476 + 0.619521 * (cl / 4)

    # Neutral point from geometry alone. It must not depend on the CG: the
    # calibrated Cma regression above scales dCma/dx_cg by its slope
    # coefficient, so `xcg - cma / cla * c` yields a "neutral point" that moves
    # with the CG. Use the shared geometric estimator instead.
    x_np = estimate_aerodynamic_center_x(design_vector)

    return {
        "CL": cl, "CD": cd, "Cma": cma, "Cmq": cmq,
        "CYb": cyb, "CYr": cyr, "Clb": clb, "Clp": clp,
        "Clr": clr, "Cnb": cnb, "Cnr": cnr, "CLa": cla,
        "x_np": x_np,
        # Legacy diagnostic; see StabilityResult.static_margin_from_cma.
        "x_np_from_cma": xcg - cma / cla * c,
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

    # get_modes' spiral root divides by Clb and is singular on this zero-dihedral
    # geometry; replace it with the 4-state solve. See stability_criteria.
    lateral = lateral_modes(
        lateral_state_matrix(
            CYb=aero["CYb"], CYr=aero["CYr"],
            Clb=aero["Clb"], Clp=aero["Clp"], Clr=aero["Clr"],
            Cnb=aero["Cnb"], Cnr=aero["Cnr"], CL=aero["CL"],
            dynamic_pressure_pa=require_scalar(
                cruise_condition.operating_point.dynamic_pressure()
            ),
            wing_area_m2=s,
            wing_span_m=b,
            mass_kg=require_scalar(mass_props.mass),
            Ixx=require_scalar(mass_props.Ixx),
            Izz=require_scalar(mass_props.Izz),
            velocity_m_s=require_scalar(cruise_condition.operating_point.velocity),
            gravity_m_s2=9.806,
        )
    )
    modes["spiral"] = {
        "eigenvalue_real": lateral.spiral_eigenvalue_real,
        "eigenvalue_imag": 0.0,
        "damping_ratio": -np.sign(lateral.spiral_eigenvalue_real),
    }

    x_cg = require_scalar(mass_props.x_cg)
    return StabilityResult(
        phugoid=dict_to_mode_result(modes["phugoid"]),
        short_period=dict_to_mode_result(modes["short_period"]),
        dutch_roll=dict_to_mode_result(modes["dutch_roll"]),
        spiral=dict_to_mode_result(modes["spiral"]),
        roll_subsidence=dict_to_mode_result(modes["roll_subsidence"]),
        Cma=float(aero["Cma"]),
        Cnb=float(aero["Cnb"]),
        static_margin=float((aero["x_np"] - x_cg) / c),
        neutral_point_x_m=float(aero["x_np"]),
        static_margin_from_cma=float(-aero["Cma"] / aero["CLa"]),
        spiral_time_to_double_s=time_to_double_s(lateral.spiral_eigenvalue_real),
    )
