from types import SimpleNamespace

import numpy as np
from aerosandbox.dynamics.flight_dynamics.airplane import get_modes
from aerosandbox.weights.mass_properties import MassProperties

from src.aero.custom_classes import CruiseCondition, StabilityResult
from src.aero.utils import dict_to_mode_result, require_scalar
from src.vectors import DesignVector


def stability_analysis_coarse(
    design_vector: DesignVector,
    cruise_condition: CruiseCondition,
    mass_props: MassProperties,
) -> StabilityResult:
    """Estimate stability from conceptual-design formulas."""
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
    cla = aw + eta * st / s * at * (1 - downwash)
    cma = aw * (xcg - xw) / c - eta * st / s * at * (1 - downwash) * lt / c
    cmq = -1.6 * eta * at * (st * lt / (s * c)) * lt / c  # Tail-dominated pitch damping.

    # Trim lift and parabolic drag estimates.
    q = require_scalar(cruise_condition.operating_point.dynamic_pressure())
    cl = require_scalar(mass_props.mass) * 9.806 / (q * s)
    cd = 0.02 + cl**2 / (np.pi * 0.85 * b**2 / s)

    # Vertical-tail derivatives; assumes 90% sidewash efficiency.
    cyb = -eta * av * sv / s
    cnb = -cyb * lt / b - 0.05  # Tail stability minus fuselage estimate.
    cnr = 2 * cyb * (lt / b) ** 2
    cyr = -2 * cyb * lt / b

    # Rectangular-wing damping; tail height supplies dihedral effect.
    clb = -cyb * (0.5 * design_vector.vstab_span - require_scalar(mass_props.z_cg)) / b
    clp = -aw / 8
    clr = cl / 4

    aero = {
        "CL": cl, "CD": cd, "Cma": cma, "Cmq": cmq,
        "CYb": cyb, "CYr": cyr, "Clb": clb, "Clp": clp,
        "Clr": clr, "Cnb": cnb, "Cnr": cnr,
    }
    airplane = SimpleNamespace(s_ref=s, c_ref=c, b_ref=b)
    modes = get_modes(airplane, cruise_condition.operating_point, mass_props, aero)

    return StabilityResult(
        phugoid=dict_to_mode_result(modes["phugoid"]),
        short_period=dict_to_mode_result(modes["short_period"]),
        dutch_roll=dict_to_mode_result(modes["dutch_roll"]),
        spiral=dict_to_mode_result(modes["spiral"]),
        roll_subsidence=dict_to_mode_result(modes["roll_subsidence"]),
        Cma=float(cma),
        Cnb=float(cnb),
        static_margin=float(-cma / cla),
    )
