"""Shared static derivative fit, in radians, calibrated on 2026-09-06.

Factors come from the 105 training cases in static_calibration_20260906/
under_55lb/summary.json (coupled_enriched); the 45 holdout cases were excluded.
Scope: NACA2412, default tails, resolved fuselage, alpha -2..10 degrees,
elevator -8..8 degrees, speed 20..42 m/s, and M2 mass <=55 lb.
Extrapolation is provisional, especially CLa. This is a derivative surrogate,
not a recalibration of the cruise trim force model or dynamic derivatives.
"""
from functools import lru_cache

import numpy as np

from src.aero.utils import require_scalar
from src.vectors import ASBDesignVector, DesignVector

WING_GAIN = 0.9774884788988752
TAIL_GAIN = 1.2941990573838225
BODY_LIFT_GAIN = 5.004886074919616
BODY_PITCH_GAIN = 1.7843560019558957
FIN_GAIN = 0.8886195019863597
BODY_YAW_GAIN = 1.937582987540451


@lru_cache(maxsize=4096)
def _body_geometry(design_values):
    design = DesignVector(*design_values)
    body = ASBDesignVector.from_design_vector(design).make_fuselage(
        wing_le_x=0.0, wing_te_x=design.wing_chord,
        tail_te_x=design.tail_arm + max(design.hstab_chord, design.vstab_chord),
    )
    xs = np.array([require_scalar(section.xyz_c[0]) for section in body.xsecs])
    areas = np.array([require_scalar(section.xsec_area()) for section in body.xsecs])
    centroid = float(np.trapezoid(xs * areas, xs) / np.trapezoid(areas, xs))
    return float(body.length()), float(body.volume()), centroid


def calibrated_static_derivatives(design, alpha_deg, elevator_deg, x_cg):
    """Return CLa, Cma, Cnb with shared force slopes and actual CG arms.

    Pitch translation uses CLa*cos(alpha), a low-angle normal-force
    approximation that omits axial-force derivatives. Total full-model yaw
    need not share the fin-only CG trend; body sideforce is approximated here.
    """
    values = tuple(getattr(design, name) for name, field in
                   design.__dataclass_fields__.items() if field.init)
    length, volume, body_x = _body_geometry(values)
    s, c, b = design.wing_area, design.wing_chord, design.wing_span
    aw = 2 * np.pi / (1 + 2 / (b*b/s))
    ah = .90 * .85 * 2 * np.pi / (1 + 2 / (design.hstab_span**2/design.hstab_area)) * design.hstab_area/s
    av = .90 * 2 * np.pi / (1 + 2 / (design.vstab_span**2/design.vstab_area)) * design.vstab_area/s
    alpha = np.deg2rad(alpha_deg)
    wing = WING_GAIN * aw * max(0., 1 - (alpha_deg/15)**2)
    tail = TAIL_GAIN * ah * max(0., 1 - ((alpha_deg + .65*elevator_deg)/20)**2)
    body = BODY_LIFT_GAIN * design.fuselage_width * length/s * abs(np.sin(alpha))
    cla = wing + tail + body
    cma = np.cos(alpha) * (
        wing * (x_cg - .25*c)/c
        + tail * (x_cg - design.tail_arm - .25*design.hstab_chord)/c
        + body * (x_cg - body_x)/c
    ) + BODY_PITCH_GAIN * volume/(s*c)
    cnb = (FIN_GAIN * av * (design.tail_arm + .25*design.vstab_chord - x_cg)/b
           - BODY_YAW_GAIN * volume/(s*b))
    return float(cla), float(cma), float(cnb)
