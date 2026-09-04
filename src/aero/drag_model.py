"""Fast drag build-up shared by algebraic cruise solvers."""

from functools import lru_cache

import aerosandbox.numpy as np
from aerosandbox.aerodynamics.aero_3D.aero_buildup_submodels.fuselage_aerodynamics_utilities import (
    fuselage_base_drag_coefficient,
    fuselage_form_factor,
)

from src.vectors import (
    ASBDesignVector,
    DesignVector,
    ParameterVector,
    SENSOR_DIAMETER_M,
)


MU = 1.81e-5
SENSOR_CD = 0.137
SENSOR_RADIUS_M = 0.5 * SENSOR_DIAMETER_M


def sensor_drag_force(
    design: DesignVector,
    parameters: ParameterVector,
    velocity,
):
    """Return sensor drag using the projected side area of a cylinder.

    The M3 flown weight determines rod length; diameter remains fixed.
    """
    sensor_length = design.mission3_sensor_length_m
    sensor_area = 2.0 * SENSOR_RADIUS_M * sensor_length
    return 0.5 * parameters.rho * velocity**2 * SENSOR_CD * sensor_area


@lru_cache(maxsize=4096)
def _fuselage_drag_geometry_cached(design_values: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Build fuselage geometry once for all mission evaluations of a design."""
    design = DesignVector(*design_values)
    tail_te = design.tail_arm + max(design.hstab_chord, design.vstab_chord)
    fuselage = ASBDesignVector.from_design_vector(design).make_fuselage(
        wing_le_x=0.0, wing_te_x=design.wing_chord, tail_te_x=tail_te
    )
    return (
        float(fuselage.length()),
        float(fuselage_form_factor(float(fuselage.fineness_ratio()), 0.5)),
        float(fuselage.area_wetted()),
        float(fuselage.area_base()),
    )


def fuselage_drag_geometry(design: DesignVector) -> tuple[float, float, float, float]:
    """Precompute the fuselage geometry used by its profile-drag model."""
    init_fields = tuple(
        getattr(design, name)
        for name, field in design.__dataclass_fields__.items()
        if field.init
    )
    return _fuselage_drag_geometry_cached(init_fields)


def drag_coefficients(
    design: DesignVector,
    parameters: ParameterVector,
    velocity,
    wing_cl,
    tail_cl,
    fuselage_geometry: tuple[float, float, float, float] | None = None,
) -> dict:
    """Return calibrated profile, induced, and fuselage drag coefficients."""
    s, st, sv = design.wing_area, design.hstab_area, design.vstab_area
    ar = design.wing_span**2 / s
    art, ratio = design.hstab_span**2 / st, st / s
    re_w = parameters.rho * velocity * design.wing_chord / MU
    re_h = parameters.rho * velocity * design.hstab_chord / MU
    re_v = parameters.rho * velocity * design.vstab_chord / MU
    wing_profile = 0.001870 + 3.66232 / np.sqrt(re_w)
    tail_profile = (
        0.000789379141 * (st + sv)
        + 4.21981706 * (st / np.sqrt(re_h) + sv / np.sqrt(re_v))
    ) / s

    wing_induced = 1.20250220 * wing_cl**2 / (np.pi * ar)
    tail_induced = 1.34448532 * ratio * tail_cl**2 / (np.pi * art)
    interaction = 2 * ratio * wing_cl * tail_cl / (np.pi * np.sqrt(ar * art))
    interaction *= 1.38762007 - 0.09307174 * design.wing_chord / design.tail_arm

    length, form_factor, wetted_area, base_area = fuselage_geometry or fuselage_drag_geometry(design)
    re_f = parameters.rho * velocity * length / MU
    skin_friction = (3.46 * np.log10(re_f) - 5.6) ** -2
    skin_friction *= form_factor
    fuselage_drag_area = skin_friction * wetted_area
    fuselage_drag_area += fuselage_base_drag_coefficient(velocity / 340.0) * base_area
    body = 1.05536867 * fuselage_drag_area / s
    return {
        "wing_profile": wing_profile,
        "tail_profile": tail_profile,
        "wing_induced": wing_induced,
        "tail_induced": tail_induced,
        "interaction": interaction,
        "body": body,
    }
