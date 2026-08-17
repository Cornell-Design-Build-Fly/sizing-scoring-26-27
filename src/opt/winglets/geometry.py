from __future__ import annotations

from dataclasses import dataclass

import aerosandbox as asb
import aerosandbox.numpy as np

from src.vectors import ASBDesignVector, DesignVector


WINGLET_REGION_LIMIT_M = 0.250
WINGLET_BLEND_XSECS = 30

WINGLET_VARIABLES: list[tuple[str, tuple[float, float]]] = [
    ("blend_length_m", (0.050, WINGLET_REGION_LIMIT_M)),
    ("height_m", (0.020, 0.100)),
    ("tip_inset_m", (0.000, 0.120)),
    ("cant_angle_deg", (35.0, 130.0)),
    ("leading_edge_sweep_m", (0.000, 0.180)),
    ("sweep_exponent", (0.100, 2.500)),
    ("tip_chord_ratio", (0.050, 0.800)),
    ("taper_exponent", (0.100, 2.500)),
    ("toe_angle_deg", (-6.0, 6.0)),
    ("tip_incidence_deg", (-6.0, 6.0)),
    ("blend_tension", (0.500, 2.000)),
]


@dataclass(frozen=True)
class WingletGeometry:
    """Practical blended-winglet shape controls, all dimensional values in meters."""

    blend_length_m: float = 0.180
    height_m: float = 0.070
    tip_inset_m: float = 0.000
    cant_angle_deg: float = 92.0
    leading_edge_sweep_m: float = 0.040
    sweep_exponent: float = 2.000
    tip_chord_ratio: float = 0.400
    taper_exponent: float = 1.500
    toe_angle_deg: float = -1.5
    tip_incidence_deg: float = -2.0
    blend_tension: float = 1.000

    @staticmethod
    def bounds() -> list[tuple[float, float]]:
        return [bounds for _, bounds in WINGLET_VARIABLES]

    @staticmethod
    def opt_names() -> list[str]:
        return [name for name, _ in WINGLET_VARIABLES]

    @classmethod
    def from_array(cls, x) -> "WingletGeometry":
        if len(x) != len(WINGLET_VARIABLES):
            raise ValueError(
                f"Input array must have length {len(WINGLET_VARIABLES)}, but got {len(x)}."
            )
        return cls(**{name: value for value, (name, _) in zip(x, WINGLET_VARIABLES)})

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.opt_names()}


def _remorphed_main_wing(
    design_vector: DesignVector,
    winglet: WingletGeometry,
    *,
    wing_le: tuple[float, float, float] = (0.0, 0.0, 0.0),
    winglet_airfoil: str = "naca0012",
) -> asb.Wing:
    """Builds the main wing with its final spanwise region remorphed into a winglet."""

    span_semilength = design_vector.wing_span / 2.0
    chord = design_vector.wing_chord
    root_y = span_semilength - winglet.blend_length_m
    tip_y = span_semilength - winglet.tip_inset_m

    cant_rad = winglet.cant_angle_deg * np.pi / 180.0
    root_handle = 0.40 * winglet.blend_tension * winglet.blend_length_m
    tip_handle = 0.50 * winglet.blend_tension * np.minimum(
        winglet.blend_length_m,
        winglet.height_m,
    )

    p0 = np.array([root_y, wing_le[2]])
    p1 = np.array([root_y + root_handle, wing_le[2]])
    p3 = np.array([tip_y, wing_le[2] + winglet.height_m])
    p2 = p3 - tip_handle * np.array([np.cos(cant_rad), np.sin(cant_rad)])
    root_te = np.array([wing_le[0] + chord, wing_le[2]])
    tip_le_x = wing_le[0] + winglet.leading_edge_sweep_m
    tip_twist_rad = (winglet.tip_incidence_deg + winglet.toe_angle_deg) * np.pi / 180.0
    tip_chord = chord * winglet.tip_chord_ratio
    tip_te = np.array(
        [
            tip_le_x + tip_chord * np.cos(tip_twist_rad),
            wing_le[2] + winglet.height_m - tip_chord * np.sin(tip_twist_rad),
        ]
    )

    wing_airfoil = asb.Airfoil(design_vector.wing_airfoil)
    tip_airfoil = asb.Airfoil(winglet_airfoil)
    xsecs = [
        asb.WingXSec(
            xyz_le=list(wing_le),
            chord=chord,
            twist=0.0,
            airfoil=wing_airfoil,
        ),
        asb.WingXSec(
            xyz_le=[wing_le[0], root_y, wing_le[2]],
            chord=chord,
            twist=0.0,
            airfoil=wing_airfoil,
        ),
    ]

    for s in np.linspace(0.0, 1.0, WINGLET_BLEND_XSECS + 1)[1:]:
        bezier = (
            (1.0 - s) ** 3 * p0
            + 3.0 * (1.0 - s) ** 2 * s * p1
            + 3.0 * (1.0 - s) * s**2 * p2
            + s**3 * p3
        )
        smooth_s = s**2 * (3.0 - 2.0 * s)
        # Start both edges tangent to the flat wing. These functions have
        # zero slope at s=0 but do not force zero slope at the tip.
        sweep_progress = s ** (1.0 + winglet.sweep_exponent)
        te_progress = s ** (1.0 + winglet.taper_exponent)
        trailing_edge = root_te * (1.0 - te_progress) + tip_te * te_progress
        leading_edge_x = wing_le[0] + winglet.leading_edge_sweep_m * sweep_progress
        dx_chord = trailing_edge[0] - leading_edge_x
        dz_chord = trailing_edge[1] - bezier[1]
        section_chord = (dx_chord**2 + dz_chord**2) ** 0.5
        section_twist = -np.arctan2(dz_chord, dx_chord) * 180.0 / np.pi
        airfoil_blend = np.clip((s - 0.15) / 0.55, 0.0, 1.0)
        airfoil_blend = airfoil_blend**2 * (3.0 - 2.0 * airfoil_blend)
        airfoil = (
            wing_airfoil
            if airfoil_blend == 0.0
            else (
                tip_airfoil
                if airfoil_blend == 1.0
                else wing_airfoil.blend_with_another_airfoil(
                    airfoil=tip_airfoil,
                    blend_fraction=float(airfoil_blend),
                )
            )
        )
        xsecs.append(
            asb.WingXSec(
                xyz_le=[
                    leading_edge_x,
                    bezier[0],
                    bezier[1],
                ],
                chord=section_chord,
                twist=section_twist,
                airfoil=airfoil,
            )
        )

    return asb.Wing(
        name="Main Wing with Remorphed Tips",
        symmetric=True,
        xsecs=xsecs,
    )


def add_winglets_to_design_vector_airplane(
    airplane: asb.Airplane,
    design_vector: DesignVector,
    winglet: WingletGeometry,
    *,
    wing_le: tuple[float, float, float] = (0.0, 0.0, 0.0),
    winglet_airfoil: str = "naca0012",
) -> asb.Airplane:
    """Returns a copy of a design-vector airplane with the main wingtips remorphed."""

    remorphed_main_wing = _remorphed_main_wing(
        design_vector,
        winglet,
        wing_le=wing_le,
        winglet_airfoil=winglet_airfoil,
    )
    other_wings = [wing for wing in airplane.wings if wing.name != "Main Wing"]

    return asb.Airplane(
        name=f"{airplane.name} + Winglets",
        xyz_ref=airplane.xyz_ref,
        wings=[remorphed_main_wing, *other_wings],
        fuselages=airplane.fuselages,
        s_ref=airplane.s_ref,
        c_ref=airplane.c_ref,
        b_ref=airplane.b_ref,
    )


def make_winglet_airplane(
    design_vector: DesignVector,
    winglet: WingletGeometry,
    *,
    name: str = "Winglet Candidate",
    winglet_airfoil: str = "naca0012",
) -> asb.Airplane:
    """Builds the repo's standard ASB airplane with optimized main-wing tip geometry."""

    asb_design_vector = ASBDesignVector.from_design_vector(design_vector)
    base_airplane = asb_design_vector.make_airplane(name=name)
    return add_winglets_to_design_vector_airplane(
        base_airplane,
        design_vector,
        winglet,
        winglet_airfoil=winglet_airfoil,
    )
