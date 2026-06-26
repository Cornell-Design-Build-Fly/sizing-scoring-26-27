from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import aerosandbox as asb

# Constants from DFO baseline
V_H  = 0.50
V_V  = 0.075
AR_H = 3
AR_V = 1.75
FUSELAGE_BOX_SIZE = 0.13
FUSELAGE_SHAPE = 8.0
FUSELAGE_TIP_SIZE = 0.01

OPT_VARS = [
    ("wing_span", (0.8, 1.8)),
    ("wing_chord", (0.12, 0.35)),
    ("tail_arm", (0.3, 0.9)),
    ("nose_length", (0.08, 0.3)),
    ("ducks_num", (3, 10)),
    ("pucks_num", (1, 10)),
    ("banner_length", (0.5, 5.0)),
    ("batt_capacity", (1.0, 10.0)),
]

@dataclass
class DesignVector:
    """Baseline aircraft sizing vector in meters."""
    # The only four things you actually set
    # Aero geometries
    wing_span: float = 1.181 # [m]
    wing_chord: float = 0.307 # [m]
    tail_arm: float = 0.845 # [m]
    nose_length: float = 0.254 # [m]

    # Mission payloads
    ducks_num: float = 3
    pucks_num: float = 1
    banner_length: float = 3.8 # [m]

    # Prop components
    batt_capacity: float = 4.5 # [Ah]
    prop_diameter_in: float = 14.0  # [in]
    prop_pitch_in: float = 10.0  # [in]
    motor_kv: float = 335.0  # [RPM/V]
    motor_max_power: float = 2200.0  # [W]
    cruise_throttle: float = 0.90
    mission3_cruise_throttle: float = 0.85


    # Derived, do not set manually
    wing_area:        float = field(init=False)
    hstab_area:       float = field(init=False)
    hstab_span:       float = field(init=False)
    hstab_chord:      float = field(init=False)
    vstab_area:       float = field(init=False)
    vstab_span:       float = field(init=False)
    vstab_chord:      float = field(init=False)
    fuselage_width:   float = field(init=False)
    fuselage_height:  float = field(init=False)
    batt_energy:      float = field(init=False)


    def __post_init__(self):
        """Calculates derived parameters and checks for validity."""
        if self.wing_span <= 0 or self.wing_chord <= 0 or self.tail_arm <= 0 or self.nose_length <= 0:
            raise ValueError("All DesignVector primary dimensions must be positive and expressed in meters.")

        self.wing_area   = self.wing_span * self.wing_chord

        self.hstab_area  = V_H * self.wing_area * self.wing_chord / self.tail_arm
        self.hstab_span  = np.sqrt(AR_H * self.hstab_area)
        self.hstab_chord = self.hstab_area / self.hstab_span

        self.vstab_area  = V_V * self.wing_area * self.wing_span / self.tail_arm
        self.vstab_span  = np.sqrt(AR_V * self.vstab_area)
        self.vstab_chord = self.vstab_area / self.vstab_span

        self.fuselage_width = FUSELAGE_BOX_SIZE
        self.fuselage_height = FUSELAGE_BOX_SIZE

        self.batt_energy = self.batt_capacity * ParameterVector.voltage

    def to_array(self) -> np.ndarray:
        """Returns the optimizer variables in the same order as bounds()."""
        return np.array([getattr(self, name) for name, _ in OPT_VARS], dtype=float)

    @staticmethod
    def from_array(x):
        """Builds a design vector from an optimizer array."""
        if len(x) != len(OPT_VARS):
            raise ValueError(f"Input array must have length {len(OPT_VARS)}, but got {len(x)}.")
        kwargs = {name: float(value) for value, (name, _) in zip(x, OPT_VARS)}
        return DesignVector(**kwargs)

    @staticmethod
    def bounds() -> list[tuple[float, float]]:
        """Returns SciPy-style bounds in the same order as to_array()."""
        return [bounds for _, bounds in OPT_VARS]

    @staticmethod
    def opt_names() -> list[str]:
        """Returns the optimizer variable names in array order."""
        return [name for name, _ in OPT_VARS]
    
# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------

@dataclass
class ParameterVector:
    """A vector of parameters that can be used for non-geometry optimization."""
    gravity = 9.806 # [m/s^2]
    rho = 1.225 # [kg/m^3]
    voltage = 22.2 # [V]
    temp = 20.0 # [C]
    press = 101325 # [Pa]


# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------


@dataclass
class ASBDesignVector(DesignVector):
    """Metric design vector with helpers to build AeroSandbox geometry."""

    @classmethod
    def from_design_vector(
        cls,
        design_vector: DesignVector,
        unit_scale: float = 1.0,
    ) -> "ASBDesignVector":
        """Promotes any existing design vector into an ASB-ready one."""
        return cls(
            wing_span=design_vector.wing_span * unit_scale,
            wing_chord=design_vector.wing_chord * unit_scale,
            tail_arm=design_vector.tail_arm * unit_scale,
            nose_length=design_vector.nose_length * unit_scale,
        )

    def make_airplane(
        self,
        *,
        name: str = "Design Vector Plane",
        wing_airfoil: str = "naca2412",
        tail_airfoil: str = "naca0012",
        wing_le: tuple[float, float, float] = (0.0, 0.0, 0.0),
        tail_waterline: float = 0.00,
    ) -> asb.Airplane:
        """Builds a simple AeroSandbox airplane from the design-vector geometry."""
        import aerosandbox as asb

        wing_qc_x = wing_le[0] + 0.25 * self.wing_chord
        wing_te_x = wing_le[0] + self.wing_chord
        tail_qc_x = wing_qc_x + self.tail_arm
        horizontal_tail_le_x = tail_qc_x - 0.25 * self.hstab_chord
        vertical_tail_le_x = tail_qc_x - 0.25 * self.vstab_chord
        tail_te_x = max(
            horizontal_tail_le_x + self.hstab_chord,
            vertical_tail_le_x + self.vstab_chord,
        )
        fuselage = self.make_fuselage(
            wing_le_x=wing_le[0],
            wing_te_x=wing_te_x,
            tail_te_x=tail_te_x,
        )

        wing_airfoil_obj = asb.Airfoil(wing_airfoil)
        tail_airfoil_obj = asb.Airfoil(tail_airfoil)

        main_wing = asb.Wing(
            name="Main Wing",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=list(wing_le),
                    chord=self.wing_chord,
                    twist=0.0,
                    airfoil=wing_airfoil_obj,
                ),
                asb.WingXSec(
                    xyz_le=[wing_le[0], self.wing_span / 2.0, wing_le[2]],
                    chord=self.wing_chord,
                    twist=0.0,
                    airfoil=wing_airfoil_obj,
                ),
            ],
        )

        horizontal_tail = asb.Wing(
            name="Horizontal Tail",
            symmetric=True,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[horizontal_tail_le_x, 0.0, tail_waterline],
                    chord=self.hstab_chord,
                    twist=0.0,
                    airfoil=tail_airfoil_obj,
                ),
                asb.WingXSec(
                    xyz_le=[horizontal_tail_le_x, self.hstab_span / 2.0, tail_waterline],
                    chord=self.hstab_chord,
                    twist=0.0,
                    airfoil=tail_airfoil_obj,
                ),
            ],
        )

        vertical_tail = asb.Wing(
            name="Vertical Tail",
            symmetric=False,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[vertical_tail_le_x, 0.0, 0.0],
                    chord=self.vstab_chord,
                    twist=0.0,
                    airfoil=tail_airfoil_obj,
                ),
                asb.WingXSec(
                    xyz_le=[vertical_tail_le_x, 0.0, self.vstab_span],
                    chord=self.vstab_chord,
                    twist=0.0,
                    airfoil=tail_airfoil_obj,
                ),
            ],
        )

        airplane = asb.Airplane(
            name=name,
            xyz_ref=[wing_qc_x, 0.0, 0.0],
            wings=[main_wing, horizontal_tail, vertical_tail],
            fuselages=[fuselage],
            s_ref=float(self.wing_area),
            c_ref=float(self.wing_chord),
            b_ref=float(self.wing_span),
        )

        return airplane

    def make_fuselage(
        self,
        *,
        wing_le_x: float,
        wing_te_x: float,
        tail_te_x: float,
    ) -> "asb.Fuselage":
        """Builds a fuselage from nose tip to tail tip using the design vector."""
        import aerosandbox as asb

        nose_tip_x = wing_le_x - self.nose_length
        nose_transition_x = nose_tip_x + 0.35 * self.nose_length
        aft_mid_x = wing_te_x + 0.65 * (tail_te_x - wing_te_x)

        return asb.Fuselage(
            name="Fuselage",
            xsecs=[
                asb.FuselageXSec(
                    xyz_c=[nose_tip_x, 0.0, -self.fuselage_height / 2.0],
                    width=FUSELAGE_TIP_SIZE,
                    height=FUSELAGE_TIP_SIZE,
                    shape=2.0,
                ),
                asb.FuselageXSec(
                    xyz_c=[nose_transition_x, 0.0,  -self.fuselage_height / 2.0],
                    width=self.fuselage_width,
                    height=self.fuselage_height,
                    shape=FUSELAGE_SHAPE,
                ),
                asb.FuselageXSec(
                    xyz_c=[wing_te_x, 0.0,  -self.fuselage_height / 2.0],
                    width=self.fuselage_width,
                    height=self.fuselage_height,
                    shape=FUSELAGE_SHAPE,
                ),
                asb.FuselageXSec(
                    xyz_c=[aft_mid_x, 0.0,  -self.fuselage_height / 4.0],
                    width=0.06,
                    height=0.06,
                    shape=FUSELAGE_SHAPE,
                ),
                asb.FuselageXSec(
                    xyz_c=[tail_te_x, 0.0, 0.0],
                    width=FUSELAGE_TIP_SIZE,
                    height=FUSELAGE_TIP_SIZE,
                    shape=2.0,
                ),
            ],
        )
