from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import aerosandbox as asb

from src.prop.prop_classes import (
    DEFAULT_BATTERY_CELL_COUNT,
    battery_energy_wh,
    battery_nominal_voltage_v,
    normalize_battery_cell_count,
)

# Constants from DFO baseline
V_H  = 0.50
V_V  = 0.036   # tail volume coeff; 0.075 was 2× oversize vs actual DF1 (0.036)
AR_H = 3
AR_V = 0.89    # actual DF1 fin: 5.20in × 5.83in → span/chord = 0.89; 1.75 was wrong shape
FUSELAGE_BOX_SIZE = 0.13
FUSELAGE_START_WIDTH = 5.0 * 0.0254
FUSELAGE_SHAPE = 8.0
FUSELAGE_TIP_SIZE = 0.01
MAX_EXTRA_SHIPPING_CONTAINERS = 10
INCH_M = 0.0254
POUND_KG = 0.45359237
# Nominal sensor diameter, kept as the default and as the container's minimum
# cross-section. The flown diameter is a design variable (see OPT_VARS) because
# freezing it made "heavy" reachable only by making the sensor LONG, which then
# lengthened the container, the fuselage and the takeoff roll. That coupling is
# an artifact of the constant, not physics: a short fat sensor is equally legal.
SENSOR_DIAMETER_M = 3.0 * INCH_M
MIN_SENSOR_DIAMETER_M = 0.5 * INCH_M
MAX_SENSOR_DIAMETER_M = 6.0 * INCH_M
# Density ceiling for the sensor. Not a guess: the rules require the lights,
# battery and control electronics to live inside the sensor, so it cannot be
# solid ballast. Lead is 11340 kg/m^3; at roughly 70% packing around the
# electronics that lands near 7900, so solid steel is a good proxy for "as
# dense as this could actually be built".
SENSOR_STEEL_DENSITY_KG_M3 = 7850.0
# Rules 3.1.1 require the sensor to be at least 6 inches long.  Length and
# weight are independent, so enforcing that geometry no longer creates the old
# artificial 12.03 lb minimum sensor weight.
MIN_SENSOR_LENGTH_M = 6.0 * INCH_M
MAX_SENSOR_LENGTH_M = 24.0 * INCH_M
MIN_SENSOR_WEIGHT_KG = 0.05
MIN_MISSION3_SENSOR_WEIGHT_KG = MIN_SENSOR_WEIGHT_KG


def maximum_sensor_weight_kg(
    sensor_length_m: float,
    sensor_diameter_m: float = SENSOR_DIAMETER_M,
) -> float:
    """Heaviest physically realizable sensor of a given length and diameter.

    Weight, length and diameter are all free; the sensor simply cannot be denser
    than SENSOR_STEEL_DENSITY_KG_M3. Without this bound the optimizer would take
    maximum weight in minimum volume -- Mission 3 and the Ground Mission both
    reward weight -- and get an arbitrarily small, arbitrarily heavy payload.
    """

    sensor_length_m = float(sensor_length_m)
    sensor_diameter_m = float(sensor_diameter_m)
    if not np.isfinite(sensor_length_m) or sensor_length_m <= 0.0:
        raise ValueError("sensor_length_m must be finite and positive.")
    if not np.isfinite(sensor_diameter_m) or sensor_diameter_m <= 0.0:
        raise ValueError("sensor_diameter_m must be finite and positive.")
    cross_section_m2 = np.pi * (0.5 * sensor_diameter_m) ** 2
    return float(SENSOR_STEEL_DENSITY_KG_M3 * cross_section_m2 * sensor_length_m)


# Differential evolution requires finite box bounds. This is not an
# independent sensor-weight cap: it is the density limit evaluated at the
# largest sensor length the optimizer can select. The nonlinear density
# constraint below tightens the bound for every shorter sensor.
OPTIMIZER_SENSOR_WEIGHT_UPPER_KG = maximum_sensor_weight_kg(
    MAX_SENSOR_LENGTH_M, MAX_SENSOR_DIAMETER_M
)


def sensor_length_from_weight_kg(sensor_weight_kg: float) -> float:
    """Length of a solid-steel sensor of the given weight (legacy helper)."""

    sensor_weight_kg = float(sensor_weight_kg)
    if not np.isfinite(sensor_weight_kg) or sensor_weight_kg <= 0.0:
        raise ValueError("sensor_weight_kg must be finite and positive.")
    cross_section_m2 = np.pi * (0.5 * SENSOR_DIAMETER_M) ** 2
    return float(sensor_weight_kg / (SENSOR_STEEL_DENSITY_KG_M3 * cross_section_m2))

# Rules 3.2.3.2: total propulsion energy may not exceed 100 Wh. With the cell
# count fixed at 8S this converts directly into a capacity ceiling, so the
# optimizer box is entirely feasible and no separate energy constraint is
# needed. Derived rather than hardcoded so it tracks DEFAULT_BATTERY_CELL_COUNT.
MAX_PROPULSION_ENERGY_WH = 100.0
MAX_BATT_CAPACITY_AH = MAX_PROPULSION_ENERGY_WH / battery_nominal_voltage_v(
    DEFAULT_BATTERY_CELL_COUNT
)

OPT_VARS = [
    ("wing_span", (0.914, 1.8288)),
    ("wing_chord", (0.12, 0.40)),
    ("tail_arm", (0.3, 0.9)),
    ("nose_length", (0.08, 0.3)),
    ("extra_shipping_containers", (0, MAX_EXTRA_SHIPPING_CONTAINERS)),
    ("sensor_length_m", (MIN_SENSOR_LENGTH_M, MAX_SENSOR_LENGTH_M)),
    ("sensor_diameter_m", (MIN_SENSOR_DIAMETER_M, MAX_SENSOR_DIAMETER_M)),
    (
        "sensor_weight_kg",
        (MIN_SENSOR_WEIGHT_KG, OPTIMIZER_SENSOR_WEIGHT_UPPER_KG),
    ),
    (
        "mission3_sensor_weight_kg",
        (MIN_MISSION3_SENSOR_WEIGHT_KG, OPTIMIZER_SENSOR_WEIGHT_UPPER_KG),
    ),
    ("batt_capacity", (1.0, MAX_BATT_CAPACITY_AH)),
    ("prop_diameter_in", (10.0, 25.0)),
    # The database reaches 3 in and the P/D >= 0.4 constraint imposes an
    # effective 4 in floor at the 10 in minimum diameter.  A 5 in box bound was
    # therefore an artificial active cap in the latest optimum.
    ("prop_pitch_in", (4.0, 18.0)),
    ("motor_kv", (200.0, 650.0)),
    ("motor_max_power", (1000.0, 3000.0)),
    ("cruise_throttle", (0.5, 1.0)),
    ("mission3_cruise_throttle", (0.5, 1.0)),
]

@dataclass
class DesignVector:
    """Baseline aircraft sizing vector in meters.

    ``tail_arm`` is the leading-edge-to-leading-edge distance from the main
    wing to the common horizontal/vertical-tail leading-edge station.
    """
    # The only four things you actually set
    # Aero geometries
    wing_span: float = 1.181 # [m]
    wing_chord: float = 0.307 # [m]
    tail_arm: float = 0.845 # [m]
    nose_length: float = 0.254 # [m]

    # Mission payloads. There is always one sensor shipping container in M2;
    # this variable controls only the additional container simulators.
    # ``sensor_weight_kg`` is the maximum declared sensor weight used by M2 and
    # the Ground Mission. Mission 3 may fly at any positive weight up to that
    # declared maximum. Both lengths follow from their respective weights.
    extra_shipping_containers: float = 0
    sensor_length_m: float = 6.0 * INCH_M
    sensor_diameter_m: float = SENSOR_DIAMETER_M
    sensor_weight_kg: float = 1.0
    mission3_sensor_weight_kg: float | None = None

    # Prop components
    batt_capacity: float = MAX_BATT_CAPACITY_AH  # [Ah]
    battery_cell_count: int = DEFAULT_BATTERY_CELL_COUNT
    prop_diameter_in: float = 14.0  # [in]
    prop_pitch_in: float = 10.0  # [in]
    motor_kv: float = 335.0  # [RPM/V]
    motor_max_power: float = 2200.0  # [W]
    cruise_throttle: float = 0.90
    mission3_cruise_throttle: float = 0.85

    # Packaging geometry. The mechanical module expands this starting
    # cross-section as needed to enclose the M2 container arrangement.
    # These inputs are not currently included in OPT_VARS.
    fuselage_width: float = FUSELAGE_START_WIDTH
    fuselage_height: float = FUSELAGE_BOX_SIZE
    # A nonpositive value means the constant-width body ends at the wing TE.
    # Mechanical packaging resolves this to its actual aft edge downstream.
    fuselage_box_back_x_m: float = 0.0
    wing_airfoil: str = "naca2412"

    # Derived, do not set manually
    wing_area:        float = field(init=False)
    hstab_area:       float = field(init=False)
    hstab_span:       float = field(init=False)
    hstab_chord:      float = field(init=False)
    vstab_area:       float = field(init=False)
    vstab_span:       float = field(init=False)
    vstab_chord:      float = field(init=False)
    battery_nominal_voltage_v: float = field(init=False)
    batt_energy:      float = field(init=False)
    # Mission 3 flies the SAME physical sensor at a possibly lower weight, so
    # its length is the declared length. Rules 3.1.1 require the sensor to keep
    # the same external geometry for every mission and any added weight to be
    # internal, so deriving a shorter M3 body from a lighter M3 weight (as the
    # previous solid-rod model did) was not physical.
    mission3_sensor_length_m: float = field(init=False)


    def __post_init__(self):
        """Calculates derived parameters and checks for validity."""
        if (
            self.wing_span <= 0
            or self.wing_chord <= 0
            or self.tail_arm <= 0
            or self.nose_length <= 0
            or self.fuselage_width <= 0
            or self.fuselage_height <= 0
            or not np.isfinite(self.fuselage_box_back_x_m)
            or self.fuselage_box_back_x_m < 0
            or not np.isfinite(self.sensor_weight_kg)
            or self.sensor_weight_kg < MIN_SENSOR_WEIGHT_KG
            or not np.isfinite(self.sensor_length_m)
            or self.sensor_length_m < MIN_SENSOR_LENGTH_M
        ):
            raise ValueError(
                "All DesignVector primary dimensions must be positive, "
                f"sensor_weight_kg must be at least {MIN_SENSOR_WEIGHT_KG} kg, "
                f"and sensor_length_m at least {MIN_SENSOR_LENGTH_M} m."
            )
        if self.sensor_weight_kg > maximum_sensor_weight_kg(
            self.sensor_length_m, self.sensor_diameter_m
        ):
            raise ValueError(
                "sensor_weight_kg exceeds a solid steel rod of the declared "
                "length and diameter; the sensor would be denser than steel."
            )
        if self.mission3_sensor_weight_kg is None:
            self.mission3_sensor_weight_kg = float(self.sensor_weight_kg)
        if (
            not np.isfinite(self.mission3_sensor_weight_kg)
            or self.mission3_sensor_weight_kg < MIN_MISSION3_SENSOR_WEIGHT_KG
            or self.mission3_sensor_weight_kg > self.sensor_weight_kg
        ):
            raise ValueError(
                "mission3_sensor_weight_kg must represent at least a 6-inch "
                "sensor and cannot exceed the maximum declared sensor_weight_kg."
            )
        self.mission3_sensor_length_m = float(self.sensor_length_m)
        if (
            not np.isfinite(self.extra_shipping_containers)
            or not (
                0
                <= self.extra_shipping_containers
                <= MAX_EXTRA_SHIPPING_CONTAINERS
            )
        ):
            raise ValueError(
                "extra_shipping_containers must lie in "
                f"[0, {MAX_EXTRA_SHIPPING_CONTAINERS}]."
            )

        self.wing_area   = self.wing_span * self.wing_chord

        self.hstab_area  = V_H * self.wing_area * self.wing_chord / self.tail_arm
        self.hstab_span  = np.sqrt(AR_H * self.hstab_area)
        self.hstab_chord = self.hstab_area / self.hstab_span

        self.vstab_area  = V_V * self.wing_area * self.wing_span / self.tail_arm
        self.vstab_span  = np.sqrt(AR_V * self.vstab_area)
        self.vstab_chord = self.vstab_area / self.vstab_span

        self.battery_cell_count = normalize_battery_cell_count(
            self.battery_cell_count
        )
        self.battery_nominal_voltage_v = battery_nominal_voltage_v(
            self.battery_cell_count
        )
        self.batt_energy = battery_energy_wh(
            self.batt_capacity,
            self.battery_cell_count,
        )
        if self.batt_energy > MAX_PROPULSION_ENERGY_WH + 1e-9:
            raise ValueError("Total propulsion battery energy cannot exceed 100 Wh.")

    def to_array(self) -> np.ndarray:
        """Returns the optimizer variables in the same order as bounds()."""
        return np.array([getattr(self, name) for name, _ in OPT_VARS], dtype=float)

    @staticmethod
    def from_array(x):
        """Builds a design vector from an optimizer array."""
        if len(x) != len(OPT_VARS):
            raise ValueError(f"Input array must have length {len(OPT_VARS)}, but got {len(x)}.")
        kwargs = {name: float(value) for value, (name, _) in zip(x, OPT_VARS)}
        return DesignVector(**kwargs) # type: ignore

    @staticmethod
    def bounds() -> list[tuple[float, float]]:
        """Returns SciPy-style bounds in the same order as to_array()."""
        return [bounds for _, bounds in OPT_VARS]

    @staticmethod
    def opt_names() -> list[str]:
        """Returns the optimizer variable names in array order."""
        return [name for name, _ in OPT_VARS]

    def disp_vars(self, optimization_names: list[str] | None = None) -> str:
        """Returns a formatted display of optimization, fixed, and derived variables."""
        ordered_opt_names = (
            self.opt_names() if optimization_names is None else optimization_names
        )
        unknown_names = set(ordered_opt_names) - set(self.__dataclass_fields__)
        if unknown_names:
            raise ValueError(
                "Unknown optimization variable names: "
                + ", ".join(sorted(unknown_names))
            )
        opt_names = set(ordered_opt_names)
        derived_names = [
            name for name, dataclass_field in self.__dataclass_fields__.items()
            if not dataclass_field.init
        ]
        fixed_names = [
            name for name, dataclass_field in self.__dataclass_fields__.items()
            if dataclass_field.init and name not in opt_names
        ]

        sections = [
            ("--- Optimization Variables ---", ordered_opt_names),
            ("--- Fixed Variables ---", fixed_names),
            ("--- Derived Variables ---", derived_names),
        ]

        lines = []
        for title, names in sections:
            lines.append(f"{title}")
            lines.extend(f"  {name}: {getattr(self, name)}" for name in names)

        return "\n".join(lines)
    
# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------

@dataclass
class ParameterVector:
    """Environmental parameters shared by the analysis modules."""
    gravity = 9.806 # [m/s^2]
    rho = 1.225 # [kg/m^3]
    temp = 20.0 # [C]
    pressure = 101325 # [Pa]


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
        promoted = cls(
            wing_span=design_vector.wing_span * unit_scale,
            wing_chord=design_vector.wing_chord * unit_scale,
            tail_arm=design_vector.tail_arm * unit_scale,
            nose_length=design_vector.nose_length * unit_scale,
            extra_shipping_containers=design_vector.extra_shipping_containers,
            sensor_length_m=design_vector.sensor_length_m * unit_scale,
            sensor_diameter_m=design_vector.sensor_diameter_m * unit_scale,
            sensor_weight_kg=design_vector.sensor_weight_kg,
            mission3_sensor_weight_kg=design_vector.mission3_sensor_weight_kg,
            batt_capacity=design_vector.batt_capacity,
            battery_cell_count=design_vector.battery_cell_count,
            prop_diameter_in=design_vector.prop_diameter_in,
            prop_pitch_in=design_vector.prop_pitch_in,
            motor_kv=design_vector.motor_kv,
            motor_max_power=design_vector.motor_max_power,
            cruise_throttle=design_vector.cruise_throttle,
            mission3_cruise_throttle=design_vector.mission3_cruise_throttle,
            fuselage_width=design_vector.fuselage_width * unit_scale,
            fuselage_height=design_vector.fuselage_height * unit_scale,
            fuselage_box_back_x_m=(
                design_vector.fuselage_box_back_x_m * unit_scale
            ),
            wing_airfoil=design_vector.wing_airfoil,
        )
        promoted.mission3_sensor_length_m = (
            design_vector.mission3_sensor_length_m * unit_scale
        )
        return promoted

    def make_airplane(
        self,
        *,
        name: str = "Design Vector Plane",
        tail_airfoil: str = "naca0012",
        wing_le: tuple[float, float, float] = (0.0, 0.0, 0.0),
        tail_waterline: float = 0.00,
        elevator_deflection=0.0,
        tail_incidence=0.0,
    ) -> asb.Airplane:
        """Builds a simple AeroSandbox airplane from the design-vector geometry."""

        wing_qc_x = wing_le[0] + 0.25 * self.wing_chord
        wing_te_x = wing_le[0] + self.wing_chord
        horizontal_tail_le_x = wing_le[0] + self.tail_arm
        vertical_tail_le_x = wing_le[0] + self.tail_arm
        tail_te_x = max(
            horizontal_tail_le_x + self.hstab_chord,
            vertical_tail_le_x + self.vstab_chord,
        )
        fuselage = self.make_fuselage(
            wing_le_x=wing_le[0],
            wing_te_x=wing_te_x,
            tail_te_x=tail_te_x,
        )

        wing_airfoil_obj = asb.Airfoil(self.wing_airfoil)
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
                    twist=tail_incidence,
                    airfoil=tail_airfoil_obj,
                    control_surfaces=[
                        asb.ControlSurface(
                            name="Elevator",
                            deflection=elevator_deflection,
                            hinge_point=0.75,
                        )
                    ],
                ),
                asb.WingXSec(
                    xyz_le=[horizontal_tail_le_x, self.hstab_span / 2.0, tail_waterline],
                    chord=self.hstab_chord,
                    twist=tail_incidence,
                    airfoil=tail_airfoil_obj,
                    control_surfaces=[
                        asb.ControlSurface(
                            name="Elevator",
                            deflection=elevator_deflection,
                            hinge_point=0.75,
                        )
                    ],
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

        nose_tip_x = wing_le_x - self.nose_length
        nose_transition_x = nose_tip_x + 0.35 * self.nose_length
        box_back_x = max(wing_te_x, self.fuselage_box_back_x_m)
        if box_back_x >= tail_te_x:
            raise ValueError("The constant-width fuselage must end before the tail tip.")
        aft_mid_x = box_back_x + 0.65 * (tail_te_x - box_back_x)

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
                    xyz_c=[box_back_x, 0.0,  -self.fuselage_height / 2.0],
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
