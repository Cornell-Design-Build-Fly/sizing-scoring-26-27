"""Data models and configuration for the mechanical mass-properties module.

Coordinate convention
---------------------
The mechanical module uses the same body-axis convention as the geometry in
``src.vectors``:

* ``x`` is positive aft.
* ``y`` is positive toward the right wing.
* ``z`` is positive upward.
* The main-wing root leading edge is ``(0, 0, 0)``.

All distances are in metres, masses in kilograms, and inertia values in
``kg m^2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from src.mech.electronics import (
    ElectronicsLayout,
    ElectronicsPackagingConfig,
)

ALL_MISSIONS = frozenset({"M1", "M2", "M3"})


def _vector3(value: Iterable[float], *, name: str) -> np.ndarray:
    array = np.asarray(tuple(value), dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values; got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True)
class MassItem:
    """One mass element in the airplane component ledger.

    ``dimensions_m`` are the axis-aligned full dimensions ``(length_x,
    width_y, height_z)``. They are used to estimate each component's intrinsic
    inertia as a rectangular prism. Set all three dimensions to zero to model
    an item as a point mass. A custom intrinsic tensor may instead be supplied
    through ``intrinsic_inertia_kg_m2``.
    """

    name: str
    mass_kg: float
    position_m: tuple[float, float, float] | np.ndarray
    missions: frozenset[str] = ALL_MISSIONS
    dimensions_m: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0)
    category: str = "unspecified"
    notes: str = ""
    intrinsic_inertia_kg_m2: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MassItem.name cannot be empty.")
        if not np.isfinite(self.mass_kg) or self.mass_kg < 0:
            raise ValueError(f"Mass for {self.name!r} must be finite and nonnegative.")

        position = _vector3(self.position_m, name=f"{self.name}.position_m")
        dimensions = _vector3(self.dimensions_m, name=f"{self.name}.dimensions_m")
        if np.any(dimensions < 0):
            raise ValueError(f"Dimensions for {self.name!r} must be nonnegative.")

        missions = frozenset(self.missions)
        invalid_missions = missions.difference(ALL_MISSIONS)
        if invalid_missions:
            raise ValueError(
                f"MassItem {self.name!r} contains invalid missions: {sorted(invalid_missions)}."
            )

        inertia = self.intrinsic_inertia_kg_m2
        if inertia is not None:
            inertia = np.asarray(inertia, dtype=float)
            if inertia.shape != (3, 3):
                raise ValueError(
                    f"Intrinsic inertia for {self.name!r} must be a 3x3 matrix."
                )
            if not np.all(np.isfinite(inertia)):
                raise ValueError(
                    f"Intrinsic inertia for {self.name!r} must contain finite values."
                )
            if not np.allclose(inertia, inertia.T, atol=1e-12):
                raise ValueError(
                    f"Intrinsic inertia for {self.name!r} must be symmetric."
                )

        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "dimensions_m", dimensions)
        object.__setattr__(self, "missions", missions)
        object.__setattr__(self, "intrinsic_inertia_kg_m2", inertia)

    def intrinsic_inertia(self) -> np.ndarray:
        """Return this item's inertia tensor about its own center of mass."""
        if self.intrinsic_inertia_kg_m2 is not None:
            return self.intrinsic_inertia_kg_m2.copy()

        lx, ly, lz = self.dimensions_m
        mass = self.mass_kg
        return np.diag(
            [
                mass * (ly**2 + lz**2) / 12.0,
                mass * (lx**2 + lz**2) / 12.0,
                mass * (lx**2 + ly**2) / 12.0,
            ]
        )


@dataclass(frozen=True)
class StaticMarginConfig:
    """General static-margin limits and reference target.

    Mission 1 uses ``maximum`` as its one-sided acceptance limit. Mission 2 has
    its own exact placement target in :class:`Mission2Config`.
    """

    minimum: float = 0.10
    target: float = 0.20
    maximum: float = 0.23
    optimizer_penalty_buffer: float = 0.15
    optimizer_penalty_scale: float = 0.15

    def __post_init__(self) -> None:
        if not np.all(
            np.isfinite(
                [
                    self.minimum,
                    self.target,
                    self.maximum,
                    self.optimizer_penalty_buffer,
                    self.optimizer_penalty_scale,
                ]
            )
        ):
            raise ValueError("Static margins must be finite.")
        if not (0 <= self.minimum <= self.target <= self.maximum):
            raise ValueError(
                "Static margins must satisfy 0 <= minimum <= target <= maximum."
            )
        if not np.isclose(self.target, 0.20, rtol=0.0, atol=1e-12):
            raise ValueError("Mission-1 target static margin is fixed at exactly 0.20.")
        if self.optimizer_penalty_buffer < 0:
            raise ValueError("Static-margin optimizer penalty buffer cannot be negative.")
        if self.optimizer_penalty_scale <= 0:
            raise ValueError("Static-margin optimizer penalty scale must be positive.")


@dataclass(frozen=True)
class NeutralPointConfig:
    """Deprecated neutral-point inputs retained for configuration compatibility.

    The formula-sheet aerodynamic-center estimator no longer uses these
    empirical parameters. Existing configurations may continue to provide them
    without changing the result.
    """

    two_dimensional_lift_curve_slope_per_rad: float = 2.0 * np.pi
    wing_oswald_efficiency: float = 0.85
    tail_oswald_efficiency: float = 0.80
    tail_efficiency: float = 0.90
    tail_dynamic_pressure_ratio: float = 0.95
    downwash_gradient: float = 0.35
    fuselage_shift_chords: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "two_dimensional_lift_curve_slope_per_rad": self.two_dimensional_lift_curve_slope_per_rad,
            "wing_oswald_efficiency": self.wing_oswald_efficiency,
            "tail_oswald_efficiency": self.tail_oswald_efficiency,
            "tail_efficiency": self.tail_efficiency,
            "tail_dynamic_pressure_ratio": self.tail_dynamic_pressure_ratio,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0 <= self.downwash_gradient < 1:
            raise ValueError("downwash_gradient must lie in [0, 1).")
        if not np.isfinite(self.fuselage_shift_chords):
            raise ValueError("fuselage_shift_chords must be finite.")


@dataclass(frozen=True)
class BatteryMassModel:
    """Battery-mass regression using capacity and nominal pack voltage.

    The supplied single-cell fit is ``28.4 * capacity_ah + 0.63`` grams.
    Multiplying by ``nominal_voltage_v / 3.7`` accounts for the cell count;
    ``mass_kg`` then converts the result to kilograms for the mass ledger.
    """

    slope_g_per_ah: float = 28.4
    intercept_g: float = 0.63
    cell_nominal_voltage_v: float = 3.7

    def __post_init__(self) -> None:
        values = (self.slope_g_per_ah, self.intercept_g, self.cell_nominal_voltage_v)
        if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0):
            raise ValueError("Battery mass-model values must be finite and positive.")

    def mass_kg(self, capacity_ah: float, nominal_voltage_v: float) -> float:
        """Estimate total battery-pack mass in kilograms."""

        if not np.isfinite(capacity_ah) or capacity_ah <= 0:
            raise ValueError("Battery capacity must be finite and positive.")
        if not np.isfinite(nominal_voltage_v) or nominal_voltage_v <= 0:
            raise ValueError("Battery nominal voltage must be finite and positive.")
        mass_g = (self.slope_g_per_ah * capacity_ah + self.intercept_g) * (
            nominal_voltage_v / self.cell_nominal_voltage_v
        )
        return float(mass_g / 1000.0)


@dataclass(frozen=True)
class MotorMassModel:
    """Quadratic motor-mass regression using motor Kv and maximum power.

    The supplied regression produces mass in grams for Kv in RPM/V and maximum
    power in watts. ``mass_kg`` converts that result to the kilograms used by
    the mechanical component ledger.
    """

    coefficients_g: tuple[float, float, float, float, float, float] = (
        49.1108060785,
        -0.0414442103,
        0.2336039917,
        0.0000566359,
        -0.0001090885,
        -0.0000104182,
    )

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients_g, dtype=float)
        if coefficients.shape != (6,) or not np.all(np.isfinite(coefficients)):
            raise ValueError("Motor mass model requires six finite coefficients.")

    def mass_kg(self, kv_rpm_per_v: float, max_power_w: float) -> float:
        """Estimate motor mass in kilograms."""

        if not np.isfinite(kv_rpm_per_v) or kv_rpm_per_v <= 0:
            raise ValueError("Motor Kv must be finite and positive.")
        if not np.isfinite(max_power_w) or max_power_w <= 0:
            raise ValueError("Motor maximum power must be finite and positive.")

        c0, c_kv, c_power, c_kv2, c_cross, c_power2 = self.coefficients_g
        mass_g = (
            c0
            + c_kv * kv_rpm_per_v
            + c_power * max_power_w
            + c_kv2 * kv_rpm_per_v**2
            + c_cross * kv_rpm_per_v * max_power_w
            + c_power2 * max_power_w**2
        )
        if mass_g < 0:
            raise ValueError(
                "Motor mass regression produced a negative mass; "
                "the requested Kv and power are outside its usable range."
            )
        return float(mass_g / 1000.0)


@dataclass(frozen=True)
class PropellerMassModel:
    """Cubic propeller-mass regression using diameter in inches.

    The supplied polynomial returns grams. ``mass_kg`` converts its result to
    the kilograms used by the mechanical component ledger.
    """

    cubic_g_per_in3: float = 0.0181235
    quadratic_g_per_in2: float = -0.192008
    linear_g_per_in: float = 1.17229
    intercept_g: float = 9.76484

    def __post_init__(self) -> None:
        coefficients = (
            self.cubic_g_per_in3,
            self.quadratic_g_per_in2,
            self.linear_g_per_in,
            self.intercept_g,
        )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("Propeller mass-model coefficients must be finite.")

    def mass_kg(self, diameter_in: float) -> float:
        """Estimate propeller mass in kilograms."""

        if not np.isfinite(diameter_in) or diameter_in <= 0:
            raise ValueError("Propeller diameter must be finite and positive.")
        mass_g = (
            self.cubic_g_per_in3 * diameter_in**3
            + self.quadratic_g_per_in2 * diameter_in**2
            + self.linear_g_per_in * diameter_in
            + self.intercept_g
        )
        if mass_g < 0:
            raise ValueError(
                "Propeller mass regression produced a negative mass; "
                "the requested diameter is outside its usable range."
            )
        return float(mass_g / 1000.0)


@dataclass(frozen=True)
class AirframeMassConfig:
    """Empirical airframe and systems mass model constants."""

    wing_areal_density_kg_m2: float = 0.356 / 0.36258
    # The supplied tail datum is a linear mass density: a 0.259 m-long
    # stabilizer weighs 49 g. It is applied independently to the full span of
    # the horizontal and vertical stabilizers.
    tail_linear_density_kg_m: float = 0.049 / 0.259
    wing_surface_thickness_m: float = 0.020
    tail_surface_thickness_m: float = 0.012

    wing_servo_mass_kg: float = 0.021
    wing_servo_dimensions_m: tuple[float, float, float] = (0.040, 0.020, 0.040)
    wing_servo_chord_fraction: float = 0.50

    # One 21 g servo is included for each stabilizer. Until exact installation
    # coordinates are available, each is placed at the geometric center of its
    # corresponding stabilizer. The location logic is isolated in main_mech.py
    # so measured positions can be substituted later without changing the mass
    # model or public API.
    tail_servo_mass_kg: float = 0.021
    tail_servo_dimensions_m: tuple[float, float, float] = (0.040, 0.020, 0.040)

    wing_integration_mass_kg: float = 0.100
    wing_integration_dimensions_m: tuple[float, float, float] = (0.080, 0.080, 0.060)

    spar_linear_density_kg_m: float = 0.202 / 1.18
    wing_spar_chord_fraction: float = 0.30
    spar_cross_section_m: tuple[float, float] = (0.020, 0.020)

    tail_integration_mass_kg: float = 0.025
    tail_integration_dimensions_m: tuple[float, float, float] = (0.060, 0.060, 0.050)

    # A 0.5 m fuselage with a 0.457 m cross-sectional perimeter weighs 0.300 kg.
    # Treat this as a shell-area density so the structural mass changes with
    # both fuselage length and cross-sectional size.
    fuselage_shell_areal_density_kg_m2: float = 0.300 / (0.5 * 0.457)

    landing_gear_mass_kg: float = 0.220
    # Vertical distance from the main-wing plane to the landing-gear center.
    landing_gear_vertical_offset_m: float = 4.0 * 0.0254
    landing_gear_dimensions_m: tuple[float, float, float] = (0.080, 0.180, 0.080)

    # Motor and propeller masses are evaluated directly from DesignVector
    # propulsion inputs using their supplied regressions.
    motor_mass_model: MotorMassModel = field(default_factory=MotorMassModel)
    propeller_mass_model: PropellerMassModel = field(
        default_factory=PropellerMassModel
    )
    esc_mass_kg: float = 0.118
    other_electronics_mass_kg: float = 0.050
    battery_model: BatteryMassModel = field(default_factory=BatteryMassModel)
    electronics_packaging: ElectronicsPackagingConfig = field(
        default_factory=ElectronicsPackagingConfig
    )
    # Deprecated compatibility inputs. The electronics envelope and its
    # three-inch vertical station are now authoritative, so non-default
    # overrides are rejected instead of producing contradictory geometry.
    electronics_dimensions_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    electronics_y_m: float = 0.0
    electronics_z_m: float | None = None
    # Optional user-imposed packaging bounds. ``None`` means unbounded: the
    # module uses the exact equivalent electronics-group CG required to hit the
    # target M1 static margin, even when that location lies ahead of the current
    # modeled nose. Supplying a tuple adds a hard feasibility check; it never
    # clips the solved CM away from the exact static-margin target.
    electronics_x_bounds_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        nonnegative_values = {
            "wing_areal_density_kg_m2": self.wing_areal_density_kg_m2,
            "tail_linear_density_kg_m": self.tail_linear_density_kg_m,
            "wing_surface_thickness_m": self.wing_surface_thickness_m,
            "tail_surface_thickness_m": self.tail_surface_thickness_m,
            "wing_servo_mass_kg": self.wing_servo_mass_kg,
            "tail_servo_mass_kg": self.tail_servo_mass_kg,
            "wing_integration_mass_kg": self.wing_integration_mass_kg,
            "spar_linear_density_kg_m": self.spar_linear_density_kg_m,
            "tail_integration_mass_kg": self.tail_integration_mass_kg,
            "fuselage_shell_areal_density_kg_m2": (
                self.fuselage_shell_areal_density_kg_m2
            ),
            "landing_gear_mass_kg": self.landing_gear_mass_kg,
            "landing_gear_vertical_offset_m": self.landing_gear_vertical_offset_m,
            "esc_mass_kg": self.esc_mass_kg,
            "other_electronics_mass_kg": self.other_electronics_mass_kg,
        }
        for name, value in nonnegative_values.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not 0 <= self.wing_servo_chord_fraction <= 1:
            raise ValueError("wing_servo_chord_fraction must lie in [0, 1].")
        if not 0 <= self.wing_spar_chord_fraction <= 1:
            raise ValueError("wing_spar_chord_fraction must lie in [0, 1].")
        for name, dimensions in {
            "wing_servo_dimensions_m": self.wing_servo_dimensions_m,
            "tail_servo_dimensions_m": self.tail_servo_dimensions_m,
            "wing_integration_dimensions_m": self.wing_integration_dimensions_m,
            "tail_integration_dimensions_m": self.tail_integration_dimensions_m,
            "landing_gear_dimensions_m": self.landing_gear_dimensions_m,
            "electronics_dimensions_m": self.electronics_dimensions_m,
        }.items():
            array = np.asarray(dimensions, dtype=float)
            if array.shape != (3,) or not np.all(np.isfinite(array)) or np.any(array < 0):
                raise ValueError(f"{name} must contain three finite nonnegative values.")
        if not np.allclose(self.electronics_dimensions_m, 0.0):
            raise ValueError(
                "electronics_dimensions_m can no longer override the skinny/fat "
                "electronics envelope."
            )
        spar_section = np.asarray(self.spar_cross_section_m, dtype=float)
        if (
            spar_section.shape != (2,)
            or not np.all(np.isfinite(spar_section))
            or np.any(spar_section < 0)
        ):
            raise ValueError("spar_cross_section_m must contain two finite nonnegative values.")
        if not np.isfinite(self.electronics_y_m):
            raise ValueError("electronics_y_m must be finite.")
        if self.electronics_z_m is not None:
            raise ValueError(
                "electronics_z_m can no longer override the required three-inch "
                "station; configure ElectronicsPackagingConfig instead."
            )
        if self.electronics_x_bounds_m is not None:
            lower, upper = self.electronics_x_bounds_m
            if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
                raise ValueError("electronics_x_bounds_m must be finite and increasing.")
        if not isinstance(self.motor_mass_model, MotorMassModel):
            raise ValueError(
                "motor_mass_model must be a MotorMassModel; one-dimensional "
                "motor-mass interpolation is no longer supported."
            )
        if not isinstance(self.propeller_mass_model, PropellerMassModel):
            raise ValueError(
                "propeller_mass_model must be a PropellerMassModel; "
                "propeller-mass interpolation is no longer supported."
            )
        if not isinstance(self.battery_model, BatteryMassModel):
            raise ValueError(
                "battery_model must be a BatteryMassModel; capacity-only "
                "battery-mass interpolation is no longer supported."
            )

    def electronics_component_masses_kg(
        self,
        battery_capacity_ah: float,
        battery_nominal_voltage_v: float,
        motor_kv_rpm_per_v: float,
        motor_max_power_w: float,
        propeller_diameter_in: float,
    ) -> tuple[tuple[str, float], ...]:
        """Resolve the permanent electronics ledger without double-counting."""

        components: list[tuple[str, float]] = [
            (
                "Battery",
                self.battery_model.mass_kg(
                    battery_capacity_ah, battery_nominal_voltage_v
                ),
            )
        ]
        components.extend(
            [
                (
                    "Motor",
                    self.motor_mass_model.mass_kg(
                        motor_kv_rpm_per_v, motor_max_power_w
                    ),
                ),
                (
                    "Propeller",
                    self.propeller_mass_model.mass_kg(propeller_diameter_in),
                ),
            ]
        )
        components.extend(
            [
                ("ESC", self.esc_mass_kg),
                ("Other electronics", self.other_electronics_mass_kg),
            ]
        )
        return tuple(components)

    def electronics_mass_kg(
        self,
        battery_capacity_ah: float,
        battery_nominal_voltage_v: float,
        motor_kv_rpm_per_v: float,
        motor_max_power_w: float,
        propeller_diameter_in: float,
    ) -> float:
        return float(
            sum(
                mass
                for _, mass in self.electronics_component_masses_kg(
                    battery_capacity_ah,
                    battery_nominal_voltage_v,
                    motor_kv_rpm_per_v,
                    motor_max_power_w,
                    propeller_diameter_in,
                )
            )
        )


@dataclass(frozen=True)
class SensorConfig:
    """Shared physical model for the M2/M3 sensor and release mechanism."""

    diameter_m: float = 3.0 * 0.0254
    steel_density_kg_m3: float = 7850.0
    release_mechanism_mass_ratio: float = 1.0 / 20.0

    def __post_init__(self) -> None:
        values = (
            self.diameter_m,
            self.steel_density_kg_m3,
            self.release_mechanism_mass_ratio,
        )
        if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0):
            raise ValueError("Sensor configuration values must be finite and positive.")

    def length_m(self, sensor_mass_kg: float) -> float:
        """Return the length of a solid steel rod with the configured diameter."""

        if not np.isfinite(sensor_mass_kg) or sensor_mass_kg <= 0:
            raise ValueError("sensor_mass_kg must be finite and positive.")
        radius_m = 0.5 * self.diameter_m
        cross_section_m2 = np.pi * radius_m**2
        return float(sensor_mass_kg / (self.steel_density_kg_m3 * cross_section_m2))

    def release_mechanism_mass_kg(self, sensor_mass_kg: float) -> float:
        if not np.isfinite(sensor_mass_kg) or sensor_mass_kg <= 0:
            raise ValueError("sensor_mass_kg must be finite and positive.")
        return float(self.release_mechanism_mass_ratio * sensor_mass_kg)


@dataclass(frozen=True)
class Mission2Config:
    """Mission-2 shipping-container geometry and local packing rules."""

    container_width_m: float = 5.0 * 0.0254
    container_height_m: float = 5.0 * 0.0254
    container_length_padding_m: float = 2.0 * 0.0254
    empty_container_mass_kg: float = 0.5 * 0.45359237
    containers_across: int = 3
    maximum_stack: int = 2
    maximum_extra_containers: int = 10
    target_static_margin: float = 0.12
    clearance_m: float = 0.0
    electronics_aft_clearance_m: float = 0.0
    tail_leading_edge_clearance_m: float = 0.0

    def __post_init__(self) -> None:
        positive_values = (
            self.container_width_m,
            self.container_height_m,
            self.container_length_padding_m,
            self.empty_container_mass_kg,
        )
        nonnegative_values = (
            self.clearance_m,
            self.electronics_aft_clearance_m,
            self.tail_leading_edge_clearance_m,
        )
        if not np.all(np.isfinite(positive_values)) or np.any(
            np.asarray(positive_values) <= 0
        ):
            raise ValueError("Mission-2 container values must be finite and positive.")
        if not np.all(np.isfinite(nonnegative_values)) or np.any(
            np.asarray(nonnegative_values) < 0
        ):
            raise ValueError("Mission-2 clearances must be finite and nonnegative.")
        for name, value in {
            "containers_across": self.containers_across,
            "maximum_stack": self.maximum_stack,
            "maximum_extra_containers": self.maximum_extra_containers,
        }.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 <= self.target_static_margin <= 1:
            raise ValueError("Mission-2 target static margin must lie in [0, 1].")

    def container_dimensions_m(self, sensor_length_m: float) -> tuple[float, float, float]:
        if not np.isfinite(sensor_length_m) or sensor_length_m <= 0:
            raise ValueError("sensor_length_m must be finite and positive.")
        return (
            float(sensor_length_m + self.container_length_padding_m),
            self.container_width_m,
            self.container_height_m,
        )


@dataclass(frozen=True)
class Mission3Config:
    """Mission-3 sensor placement options."""

    # The sensor is placed at the installed M1 airplane CG by default.
    center_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        offset = np.asarray(self.center_offset_m, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("center_offset_m must contain three finite values.")


@dataclass(frozen=True)
class MechanicalModuleConfig:
    static_margin: StaticMarginConfig = field(default_factory=StaticMarginConfig)
    neutral_point: NeutralPointConfig = field(default_factory=NeutralPointConfig)
    airframe: AirframeMassConfig = field(default_factory=AirframeMassConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    mission2: Mission2Config = field(default_factory=Mission2Config)
    mission3: Mission3Config = field(default_factory=Mission3Config)

    def __post_init__(self) -> None:
        if not (
            self.static_margin.minimum
            <= self.mission2.target_static_margin
            <= self.static_margin.maximum
        ):
            raise ValueError(
                "Mission-2 target static margin must lie inside the configured "
                "general static-margin range."
            )


@dataclass(frozen=True)
class MissionMassProperties:
    mission: str
    items: tuple[MassItem, ...]
    total_mass_kg: float
    weight_n: float
    cg_m: tuple[float, float, float]
    inertia_tensor_kg_m2: np.ndarray
    static_margin: float
    static_margin_feasible: bool
    placement_feasible: bool = True
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MechanicalResult:
    """Mechanical outputs with explicit starting and resolved geometry."""

    neutral_point_x_m: float
    wing_aerodynamic_center_x_m: float
    horizontal_tail_aerodynamic_center_x_m: float
    target_cg_x_m: float
    acceptable_cg_x_range_m: tuple[float, float]
    electronics_position_m: np.ndarray
    electronics_layout: ElectronicsLayout
    input_fuselage_width_m: float
    input_fuselage_height_m: float
    resolved_fuselage_width_m: float
    resolved_fuselage_height_m: float
    fuselage_width_increases: int
    all_items: tuple[MassItem, ...]
    missions: dict[str, MissionMassProperties]
    warnings: tuple[str, ...] = ()
    penalty: float = 0.0
    penalty_static_margin: float = 0.0

    @property
    def fuselage_width_m(self) -> float:
        """Backward-compatible alias for the resolved fuselage width."""

        return self.resolved_fuselage_width_m

    @property
    def fuselage_height_m(self) -> float:
        """Backward-compatible alias for the resolved fuselage height."""

        return self.resolved_fuselage_height_m

    def for_mission(self, mission: str) -> MissionMassProperties:
        key = mission.upper()
        try:
            return self.missions[key]
        except KeyError as exc:
            raise ValueError(f"Unknown mission {mission!r}; expected M1, M2, or M3.") from exc

    def component_array(self, mission: str | None = None) -> np.ndarray:
        """Return a structured NumPy array of the component mass ledger."""
        if mission is None:
            items = self.all_items
        else:
            mission_key = mission.upper()
            if mission_key not in ALL_MISSIONS:
                raise ValueError(
                    f"Unknown mission {mission!r}; expected M1, M2, or M3."
                )
            items = tuple(item for item in self.all_items if mission_key in item.missions)

        dtype = np.dtype(
            [
                ("name", "U64"),
                ("category", "U32"),
                ("missions", "U16"),
                ("mass_kg", "f8"),
                ("x_m", "f8"),
                ("y_m", "f8"),
                ("z_m", "f8"),
                ("length_x_m", "f8"),
                ("width_y_m", "f8"),
                ("height_z_m", "f8"),
            ]
        )
        records = []
        for item in items:
            records.append(
                (
                    item.name,
                    item.category,
                    ",".join(sorted(item.missions)),
                    item.mass_kg,
                    *item.position_m,
                    *item.dimensions_m,
                )
            )
        return np.array(records, dtype=dtype)
