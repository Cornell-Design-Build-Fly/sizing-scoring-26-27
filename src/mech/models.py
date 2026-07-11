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
    LinearMassModel,
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
    """Static-margin limits, with the M1 target fixed at 20% MAC."""

    minimum: float = 0.10
    target: float = 0.20
    maximum: float = 0.20

    def __post_init__(self) -> None:
        if not np.all(np.isfinite([self.minimum, self.target, self.maximum])):
            raise ValueError("Static margins must be finite.")
        if not (0 <= self.minimum <= self.target <= self.maximum):
            raise ValueError(
                "Static margins must satisfy 0 <= minimum <= target <= maximum."
            )
        if not np.isclose(self.target, 0.20, rtol=0.0, atol=1e-12):
            raise ValueError("Mission-1 target static margin is fixed at exactly 0.20.")


@dataclass(frozen=True)
class NeutralPointConfig:
    """Parameters for the wing-plus-horizontal-tail neutral-point estimate.

    The estimate weights the wing and horizontal-tail aerodynamic centers by
    their finite-wing lift-curve slopes, areas, tail dynamic-pressure ratio,
    tail efficiency, and downwash. This is more appropriate for static margin
    than a simple geometric area average. Fuselage effects can later be
    calibrated with ``fuselage_shift_chords`` using measured or higher-fidelity
    stability data.
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
    """Linear battery-capacity-to-mass model.

    The default line passes through the supplied 4.5 Ah, 0.690 kg baseline and
    the origin. This prevents the optimizer from receiving additional battery
    capacity with no mass penalty. When CUDBF has measured several candidate
    packs, replace ``slope_kg_per_ah`` with a regression slope; the reference
    point guarantees the baseline pack remains exactly 0.690 kg.

    Set ``slope_kg_per_ah=0`` to temporarily recover a fixed 0.690 kg battery.
    """

    reference_capacity_ah: float = 4.5
    reference_mass_kg: float = 0.690
    slope_kg_per_ah: float = 0.690 / 4.5
    minimum_mass_kg: float = 0.0

    def __post_init__(self) -> None:
        values = [
            self.reference_capacity_ah,
            self.reference_mass_kg,
            self.slope_kg_per_ah,
            self.minimum_mass_kg,
        ]
        if not np.all(np.isfinite(values)):
            raise ValueError("Battery mass-model values must be finite.")
        if self.reference_capacity_ah <= 0:
            raise ValueError("reference_capacity_ah must be positive.")
        if (
            self.reference_mass_kg < 0
            or self.minimum_mass_kg < 0
            or self.slope_kg_per_ah < 0
        ):
            raise ValueError("Battery masses and capacity slope cannot be negative.")

    def mass_kg(self, capacity_ah: float) -> float:
        if not np.isfinite(capacity_ah) or capacity_ah <= 0:
            raise ValueError("Battery capacity must be finite and positive.")
        mass = self.reference_mass_kg + self.slope_kg_per_ah * (
            capacity_ah - self.reference_capacity_ah
        )
        return max(float(mass), self.minimum_mass_kg)


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
    spar_cross_section_m: tuple[float, float] = (0.010, 0.010)

    tail_integration_mass_kg: float = 0.025
    tail_integration_dimensions_m: tuple[float, float, float] = (0.060, 0.060, 0.050)

    fuselage_linear_density_kg_m: float = 0.300 / 0.5

    landing_gear_mass_kg: float = 0.220
    landing_gear_vertical_offset_m: float = 4.0 * 0.0254
    landing_gear_dimensions_m: tuple[float, float, float] = (0.080, 0.180, 0.080)

    motor_prop_mass_kg: float = 0.390
    # The supplied baseline combines motor and propeller mass.  Leave these
    # models as ``None`` to use that exact 0.390 kg combined item.  Once two
    # catalogue fits and their sizing inputs are available, supply both models
    # to list the motor and propeller separately without double-counting the
    # combined baseline.
    motor_mass_model: LinearMassModel | None = None
    propeller_mass_model: LinearMassModel | None = None
    motor_sizing_value: float | None = None
    propeller_sizing_value: float | None = None
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
            "fuselage_linear_density_kg_m": self.fuselage_linear_density_kg_m,
            "landing_gear_mass_kg": self.landing_gear_mass_kg,
            "landing_gear_vertical_offset_m": self.landing_gear_vertical_offset_m,
            "motor_prop_mass_kg": self.motor_prop_mass_kg,
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

        interpolation_fields = (
            self.motor_mass_model,
            self.propeller_mass_model,
            self.motor_sizing_value,
            self.propeller_sizing_value,
        )
        if any(value is not None for value in interpolation_fields):
            if any(value is None for value in interpolation_fields):
                raise ValueError(
                    "Motor and propeller interpolation requires both mass models "
                    "and both sizing values."
                )
            if not np.all(
                np.isfinite((self.motor_sizing_value, self.propeller_sizing_value))
            ):
                raise ValueError("Motor and propeller sizing values must be finite.")

    def electronics_component_masses_kg(
        self, battery_capacity_ah: float
    ) -> tuple[tuple[str, float], ...]:
        """Resolve the permanent electronics ledger without double-counting."""

        components: list[tuple[str, float]] = [
            ("Battery", self.battery_model.mass_kg(battery_capacity_ah))
        ]
        if self.motor_mass_model is None:
            components.append(("Motor and propeller", self.motor_prop_mass_kg))
        else:
            # Validation guarantees that the paired values are present.
            motor_model = self.motor_mass_model
            propeller_model = self.propeller_mass_model
            motor_input = self.motor_sizing_value
            propeller_input = self.propeller_sizing_value
            assert propeller_model is not None
            assert motor_input is not None
            assert propeller_input is not None
            components.extend(
                [
                    (
                        "Motor",
                        motor_model.mass_kg(motor_input),
                    ),
                    (
                        "Propeller",
                        propeller_model.mass_kg(propeller_input),
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

    def electronics_mass_kg(self, battery_capacity_ah: float) -> float:
        return float(
            sum(
                mass
                for _, mass in self.electronics_component_masses_kg(
                    battery_capacity_ah
                )
            )
        )


@dataclass(frozen=True)
class PlacementRules:
    """Per-type direction controls retained around the center-out anchor.

    The current fixed-layer process uses ``allow_forward`` and ``allow_aft``.
    The older vertical/stacking flags remain in the public configuration for
    compatibility; vertical ordering is now set by ``RelativePayloadRules``.
    """

    allow_forward: bool = True
    allow_aft: bool = True
    allow_above: bool = True
    allow_below: bool = True
    allow_stacking: bool = True

    def __post_init__(self) -> None:
        if not (self.allow_forward or self.allow_aft):
            raise ValueError("At least one of allow_forward or allow_aft must be True.")
        if not (self.allow_above or self.allow_below):
            raise ValueError("At least one of allow_above or allow_below must be True.")


@dataclass(frozen=True)
class RelativePayloadRules:
    """Relative duck/puck ordering for the fixed-layer M2 process.

    The default Mission-2 configuration selects ``pucks_below_ducks``.
    Longitudinal fields are retained for configuration compatibility, but the
    center-out process rejects them because both first items must share the
    starting-CG plane.
    """

    pucks_forward_of_ducks: bool = False
    pucks_aft_of_ducks: bool = False
    pucks_above_ducks: bool = False
    pucks_below_ducks: bool = False

    def __post_init__(self) -> None:
        if self.pucks_forward_of_ducks and self.pucks_aft_of_ducks:
            raise ValueError(
                "Pucks cannot be required both forward and aft of all ducks."
            )
        if self.pucks_above_ducks and self.pucks_below_ducks:
            raise ValueError(
                "Pucks cannot be required both above and below all ducks."
            )


@dataclass(frozen=True)
class PayloadTypeConfig:
    """Mass, bounding box, and placement rules for one M2 payload type."""

    label: str
    mass_kg: float
    dimensions_m: tuple[float, float, float]
    rules: PlacementRules = field(default_factory=PlacementRules)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Payload label cannot be empty.")
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError(f"Payload mass for {self.label!r} must be finite and positive.")
        dimensions = np.asarray(self.dimensions_m, dtype=float)
        if (
            dimensions.shape != (3,)
            or not np.all(np.isfinite(dimensions))
            or np.any(dimensions <= 0)
        ):
            raise ValueError(
                f"Payload dimensions for {self.label!r} must be three positive values."
            )


@dataclass(frozen=True)
class Mission2Config:
    """Mission-2 deterministic center-out payload configuration.

    The first duck and puck share the M1-CG x/y location.  Each type then grows
    outward on its own bounding-box lattice.  The default vertical relationship
    puts ducks three inches below the wing and pucks immediately below them.
    """

    duck: PayloadTypeConfig = field(
        default_factory=lambda: PayloadTypeConfig(
            label="Duck",
            mass_kg=0.018,
            dimensions_m=(0.053, 0.053, 0.053),
        )
    )
    puck: PayloadTypeConfig = field(
        default_factory=lambda: PayloadTypeConfig(
            label="Puck",
            mass_kg=0.170,
            dimensions_m=(0.0762, 0.0762, 0.0254),
        )
    )
    relative_payload_rules: RelativePayloadRules = field(
        default_factory=lambda: RelativePayloadRules(pucks_below_ducks=True)
    )
    compartment_x_bounds_m: tuple[float, float] | None = None
    electronics_aft_clearance_m: float = 0.0
    tail_leading_edge_clearance_m: float = 0.0
    # ``None`` uses the full fuselage width.  A value may make the payload bay
    # narrower, but main_mech always clips it to the actual fuselage sidewalls.
    maximum_width_m: float | None = None
    compartment_center_y_m: float = 0.0
    duck_center_z_m: float = -3.0 * 0.0254
    relative_reference_x_m: float | None = None
    clearance_m: float = 0.0
    vertical_clearance_m: float = 0.0

    def __post_init__(self) -> None:
        scalar_values = [
            self.compartment_center_y_m,
            self.duck_center_z_m,
            self.electronics_aft_clearance_m,
            self.tail_leading_edge_clearance_m,
            self.clearance_m,
            self.vertical_clearance_m,
        ]
        if not np.all(np.isfinite(scalar_values)):
            raise ValueError("Mission-2 scalar configuration values must be finite.")
        if self.maximum_width_m is not None and (
            not np.isfinite(self.maximum_width_m) or self.maximum_width_m <= 0
        ):
            raise ValueError("Mission-2 maximum width must be positive when supplied.")
        if self.electronics_aft_clearance_m < 0 or self.tail_leading_edge_clearance_m < 0:
            raise ValueError("Mission-2 longitudinal keep-out distances cannot be negative.")
        if self.clearance_m < 0 or self.vertical_clearance_m < 0:
            raise ValueError("Mission-2 clearances cannot be negative.")
        if self.compartment_x_bounds_m is not None:
            lower, upper = self.compartment_x_bounds_m
            if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
                raise ValueError("compartment_x_bounds_m must be finite and increasing.")
        if self.relative_reference_x_m is not None:
            raise ValueError(
                "relative_reference_x_m is no longer configurable; Mission 2 "
                "must start at the actual Mission-1 CG plane."
            )


@dataclass(frozen=True)
class Mission3Config:
    """Mission-3 three-mass banner-system model.

    The current-year defaults include two 100 g mechanisms and a banner areal
    density based on a 0.233 kg, 2.9 m^2 reference banner. Banner area is
    ``banner_length_m * banner_height_m``. A fixed ``banner_mass_kg`` overrides
    all density models. ``banner_linear_density_kg_m`` is retained as a legacy
    override for callers that already have a measured mass per unit length.

    A value of ``None`` for ``banner_center_x_m`` lets the module solve for the
    best longitudinal location; otherwise the supplied position is used
    exactly (subject to optional bounds).  The two mechanisms use explicit,
    fixed longitudinal distances from the banner instead of deriving those
    distances from banner height.
    """

    forward_mechanism_mass_kg: float = 0.100
    aft_mechanism_mass_kg: float = 0.100
    banner_mass_kg: float | None = None
    banner_areal_density_kg_m2: float = 0.233 / 2.9
    banner_linear_density_kg_m: float | None = None
    banner_height_m: float = 0.10
    banner_center_x_m: float | None = None
    banner_center_y_m: float = 0.0
    banner_center_z_m: float = 0.0
    banner_center_x_bounds_m: tuple[float, float] | None = None
    forward_mechanism_distance_m: float = 0.075
    aft_mechanism_distance_m: float = 0.075
    forward_mechanism_dimensions_m: tuple[float, float, float] = (0.050, 0.050, 0.050)
    aft_mechanism_dimensions_m: tuple[float, float, float] = (0.050, 0.050, 0.050)
    banner_packed_dimensions_m: tuple[float, float, float] = (0.100, 0.060, 0.060)

    def __post_init__(self) -> None:
        masses = [
            self.forward_mechanism_mass_kg,
            self.aft_mechanism_mass_kg,
            self.banner_areal_density_kg_m2,
        ]
        if self.banner_mass_kg is not None:
            masses.append(self.banner_mass_kg)
        if self.banner_linear_density_kg_m is not None:
            masses.append(self.banner_linear_density_kg_m)
        if not np.all(np.isfinite(masses)) or np.any(np.asarray(masses) < 0):
            raise ValueError("Mission-3 masses and mass density must be finite and nonnegative.")
        distances = (
            self.banner_height_m,
            self.forward_mechanism_distance_m,
            self.aft_mechanism_distance_m,
        )
        if not np.all(np.isfinite(distances)) or np.any(np.asarray(distances) <= 0):
            raise ValueError("Mission-3 banner height and fixed distances must be positive.")
        for value_name, value in {
            "banner_center_x_m": self.banner_center_x_m,
            "banner_center_y_m": self.banner_center_y_m,
            "banner_center_z_m": self.banner_center_z_m,
        }.items():
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{value_name} must be finite when supplied.")
        if self.banner_center_x_bounds_m is not None:
            lower, upper = self.banner_center_x_bounds_m
            if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
                raise ValueError("banner_center_x_bounds_m must be finite and increasing.")
        for name, dimensions in {
            "forward_mechanism_dimensions_m": self.forward_mechanism_dimensions_m,
            "aft_mechanism_dimensions_m": self.aft_mechanism_dimensions_m,
            "banner_packed_dimensions_m": self.banner_packed_dimensions_m,
        }.items():
            array = np.asarray(dimensions, dtype=float)
            if array.shape != (3,) or not np.all(np.isfinite(array)) or np.any(array < 0):
                raise ValueError(f"{name} must contain three finite nonnegative values.")

    def resolved_banner_mass_kg(self, banner_length_m: float) -> float:
        if not np.isfinite(banner_length_m) or banner_length_m < 0:
            raise ValueError("banner_length_m must be finite and nonnegative.")
        if self.banner_mass_kg is not None:
            return float(self.banner_mass_kg)
        if self.banner_linear_density_kg_m is not None:
            return float(self.banner_linear_density_kg_m * banner_length_m)
        banner_area_m2 = banner_length_m * self.banner_height_m
        return float(self.banner_areal_density_kg_m2 * banner_area_m2)


@dataclass(frozen=True)
class MechanicalModuleConfig:
    static_margin: StaticMarginConfig = field(default_factory=StaticMarginConfig)
    neutral_point: NeutralPointConfig = field(default_factory=NeutralPointConfig)
    airframe: AirframeMassConfig = field(default_factory=AirframeMassConfig)
    mission2: Mission2Config = field(default_factory=Mission2Config)
    mission3: Mission3Config = field(default_factory=Mission3Config)


@dataclass(frozen=True)
class MissionMassProperties:
    mission: str
    items: tuple[MassItem, ...]
    total_mass_kg: float
    weight_n: float
    cg_m: np.ndarray
    inertia_tensor_kg_m2: np.ndarray
    static_margin: float
    static_margin_feasible: bool
    placement_feasible: bool = True
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MechanicalResult:
    neutral_point_x_m: float
    wing_aerodynamic_center_x_m: float
    horizontal_tail_aerodynamic_center_x_m: float
    target_cg_x_m: float
    acceptable_cg_x_range_m: tuple[float, float]
    electronics_position_m: np.ndarray
    electronics_layout: ElectronicsLayout
    all_items: tuple[MassItem, ...]
    missions: dict[str, MissionMassProperties]
    warnings: tuple[str, ...] = ()

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
