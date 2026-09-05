"""Takeoff, climb, and mission-energy checks for propulsion sizing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from src.aero.drag_model import drag_coefficients, fuselage_drag_geometry, sensor_drag_force
from src.prop.continuous_prop_database import ContinuousPropDatabase
from src.prop.prop_cruise_values import solve_cruise_samples
from src.prop.prop_helper_functions import make_battery_from_design, make_motor_from_design
from src.vectors import DesignVector, ParameterVector


@dataclass(frozen=True)
class PropulsionRequirements:
    """Editable operational requirements used by the optimizer."""

    maximum_takeoff_distance_m: float = 60.0
    minimum_climb_rate_mps: float = 2.0
    climb_altitude_m: float = 15.24  # 50 ft
    liftoff_stall_speed_factor: float = 1.20
    climb_stall_speed_factor: float = 1.30
    rolling_friction_coefficient: float = 0.04
    takeoff_lift_coefficient_fraction: float = 0.80
    reacceleration_efficiency: float = 0.65
    usable_energy_margin_fraction: float = 0.05
    maximum_propeller_tip_mach: float = 0.75
    speed_of_sound_mps: float = 343.0

    def __post_init__(self) -> None:
        positive = (
            self.maximum_takeoff_distance_m,
            self.minimum_climb_rate_mps,
            self.climb_altitude_m,
            self.liftoff_stall_speed_factor,
            self.climb_stall_speed_factor,
            self.reacceleration_efficiency,
            self.maximum_propeller_tip_mach,
            self.speed_of_sound_mps,
        )
        if not np.all(np.isfinite(positive)) or np.any(np.asarray(positive) <= 0.0):
            raise ValueError("Propulsion requirement values must be finite and positive.")
        fractions = (
            self.rolling_friction_coefficient,
            self.takeoff_lift_coefficient_fraction,
            self.reacceleration_efficiency,
            self.usable_energy_margin_fraction,
        )
        if not all(0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("Propulsion requirement fractions must lie in [0, 1].")


DEFAULT_PROPULSION_REQUIREMENTS = PropulsionRequirements()


@dataclass(frozen=True)
class MissionPropulsionPerformance:
    mission: int
    feasible: bool
    penalty: float
    static_thrust_n: float
    static_thrust_to_weight: float
    liftoff_speed_mps: float
    takeoff_distance_m: float
    optimistic_takeoff_distance_lower_bound_m: float
    takeoff_screened_early: bool
    takeoff_time_s: float
    climb_speed_mps: float
    climb_rate_mps: float
    maximum_propeller_tip_mach: float
    cruise_power_w: float
    takeoff_energy_wh: float
    climb_energy_wh: float
    reacceleration_energy_wh: float
    cruise_energy_wh: float
    required_energy_wh: float
    allowed_energy_wh: float
    energy_margin_wh: float
    takeoff_distance_margin_m: float
    climb_rate_margin_mps: float
    limiting_constraint: str

    def to_dict(self) -> dict:
        return asdict(self)


def _drag_n(
    design: DesignVector,
    parameters: ParameterVector,
    speed_mps: np.ndarray | float,
    weight_n: float,
    mission: int,
    *,
    lift_coefficient: np.ndarray | float | None = None,
):
    speed = np.asarray(speed_mps, dtype=float)
    q = 0.5 * parameters.rho * np.maximum(speed, 0.1) ** 2
    if lift_coefficient is None:
        lift_coefficient = weight_n / (q * design.wing_area)
    coefficients = drag_coefficients(
        design,
        parameters,
        speed,
        lift_coefficient,
        0.0,
        fuselage_drag_geometry(design),
    )
    drag = q * design.wing_area * sum(coefficients.values())
    if mission == 3:
        drag += sensor_drag_force(design, parameters, speed)
    return np.asarray(drag, dtype=float)


def _penalty_and_limit(
    takeoff_distance_m: float,
    climb_rate_mps: float,
    required_energy_wh: float,
    allowed_energy_wh: float,
    propeller_tip_mach: float,
    requirements: PropulsionRequirements,
) -> tuple[float, str]:
    violations = {
        "takeoff_distance": max(
            0.0,
            takeoff_distance_m / requirements.maximum_takeoff_distance_m - 1.0,
        ),
        "climb_rate": max(
            0.0,
            1.0 - climb_rate_mps / requirements.minimum_climb_rate_mps,
        ),
        "mission_energy": max(0.0, required_energy_wh / allowed_energy_wh - 1.0),
        "propeller_tip_mach": max(
            0.0,
            propeller_tip_mach / requirements.maximum_propeller_tip_mach - 1.0,
        ),
    }
    limiting = max(violations, key=violations.get)
    worst = violations[limiting]
    return 10.0 * (1.0 - math.exp(-2.0 * worst)), limiting


def evaluate_mission_propulsion(
    design: DesignVector,
    parameters: ParameterVector,
    *,
    mission: int,
    mass_kg: float,
    cruise_speed_mps: float,
    stall_speed_mps: float,
    lap_time_s: float,
    prop_database: ContinuousPropDatabase,
    requirements: PropulsionRequirements = DEFAULT_PROPULSION_REQUIREMENTS,
) -> MissionPropulsionPerformance:
    """Evaluate full-throttle takeoff/climb and complete-mission energy."""
    if mission not in (1, 2, 3):
        raise ValueError("mission must be 1, 2, or 3.")
    values = (mass_kg, cruise_speed_mps, stall_speed_mps, lap_time_s)
    if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0.0):
        raise ValueError("Mass, speeds, and lap time must be finite and positive.")

    motor = make_motor_from_design(design, parameters)
    battery = make_battery_from_design(design, parameters)
    liftoff_speed = requirements.liftoff_stall_speed_factor * stall_speed_mps
    climb_speed = requirements.climb_stall_speed_factor * stall_speed_mps
    current_limit = min(motor.max_current, battery.get_max_current())
    static = solve_cruise_samples(
        design.prop_diameter_in,
        design.prop_pitch_in,
        (0.01,),
        motor,
        battery,
        current_limit,
        1.0,
        prop_database,
        min_rpm=3000,
        max_rpm=20000,
        rpm_step=100,
    )
    weight_n = mass_kg * parameters.gravity
    static_thrust = float(static.thrust_samples_n[0])
    static_tip_mach = (
        math.pi
        * design.prop_diameter_in
        * 0.0254
        * float(static.selected_rpm[0])
        / 60.0
        / requirements.speed_of_sound_mps
    )
    if static_thrust <= 0.0 or static.failed_mask[0]:
        optimistic_takeoff_distance = math.inf
    else:
        optimistic_takeoff_distance = (
            mass_kg * liftoff_speed**2 / (2.0 * static_thrust)
        )
    # This deliberately ignores drag, rolling resistance, and thrust loss with
    # airspeed. Failure is therefore conclusive, while passing only authorizes
    # the more detailed (and more expensive) calculation below.
    if optimistic_takeoff_distance > requirements.maximum_takeoff_distance_m:
        violation = (
            optimistic_takeoff_distance / requirements.maximum_takeoff_distance_m
            - 1.0
        )
        penalty = 10.0 * (1.0 - math.exp(-2.0 * violation))
        allowed_energy = (
            battery.vnom
            * battery.get_useable_capacity()
            * (1.0 - requirements.usable_energy_margin_fraction)
        )
        return MissionPropulsionPerformance(
            mission=mission,
            feasible=False,
            penalty=penalty,
            static_thrust_n=static_thrust,
            static_thrust_to_weight=static_thrust / weight_n,
            liftoff_speed_mps=liftoff_speed,
            takeoff_distance_m=optimistic_takeoff_distance,
            optimistic_takeoff_distance_lower_bound_m=optimistic_takeoff_distance,
            takeoff_screened_early=True,
            takeoff_time_s=math.inf,
            climb_speed_mps=climb_speed,
            climb_rate_mps=0.0,
            maximum_propeller_tip_mach=static_tip_mach,
            cruise_power_w=0.0,
            takeoff_energy_wh=math.inf,
            climb_energy_wh=math.inf,
            reacceleration_energy_wh=math.inf,
            cruise_energy_wh=math.inf,
            required_energy_wh=math.inf,
            allowed_energy_wh=allowed_energy,
            energy_margin_wh=-math.inf,
            takeoff_distance_margin_m=(
                requirements.maximum_takeoff_distance_m
                - optimistic_takeoff_distance
            ),
            climb_rate_margin_mps=-requirements.minimum_climb_rate_mps,
            limiting_constraint="takeoff_distance_lower_bound",
        )

    takeoff_speeds = np.linspace(0.01, liftoff_speed, 41)
    sample_speeds = np.unique(
        np.concatenate((takeoff_speeds, (climb_speed, cruise_speed_mps)))
    )
    full = solve_cruise_samples(
        design.prop_diameter_in,
        design.prop_pitch_in,
        sample_speeds,
        motor,
        battery,
        current_limit,
        1.0,
        prop_database,
        min_rpm=3000,
        max_rpm=20000,
        rpm_step=100,
    )
    cruise_throttle = (
        design.cruise_throttle if mission in (1, 2) else design.mission3_cruise_throttle
    )
    cruise = solve_cruise_samples(
        design.prop_diameter_in,
        design.prop_pitch_in,
        (cruise_speed_mps,),
        motor,
        battery,
        current_limit,
        cruise_throttle,
        prop_database,
        min_rpm=3000,
        max_rpm=20000,
        rpm_step=100,
    )

    wing_ar = design.wing_span**2 / design.wing_area
    cl_max = 1.45 * wing_ar / (wing_ar + 2.0)
    takeoff_cl = requirements.takeoff_lift_coefficient_fraction * cl_max
    takeoff_q = 0.5 * parameters.rho * takeoff_speeds**2
    lift_n = np.minimum(
        0.98 * weight_n,
        takeoff_q * design.wing_area * takeoff_cl,
    )
    drag_n = _drag_n(
        design,
        parameters,
        takeoff_speeds,
        weight_n,
        mission,
        lift_coefficient=takeoff_cl,
    )
    takeoff_indices = np.searchsorted(sample_speeds, takeoff_speeds)
    takeoff_thrust_n = full.thrust_samples_n[takeoff_indices]
    takeoff_power_w = full.selected_power_w[takeoff_indices]
    net_force_n = (
        takeoff_thrust_n
        - drag_n
        - requirements.rolling_friction_coefficient * (weight_n - lift_n)
    )
    acceleration = net_force_n / mass_kg
    if np.any(acceleration <= 0.0) or np.any(full.failed_mask[takeoff_indices]):
        takeoff_distance = math.inf
        takeoff_time = math.inf
        takeoff_energy = math.inf
    else:
        inverse_acceleration = 1.0 / acceleration
        takeoff_distance = float(np.trapezoid(takeoff_speeds * inverse_acceleration, takeoff_speeds))
        takeoff_time = float(np.trapezoid(inverse_acceleration, takeoff_speeds))
        takeoff_energy = float(
            np.trapezoid(takeoff_power_w * inverse_acceleration, takeoff_speeds)
            / 3600.0
        )

    climb_index = int(np.searchsorted(sample_speeds, climb_speed))
    climb_thrust_n = float(full.thrust_samples_n[climb_index])
    climb_power_w = float(full.selected_power_w[climb_index])
    climb_drag_n = float(
        _drag_n(design, parameters, climb_speed, weight_n, mission)
    )
    climb_rate = max(0.0, (climb_thrust_n - climb_drag_n) * climb_speed / weight_n)
    if climb_rate <= 0.0 or full.failed_mask[climb_index]:
        climb_energy = math.inf
    else:
        climb_energy = climb_power_w * requirements.climb_altitude_m / climb_rate / 3600.0

    cruise_power = float(cruise.selected_power_w[0])
    mission_laps = {1: 3, 2: 5, 3: max(1, int(300.0 // lap_time_s))}[mission]
    cruise_duration_s = {1: 3.0 * lap_time_s, 2: 5.0 * lap_time_s, 3: 300.0}[mission]
    cruise_energy = cruise_power * cruise_duration_s / 3600.0
    turn_speed = min(cruise_speed_mps, math.sqrt(2.5) * stall_speed_mps)
    delta_ke_j = 0.5 * mass_kg * max(0.0, cruise_speed_mps**2 - turn_speed**2)
    reacceleration_energy = (
        3.0 * mission_laps * delta_ke_j
        / requirements.reacceleration_efficiency
        / 3600.0
    )
    required_energy = takeoff_energy + climb_energy + reacceleration_energy + cruise_energy
    usable_energy = battery.vnom * battery.get_useable_capacity()
    allowed_energy = usable_energy * (1.0 - requirements.usable_energy_margin_fraction)
    propeller_tip_speed_mps = (
        math.pi
        * design.prop_diameter_in
        * 0.0254
        * float(np.max(full.selected_rpm))
        / 60.0
    )
    propeller_tip_mach = propeller_tip_speed_mps / requirements.speed_of_sound_mps
    penalty, limiting = _penalty_and_limit(
        takeoff_distance,
        climb_rate,
        required_energy,
        allowed_energy,
        propeller_tip_mach,
        requirements,
    )
    feasible = (
        takeoff_distance <= requirements.maximum_takeoff_distance_m
        and climb_rate >= requirements.minimum_climb_rate_mps
        and required_energy <= allowed_energy
        and propeller_tip_mach <= requirements.maximum_propeller_tip_mach
    )
    return MissionPropulsionPerformance(
        mission=mission,
        feasible=feasible,
        penalty=penalty,
        static_thrust_n=static_thrust,
        static_thrust_to_weight=static_thrust / weight_n,
        liftoff_speed_mps=liftoff_speed,
        takeoff_distance_m=takeoff_distance,
        optimistic_takeoff_distance_lower_bound_m=optimistic_takeoff_distance,
        takeoff_screened_early=False,
        takeoff_time_s=takeoff_time,
        climb_speed_mps=climb_speed,
        climb_rate_mps=climb_rate,
        maximum_propeller_tip_mach=propeller_tip_mach,
        cruise_power_w=cruise_power,
        takeoff_energy_wh=takeoff_energy,
        climb_energy_wh=climb_energy,
        reacceleration_energy_wh=reacceleration_energy,
        cruise_energy_wh=cruise_energy,
        required_energy_wh=required_energy,
        allowed_energy_wh=allowed_energy,
        energy_margin_wh=allowed_energy - required_energy,
        takeoff_distance_margin_m=requirements.maximum_takeoff_distance_m - takeoff_distance,
        climb_rate_margin_mps=climb_rate - requirements.minimum_climb_rate_mps,
        limiting_constraint=limiting,
    )


def propulsion_margin_bonus(
    results: tuple[MissionPropulsionPerformance, ...],
    requirements: PropulsionRequirements = DEFAULT_PROPULSION_REQUIREMENTS,
) -> float:
    """Small optimizer-only tie-breaker; never changes official scores."""
    if not results or not all(result.feasible for result in results):
        return 0.0
    takeoff = min(
        1.0,
        min(
            1.0 - result.takeoff_distance_m / requirements.maximum_takeoff_distance_m
            for result in results
        ),
    )
    climb = min(
        1.0,
        min(
            result.climb_rate_mps / requirements.minimum_climb_rate_mps - 1.0
            for result in results
        ),
    )
    energy = min(
        1.0,
        min(result.energy_margin_wh / result.allowed_energy_wh for result in results) / 0.20,
    )
    return 0.05 * (
        0.4 * max(0.0, takeoff)
        + 0.3 * max(0.0, climb)
        + 0.3 * max(0.0, energy)
    )


__all__ = [
    "DEFAULT_PROPULSION_REQUIREMENTS",
    "MissionPropulsionPerformance",
    "PropulsionRequirements",
    "evaluate_mission_propulsion",
    "propulsion_margin_bonus",
]
