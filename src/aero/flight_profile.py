"""DBF takeoff, lap, climb, turn, and landing time model."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from src.aero.drag_model import sensor_drag_force
from src.vectors import DesignVector, ParameterVector


@dataclass(frozen=True)
class FlightProfileConfig:
    runway_length_m: float = 60.0
    first_turn_altitude_m: float = 200.0 * 0.3048
    straight_length_m: float = 500.0 * 0.3048
    straights_per_lap: int = 4
    turn_angles_rad: tuple[float, ...] = (math.pi, 2.0 * math.pi, math.pi)
    maximum_bank_deg: float = 35.0
    maximum_load_factor: float = 2.5
    roll_time_s: float = 1.5
    liftoff_speed_factor: float = 1.20
    approach_speed_factor: float = 1.30
    touchdown_speed_factor: float = 1.15
    rolling_friction_coefficient: float = 0.03
    braking_friction_coefficient: float = 0.25
    ground_drag_coefficient: float = 0.035
    oswald_efficiency: float = 0.80
    turn_deceleration_m_s2: float = 2.0
    turn_acceleration_m_s2: float = 1.5
    approach_deceleration_m_s2: float = 1.5
    # Reaching 200 ft within the 500 ft first leg geometrically requires at
    # least about 22 degrees even before accounting for takeoff roll.
    maximum_climb_angle_deg: float = 30.0
    usable_battery_fraction: float = 0.85
    climb_energy_efficiency: float = 0.70
    landing_power_fraction: float = 0.15
    integration_points: int = 240


DEFAULT_FLIGHT_PROFILE_CONFIG = FlightProfileConfig()


@dataclass(frozen=True)
class FlightProfileResult:
    feasible: bool
    lap_time_s: float
    takeoff_time_s: float
    takeoff_distance_m: float
    climb_time_s: float
    climb_horizontal_distance_m: float
    turn_time_s: float
    straight_time_s: float
    speed_transition_time_s: float
    landing_time_s: float
    landing_distance_m: float
    turn_speed_m_s: float
    turn_bank_deg: float
    battery_energy_available_wh: float
    mission_energy_required_wh: float
    potential_energy_gained_wh: float
    landing_potential_energy_dissipated_wh: float
    energy_feasible: bool
    reason: str = ""

    @property
    def non_lap_time_s(self) -> float:
        return self.takeoff_time_s + self.climb_time_s + self.landing_time_s


def _failed(reason: str) -> FlightProfileResult:
    return FlightProfileResult(
        feasible=False,
        lap_time_s=1e6,
        takeoff_time_s=1e6,
        takeoff_distance_m=math.inf,
        climb_time_s=1e6,
        climb_horizontal_distance_m=math.inf,
        turn_time_s=1e6,
        straight_time_s=1e6,
        speed_transition_time_s=1e6,
        landing_time_s=1e6,
        landing_distance_m=math.inf,
        turn_speed_m_s=0.0,
        turn_bank_deg=0.0,
        battery_energy_available_wh=0.0,
        mission_energy_required_wh=math.inf,
        potential_energy_gained_wh=math.inf,
        landing_potential_energy_dissipated_wh=math.inf,
        energy_feasible=False,
        reason=reason,
    )


def _thrust(thrust_velocity: tuple[float, float, float], velocity):
    a, b, c = thrust_velocity
    return a * velocity**2 + b * velocity + c


def _drag_estimate(
    design: DesignVector,
    parameters: ParameterVector,
    velocity,
    lift,
    *,
    sensor_deployed: bool,
    config: FlightProfileConfig,
):
    q = 0.5 * parameters.rho * np.maximum(velocity, 0.1) ** 2
    aspect_ratio = design.wing_span**2 / design.wing_area
    parasite = q * design.wing_area * config.ground_drag_coefficient
    induced = lift**2 / (
        q * design.wing_area * math.pi * aspect_ratio * config.oswald_efficiency
    )
    tow = sensor_drag_force(design, parameters, velocity) if sensor_deployed else 0.0
    return parasite + induced + tow


def _takeoff(
    design: DesignVector,
    parameters: ParameterVector,
    thrust_velocity: tuple[float, float, float],
    mass_kg: float,
    stall_speed_m_s: float,
    mission: int,
    config: FlightProfileConfig,
) -> tuple[float, float] | None:
    liftoff_speed = config.liftoff_speed_factor * stall_speed_m_s
    velocity = np.linspace(max(0.05, liftoff_speed / config.integration_points), liftoff_speed, config.integration_points)
    weight = mass_kg * parameters.gravity
    # Lift builds from zero to weight as the airplane accelerates and rotates.
    lift = weight * (velocity / liftoff_speed) ** 2
    drag = _drag_estimate(
        design, parameters, velocity, lift, sensor_deployed=False, config=config
    )
    rolling = config.rolling_friction_coefficient * np.maximum(weight - lift, 0.0)
    acceleration = (_thrust(thrust_velocity, velocity) - drag - rolling) / mass_kg
    if np.any(~np.isfinite(acceleration)) or np.any(acceleration <= 0.0):
        return None
    takeoff_time = float(np.trapezoid(1.0 / acceleration, velocity))
    takeoff_distance = float(np.trapezoid(velocity / acceleration, velocity))
    return takeoff_time, takeoff_distance


def _climb_time(
    design: DesignVector,
    parameters: ParameterVector,
    thrust_velocity: tuple[float, float, float],
    mass_kg: float,
    stall_speed_m_s: float,
    cruise_speed_m_s: float,
    mission: int,
    config: FlightProfileConfig,
) -> tuple[float, float, float] | None:
    climb_speed = min(cruise_speed_m_s, max(1.30 * stall_speed_m_s, 0.8 * cruise_speed_m_s))
    weight = mass_kg * parameters.gravity
    drag = float(_drag_estimate(
        design,
        parameters,
        climb_speed,
        weight,
        sensor_deployed=False,
        config=config,
    ))
    excess_power = (float(_thrust(thrust_velocity, climb_speed)) - drag) * climb_speed
    if excess_power <= 0.0:
        return None
    climb_rate = min(
        excess_power / weight,
        climb_speed * math.sin(math.radians(config.maximum_climb_angle_deg)),
    )
    if climb_rate <= 0.0:
        return None
    climb_time = config.first_turn_altitude_m / climb_rate
    horizontal_speed = math.sqrt(max(0.0, climb_speed**2 - climb_rate**2))
    return climb_time, horizontal_speed * climb_time, climb_speed


def _electrical_power_w(
    flight_time_fit: tuple[float, float, float],
    velocity_m_s: float,
    usable_energy_wh: float,
) -> float | None:
    endurance_s = float(np.polyval(flight_time_fit, velocity_m_s))
    if not math.isfinite(endurance_s) or endurance_s <= 0.0:
        return None
    return usable_energy_wh * 3600.0 / endurance_s


def _lap_phases(
    cruise_speed_m_s: float,
    stall_speed_m_s: float,
    parameters: ParameterVector,
    config: FlightProfileConfig,
) -> tuple[float, float, float, float, float] | None:
    if cruise_speed_m_s <= stall_speed_m_s:
        return None
    structural_corner_speed = math.sqrt(config.maximum_load_factor) * stall_speed_m_s
    turn_speed = min(cruise_speed_m_s, structural_corner_speed)
    available_n = (turn_speed / stall_speed_m_s) ** 2
    if available_n <= 1.0:
        return None
    bank_limit = min(
        math.radians(config.maximum_bank_deg),
        math.acos(1.0 / config.maximum_load_factor),
        math.acos(1.0 / available_n),
    )
    if bank_limit <= 0.0:
        return None

    # Half-cosine roll ramps, matching the supplied course/rope model.
    ramp_t = np.linspace(0.0, config.roll_time_s, 101)
    ramp_bank = bank_limit * 0.5 * (1.0 - np.cos(math.pi * ramp_t / config.roll_time_s))
    ramp_heading = float(np.trapezoid(
        parameters.gravity * np.tan(ramp_bank) / turn_speed,
        ramp_t,
    ))
    maximum_turn_rate = parameters.gravity * math.tan(bank_limit) / turn_speed
    turn_time = 0.0
    for angle in config.turn_angles_rad:
        remaining_heading = angle - 2.0 * ramp_heading
        if remaining_heading < 0.0:
            return None
        turn_time += 2.0 * config.roll_time_s + remaining_heading / maximum_turn_rate

    delta_v = max(0.0, cruise_speed_m_s - turn_speed)
    decel_time = delta_v / config.turn_deceleration_m_s2
    accel_time = delta_v / config.turn_acceleration_m_s2
    transitions = len(config.turn_angles_rad)
    transition_time = transitions * (decel_time + accel_time)
    transition_distance = transitions * (
        0.5 * (cruise_speed_m_s + turn_speed) * (decel_time + accel_time)
    )
    total_straight_distance = config.straights_per_lap * config.straight_length_m
    if transition_distance >= total_straight_distance:
        return None
    straight_time = (total_straight_distance - transition_distance) / cruise_speed_m_s
    return turn_time + transition_time + straight_time, turn_time, straight_time, transition_time, turn_speed


def _landing(
    design: DesignVector,
    parameters: ParameterVector,
    mass_kg: float,
    stall_speed_m_s: float,
    cruise_speed_m_s: float,
    config: FlightProfileConfig,
) -> tuple[float, float]:
    approach_speed = config.approach_speed_factor * stall_speed_m_s
    touchdown_speed = config.touchdown_speed_factor * stall_speed_m_s
    approach_transition_time = abs(cruise_speed_m_s - approach_speed) / config.approach_deceleration_m_s2
    velocity = np.linspace(touchdown_speed, 0.05, config.integration_points)
    weight = mass_kg * parameters.gravity
    q = 0.5 * parameters.rho * velocity**2
    drag = q * design.wing_area * config.ground_drag_coefficient
    deceleration = (drag + config.braking_friction_coefficient * weight) / mass_kg
    rollout_time = float(abs(np.trapezoid(1.0 / deceleration, velocity)))
    rollout_distance = float(abs(np.trapezoid(velocity / deceleration, velocity)))
    return approach_transition_time + rollout_time, rollout_distance


def compute_flight_profile(
    cruise_speed_m_s: float,
    stall_speed_m_s: float,
    design: DesignVector,
    parameters: ParameterVector,
    thrust_velocity: tuple[float, float, float],
    mass_kg: float,
    mission: int,
    flight_time_fit: tuple[float, float, float],
    config: FlightProfileConfig = DEFAULT_FLIGHT_PROFILE_CONFIG,
) -> FlightProfileResult:
    """Compute physical phase times for one DBF mission configuration."""
    usable_energy_wh = design.batt_energy * config.usable_battery_fraction
    if mission not in (1, 2, 3):
        return _failed("Mission must be 1, 2, or 3.")
    if min(cruise_speed_m_s, stall_speed_m_s, mass_kg) <= 0.0:
        return _failed("Speed and mass inputs must be positive.")

    takeoff = _takeoff(
        design, parameters, thrust_velocity, mass_kg, stall_speed_m_s, mission, config
    )
    if takeoff is None:
        return _failed("Insufficient positive acceleration during takeoff roll.")
    takeoff_time, takeoff_distance = takeoff
    if takeoff_distance > config.runway_length_m:
        return replace(
            _failed(
                f"Takeoff distance {takeoff_distance:.1f} m exceeds the 60 m runway."
            ),
            takeoff_time_s=takeoff_time,
            takeoff_distance_m=takeoff_distance,
            battery_energy_available_wh=usable_energy_wh,
        )

    climb = _climb_time(
        design, parameters, thrust_velocity, mass_kg, stall_speed_m_s,
        cruise_speed_m_s, mission, config
    )
    if climb is None:
        return replace(
            _failed("Insufficient excess power to reach the first-turn altitude."),
            takeoff_time_s=takeoff_time,
            takeoff_distance_m=takeoff_distance,
            battery_energy_available_wh=usable_energy_wh,
        )
    climb_time, climb_horizontal_distance, climb_speed = climb
    distance_before_first_turn = takeoff_distance + climb_horizontal_distance
    if distance_before_first_turn > config.straight_length_m:
        return replace(
            _failed(
                "The airplane does not reach 200 ft before the first turn: "
                f"it needs {distance_before_first_turn:.1f} m, but only "
                f"{config.straight_length_m:.1f} m is available."
            ),
            takeoff_time_s=takeoff_time,
            takeoff_distance_m=takeoff_distance,
            climb_time_s=climb_time,
            climb_horizontal_distance_m=climb_horizontal_distance,
            battery_energy_available_wh=usable_energy_wh,
        )
    lap = _lap_phases(cruise_speed_m_s, stall_speed_m_s, parameters, config)
    if lap is None:
        return _failed("The requested turn/transition profile is infeasible.")
    lap_time, turn_time, straight_time, transition_time, turn_speed = lap
    landing_time, landing_distance = _landing(
        design, parameters, mass_kg, stall_speed_m_s, cruise_speed_m_s, config
    )
    if landing_distance > config.runway_length_m:
        return _failed(
            f"Landing rollout {landing_distance:.1f} m exceeds the 60 m runway."
        )

    liftoff_speed = config.liftoff_speed_factor * stall_speed_m_s
    approach_speed = config.approach_speed_factor * stall_speed_m_s
    representative_transition_speed = 0.5 * (cruise_speed_m_s + turn_speed)
    phase_speeds = (
        0.5 * liftoff_speed,
        climb_speed,
        cruise_speed_m_s,
        turn_speed,
        representative_transition_speed,
        approach_speed,
    )
    phase_powers = tuple(
        _electrical_power_w(flight_time_fit, speed, usable_energy_wh)
        for speed in phase_speeds
    )
    if any(power is None for power in phase_powers):
        return _failed("Propulsion endurance is invalid at a required flight phase.")
    takeoff_power, climb_power, straight_power, turn_power, transition_power, landing_power = phase_powers
    lap_energy_wh = (
        straight_power * straight_time
        + turn_power * turn_time
        + transition_power * transition_time
    ) / 3600.0
    potential_energy_wh = (
        mass_kg * parameters.gravity * config.first_turn_altitude_m / 3600.0
    )
    overhead_energy_wh = (
        takeoff_power * takeoff_time
        + climb_power * climb_time
        + config.landing_power_fraction * landing_power * landing_time
    ) / 3600.0
    if mission == 1:
        mission_energy_wh = overhead_energy_wh + 3.0 * lap_energy_wh
    elif mission == 2:
        mission_energy_wh = overhead_energy_wh + 5.0 * lap_energy_wh
    else:
        usable_lap_time = max(0.0, 300.0 - takeoff_time - climb_time - landing_time)
        mission_energy_wh = overhead_energy_wh + usable_lap_time * lap_energy_wh / lap_time
    energy_feasible = mission_energy_wh <= usable_energy_wh

    available_n = (turn_speed / stall_speed_m_s) ** 2
    turn_bank = min(
        math.radians(config.maximum_bank_deg),
        math.acos(1.0 / config.maximum_load_factor),
        math.acos(1.0 / available_n),
    )
    return FlightProfileResult(
        feasible=energy_feasible,
        lap_time_s=lap_time,
        takeoff_time_s=takeoff_time,
        takeoff_distance_m=takeoff_distance,
        climb_time_s=climb_time,
        climb_horizontal_distance_m=climb_horizontal_distance,
        turn_time_s=turn_time,
        straight_time_s=straight_time,
        speed_transition_time_s=transition_time,
        landing_time_s=landing_time,
        landing_distance_m=landing_distance,
        turn_speed_m_s=turn_speed,
        turn_bank_deg=math.degrees(turn_bank),
        battery_energy_available_wh=usable_energy_wh,
        mission_energy_required_wh=mission_energy_wh,
        potential_energy_gained_wh=potential_energy_wh,
        landing_potential_energy_dissipated_wh=potential_energy_wh,
        energy_feasible=energy_feasible,
        reason=(
            ""
            if energy_feasible
            else f"Mission needs {mission_energy_wh:.1f} Wh but only "
            f"{usable_energy_wh:.1f} Wh is usable."
        ),
    )


__all__ = [
    "DEFAULT_FLIGHT_PROFILE_CONFIG",
    "FlightProfileConfig",
    "FlightProfileResult",
    "compute_flight_profile",
]
