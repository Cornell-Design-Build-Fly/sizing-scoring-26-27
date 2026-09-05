"""Takeoff, climb, and mission-energy checks for propulsion sizing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from src.aero.aero_score import (
    N_ZS,
    STRAIGHT_LENGTH_M,
    STRAIGHTS_PER_LAP,
    TURN_180_COUNT,
    TURN_180_RAD,
    TURN_360_COUNT,
    TURN_360_RAD,
)
from src.aero.drag_model import drag_coefficients, fuselage_drag_geometry, sensor_drag_force
from src.prop.continuous_prop_database import ContinuousPropDatabase
from src.prop.prop_cruise_values import solve_cruise_samples
from src.prop.prop_helper_functions import make_battery_from_design, make_motor_from_design
from src.vectors import DesignVector, ParameterVector


PROPULSION_INFEASIBLE_BASE_PENALTY = 10.0
PROPULSION_VIOLATION_PENALTY_SCALE = 10.0


@dataclass(frozen=True)
class PropulsionRequirements:
    """Editable operational requirements used by the optimizer."""

    # The rules do not specify a runway limit, but the airplane still needs a
    # finite operating runway. Keep the team's 60 m engineering requirement.
    # The rules also specify no minimum course altitude, so the old 200 ft
    # climb requirement is disabled unless explicitly requested for a study.
    maximum_takeoff_distance_m: float | None = 60.0
    climb_altitude_m: float = 0.0
    climb_distance_m: float | None = None
    climb_distance_includes_takeoff_roll: bool = False
    minimum_climb_rate_mps: float = 2.0
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
            self.minimum_climb_rate_mps,
            self.liftoff_stall_speed_factor,
            self.climb_stall_speed_factor,
            self.reacceleration_efficiency,
            self.maximum_propeller_tip_mach,
            self.speed_of_sound_mps,
        )
        if not np.all(np.isfinite(positive)) or np.any(np.asarray(positive) <= 0.0):
            raise ValueError("Propulsion requirement values must be finite and positive.")
        optional_positive = (
            self.maximum_takeoff_distance_m,
            self.climb_distance_m,
        )
        if any(
            value is not None and (not math.isfinite(value) or value <= 0.0)
            for value in optional_positive
        ):
            raise ValueError("Optional propulsion distances must be positive.")
        if not math.isfinite(self.climb_altitude_m) or self.climb_altitude_m < 0.0:
            raise ValueError("Climb altitude must be finite and nonnegative.")
        if self.climb_altitude_m > 0.0 and self.climb_distance_m is None:
            raise ValueError("A positive climb altitude requires a climb distance.")
        fractions = (
            self.rolling_friction_coefficient,
            self.takeoff_lift_coefficient_fraction,
            self.reacceleration_efficiency,
            self.usable_energy_margin_fraction,
        )
        if not all(0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("Propulsion requirement fractions must lie in [0, 1].")


    @property
    def required_climb_gradient(self) -> float:
        """Altitude gained per unit ground distance, i.e. tan(climb angle)."""

        if self.climb_distance_m is None:
            return 0.0
        return self.climb_altitude_m / self.climb_distance_m


DEFAULT_PROPULSION_REQUIREMENTS = PropulsionRequirements()


@dataclass(frozen=True)
class MissionPropulsionPerformance:
    mission: int
    feasible: bool
    penalty: float
    inertial_mass_kg: float
    supported_mass_kg: float
    operating_points_feasible: bool
    static_thrust_n: float
    static_thrust_to_weight: float
    liftoff_speed_mps: float
    takeoff_distance_m: float
    optimistic_takeoff_distance_lower_bound_m: float
    takeoff_screened_early: bool
    takeoff_time_s: float
    climb_speed_mps: float
    climb_rate_mps: float
    climb_gradient: float
    climb_distance_required_m: float
    climb_distance_allowed_m: float
    maximum_propeller_tip_mach: float
    cruise_power_w: float  # nominal-equivalent battery depletion power
    takeoff_energy_wh: float
    climb_energy_wh: float
    reacceleration_energy_wh: float
    cruise_energy_wh: float
    required_energy_wh: float
    allowed_energy_wh: float
    energy_margin_wh: float
    takeoff_distance_margin_m: float
    climb_rate_margin_mps: float
    climb_distance_margin_m: float
    limiting_constraint: str
    aerodynamic_lap_time_s: float
    modeled_lap_time_s: float
    straight_time_per_lap_s: float
    turn_time_per_lap_s: float
    turn_speed_mps: float
    turn_load_factor: float
    turn_power_w: float  # nominal-equivalent battery depletion power
    straight_energy_wh: float
    turn_energy_wh: float

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


@dataclass(frozen=True)
class _TurnPerformance:
    feasible: bool
    speed_mps: float
    load_factor: float
    angular_rate_rad_s: float
    required_thrust_n: float
    battery_power_w: float
    maximum_rpm: float


def _propulsion_limited_turn(
    design: DesignVector,
    parameters: ParameterVector,
    *,
    mission: int,
    supported_weight_n: float,
    cruise_speed_mps: float,
    stall_speed_mps: float,
    motor,
    battery,
    current_limit_a: float,
    prop_database: ContinuousPropDatabase,
) -> _TurnPerformance:
    """Find the quickest sustainable level turn and its battery power.

    The old course model always used the structural corner speed and load
    factor, even when the propulsion system could not overcome the associated
    induced drag.  This search applies lift, structural, thrust, current,
    voltage, and motor-power limits together.  At the selected force demand it
    then chooses the least-power propeller RPM that sustains the turn.
    """

    maximum_turn_speed = min(
        cruise_speed_mps,
        math.sqrt(N_ZS) * stall_speed_mps,
    )
    minimum_turn_speed = 1.001 * stall_speed_mps
    if maximum_turn_speed <= minimum_turn_speed:
        return _TurnPerformance(False, math.nan, 1.0, 0.0, math.inf, math.inf, 0.0)

    turn_speeds = np.linspace(minimum_turn_speed, maximum_turn_speed, 31)
    maximum_propulsion = solve_cruise_samples(
        design.prop_diameter_in,
        design.prop_pitch_in,
        turn_speeds,
        motor,
        battery,
        current_limit_a,
        1.0,
        prop_database,
        min_rpm=3000,
        max_rpm=20000,
        rpm_step=100,
    )

    lift_limited_load_factor = (turn_speeds / stall_speed_mps) ** 2
    upper_load_factor = np.minimum(N_ZS, lift_limited_load_factor)
    lower_load_factor = np.ones_like(turn_speeds)
    q = 0.5 * parameters.rho * turn_speeds**2

    drag_at_one_g = _drag_n(
        design,
        parameters,
        turn_speeds,
        supported_weight_n,
        mission,
        lift_coefficient=supported_weight_n / (q * design.wing_area),
    )
    sustainable = (
        ~maximum_propulsion.failed_mask
        & (upper_load_factor > 1.0)
        & (drag_at_one_g <= maximum_propulsion.thrust_samples_n)
    )
    if not np.any(sustainable):
        return _TurnPerformance(False, math.nan, 1.0, 0.0, math.inf, math.inf, 0.0)

    # Drag rises monotonically with lift coefficient in this model, so a short
    # vectorized bisection gives the thrust-limited load factor at every speed.
    for _ in range(24):
        midpoint = 0.5 * (lower_load_factor + upper_load_factor)
        midpoint_drag = _drag_n(
            design,
            parameters,
            turn_speeds,
            supported_weight_n,
            mission,
            lift_coefficient=(
                midpoint * supported_weight_n / (q * design.wing_area)
            ),
        )
        can_hold = sustainable & (
            midpoint_drag <= maximum_propulsion.thrust_samples_n
        )
        lower_load_factor = np.where(can_hold, midpoint, lower_load_factor)
        upper_load_factor = np.where(can_hold, upper_load_factor, midpoint)

    angular_rates = np.zeros_like(turn_speeds)
    angular_rates[sustainable] = (
        parameters.gravity
        * np.sqrt(np.maximum(lower_load_factor[sustainable] ** 2 - 1.0, 0.0))
        / turn_speeds[sustainable]
    )
    best_index = int(np.argmax(angular_rates))
    if angular_rates[best_index] <= 0.0:
        return _TurnPerformance(False, math.nan, 1.0, 0.0, math.inf, math.inf, 0.0)

    turn_speed = float(turn_speeds[best_index])
    turn_load_factor = float(lower_load_factor[best_index])
    turn_q = float(q[best_index])
    required_thrust = float(
        _drag_n(
            design,
            parameters,
            turn_speed,
            supported_weight_n,
            mission,
            lift_coefficient=(
                turn_load_factor
                * supported_weight_n
                / (turn_q * design.wing_area)
            ),
        )
    )
    turn_operating_point = solve_cruise_samples(
        design.prop_diameter_in,
        design.prop_pitch_in,
        (turn_speed,),
        motor,
        battery,
        current_limit_a,
        1.0,
        prop_database,
        min_rpm=3000,
        max_rpm=20000,
        rpm_step=100,
        minimum_thrust_n=(required_thrust,),
    )
    if turn_operating_point.failed_mask[0]:
        return _TurnPerformance(
            False,
            turn_speed,
            turn_load_factor,
            float(angular_rates[best_index]),
            required_thrust,
            math.inf,
            float(maximum_propulsion.selected_rpm[best_index]),
        )

    return _TurnPerformance(
        True,
        turn_speed,
        turn_load_factor,
        float(angular_rates[best_index]),
        required_thrust,
        float(turn_operating_point.selected_current_a[0] * battery.vnom),
        max(
            float(maximum_propulsion.selected_rpm[best_index]),
            float(turn_operating_point.selected_rpm[0]),
        ),
    )


def _penalty_and_limit(
    takeoff_distance_m: float,
    climb_rate_mps: float,
    climb_distance_required_m: float,
    climb_distance_allowed_m: float,
    required_energy_wh: float,
    allowed_energy_wh: float,
    propeller_tip_mach: float,
    requirements: PropulsionRequirements,
    *,
    operating_point_failed: bool = False,
) -> tuple[float, str]:
    raw_violations = {
        "takeoff_distance": (
            max(
                0.0,
                takeoff_distance_m / requirements.maximum_takeoff_distance_m
                - 1.0,
            )
            if requirements.maximum_takeoff_distance_m is not None
            else 0.0
        ),
        "climb_rate": max(
            0.0,
            1.0 - climb_rate_mps / requirements.minimum_climb_rate_mps,
        ),
        "climb_to_pattern_altitude": (
            max(
                0.0,
                climb_distance_required_m / climb_distance_allowed_m - 1.0,
            )
            if requirements.climb_altitude_m > 0.0
            and climb_distance_allowed_m > 0.0
            else 0.0
        ),
        "mission_energy": max(0.0, required_energy_wh / allowed_energy_wh - 1.0),
        "propeller_tip_mach": max(
            0.0,
            propeller_tip_mach / requirements.maximum_propeller_tip_mach - 1.0,
        ),
        "propulsion_operating_point": float(operating_point_failed),
    }
    violations = {
        name: value if math.isfinite(value) else math.inf
        for name, value in raw_violations.items()
    }
    limiting = max(violations, key=violations.get)
    worst = violations[limiting]
    if worst <= 0.0:
        return 0.0, "none"
    severity = PROPULSION_VIOLATION_PENALTY_SCALE * (
        1.0 - math.exp(-2.0 * worst)
    )
    return PROPULSION_INFEASIBLE_BASE_PENALTY + severity, limiting


def evaluate_mission_propulsion(
    design: DesignVector,
    parameters: ParameterVector,
    *,
    mission: int,
    mass_kg: float,
    supported_mass_kg: float | None = None,
    cruise_speed_mps: float,
    stall_speed_mps: float,
    lap_time_s: float,
    prop_database: ContinuousPropDatabase,
    requirements: PropulsionRequirements = DEFAULT_PROPULSION_REQUIREMENTS,
) -> MissionPropulsionPerformance:
    """Evaluate full-throttle takeoff/climb and complete-mission energy.

    ``mass_kg`` is the mass accelerated along the flight path. The optional
    ``supported_mass_kg`` is the equivalent vertical load used for lift,
    thrust-to-weight, and climb; it differs for the towed-sensor mission.
    """
    if mission not in (1, 2, 3):
        raise ValueError("mission must be 1, 2, or 3.")
    if supported_mass_kg is None:
        supported_mass_kg = mass_kg
    values = (
        mass_kg,
        supported_mass_kg,
        cruise_speed_mps,
        stall_speed_mps,
        lap_time_s,
    )
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
    weight_n = supported_mass_kg * parameters.gravity
    static_thrust = float(static.thrust_samples_n[0])
    if static_thrust <= 0.0 or static.failed_mask[0]:
        optimistic_takeoff_distance = math.inf
    else:
        optimistic_takeoff_distance = (
            mass_kg * liftoff_speed**2 / (2.0 * static_thrust)
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
    # Capacity is an amp-hour limit. Convert current to nominal-equivalent pack
    # power for depletion accounting; terminal I*V_sag omits internal-resistance
    # loss and cannot be compared directly with nominal V*Ah pack energy.
    takeoff_battery_power_w = (
        full.selected_current_a[takeoff_indices] * battery.vnom
    )
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
            np.trapezoid(
                takeoff_battery_power_w * inverse_acceleration,
                takeoff_speeds,
            )
            / 3600.0
        )

    climb_index = int(np.searchsorted(sample_speeds, climb_speed))
    climb_thrust_n = float(full.thrust_samples_n[climb_index])
    climb_power_w = float(full.selected_current_a[climb_index] * battery.vnom)
    climb_drag_n = float(
        _drag_n(design, parameters, climb_speed, weight_n, mission)
    )
    # Steady climb: sin(gamma) = (T - D) / W.
    excess_thrust_fraction = (climb_thrust_n - climb_drag_n) / weight_n
    sin_gamma = float(np.clip(excess_thrust_fraction, 0.0, 0.999))
    climb_angle_rad = math.asin(sin_gamma)
    climb_rate = max(0.0, climb_speed * sin_gamma)
    horizontal_speed = climb_speed * math.cos(climb_angle_rad)
    climb_gradient = (
        climb_rate / horizontal_speed if horizontal_speed > 1e-9 else 0.0
    )
    if requirements.climb_altitude_m <= 0.0:
        climb_distance_required = 0.0
    elif climb_gradient > 1e-9:
        climb_distance_required = requirements.climb_altitude_m / climb_gradient
    else:
        climb_distance_required = math.inf
    climb_distance_allowed = (
        math.inf
        if requirements.climb_distance_m is None
        else requirements.climb_distance_m
    )
    if (
        requirements.climb_distance_includes_takeoff_roll
        and math.isfinite(climb_distance_allowed)
    ):
        climb_distance_allowed = max(
            0.0, requirements.climb_distance_m - takeoff_distance
        )
    if requirements.climb_altitude_m <= 0.0:
        climb_energy = 0.0
    elif climb_rate <= 0.0 or full.failed_mask[climb_index]:
        climb_energy = math.inf
    else:
        climb_energy = climb_power_w * requirements.climb_altitude_m / climb_rate / 3600.0

    cruise_power = (
        math.inf
        if cruise.failed_mask[0]
        else float(cruise.selected_current_a[0] * battery.vnom)
    )
    turn = _propulsion_limited_turn(
        design,
        parameters,
        mission=mission,
        supported_weight_n=weight_n,
        cruise_speed_mps=cruise_speed_mps,
        stall_speed_mps=stall_speed_mps,
        motor=motor,
        battery=battery,
        current_limit_a=current_limit,
        prop_database=prop_database,
    )
    total_turn_angle_rad = (
        TURN_180_COUNT * TURN_180_RAD
        + TURN_360_COUNT * TURN_360_RAD
    )
    straight_time_per_lap = (
        STRAIGHTS_PER_LAP * STRAIGHT_LENGTH_M / cruise_speed_mps
    )
    turn_time_per_lap = (
        total_turn_angle_rad / turn.angular_rate_rad_s
        if turn.feasible
        else math.inf
    )
    modeled_lap_time = straight_time_per_lap + turn_time_per_lap
    if not turn.feasible:
        straight_energy = math.inf
        turn_energy = math.inf
        cruise_energy = math.inf
        reacceleration_energy = math.inf
    else:
        required_laps = {1: 3, 2: 5}.get(mission)
        lap_equivalents = (
            max(1, int(300.0 // modeled_lap_time))
            if required_laps is None
            else required_laps
        )
        straight_duration_s = lap_equivalents * straight_time_per_lap
        turn_duration_s = lap_equivalents * turn_time_per_lap
        straight_energy = cruise_power * straight_duration_s / 3600.0
        turn_energy = turn.battery_power_w * turn_duration_s / 3600.0
        # Kept as a compatibility/reporting aggregate: it now means all steady
        # course-flight energy, rather than straight power times every second.
        cruise_energy = straight_energy + turn_energy
        delta_ke_j = 0.5 * mass_kg * max(
            0.0,
            cruise_speed_mps**2 - turn.speed_mps**2,
        )
        reacceleration_energy = (
            3.0 * lap_equivalents * delta_ke_j
            / requirements.reacceleration_efficiency
            / 3600.0
        )
    required_energy = takeoff_energy + climb_energy + reacceleration_energy + cruise_energy
    usable_energy = battery.vnom * battery.get_useable_capacity()
    allowed_energy = usable_energy * (1.0 - requirements.usable_energy_margin_fraction)
    maximum_rpm = max(float(np.max(full.selected_rpm)), turn.maximum_rpm)
    propeller_tip_speed_mps = (
        math.pi
        * design.prop_diameter_in
        * 0.0254
        * maximum_rpm
        / 60.0
    )
    propeller_tip_mach = propeller_tip_speed_mps / requirements.speed_of_sound_mps
    operating_point_failed = bool(
        static.failed_mask[0]
        or np.any(full.failed_mask)
        or cruise.failed_mask[0]
        or not turn.feasible
        or not math.isfinite(takeoff_distance)
    )
    penalty, limiting = _penalty_and_limit(
        takeoff_distance,
        climb_rate,
        climb_distance_required,
        climb_distance_allowed,
        required_energy,
        allowed_energy,
        propeller_tip_mach,
        requirements,
        operating_point_failed=operating_point_failed,
    )
    feasible = (
        not operating_point_failed
        and (
            requirements.maximum_takeoff_distance_m is None
            or takeoff_distance <= requirements.maximum_takeoff_distance_m
        )
        and climb_rate >= requirements.minimum_climb_rate_mps
        and (
            requirements.climb_altitude_m <= 0.0
            or climb_distance_required <= climb_distance_allowed
        )
        and required_energy <= allowed_energy
        and propeller_tip_mach <= requirements.maximum_propeller_tip_mach
    )
    return MissionPropulsionPerformance(
        mission=mission,
        feasible=feasible,
        penalty=penalty,
        inertial_mass_kg=mass_kg,
        supported_mass_kg=supported_mass_kg,
        operating_points_feasible=not operating_point_failed,
        static_thrust_n=static_thrust,
        static_thrust_to_weight=static_thrust / weight_n,
        liftoff_speed_mps=liftoff_speed,
        takeoff_distance_m=takeoff_distance,
        optimistic_takeoff_distance_lower_bound_m=optimistic_takeoff_distance,
        takeoff_screened_early=False,
        takeoff_time_s=takeoff_time,
        climb_speed_mps=climb_speed,
        climb_rate_mps=climb_rate,
        climb_gradient=climb_gradient,
        climb_distance_required_m=climb_distance_required,
        climb_distance_allowed_m=climb_distance_allowed,
        maximum_propeller_tip_mach=propeller_tip_mach,
        cruise_power_w=cruise_power,
        takeoff_energy_wh=takeoff_energy,
        climb_energy_wh=climb_energy,
        reacceleration_energy_wh=reacceleration_energy,
        cruise_energy_wh=cruise_energy,
        required_energy_wh=required_energy,
        allowed_energy_wh=allowed_energy,
        energy_margin_wh=allowed_energy - required_energy,
        takeoff_distance_margin_m=(
            math.inf
            if requirements.maximum_takeoff_distance_m is None
            else requirements.maximum_takeoff_distance_m - takeoff_distance
        ),
        climb_rate_margin_mps=climb_rate - requirements.minimum_climb_rate_mps,
        climb_distance_margin_m=climb_distance_allowed - climb_distance_required,
        limiting_constraint=limiting,
        aerodynamic_lap_time_s=lap_time_s,
        modeled_lap_time_s=modeled_lap_time,
        straight_time_per_lap_s=straight_time_per_lap,
        turn_time_per_lap_s=turn_time_per_lap,
        turn_speed_mps=turn.speed_mps,
        turn_load_factor=turn.load_factor,
        turn_power_w=turn.battery_power_w,
        straight_energy_wh=straight_energy,
        turn_energy_wh=turn_energy,
    )


def propulsion_margin_bonus(
    results: tuple[MissionPropulsionPerformance, ...],
    requirements: PropulsionRequirements = DEFAULT_PROPULSION_REQUIREMENTS,
) -> float:
    """Small optimizer-only tie-breaker; never changes official scores."""
    if (
        {result.mission for result in results} != {1, 2, 3}
        or not all(result.feasible for result in results)
    ):
        return 0.0
    takeoff = (
        min(
            1.0,
            min(
                1.0
                - result.takeoff_distance_m
                / requirements.maximum_takeoff_distance_m
                for result in results
            ),
        )
        if requirements.maximum_takeoff_distance_m is not None
        else 0.0
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
    "PROPULSION_INFEASIBLE_BASE_PENALTY",
    "PROPULSION_VIOLATION_PENALTY_SCALE",
    "PropulsionRequirements",
    "evaluate_mission_propulsion",
    "propulsion_margin_bonus",
]
