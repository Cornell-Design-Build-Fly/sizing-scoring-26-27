"""Takeoff, climb, and mission-energy checks for propulsion sizing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from src.aero.aero_score import (
    FLIGHT_WINDOW_S,
    GROUND_TIME_S,
    N_ZS,
    STRAIGHT_LENGTH_M,
    STRAIGHTS_PER_LAP,
    TURN_180_COUNT,
    TURN_180_RAD,
    TURN_360_COUNT,
    TURN_360_RAD,
)
from src.aero.drag_model import drag_coefficients, fuselage_drag_geometry, sensor_drag_force
from src.aero.flaps import DEFAULT_FLAPS, FlapConfig, clean_cl_max
from src.prop.continuous_prop_database import ContinuousPropDatabase
from src.prop.prop_cruise_values import (
    CruiseGrid,
    evaluate_cruise_grid,
    select_cruise_points,
    solve_cruise_samples,
)
from src.prop.prop_helper_functions import make_battery_from_design, make_motor_from_design
from src.vectors import DesignVector, ParameterVector


PROPULSION_INFEASIBLE_BASE_PENALTY = 10.0
PROPULSION_VIOLATION_PENALTY_SCALE = 10.0


@dataclass(frozen=True)
class PropulsionRequirements:
    """Editable operational requirements used by the optimizer."""

    # The rules do not specify a runway limit, but the airplane still needs a
    # finite operating runway. Keep the team's 75 m engineering requirement.
    # The rules also specify no minimum course altitude, so the old 200 ft
    # climb requirement is disabled unless explicitly requested for a study.
    maximum_takeoff_distance_m: float | None = 75.0
    # Pattern altitude. The aircraft climbs to it on every mission, retracts the
    # flaps there and flies the course clean, so this is an operating point, not
    # a rules gate -- the rules set no minimum course altitude. 200 ft is the
    # team's own pattern-altitude number, kept from the old climb requirement.
    # Charging it closed a real hole: the previous default of zero climb meant
    # the model paid nothing at all to get to altitude.
    cruise_altitude_m: float = 60.96
    # Mission clock and the part of it consumed by takeoff and landing. The
    # remainder is the window that laps and the propulsion energy budget must
    # both fit inside. See src/aero/aero_score.py.
    flight_window_s: float = FLIGHT_WINDOW_S
    ground_time_s: float = GROUND_TIME_S
    # Number of bisection steps used to find the highest cruise power the pack
    # can sustain for a whole mission. Every step is a pure re-selection over
    # cached propeller grids, so this is nearly free.
    energy_budget_iterations: int = 24
    # Loose guard rail on turn geometry. Nothing else in the model keeps the
    # airplane on the field: lap time alone lets the energy budget buy laps by
    # flying enormous, gentle, slow turns that would leave the course. Half a
    # 500 ft straight leg is a generous upper bound, not a real field limit --
    # a hard-turning airplane at corner speed naturally sits near 10-20 m.
    # EDIT: tighten to the actual flying-site limit once the team has one.
    maximum_turn_radius_m: float | None = STRAIGHT_LENGTH_M / 2.0
    # Plain flaps, deployed for the ground roll and for the approach only.
    flaps: FlapConfig = DEFAULT_FLAPS
    # Optional gate: reach ``cruise_altitude_m`` inside this ground distance.
    # None leaves the climb unconstrained in distance, which is the default
    # because no rule requires altitude by any point on the course.
    climb_distance_m: float | None = None
    climb_distance_includes_takeoff_roll: bool = False
    minimum_climb_rate_mps: float = 2.0
    liftoff_stall_speed_factor: float = 1.20
    climb_stall_speed_factor: float = 1.30
    rolling_friction_coefficient: float = 0.04
    takeoff_lift_coefficient_fraction: float = 0.80
    # The detailed wing height and landing-gear compression are not design
    # variables yet. Apply only a modest 10% reduction to lift-induced drag on
    # the ground roll; profile, flap and fuselage drag receive no credit.
    ground_effect_induced_drag_factor: float = 0.90
    reacceleration_efficiency: float = 0.65
    usable_energy_margin_fraction: float = 0.05
    # APC's published maximum for Thin Electric propellers is RPM = 150,000 / D
    # with D in inches. Apply an additional design margin so the optimizer does
    # not deliberately operate on the manufacturer's absolute ceiling.
    maximum_propeller_rpm_diameter_product: float = 150_000.0
    propeller_rpm_limit_safety_factor: float = 0.90
    maximum_propeller_tip_mach: float = 0.75
    speed_of_sound_mps: float = 343.0

    def __post_init__(self) -> None:
        positive = (
            self.minimum_climb_rate_mps,
            self.liftoff_stall_speed_factor,
            self.climb_stall_speed_factor,
            self.reacceleration_efficiency,
            self.maximum_propeller_rpm_diameter_product,
            self.propeller_rpm_limit_safety_factor,
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
        if not math.isfinite(self.cruise_altitude_m) or self.cruise_altitude_m <= 0.0:
            raise ValueError("Cruise altitude must be finite and positive.")
        if not math.isfinite(self.flight_window_s) or self.flight_window_s <= 0.0:
            raise ValueError("Flight window must be finite and positive.")
        if (
            not math.isfinite(self.ground_time_s)
            or self.ground_time_s < 0.0
            or self.ground_time_s >= self.flight_window_s
        ):
            raise ValueError("Ground time must fit inside the flight window.")
        if self.energy_budget_iterations < 1:
            raise ValueError("At least one energy-budget iteration is required.")
        if self.maximum_turn_radius_m is not None and (
            not math.isfinite(self.maximum_turn_radius_m)
            or self.maximum_turn_radius_m <= 0.0
        ):
            raise ValueError("Maximum turn radius must be finite and positive.")
        fractions = (
            self.rolling_friction_coefficient,
            self.takeoff_lift_coefficient_fraction,
            self.reacceleration_efficiency,
            self.usable_energy_margin_fraction,
            self.propeller_rpm_limit_safety_factor,
        )
        if not all(0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("Propulsion requirement fractions must lie in [0, 1].")
        if not 0.0 < self.ground_effect_induced_drag_factor <= 1.0:
            raise ValueError("Ground-effect induced-drag factor must lie in (0, 1].")


    @property
    def usable_window_s(self) -> float:
        """Mission clock left for scored laps after takeoff and landing."""

        return self.flight_window_s - self.ground_time_s

    @property
    def required_climb_gradient(self) -> float:
        """Altitude gained per unit ground distance, i.e. tan(climb angle)."""

        if self.climb_distance_m is None:
            return 0.0
        return self.cruise_altitude_m / self.climb_distance_m


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
    # Level acceleration from liftoff to climb speed, flaps still down.
    acceleration_distance_m: float
    acceleration_time_s: float
    acceleration_energy_wh: float
    climb_speed_mps: float
    climb_rate_mps: float
    climb_gradient: float
    climb_distance_required_m: float
    climb_distance_allowed_m: float
    cruise_altitude_m: float
    climb_time_s: float
    # Accelerating to cruise speed after the flaps come up at cruise altitude.
    flap_retraction_energy_wh: float
    maximum_propeller_tip_mach: float
    maximum_propeller_rpm: float
    manufacturer_propeller_rpm_limit: float
    operating_propeller_rpm_limit: float
    propeller_rpm_margin: float
    maximum_propeller_shaft_power_w: float
    maximum_propeller_disk_power_loading_w_m2: float
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
    # Propeller actually flown on this mission (M1/M2 share one; M3 has its own).
    propeller_key: str
    propeller_blade_count: int
    propeller_diameter_in: float
    propeller_pitch_in: float
    # ``aerodynamic_cruise_speed_mps`` is the trimmed speed at the propeller's
    # unrestricted thrust curve. ``cruise_speed_mps`` is what the aircraft can
    # actually hold once the pack has to last the whole mission window;
    # ``energy_limited`` records whether the budget bound the speed down.
    aerodynamic_cruise_speed_mps: float
    cruise_speed_mps: float
    cruise_power_cap_w: float
    energy_limited: bool
    completed_laps: int
    mission_flight_time_s: float
    usable_window_s: float
    # Flap configuration. The clean stall speed is the one cruise and the turns
    # fly on; takeoff gets its own because the flaps are down for the roll.
    clean_stall_speed_mps: float
    takeoff_stall_speed_mps: float
    takeoff_flap_deflection_deg: float
    # Reported only. Nothing scores or gates on how fast the airplane lands --
    # the team judged a modelled landing-speed limit not worth having, so this
    # is here to be read, not to constrain.
    landing_stall_speed_mps: float
    landing_flap_deflection_deg: float

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
    flap_deflection_deg: float = 0.0,
    flaps: FlapConfig = DEFAULT_FLAPS,
    induced_drag_factor: float = 1.0,
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
        flap_deflection_deg=flap_deflection_deg,
        flaps=flaps,
    )
    induced_keys = {"wing_induced", "tail_induced", "interaction"}
    drag_coefficient = sum(
        value * induced_drag_factor if name in induced_keys else value
        for name, value in coefficients.items()
    )
    drag = q * design.wing_area * drag_coefficient
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
    maximum_shaft_power_w: float


_INFEASIBLE_TURN = _TurnPerformance(
    False, math.nan, 1.0, 0.0, math.inf, math.inf, 0.0, 0.0
)


@dataclass(frozen=True)
class _TurnEnvelope:
    """Everything about the turn that does not depend on the power cap.

    The propeller-database query and the drag table are the expensive parts and
    neither changes when the mission energy budget lowers the available power,
    so both are built once and re-selected as the budget search moves.
    """

    speeds_mps: np.ndarray
    load_factors: np.ndarray
    # drag_table[speed_index, load_factor_index], monotonic along axis 1.
    drag_table_n: np.ndarray
    lift_limited_load_factor: np.ndarray
    grid: CruiseGrid | None


def _build_turn_envelope(
    design: DesignVector,
    parameters: ParameterVector,
    *,
    mission: int,
    supported_weight_n: float,
    maximum_cruise_speed_mps: float,
    stall_speed_mps: float,
    motor,
    battery,
    prop_database: ContinuousPropDatabase,
    diameter_in: float,
    pitch_in: float,
    speed_samples: int = 31,
    load_factor_samples: int = 61,
) -> _TurnEnvelope:
    """Tabulate turn drag and the propeller grid over the turn envelope."""

    maximum_turn_speed = min(
        maximum_cruise_speed_mps,
        math.sqrt(N_ZS) * stall_speed_mps,
    )
    minimum_turn_speed = 1.001 * stall_speed_mps
    if not (maximum_turn_speed > minimum_turn_speed):
        empty = np.zeros(0, dtype=float)
        return _TurnEnvelope(empty, empty, np.zeros((0, 0)), empty, None)

    speeds = np.linspace(minimum_turn_speed, maximum_turn_speed, speed_samples)
    load_factors = np.linspace(1.0, N_ZS, load_factor_samples)

    q = 0.5 * parameters.rho * speeds**2
    one_g_lift_coefficient = supported_weight_n / (q * design.wing_area)
    # Drag at every (speed, load factor) pair. Drag rises monotonically with
    # load factor at fixed speed, so the sustainable load factor under any
    # thrust limit is a search along axis 1 -- no per-cap bisection needed.
    drag_table = _drag_n(
        design,
        parameters,
        speeds[:, np.newaxis],
        supported_weight_n,
        mission,
        lift_coefficient=(
            one_g_lift_coefficient[:, np.newaxis] * load_factors[np.newaxis, :]
        ),
    )

    grid = evaluate_cruise_grid(
        diameter_in=diameter_in,
        pitch_in=pitch_in,
        velocities_mps=speeds,
        motor=motor,
        battery=battery,
        prop_database=prop_database,
        min_rpm=3000,
        max_rpm=20000,
        rpm_step=100,
    )
    return _TurnEnvelope(
        speeds_mps=speeds,
        load_factors=load_factors,
        drag_table_n=np.asarray(drag_table, dtype=float),
        lift_limited_load_factor=(speeds / stall_speed_mps) ** 2,
        grid=grid,
    )


def _solve_turn(
    envelope: _TurnEnvelope,
    parameters: ParameterVector,
    *,
    current_limit_a: float,
    throttle_limit: float,
    motor_max_power_w: float,
    battery_vnom_v: float,
    maximum_battery_power_w: float | None,
    maximum_speed_mps: float,
    maximum_radius_m: float | None = None,
) -> _TurnPerformance:
    """Pick the quickest sustainable level turn under the supplied limits.

    Lift, the 2.5 g structural limit, propeller thrust, current, voltage, motor
    power and -- new here -- the mission energy budget are applied together.
    The turn is then flown at the least-current propeller operating point that
    still sustains it, because pack depletion is an amp-hour draw.
    """

    grid = envelope.grid
    if grid is None or envelope.speeds_mps.size == 0:
        return _INFEASIBLE_TURN

    speeds = envelope.speeds_mps
    in_range = speeds <= maximum_speed_mps * (1.0 + 1.0e-9)
    if not np.any(in_range):
        return _INFEASIBLE_TURN

    available = select_cruise_points(
        grid,
        max_current_a=current_limit_a,
        cruise_throttle=throttle_limit,
        motor_max_power_w=motor_max_power_w,
        maximum_battery_power_w=maximum_battery_power_w,
    )
    maximum_thrust_n = np.where(available.failed_mask, 0.0, available.thrust_samples_n)

    upper_load_factor = np.minimum(N_ZS, envelope.lift_limited_load_factor)
    # Largest tabulated load factor whose drag the propeller can still hold.
    holdable = envelope.drag_table_n <= maximum_thrust_n[:, np.newaxis]
    holdable &= envelope.load_factors[np.newaxis, :] <= upper_load_factor[:, np.newaxis]
    holdable[available.failed_mask, :] = False
    holdable[~in_range, :] = False
    sustainable = np.any(holdable, axis=1)
    if not np.any(sustainable):
        return _INFEASIBLE_TURN

    if maximum_radius_m is not None:
        # radius = V^2 / (g * sqrt(n^2 - 1)); rearranged, a radius ceiling is a
        # load-factor floor at each speed.
        minimum_load_factor = np.sqrt(
            1.0
            + (speeds**2 / (parameters.gravity * maximum_radius_m)) ** 2
        )
        holdable &= (
            envelope.load_factors[np.newaxis, :]
            >= minimum_load_factor[:, np.newaxis]
        )
        sustainable = np.any(holdable, axis=1)
        if not np.any(sustainable):
            return _INFEASIBLE_TURN

    best_load_factor_index = (
        envelope.load_factors.size - 1 - np.argmax(holdable[:, ::-1], axis=1)
    )
    load_factor = np.where(
        sustainable, envelope.load_factors[best_load_factor_index], 1.0
    )
    angular_rate = np.zeros_like(speeds)
    turning = sustainable & (load_factor > 1.0)
    if not np.any(turning):
        return _INFEASIBLE_TURN
    angular_rate[turning] = (
        parameters.gravity
        * np.sqrt(np.maximum(load_factor[turning] ** 2 - 1.0, 0.0))
        / speeds[turning]
    )

    best = int(np.argmax(angular_rate))
    if angular_rate[best] <= 0.0:
        return _INFEASIBLE_TURN

    required_thrust = float(
        envelope.drag_table_n[best, int(best_load_factor_index[best])]
    )
    minimum_thrust = np.zeros(speeds.size, dtype=float)
    minimum_thrust[best] = required_thrust
    operating_point = select_cruise_points(
        grid,
        max_current_a=current_limit_a,
        cruise_throttle=throttle_limit,
        motor_max_power_w=motor_max_power_w,
        maximum_battery_power_w=maximum_battery_power_w,
        minimum_thrust_n=minimum_thrust,
    )
    if operating_point.failed_mask[best]:
        return _TurnPerformance(
            False,
            float(speeds[best]),
            float(load_factor[best]),
            float(angular_rate[best]),
            required_thrust,
            math.inf,
            float(available.selected_rpm[best]),
            float(available.selected_shaft_power_w[best]),
        )
    return _TurnPerformance(
        True,
        float(speeds[best]),
        float(load_factor[best]),
        float(angular_rate[best]),
        required_thrust,
        float(operating_point.selected_current_a[best] * battery_vnom_v),
        max(
            float(available.selected_rpm[best]),
            float(operating_point.selected_rpm[best]),
        ),
        max(
            float(available.selected_shaft_power_w[best]),
            float(operating_point.selected_shaft_power_w[best]),
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
    propeller_rpm: float | None = None,
    propeller_rpm_limit: float | None = None,
    operating_point_failed: bool = False,
    course_flight_failed: bool = False,
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
            0.0
            if requirements.climb_distance_m is None
            else (
                max(
                    0.0,
                    climb_distance_required_m / climb_distance_allowed_m - 1.0,
                )
                if climb_distance_allowed_m > 0.0
                # The ground roll and the climb-out acceleration ate the whole
                # allowance, so no gradient can satisfy it.
                else math.inf
            )
        ),
        # Listed before mission_energy on purpose. An airplane that cannot hold
        # level flight or sustain any legal turn reports infinite required
        # energy as a side effect; naming the battery there would send the team
        # after the wrong constraint. ``max`` keeps the first maximal key.
        "course_flight": math.inf if course_flight_failed else 0.0,
        "mission_energy": max(0.0, required_energy_wh / allowed_energy_wh - 1.0),
        "propeller_tip_mach": max(
            0.0,
            propeller_tip_mach / requirements.maximum_propeller_tip_mach - 1.0,
        ),
        "propeller_rpm": (
            max(0.0, propeller_rpm / propeller_rpm_limit - 1.0)
            if propeller_rpm is not None
            and propeller_rpm_limit is not None
            and propeller_rpm_limit > 0.0
            else 0.0
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


@dataclass(frozen=True)
class _CourseState:
    """One whole-mission flight profile evaluated at a given power cap."""

    feasible: bool
    power_cap_w: float
    cruise_speed_mps: float
    cruise_power_w: float
    turn: _TurnPerformance
    straight_time_per_lap_s: float
    turn_time_per_lap_s: float
    lap_time_s: float
    laps: int
    flight_time_s: float
    straight_energy_wh: float
    turn_energy_wh: float
    reacceleration_energy_wh: float
    # Accelerating from climb speed to cruise speed once the flaps come up at
    # cruise altitude. It lives here rather than with the fixed takeoff and
    # climb energy because it depends on the cruise speed the budget selects.
    flap_retraction_energy_wh: float

    @property
    def course_energy_wh(self) -> float:
        return (
            self.straight_energy_wh
            + self.turn_energy_wh
            + self.reacceleration_energy_wh
            + self.flap_retraction_energy_wh
        )


_INFEASIBLE_COURSE = _CourseState(
    False,
    math.inf,
    math.nan,
    math.inf,
    _INFEASIBLE_TURN,
    math.inf,
    math.inf,
    math.inf,
    0,
    math.inf,
    math.inf,
    math.inf,
    math.inf,
    math.inf,
)


TOTAL_TURN_ANGLE_RAD = (
    TURN_180_COUNT * TURN_180_RAD + TURN_360_COUNT * TURN_360_RAD
)


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
    """Evaluate full-throttle takeoff/climb and the whole-mission energy budget.

    ``mass_kg`` is the mass accelerated along the flight path. The optional
    ``supported_mass_kg`` is the equivalent vertical load used for lift,
    thrust-to-weight, and climb; it differs for the towed-sensor mission.

    ``cruise_speed_mps`` is the aerodynamically trimmed cruise speed at the
    propeller's unrestricted thrust curve, i.e. the fastest the airframe can be
    pushed.  It is an upper bound, not the flown speed: the pack holds a fixed
    number of watt-hours and the mission has a fixed clock, so the aircraft
    actually flies at the highest power those two together allow.  That is what
    makes weight limit speed here -- a heavier airplane needs more power for the
    same speed, so the same energy budget buys it a slower lap.
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
    diameter_in, pitch_in = design.propeller_for_mission(mission)
    try:
        propeller_key = prop_database.catalog.get_by_geometry(
            diameter_in, pitch_in
        ).key
    except KeyError:
        propeller_key = "continuous-geometry-study"
    throttle_limit = design.cruise_throttle_for_mission(mission)

    # Flap configuration by phase. ``stall_speed_mps`` arrives clean, from the
    # aerodynamic trim, and stays clean for cruise and for every turn: the
    # aircraft would not carry flaps through a 2.5 g turn, and crediting the
    # turn envelope with flapped lift it does not have is exactly the mistake
    # a single CLmax invites. Takeoff flaps remain down through the climb,
    # retract at cruise altitude, and the landing setting is used diagnostically
    # for the approach.
    flaps = replace(
        requirements.flaps,
        takeoff_deflection_deg=float(design.takeoff_flap_deflection_deg),
    )
    wing_aspect_ratio = design.wing_span**2 / design.wing_area
    takeoff_flap_deg = flaps.deflection_for("takeoff")
    landing_flap_deg = flaps.deflection_for("landing")
    takeoff_stall_speed = flaps.stall_speed_for(
        stall_speed_mps, wing_aspect_ratio, "takeoff"
    )
    # Reported as a diagnostic only; there is no landing-speed constraint.
    landing_stall_speed = flaps.stall_speed_for(
        stall_speed_mps, wing_aspect_ratio, "landing"
    )
    liftoff_speed = requirements.liftoff_stall_speed_factor * takeoff_stall_speed
    # The flaps stay down through the climb and retract at cruise altitude, so
    # the climb speed is referenced to the flapped stall speed too. Leaving it
    # on the clean stall speed opened a 3.4 m/s gap between liftoff and climb
    # that the model crossed for free.
    climb_speed = requirements.climb_stall_speed_factor * takeoff_stall_speed
    current_limit = min(motor.max_current, battery.get_max_current())
    motor_max_power = float(motor.max_power)
    usable_window_s = requirements.usable_window_s

    static = solve_cruise_samples(
        diameter_in,
        pitch_in,
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
    # Liftoff to climb speed, flown level on the flapped wing. Previously the
    # model jumped this gap for free: no distance, no time, no energy.
    acceleration_speeds = np.linspace(liftoff_speed, climb_speed, 21)
    # Candidate cruise speeds between just above stall and the aerodynamic
    # ceiling. The energy budget selects from this range instead of being
    # checked after the fact against one speed.
    cruise_speed_floor = min(1.02 * stall_speed_mps, cruise_speed_mps)
    cruise_sweep = np.linspace(cruise_speed_floor, cruise_speed_mps, 25)
    sample_speeds = np.unique(
        np.concatenate(
            (
                takeoff_speeds,
                acceleration_speeds,
                (climb_speed, cruise_speed_mps),
                cruise_sweep,
            )
        )
    )
    grid = evaluate_cruise_grid(
        diameter_in=diameter_in,
        pitch_in=pitch_in,
        velocities_mps=sample_speeds,
        motor=motor,
        battery=battery,
        prop_database=prop_database,
        min_rpm=3000,
        max_rpm=20000,
        rpm_step=100,
    )
    full = select_cruise_points(
        grid,
        max_current_a=current_limit,
        cruise_throttle=1.0,
        motor_max_power_w=motor_max_power,
    )

    # ---- Takeoff roll -------------------------------------------------------
    takeoff_cl = requirements.takeoff_lift_coefficient_fraction * flaps.cl_max_for(
        wing_aspect_ratio, "takeoff"
    )
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
        flap_deflection_deg=takeoff_flap_deg,
        flaps=flaps,
        induced_drag_factor=requirements.ground_effect_induced_drag_factor,
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

    # ---- Climb-out acceleration, flaps still down ---------------------------
    # Level acceleration from liftoff to climb speed. Lift equals weight, so
    # there is no rolling friction term and the induced drag is the level-flight
    # value; the flap drag increment is still charged because the flaps are down.
    acceleration_indices = np.searchsorted(sample_speeds, acceleration_speeds)
    acceleration_thrust_n = full.thrust_samples_n[acceleration_indices]
    acceleration_battery_power_w = (
        full.selected_current_a[acceleration_indices] * battery.vnom
    )
    acceleration_drag_n = _drag_n(
        design,
        parameters,
        acceleration_speeds,
        weight_n,
        mission,
        flap_deflection_deg=takeoff_flap_deg,
        flaps=flaps,
    )
    climb_out_acceleration = (
        acceleration_thrust_n - acceleration_drag_n
    ) / mass_kg
    if (
        acceleration_speeds[-1] <= acceleration_speeds[0]
        or np.any(climb_out_acceleration <= 0.0)
        or np.any(full.failed_mask[acceleration_indices])
    ):
        acceleration_distance = math.inf
        acceleration_time = math.inf
        acceleration_energy = math.inf
    else:
        inverse_climb_out = 1.0 / climb_out_acceleration
        acceleration_distance = float(
            np.trapezoid(acceleration_speeds * inverse_climb_out, acceleration_speeds)
        )
        acceleration_time = float(
            np.trapezoid(inverse_climb_out, acceleration_speeds)
        )
        acceleration_energy = float(
            np.trapezoid(
                acceleration_battery_power_w * inverse_climb_out,
                acceleration_speeds,
            )
            / 3600.0
        )

    # ---- Climb to cruise altitude, flaps still down -------------------------
    climb_index = int(np.searchsorted(sample_speeds, climb_speed))
    climb_thrust_n = float(full.thrust_samples_n[climb_index])
    climb_power_w = float(full.selected_current_a[climb_index] * battery.vnom)
    climb_drag_n = float(
        _drag_n(
            design,
            parameters,
            climb_speed,
            weight_n,
            mission,
            flap_deflection_deg=takeoff_flap_deg,
            flaps=flaps,
        )
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
    if climb_gradient > 1e-9:
        climb_distance_required = requirements.cruise_altitude_m / climb_gradient
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
            0.0,
            requirements.climb_distance_m - takeoff_distance - acceleration_distance,
        )
    if climb_rate <= 0.0 or full.failed_mask[climb_index]:
        climb_time = math.inf
        climb_energy = math.inf
    else:
        climb_time = requirements.cruise_altitude_m / climb_rate
        climb_energy = climb_power_w * climb_time / 3600.0

    # ---- Whole-mission energy budget ---------------------------------------
    usable_energy = battery.vnom * battery.get_useable_capacity()
    allowed_energy = usable_energy * (1.0 - requirements.usable_energy_margin_fraction)
    fixed_energy = takeoff_energy + acceleration_energy + climb_energy

    cruise_indices = np.searchsorted(sample_speeds, cruise_sweep)
    cruise_speeds = sample_speeds[cruise_indices]
    level_drag_n = _drag_n(design, parameters, cruise_speeds, weight_n, mission)
    required_thrust_full = np.zeros(sample_speeds.size, dtype=float)
    required_thrust_full[cruise_indices] = level_drag_n

    turn_envelope = _build_turn_envelope(
        design,
        parameters,
        mission=mission,
        supported_weight_n=weight_n,
        maximum_cruise_speed_mps=cruise_speed_mps,
        stall_speed_mps=stall_speed_mps,
        motor=motor,
        battery=battery,
        prop_database=prop_database,
        diameter_in=diameter_in,
        pitch_in=pitch_in,
    )
    required_laps = {1: 3, 2: 5}.get(mission)

    def course_state(
        straight_power_cap_w: float | None,
        turn_power_cap_w: float | None = None,
    ) -> _CourseState:
        """Flight profile flown under the given straight and turn power caps.

        The two are separate because they trade differently: throttling the
        straights saves energy roughly in proportion to drag times distance,
        while throttling the turns widens them and can cost more energy than it
        saves.  The caller searches both families and keeps the quickest lap
        that the pack can actually pay for.
        """

        cruise = select_cruise_points(
            grid,
            max_current_a=current_limit,
            cruise_throttle=throttle_limit,
            motor_max_power_w=motor_max_power,
            maximum_battery_power_w=straight_power_cap_w,
            minimum_thrust_n=required_thrust_full,
        )
        holds_level_flight = ~cruise.failed_mask[cruise_indices]
        if not np.any(holds_level_flight):
            return _INFEASIBLE_COURSE
        fastest = int(np.max(np.flatnonzero(holds_level_flight)))
        speed = float(cruise_speeds[fastest])
        cruise_power = float(
            cruise.selected_current_a[cruise_indices][fastest] * battery.vnom
        )
        turn = _solve_turn(
            turn_envelope,
            parameters,
            current_limit_a=current_limit,
            throttle_limit=throttle_limit,
            motor_max_power_w=motor_max_power,
            battery_vnom_v=float(battery.vnom),
            maximum_battery_power_w=turn_power_cap_w,
            maximum_speed_mps=speed,
            maximum_radius_m=requirements.maximum_turn_radius_m,
        )
        if not turn.feasible:
            return _INFEASIBLE_COURSE
        straight_time = STRAIGHTS_PER_LAP * STRAIGHT_LENGTH_M / speed
        turn_time = TOTAL_TURN_ANGLE_RAD / turn.angular_rate_rad_s
        lap_time = straight_time + turn_time
        if not math.isfinite(lap_time) or lap_time <= 0.0:
            return _INFEASIBLE_COURSE
        if required_laps is None:
            laps = max(1, int(usable_window_s // lap_time))
        else:
            laps = required_laps
        flight_time = laps * lap_time
        straight_energy = cruise_power * laps * straight_time / 3600.0
        turn_energy = turn.battery_power_w * laps * turn_time / 3600.0
        delta_ke_j = 0.5 * mass_kg * max(0.0, speed**2 - turn.speed_mps**2)
        reacceleration_energy = (
            3.0 * laps * delta_ke_j
            / requirements.reacceleration_efficiency
            / 3600.0
        )
        # Once per mission: flaps up at cruise altitude, then accelerate from
        # climb speed to cruise speed. Charged as recovered kinetic energy, the
        # same treatment the turn exits get.
        retraction_ke_j = 0.5 * mass_kg * max(0.0, speed**2 - climb_speed**2)
        flap_retraction_energy = (
            retraction_ke_j / requirements.reacceleration_efficiency / 3600.0
        )
        return _CourseState(
            True,
            math.inf if straight_power_cap_w is None else float(straight_power_cap_w),
            speed,
            cruise_power,
            turn,
            straight_time,
            turn_time,
            lap_time,
            laps,
            flight_time,
            straight_energy,
            turn_energy,
            reacceleration_energy,
            flap_retraction_energy,
        )

    def fits_budget(state: _CourseState) -> bool:
        return (
            state.feasible
            and fixed_energy + state.course_energy_wh <= allowed_energy
        )

    unrestricted = course_state(None)
    course = unrestricted
    energy_limited = False
    if unrestricted.feasible and not fits_budget(unrestricted):
        # The pack cannot sustain the fastest trimmed speed for the whole
        # mission, so search the power caps it can sustain and fly the quickest
        # lap among them.  Mission energy is NOT monotone in the power cap:
        # cutting power widens the turn faster than it lowers turn power, so a
        # bisection on energy walks the wrong way.  Every candidate here is a
        # re-selection over the cached propeller grids, so a scan is cheap.
        upper = max(
            unrestricted.cruise_power_w,
            unrestricted.turn.battery_power_w,
        )
        coarse_steps = max(4, requirements.energy_budget_iterations * 2 // 3)
        refine_steps = max(2, requirements.energy_budget_iterations - coarse_steps)
        caps = list(np.linspace(upper / coarse_steps, upper, coarse_steps))
        # Two families: throttle only the straights, or throttle the whole lap.
        candidates: list[tuple[float, bool, _CourseState]] = []
        for cap in caps:
            candidates.append((cap, False, course_state(cap, None)))
            candidates.append((cap, True, course_state(cap, cap)))
        affordable = [item for item in candidates if fits_budget(item[2])]
        if affordable:
            best_cap, best_capped_turn, best = min(
                affordable, key=lambda item: item[2].lap_time_s
            )
            step = caps[1] - caps[0] if len(caps) > 1 else best_cap
            for cap in np.linspace(
                max(step * 0.25, best_cap - step),
                best_cap + step,
                refine_steps + 2,
            )[1:-1]:
                candidate = course_state(
                    float(cap), float(cap) if best_capped_turn else None
                )
                if fits_budget(candidate) and candidate.lap_time_s < best.lap_time_s:
                    best = candidate
            course = best
            energy_limited = True

    turn = course.turn
    straight_time_per_lap = course.straight_time_per_lap_s
    turn_time_per_lap = course.turn_time_per_lap_s
    modeled_lap_time = course.lap_time_s
    cruise_power = course.cruise_power_w
    straight_energy = course.straight_energy_wh
    turn_energy = course.turn_energy_wh
    reacceleration_energy = course.reacceleration_energy_wh
    # Kept as a compatibility/reporting aggregate: it means all steady
    # course-flight energy, rather than straight power times every second.
    cruise_energy = straight_energy + turn_energy
    required_energy = fixed_energy + course.course_energy_wh

    maximum_rpm = max(float(np.max(full.selected_rpm)), turn.maximum_rpm)
    maximum_shaft_power_w = max(
        float(np.max(full.selected_shaft_power_w)),
        turn.maximum_shaft_power_w,
    )
    propeller_disk_area_m2 = math.pi * (diameter_in * 0.0254) ** 2 / 4.0
    manufacturer_rpm_limit = (
        requirements.maximum_propeller_rpm_diameter_product / diameter_in
    )
    operating_rpm_limit = (
        requirements.propeller_rpm_limit_safety_factor * manufacturer_rpm_limit
    )
    propeller_tip_speed_mps = (
        math.pi
        * diameter_in
        * 0.0254
        * maximum_rpm
        / 60.0
    )
    propeller_tip_mach = propeller_tip_speed_mps / requirements.speed_of_sound_mps
    operating_point_failed = bool(
        static.failed_mask[0]
        or np.any(full.failed_mask)
        or not course.feasible
        or not turn.feasible
        or not math.isfinite(takeoff_distance)
        or not math.isfinite(acceleration_distance)
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
        propeller_rpm=maximum_rpm,
        propeller_rpm_limit=operating_rpm_limit,
        operating_point_failed=operating_point_failed,
        course_flight_failed=not course.feasible,
    )
    feasible = (
        not operating_point_failed
        and (
            requirements.maximum_takeoff_distance_m is None
            or takeoff_distance <= requirements.maximum_takeoff_distance_m
        )
        and climb_rate >= requirements.minimum_climb_rate_mps
        and (
            requirements.climb_distance_m is None
            or climb_distance_required <= climb_distance_allowed
        )
        and required_energy <= allowed_energy
        and maximum_rpm <= operating_rpm_limit
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
        acceleration_distance_m=acceleration_distance,
        acceleration_time_s=acceleration_time,
        acceleration_energy_wh=acceleration_energy,
        climb_speed_mps=climb_speed,
        climb_rate_mps=climb_rate,
        climb_gradient=climb_gradient,
        climb_distance_required_m=climb_distance_required,
        climb_distance_allowed_m=climb_distance_allowed,
        cruise_altitude_m=requirements.cruise_altitude_m,
        climb_time_s=climb_time,
        flap_retraction_energy_wh=course.flap_retraction_energy_wh,
        maximum_propeller_tip_mach=propeller_tip_mach,
        maximum_propeller_rpm=maximum_rpm,
        manufacturer_propeller_rpm_limit=manufacturer_rpm_limit,
        operating_propeller_rpm_limit=operating_rpm_limit,
        propeller_rpm_margin=operating_rpm_limit - maximum_rpm,
        maximum_propeller_shaft_power_w=maximum_shaft_power_w,
        maximum_propeller_disk_power_loading_w_m2=(
            maximum_shaft_power_w / propeller_disk_area_m2
        ),
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
        propeller_key=propeller_key,
        propeller_blade_count=2,
        propeller_diameter_in=diameter_in,
        propeller_pitch_in=pitch_in,
        aerodynamic_cruise_speed_mps=float(cruise_speed_mps),
        cruise_speed_mps=course.cruise_speed_mps,
        cruise_power_cap_w=course.power_cap_w,
        energy_limited=energy_limited,
        completed_laps=course.laps,
        mission_flight_time_s=course.flight_time_s,
        usable_window_s=usable_window_s,
        clean_stall_speed_mps=float(stall_speed_mps),
        takeoff_stall_speed_mps=takeoff_stall_speed,
        takeoff_flap_deflection_deg=takeoff_flap_deg,
        landing_stall_speed_mps=landing_stall_speed,
        landing_flap_deflection_deg=landing_flap_deg,
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
