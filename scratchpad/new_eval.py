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

    @property
    def course_energy_wh(self) -> float:
        return (
            self.straight_energy_wh
            + self.turn_energy_wh
            + self.reacceleration_energy_wh
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
    throttle_limit = design.cruise_throttle_for_mission(mission)
    liftoff_speed = requirements.liftoff_stall_speed_factor * stall_speed_mps
    climb_speed = requirements.climb_stall_speed_factor * stall_speed_mps
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
    # Candidate cruise speeds between just above stall and the aerodynamic
    # ceiling. The energy budget selects from this range instead of being
    # checked after the fact against one speed.
    cruise_speed_floor = min(1.02 * stall_speed_mps, cruise_speed_mps)
    cruise_sweep = np.linspace(cruise_speed_floor, cruise_speed_mps, 25)
    sample_speeds = np.unique(
        np.concatenate(
            (takeoff_speeds, (climb_speed, cruise_speed_mps), cruise_sweep)
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

    # ---- Climb --------------------------------------------------------------
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

    # ---- Whole-mission energy budget ---------------------------------------
    usable_energy = battery.vnom * battery.get_useable_capacity()
    allowed_energy = usable_energy * (1.0 - requirements.usable_energy_margin_fraction)
    fixed_energy = takeoff_energy + climb_energy

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

    def course_state(power_cap_w: float | None) -> _CourseState:
        """Flight profile flown at or below ``power_cap_w`` of pack draw."""

        cruise = select_cruise_points(
            grid,
            max_current_a=current_limit,
            cruise_throttle=throttle_limit,
            motor_max_power_w=motor_max_power,
            maximum_battery_power_w=power_cap_w,
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
            maximum_battery_power_w=power_cap_w,
            maximum_speed_mps=speed,
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
        return _CourseState(
            True,
            math.inf if power_cap_w is None else float(power_cap_w),
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
        )

    unrestricted = course_state(None)
    course = unrestricted
    energy_limited = False
    if unrestricted.feasible:
        unrestricted_energy = fixed_energy + unrestricted.course_energy_wh
        if unrestricted_energy > allowed_energy:
            # The pack cannot sustain the fastest trimmed speed for the whole
            # mission. Find the highest pack draw that it can. Course energy
            # rises with the power cap, so a bisection converges on the
            # boundary; every step is a re-selection over the cached grids.
            lower = 0.0
            upper = max(
                unrestricted.cruise_power_w,
                unrestricted.turn.battery_power_w,
            )
            best: _CourseState | None = None
            for _ in range(requirements.energy_budget_iterations):
                middle = 0.5 * (lower + upper)
                if middle <= 0.0:
                    break
                candidate = course_state(middle)
                if not candidate.feasible:
                    # Too little power to fly the course at all.
                    lower = middle
                    continue
                if fixed_energy + candidate.course_energy_wh <= allowed_energy:
                    best = candidate
                    lower = middle
                else:
                    upper = middle
            if best is not None:
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
        propeller_diameter_in=diameter_in,
        propeller_pitch_in=pitch_in,
        aerodynamic_cruise_speed_mps=float(cruise_speed_mps),
        cruise_speed_mps=course.cruise_speed_mps,
        cruise_power_cap_w=course.power_cap_w,
        energy_limited=energy_limited,
        completed_laps=course.laps,
        mission_flight_time_s=course.flight_time_s,
        usable_window_s=usable_window_s,
    )
