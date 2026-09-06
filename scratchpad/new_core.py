@dataclass(frozen=True)
class _TurnPerformance:
    feasible: bool
    speed_mps: float
    load_factor: float
    angular_rate_rad_s: float
    required_thrust_n: float
    battery_power_w: float
    maximum_rpm: float


_INFEASIBLE_TURN = _TurnPerformance(
    False, math.nan, 1.0, 0.0, math.inf, math.inf, 0.0
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
    )
