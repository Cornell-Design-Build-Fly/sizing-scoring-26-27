from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.prop.continuous_prop_database import (
    ContinuousPropDatabase,
)
from src.prop.prop_classes import (
    Battery,
    Motor,
    MPS_TO_MPH,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

MIN_POSITIVE_CURRENT_A = 1.0e-6
MIN_POSITIVE_VOLTAGE_V = 1.0e-6


@dataclass(frozen=True, slots=True)
class CruiseGridResult:
    """Selected propulsion operating point at every airspeed."""

    velocities_mps: FloatArray
    rpm_values: FloatArray

    thrust_samples_n: FloatArray
    flight_time_samples_s: FloatArray

    selected_rpm: FloatArray
    selected_current_a: FloatArray
    selected_throttle: FloatArray
    selected_power_w: FloatArray

    valid_rpm_count: IntArray
    failed_mask: BoolArray


@dataclass(frozen=True, slots=True)
class CruiseGrid:
    """Raw velocity x RPM propulsion grid, independent of any operating cap.

    Evaluating the propeller database dominates the runtime of a design
    evaluation, and every cap the model applies (throttle, current, motor
    power, battery power) is a pure mask over this same grid.  Separating the
    evaluation from the selection lets one database query serve many candidate
    operating limits, which is what the mission energy budget needs.
    """

    velocities_mps: FloatArray
    rpm_values: FloatArray
    rpm_grid: FloatArray

    thrust_grid_n: FloatArray
    torque_grid_nm: FloatArray
    current_grid_a: FloatArray
    voltage_sag_grid_v: FloatArray
    voltage_required_grid_v: FloatArray
    power_grid_w: FloatArray
    throttle_grid: FloatArray
    flight_time_grid_s: FloatArray
    battery_power_grid_w: FloatArray

    # Validity that does not depend on any selectable cap.
    base_valid_mask: BoolArray

    battery_nominal_voltage_v: float


def _validated_velocities(velocities_mps: ArrayLike) -> FloatArray:
    velocities = np.asarray(velocities_mps, dtype=np.float64).reshape(-1)
    if velocities.size == 0:
        raise ValueError("At least one velocity must be provided.")
    if not np.all(np.isfinite(velocities)):
        raise ValueError("Velocities must all be finite.")
    if np.any(velocities < 0.0):
        raise ValueError("Velocities cannot be negative.")
    return velocities


def evaluate_cruise_grid(
    diameter_in: float,
    pitch_in: float,
    velocities_mps: ArrayLike,
    motor: Motor,
    battery: Battery,
    prop_database: ContinuousPropDatabase,
    min_rpm: int = 3000,
    max_rpm: int = 16000,
    rpm_step: int = 100,
) -> CruiseGrid:
    """Query the propeller database once and build the motor/battery grids."""

    velocities = _validated_velocities(velocities_mps)

    if diameter_in <= 0.0:
        raise ValueError("Propeller diameter must be positive.")
    if pitch_in <= 0.0:
        raise ValueError("Propeller pitch must be positive.")
    if min_rpm <= 0:
        raise ValueError("Minimum RPM must be positive.")
    if max_rpm < min_rpm:
        raise ValueError("Maximum RPM cannot be below minimum RPM.")
    if rpm_step <= 0:
        raise ValueError("RPM step must be positive.")
    if (max_rpm - min_rpm) % rpm_step != 0:
        raise ValueError("The RPM range must be divisible by rpm_step.")

    rpm_values = np.arange(min_rpm, max_rpm + rpm_step, rpm_step, dtype=np.float64)

    # The prop database accepts velocity in mph.
    velocities_mph = velocities * MPS_TO_MPH

    # Every row is one airspeed.  Every column is one RPM.
    velocity_grid_mph, rpm_grid = np.meshgrid(
        velocities_mph,
        rpm_values,
        indexing="ij",
    )

    thrust_grid, torque_grid = prop_database.evaluate(
        diameter_in=diameter_in,
        pitch_in=pitch_in,
        velocity_mph=velocity_grid_mph,
        rpm=rpm_grid,
    )

    thrust_grid = np.asarray(thrust_grid, dtype=np.float64)
    torque_grid = np.asarray(torque_grid, dtype=np.float64)

    expected_shape = (velocities.size, rpm_values.size)
    if thrust_grid.shape != expected_shape:
        raise ValueError(
            "Prop database returned thrust shape "
            f"{thrust_grid.shape}; expected {expected_shape}."
        )
    if torque_grid.shape != expected_shape:
        raise ValueError(
            "Prop database returned torque shape "
            f"{torque_grid.shape}; expected {expected_shape}."
        )

    kt_nm_per_a = float(motor.get_kt())
    motor_resistance_ohm = float(motor.get_rm())
    no_load_current_a = float(motor.get_I0())
    battery_resistance_ohm = float(battery.get_Rb())
    usable_capacity_ah = float(battery.get_useable_capacity())

    if kt_nm_per_a <= 0.0:
        raise ValueError("Motor torque constant must be positive.")
    if motor_resistance_ohm < 0.0:
        raise ValueError("Motor resistance cannot be negative.")
    if battery_resistance_ohm < 0.0:
        raise ValueError("Battery resistance cannot be negative.")
    if usable_capacity_ah <= 0.0:
        raise ValueError("Usable battery capacity must be positive.")

    # Current needed to produce the requested propeller torque.
    current_grid_a = torque_grid / kt_nm_per_a + no_load_current_a

    # Battery voltage after internal-resistance voltage drop.
    voltage_sag_grid_v = battery.vnom - current_grid_a * battery_resistance_ohm

    # Voltage required to reach the requested RPM and current.
    voltage_required_grid_v = (
        rpm_grid / motor.kv + current_grid_a * motor_resistance_ohm
    )

    # Electrical power drawn from the sagged battery voltage.
    power_grid_w = current_grid_a * voltage_sag_grid_v

    # Pack depletion is an amp-hour draw.  Charging it at the nominal pack
    # voltage keeps power and the nominal ``V * Ah`` pack energy on the same
    # basis; terminal ``I * V_sag`` silently discards the internal-resistance
    # loss that still comes out of the pack.
    battery_power_grid_w = current_grid_a * battery.vnom

    throttle_grid = np.full(expected_shape, np.inf, dtype=np.float64)
    np.divide(
        voltage_required_grid_v,
        voltage_sag_grid_v,
        out=throttle_grid,
        where=(voltage_sag_grid_v > MIN_POSITIVE_VOLTAGE_V),
    )

    flight_time_grid_s = np.full(expected_shape, np.inf, dtype=np.float64)
    np.divide(
        usable_capacity_ah * 3600.0,
        current_grid_a,
        out=flight_time_grid_s,
        where=(current_grid_a > MIN_POSITIVE_CURRENT_A),
    )

    finite_mask = (
        np.isfinite(thrust_grid)
        & np.isfinite(torque_grid)
        & np.isfinite(current_grid_a)
        & np.isfinite(voltage_sag_grid_v)
        & np.isfinite(voltage_required_grid_v)
        & np.isfinite(power_grid_w)
        & np.isfinite(throttle_grid)
        & np.isfinite(flight_time_grid_s)
    )

    base_valid_mask = (
        finite_mask
        # This model does not represent battery regeneration.
        & (current_grid_a > MIN_POSITIVE_CURRENT_A)
        # Battery voltage must remain positive.
        & (voltage_sag_grid_v > MIN_POSITIVE_VOLTAGE_V)
        # Required motor voltage must be physically positive.
        & (voltage_required_grid_v > 0.0)
        # Preserve the nominal-voltage check from the old model.
        & (voltage_required_grid_v <= battery.vnom)
        & (throttle_grid >= 0.0)
        & (power_grid_w > 0.0)
    )

    return CruiseGrid(
        velocities_mps=velocities.copy(),
        rpm_values=rpm_values,
        rpm_grid=rpm_grid,
        thrust_grid_n=thrust_grid,
        torque_grid_nm=torque_grid,
        current_grid_a=current_grid_a,
        voltage_sag_grid_v=voltage_sag_grid_v,
        voltage_required_grid_v=voltage_required_grid_v,
        power_grid_w=power_grid_w,
        throttle_grid=throttle_grid,
        flight_time_grid_s=flight_time_grid_s,
        battery_power_grid_w=battery_power_grid_w,
        base_valid_mask=base_valid_mask,
        battery_nominal_voltage_v=float(battery.vnom),
    )


def select_cruise_points(
    grid: CruiseGrid,
    *,
    max_current_a: float,
    cruise_throttle: float,
    motor_max_power_w: float,
    maximum_battery_power_w: float | None = None,
    knockdown: bool = False,
    knockdown_factor: float = 0.9,
    minimum_thrust_n: ArrayLike | None = None,
) -> CruiseGridResult:
    """Select one operating point per airspeed under the supplied limits.

    ``maximum_battery_power_w`` caps the nominal-equivalent pack draw.  It is
    how the mission energy budget is expressed as a physical limit: the
    aircraft simply cannot fly an operating point that would empty the pack
    before the mission window closes.
    """

    if max_current_a <= 0.0:
        raise ValueError("Maximum current must be positive.")
    if not 0.0 < cruise_throttle <= 1.0:
        raise ValueError("Cruise throttle must be between 0 and 1.")
    if knockdown and not 0.0 < knockdown_factor <= 1.0:
        raise ValueError("Knockdown factor must be between 0 and 1.")
    if maximum_battery_power_w is not None and not (
        np.isfinite(maximum_battery_power_w) and maximum_battery_power_w > 0.0
    ):
        raise ValueError("Maximum battery power must be finite and positive.")

    velocities = grid.velocities_mps
    expected_shape = grid.thrust_grid_n.shape

    required_thrust = None
    if minimum_thrust_n is not None:
        required_thrust = np.asarray(minimum_thrust_n, dtype=np.float64)
        if required_thrust.ndim == 0:
            required_thrust = np.full(velocities.size, float(required_thrust))
        else:
            required_thrust = required_thrust.reshape(-1)
        if required_thrust.size != velocities.size:
            raise ValueError(
                "minimum_thrust_n must be scalar or match the velocity count."
            )
        if not np.all(np.isfinite(required_thrust)) or np.any(required_thrust < 0.0):
            raise ValueError("minimum_thrust_n must be finite and nonnegative.")

    valid_mask = (
        grid.base_valid_mask
        # Current limit.
        & (grid.current_grid_a <= max_current_a)
        # Cruise-throttle limit.
        & (grid.throttle_grid <= cruise_throttle)
        # Electrical power limit.
        & (grid.power_grid_w <= motor_max_power_w)
    )
    if maximum_battery_power_w is not None:
        valid_mask = valid_mask & (
            grid.battery_power_grid_w <= maximum_battery_power_w
        )

    selectable_thrust_grid = (
        grid.thrust_grid_n * knockdown_factor if knockdown else grid.thrust_grid_n
    )
    selection_mask = valid_mask
    if required_thrust is not None:
        selection_mask = valid_mask & (
            selectable_thrust_grid >= required_thrust[:, np.newaxis]
        )

    valid_rpm_count = np.count_nonzero(selection_mask, axis=1).astype(np.int64)
    failed_mask = valid_rpm_count == 0

    if required_thrust is None:
        # Give invalid points negative infinity so argmax ignores them.
        selection_values = np.where(selection_mask, selectable_thrust_grid, -np.inf)
        best_rpm_indices = np.argmax(selection_values, axis=1)
    else:
        # At fixed nominal pack voltage, minimum current is minimum battery
        # depletion. Terminal power alone would favor voltage-sag loss.
        selection_values = np.where(selection_mask, grid.current_grid_a, np.inf)
        best_rpm_indices = np.argmin(selection_values, axis=1)

    velocity_indices = np.arange(velocities.size)

    def _take(values: FloatArray) -> FloatArray:
        return np.where(failed_mask, 0.0, values[velocity_indices, best_rpm_indices])

    return CruiseGridResult(
        velocities_mps=velocities.copy(),
        rpm_values=grid.rpm_values,
        thrust_samples_n=_take(selectable_thrust_grid),
        flight_time_samples_s=_take(grid.flight_time_grid_s),
        selected_rpm=_take(grid.rpm_grid),
        selected_current_a=_take(grid.current_grid_a),
        selected_throttle=_take(grid.throttle_grid),
        selected_power_w=_take(grid.power_grid_w),
        valid_rpm_count=valid_rpm_count,
        failed_mask=failed_mask,
    )


def solve_cruise_samples(
    diameter_in: float,
    pitch_in: float,
    velocities_mps: ArrayLike,
    motor: Motor,
    battery: Battery,
    max_current_a: float,
    cruise_throttle: float,
    prop_database: ContinuousPropDatabase,
    min_rpm: int = 3000,
    max_rpm: int = 16000,
    rpm_step: int = 100,
    knockdown: bool = False,
    knockdown_factor: float = 0.9,
    minimum_thrust_n: ArrayLike | None = None,
    maximum_battery_power_w: float | None = None,
) -> CruiseGridResult:
    """
    Evaluate the full velocity x RPM grid.

    By default, select the maximum-thrust valid RPM independently at each
    velocity.  If ``minimum_thrust_n`` is supplied, select the lowest-current
    valid RPM that supplies at least that much thrust instead.  The latter is
    used for course segments such as sustained turns, where the required
    aerodynamic force is known and maximum thrust would overstate energy use.

    ``maximum_battery_power_w`` additionally caps the nominal-equivalent pack
    draw, which is how a mission energy budget becomes a flight limit.

    Propeller-database input units:
        diameter: inches
        pitch: inches
        velocity: mph
        RPM: revolutions per minute

    Solver velocity input and output units:
        velocity: m/s
    """

    grid = evaluate_cruise_grid(
        diameter_in=diameter_in,
        pitch_in=pitch_in,
        velocities_mps=velocities_mps,
        motor=motor,
        battery=battery,
        prop_database=prop_database,
        min_rpm=min_rpm,
        max_rpm=max_rpm,
        rpm_step=rpm_step,
    )
    return select_cruise_points(
        grid,
        max_current_a=max_current_a,
        cruise_throttle=cruise_throttle,
        motor_max_power_w=float(motor.max_power),
        maximum_battery_power_w=maximum_battery_power_w,
        knockdown=knockdown,
        knockdown_factor=knockdown_factor,
        minimum_thrust_n=minimum_thrust_n,
    )
