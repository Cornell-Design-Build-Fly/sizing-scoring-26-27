"""Fast algebraic trim with a continuously solved RPM and throttle.

The velocity/RPM grids bracket solutions; accepted RPMs are root solves in
the continuous propeller database, not selections from the RPM grid.
"""

import numpy as np
from aerosandbox import OperatingPoint
from scipy.optimize import brentq

from src.aero import cruise_analysis_fast as fast
from src.aero.custom_classes import CruiseCondition
from src.aero.flaps import clean_cl_max
from src.prop.prop_cruise_values import evaluate_cruise_grid
from src.prop.prop_classes import MPS_TO_MPH
from src.prop.prop_helper_functions import (
    make_battery_from_design,
    make_motor_from_design,
    motor_check,
)


CRUISE_THROTTLE_BOUNDS = (0.0, 1.0)
SPEED_SEARCH_POINTS = 48
SPEED_TOLERANCE_MPS = 1e-3


def _speed_interval(state, stall_speed):
    """Intersect angle bounds exactly: both trim angles are affine in 1/V^2."""
    v_min, v_max = fast.CRUISE_SPEED_BOUNDS_MPS
    v_min = max(v_min, stall_speed * (1.0 + 1e-8))
    if v_min > v_max:
        return None
    lo, hi = 1.0 / v_max**2, 1.0 / v_min**2
    at_lo = np.degrees(state(v_max)[:2])
    at_hi = np.degrees(state(v_min)[:2])
    if hi == lo:
        return (v_min, v_max) if all(
            bounds[0] <= angle <= bounds[1]
            for angle, bounds in zip(at_lo, (fast.ALPHA_BOUNDS_DEG, fast.ELEVATOR_BOUNDS_DEG))
        ) else None
    slopes = (at_hi - at_lo) / (hi - lo)
    offsets = at_lo - slopes * lo
    for slope, offset, bounds in zip(
        slopes, offsets, (fast.ALPHA_BOUNDS_DEG, fast.ELEVATOR_BOUNDS_DEG)
    ):
        if abs(slope) < 1e-12:
            if not bounds[0] <= offset <= bounds[1]:
                return None
        else:
            limits = sorted((np.asarray(bounds) - offset) / slope)
            lo, hi = max(lo, limits[0]), min(hi, limits[1])
    if lo > hi:
        return None
    return float(1.0 / np.sqrt(hi)), float(1.0 / np.sqrt(lo))


def cruise_analysis_continuous(
    design_vector, parameter_vector, cg, mass, mission, prop_database,
    *, throttle_bounds=None,
):
    """Find the fastest detected feasible trim inside the shared aero bounds.

    Solve thrust(RPM, V) = trimmed drag(V), then obtain throttle from the
    motor voltage and battery sag. No propeller-data extrapolation is accepted.
    A speed grid locates feasible intervals; bisection refines the upper edge
    of the fastest detected interval. Very narrow disconnected intervals can
    still be missed by this numerical search.
    """
    if throttle_bounds is None:
        throttle_bounds = CRUISE_THROTTLE_BOUNDS
    throttle_min, throttle_max = throttle_bounds
    if not 0.0 <= throttle_min < throttle_max <= 1.0:
        raise ValueError("Throttle bounds must satisfy 0 <= minimum < maximum <= 1.")
    failed = CruiseCondition(OperatingPoint(velocity=-1.0, alpha=-999.0), None, False)
    model = fast.fast_trim_model(
        design_vector, parameter_vector, (0.0, 0.0, 0.0), cg, mass, mission
    )
    if model is None:
        return failed
    state, wing_ar, weight = model
    stall = float(np.sqrt(
        2 * weight / (parameter_vector.rho * design_vector.wing_area * clean_cl_max(wing_ar))
    ))
    interval = _speed_interval(state, stall)
    if interval is None:
        return failed
    motor = make_motor_from_design(design_vector, parameter_vector)
    battery = make_battery_from_design(design_vector, parameter_vector)
    diameter, pitch = design_vector.propeller_for_mission(mission)
    current_limit = min(motor.max_current, battery.get_max_current())

    def prop_kwargs(speed, rpm):
        return dict(diameter_in=diameter, pitch_in=pitch, velocity_mph=speed * MPS_TO_MPH, rpm=rpm)

    def grid_at(speeds):
        return evaluate_cruise_grid(
            diameter, pitch, speeds, motor, battery, prop_database,
            min_rpm=3000, max_rpm=20000, rpm_step=100,
        )

    def solve_at(speed, grid, row):
        alpha, elevator, drag, _ = state(speed)
        alpha, elevator, drag = float(np.degrees(alpha)), float(np.degrees(elevator)), float(drag)
        if drag <= 0 or not all(
            bounds[0] - 1e-8 <= value <= bounds[1] + 1e-8
            for value, bounds in ((alpha, fast.ALPHA_BOUNDS_DEG), (elevator, fast.ELEVATOR_BOUNDS_DEG))
        ):
            return None
        residuals = grid.thrust_grid_n[row] - drag
        supported = np.asarray(prop_database.contains(**prop_kwargs(speed, grid.rpm_values)), dtype=bool)
        brackets = np.flatnonzero(
            supported[:-1] & supported[1:]
            & np.isfinite(residuals[:-1]) & np.isfinite(residuals[1:])
            & (residuals[:-1] * residuals[1:] <= 0)
        )
        best = None
        for index in brackets:
            def residual(rpm):
                thrust, _ = prop_database.evaluate(**prop_kwargs(speed, rpm))
                return float(thrust) - drag

            rpm = brentq(residual, grid.rpm_values[index], grid.rpm_values[index + 1], xtol=1e-6)
            if not bool(prop_database.contains(**prop_kwargs(speed, rpm))):
                continue
            thrust, torque = prop_database.evaluate(**prop_kwargs(speed, rpm))
            point = motor_check(float(torque), rpm, motor, battery)
            if not (
                point.passed
                and 0 < point.current_a <= current_limit
                and 0 < point.power_w <= motor.max_power
                and throttle_min <= point.throttle <= throttle_max
                and abs(float(thrust) - drag) / weight <= 1e-6
            ):
                continue
            if best is None or point.current_a < best[0]:
                best = (point.current_a, CruiseCondition(
                    operating_point=OperatingPoint(
                        velocity=float(speed), alpha=float(np.clip(alpha, *fast.ALPHA_BOUNDS_DEG))
                    ),
                    stall_speed=stall, converged=True, throttle=float(point.throttle),
                    elevator_deflection=float(np.clip(elevator, *fast.ELEVATOR_BOUNDS_DEG)),
                    thrust_n=float(thrust),
                    propeller_rpm=float(rpm),
                ))
        return None if best is None else best[1]

    # Most feasible designs trim at the speed or angle ceiling. Avoid a full
    # velocity x RPM database query when that endpoint already works.
    result = solve_at(interval[1], grid_at([interval[1]]), 0)
    if result is not None:
        return result
    speeds = np.linspace(*interval, SPEED_SEARCH_POINTS)
    grid = grid_at(speeds)
    for index in range(len(speeds) - 2, -1, -1):
        result = solve_at(float(speeds[index]), grid, index)
        if result is None:
            continue
        lower, upper = float(speeds[index]), float(speeds[index + 1])
        while upper - lower > SPEED_TOLERANCE_MPS:
            middle = (lower + upper) / 2.0
            candidate = solve_at(middle, grid_at([middle]), 0)
            if candidate is None:
                upper = middle
            else:
                lower, result = middle, candidate
        return result
    return failed
