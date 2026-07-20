# and here I am, catching you slacking, looking at the codebase for the first time...
from __future__ import annotations
from pathlib import Path
import re
import json
import math
from functools import lru_cache


import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from src.vectors import DesignVector, ParameterVector
from src.prop.prop_classes import (
    Battery,
    Motor,
    MotorCheckResult,
    PropInterpolants,
    PropulsionCurveFit,
    MPS_TO_MPH,
    DEFAULT_VELOCITIES_MPS,
)

from src.prop.prop_database import (
    ContinuousPropDatabase,
    load_default_prop_database,
)

from src.prop.prop_helper_functions import battery_resistance, motor_properties, motor_check, _get_value, make_motor_from_design, make_battery_from_design




'''Cruise Values'''

def cruise_values(
    diameter_in: float,
    pitch_in: float,
    velocity_mph: float,
    motor: Motor,
    battery: Battery,
    max_current_a: float,
    cruise_throttle: float,
    prop_database: ContinuousPropDatabase,
    min_rpm: int = 3000,
    max_rpm: int = 16000,
    rpm_step: int = 100,
) -> tuple[float, float]:
    """
    Finds the highest valid thrust at a given airspeed and throttle limit.

    Inputs:
        diameter_in:
            Propeller diameter [in]

        pitch_in:
            Propeller pitch [in]

        velocity_mph:
            Aircraft forward speed [mph]

        motor:
            Motor object

        battery:
            Battery object

        max_current_a:
            Current limit [A]

        cruise_throttle:
            Maximum allowed throttle for this condition.
            Use 1.0 for max-throttle thrust.
            Use something like 0.7 or 0.9 for cruise-throttle thrust.

        prop_database:
            ContinuousPropDatabase object with thrust/torque interpolation.

    Returns:
        best_thrust_n:
            Highest valid thrust found [N]

        best_flight_time_s:
            Estimated flight time at that operating point [s]
    """

    if diameter_in <= 0:
        raise ValueError("Propeller diameter must be positive.")

    if pitch_in <= 0:
        raise ValueError("Propeller pitch must be positive.")

    if velocity_mph < 0:
        raise ValueError("Velocity cannot be negative.")

    if max_current_a <= 0:
        raise ValueError("Max current must be positive.")

    if cruise_throttle <= 0:
        return 0.0, 0.0

    # Do not allow throttle limit above 1.
    cruise_throttle = min(float(cruise_throttle), 1.0)

    best_thrust_n = -math.inf
    best_flight_time_s = math.inf

    rpm_low = int(min_rpm)
    rpm_high = int(max_rpm)

    while (rpm_high - rpm_low) >= rpm_step:
        rpm_mid = int(round((rpm_low + rpm_high) / 2))

        thrust_n = prop_database.thrust(
            diameter_in=diameter_in,
            pitch_in=pitch_in,
            velocity_mph=velocity_mph,
            rpm=rpm_mid,
        )

        torque_nm = prop_database.torque(
            diameter_in=diameter_in,
            pitch_in=pitch_in,
            velocity_mph=velocity_mph,
            rpm=rpm_mid,
        )

        if not math.isfinite(thrust_n) or not math.isfinite(torque_nm):
            rpm_low = rpm_mid + 1
            continue

        check = motor_check(
            torque=torque_nm,
            rpm=rpm_mid,
            motor=motor,
            battery=battery,
        )

        within_limits = (
            check.passed
            and check.throttle <= cruise_throttle
            and check.power_w <= motor.max_power
            and check.current_a <= max_current_a
        )

        if within_limits:
            if thrust_n > best_thrust_n:
                best_thrust_n = thrust_n
                best_flight_time_s = check.flight_time_s

            # This RPM works, so try a higher RPM.
            rpm_low = rpm_mid + 1

        else:
            # This RPM does not work, so try a lower RPM.
            rpm_high = rpm_mid - 1

    if best_thrust_n == -math.inf:
        return 0.0, 0.0

    return float(best_thrust_n), float(best_flight_time_s)

'''PROP MAIN BLOCK'''




def prop_main(
    design_vector: DesignVector,
    parameter_vector: ParameterVector = ParameterVector,
    mission: int = 1,
    prop_database: ContinuousPropDatabase | None = None,
    velocities_mps: np.ndarray | None = None,
    disp_res: bool = False,
) -> PropulsionCurveFit:
    """
    Main propulsion model.

    Continuous diameter/pitch replacement for old MATLAB propMainInterp.m.
    """

    if mission not in (1, 2, 3):
        raise ValueError("mission must be 1, 2, or 3.")

    if prop_database is None:
        prop_database = load_default_prop_database()

    if velocities_mps is None:
        velocities_mps = DEFAULT_VELOCITIES_MPS.copy()
    else:
        velocities_mps = np.asarray(velocities_mps, dtype=float).reshape(-1)

    if len(velocities_mps) < 3:
        raise ValueError("Need at least 3 velocity samples for quadratic polyfit.")

    diameter_in = float(_get_value(design_vector, "prop_diameter_in", 14.0))
    pitch_in = float(_get_value(design_vector, "prop_pitch_in", 10.0))

    if mission in (1, 2):
        cruise_throttle = float(_get_value(design_vector, "cruise_throttle", 0.90))
    else:
        cruise_throttle = float(
            _get_value(design_vector, "mission3_cruise_throttle", 0.85)
        )

    motor = make_motor_from_design(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
    )

    battery = make_battery_from_design(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
    )

    max_thrust_samples = np.zeros_like(velocities_mps, dtype=float)
    throttled_thrust_samples = np.zeros_like(velocities_mps, dtype=float)
    max_time_samples = np.zeros_like(velocities_mps, dtype=float)
    throttled_time_samples = np.zeros_like(velocities_mps, dtype=float)

    for i, velocity_mps in enumerate(velocities_mps):
        velocity_mph = float(velocity_mps * MPS_TO_MPH)

        max_thrust, max_time = cruise_values(
            diameter_in=diameter_in,
            pitch_in=pitch_in,
            velocity_mph=velocity_mph,
            motor=motor,
            battery=battery,
            max_current_a=motor.max_current,
            cruise_throttle=1.0,
            prop_database=prop_database,
        )

        throttled_thrust, throttled_time = cruise_values(
            diameter_in=diameter_in,
            pitch_in=pitch_in,
            velocity_mph=velocity_mph,
            motor=motor,
            battery=battery,
            max_current_a=motor.max_current,
            cruise_throttle=cruise_throttle,
            prop_database=prop_database,
        )

        # Match old MATLAB behavior:
        # once thrust becomes zero at a lower speed, keep later speeds at zero.
        if i > 0 and max_thrust_samples[i - 1] == 0.0:
            max_thrust_samples[i] = 0.0
            max_time_samples[i] = 0.0
        else:
            max_thrust_samples[i] = max_thrust
            max_time_samples[i] = max_time

        if i > 0 and throttled_thrust_samples[i - 1] == 0.0:
            throttled_thrust_samples[i] = 0.0
            throttled_time_samples[i] = 0.0
        else:
            throttled_thrust_samples[i] = throttled_thrust
            throttled_time_samples[i] = throttled_time

    max_time_samples = np.nan_to_num(
        max_time_samples,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    throttled_time_samples = np.nan_to_num(
        throttled_time_samples,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    max_thrust_fit = np.polyfit(velocities_mps, max_thrust_samples, 2)
    throttled_thrust_fit = np.polyfit(velocities_mps, throttled_thrust_samples, 2)

    max_time_fit = np.polyfit(velocities_mps, max_time_samples, 2)
    throttled_time_fit = np.polyfit(velocities_mps, throttled_time_samples, 2)

    result = PropulsionCurveFit(
        throttled_thrust=throttled_thrust_fit,
        max_thrust=max_thrust_fit,
        throttled_time=throttled_time_fit,
        max_time=max_time_fit,
        sample_velocities_mps=velocities_mps,
        throttled_thrust_samples=throttled_thrust_samples,
        max_thrust_samples=max_thrust_samples,
        throttled_time_samples=throttled_time_samples,
        max_time_samples=max_time_samples,
    )

    if disp_res:
        plot_propulsion_result(result)

    return result


def prop_main_interp(
    design_vector: DesignVector,
    parameter_vector: ParameterVector = ParameterVector,
    mission: int = 1,
    prop_database: ContinuousPropDatabase | None = None,
    velocities_mps: np.ndarray | None = None,
    disp_res: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB-style wrapper.

    Returns:
        p_throttled_thrust, p_max_thrust, p_throttled_t, p_max_t
    """

    result = prop_main(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
        mission=mission,
        prop_database=prop_database,
        velocities_mps=velocities_mps,
        disp_res=disp_res,
    )

    return (
        result.throttled_thrust,
        result.max_thrust,
        result.throttled_time,
        result.max_time,
    )


def evaluate_curve(coefficients: np.ndarray, velocity_mps):
    """
    Evaluates a polynomial curve fit.
    """
    return np.polyval(coefficients, velocity_mps)


def plot_propulsion_result(result: PropulsionCurveFit) -> None:
    """
    Optional debug plotting helper.
    """

    import matplotlib.pyplot as plt

    velocities = result.sample_velocities_mps

    plt.figure()
    plt.scatter(velocities, result.throttled_thrust_samples, label="Cruise samples")
    plt.scatter(velocities, result.max_thrust_samples, label="Max samples")
    plt.plot(
        velocities,
        evaluate_curve(result.throttled_thrust, velocities),
        label="Cruise fit",
    )
    plt.plot(
        velocities,
        evaluate_curve(result.max_thrust, velocities),
        label="Max fit",
    )
    plt.xlabel("Velocity [m/s]")
    plt.ylabel("Thrust [N]")
    plt.title("Propulsion thrust curve")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.scatter(velocities, result.throttled_time_samples, label="Cruise samples")
    plt.scatter(velocities, result.max_time_samples, label="Max samples")
    plt.plot(
        velocities,
        evaluate_curve(result.throttled_time, velocities),
        label="Cruise fit",
    )
    plt.plot(
        velocities,
        evaluate_curve(result.max_time, velocities),
        label="Max fit",
    )
    plt.xlabel("Velocity [m/s]")
    plt.ylabel("Flight time [s]")
    plt.title("Propulsion flight-time curve")
    plt.grid(True)
    plt.legend()

    plt.show()
