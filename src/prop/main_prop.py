from __future__ import annotations
import math
import numpy as np

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

from src.prop.prop_cruise_values import cruise_values

from src.prop.prop_helper_functions import motor_check, _get_value, make_motor_from_design, make_battery_from_design






def prop_main(
    design_vector: DesignVector,
    parameter_vector: ParameterVector,
    mission: int,
    prop_database: ContinuousPropDatabase | None = None,
    velocities_mps: np.ndarray | None = None,
    disp_res: bool = False,
    knockdown: bool = False,
    knockdown_factor: float = 0.9,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
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

    # failure = False

    highest_failed_velocity = 0.0

    for i, velocity_mps in enumerate(velocities_mps):
        velocity_mph = float(velocity_mps * MPS_TO_MPH)


        throttled_thrust, throttled_time, _, failed_velocity = cruise_values(
            diameter_in=diameter_in,
            pitch_in=pitch_in,
            velocity_mph=velocity_mph,
            motor=motor,
            battery=battery,
            max_current_a=motor.max_current,
            cruise_throttle=cruise_throttle,
            prop_database=prop_database,
            knockdown=knockdown,
        )
        # if true_fail:
        #     failure = True
        if failed_velocity > highest_failed_velocity:
            highest_failed_velocity = failed_velocity

        # Match old MATLAB behavior:
        # once thrust becomes zero at a lower speed, keep later speeds at zero.

        if i > 0 and throttled_thrust_samples[i - 1] == 0.0:
            throttled_thrust_samples[i] = 0.0
            throttled_time_samples[i] = 0.0
        else:
            throttled_thrust_samples[i] = throttled_thrust
            throttled_time_samples[i] = throttled_time

    penalty = highest_failed_velocity


    throttled_time_samples = np.nan_to_num(
        throttled_time_samples,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    throttled_thrust_fit = np.polyfit(velocities_mps, throttled_thrust_samples, 2)

    throttled_time_fit = np.polyfit(velocities_mps, throttled_time_samples, 2)


    # if disp_res:
    #     plot_propulsion_result(result)

    return (
        (
            float(throttled_thrust_fit[0]),
            float(throttled_thrust_fit[1]),
            float(throttled_thrust_fit[2]),
        ),
        (
            float(throttled_time_fit[0]),
            float(throttled_time_fit[1]),
            float(throttled_time_fit[2]),
        ),
        # failure,
        penalty,
    )