"""
Main propulsion interface.

Provides the public entry points for the propulsion model. Evaluates
propulsion performance across a range of flight velocities and returns
quadratic curve fits for thrust and estimated flight time.
"""
from __future__ import annotations

import numpy as np

from src.prop.operating_point import cruise_values

from src.prop.plotting import plot_propulsion_result

from src.vectors import DesignVector, ParameterVector

from src.prop.prop_classes import (
    PropulsionCurveFit,
    MPS_TO_MPH,
    DEFAULT_VELOCITIES_MPS,
)

from src.prop.prop_database import (
    ContinuousPropDatabase,
    load_default_prop_database,
)

from src.prop.prop_helper_functions import (
    _get_value,
    make_motor_from_design,
    make_battery_from_design,
)

def prop_main(
    design_vector: DesignVector,
    parameter_vector: ParameterVector = ParameterVector,
    mission: int = 1,
    prop_database: ContinuousPropDatabase | None = None,
    velocities_mps: np.ndarray | None = None,
    disp_res: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
# changed return type from this: -> PropulsionCurveFit:

    """
    Evaluates propulsion performance across a range of flight velocities.

    Constructs the propulsion system from the design vector, computes
    maximum- and cruise-throttle operating points, and returns quadratic
    curve fits for thrust and estimated flight time.

    Returns:
        PropulsionCurveFit containing fitted thrust and flight-time
        curves along with the sampled operating-point data.
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

    # used to be this: return result
    return throttled_thrust_fit, max_thrust_fit

'''
def prop_main_interp(
    design_vector: DesignVector,
    parameter_vector: ParameterVector = ParameterVector,
    mission: int = 1,
    prop_database: ContinuousPropDatabase | None = None,
    velocities_mps: np.ndarray | None = None,
    disp_res: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Compatibility wrapper matching the MATLAB interface.

    Returns only the quadratic curve coefficients expected by
    legacy code.
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
    '''