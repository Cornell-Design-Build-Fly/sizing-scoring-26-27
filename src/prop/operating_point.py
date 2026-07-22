"""
Propulsion operating-point solver.

Determines the highest valid operating point for a given propeller,
motor, battery, airspeed, and throttle limit while enforcing motor
and electrical constraints.
"""
from __future__ import annotations

import math

from src.prop.prop_classes import (
    Battery,
    Motor,
)

from src.prop.prop_database import (
    ContinuousPropDatabase,
)

from src.prop.prop_helper_functions import (
    motor_check,
)

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
