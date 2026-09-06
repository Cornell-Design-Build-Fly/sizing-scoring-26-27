from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from src.vectors import (
    DesignVector,
    ParameterVector,
)
from src.prop.prop_classes import (
    DEFAULT_VELOCITIES_MPS,
)
from src.prop.continuous_prop_database import (
    ContinuousPropDatabase,
    load_default_continuous_prop_database,
)
from src.prop.prop_cruise_values import (
    solve_cruise_samples,
)
from src.prop.prop_helper_functions import (
    _get_value,
    make_battery_from_design,
    make_motor_from_design,
)


CurveFit = tuple[float, float, float]



def prop_main(
    design_vector: DesignVector,
    parameter_vector: ParameterVector,
    mission: int,
    prop_database: ContinuousPropDatabase | None = None,
    velocities_mps: ArrayLike | None = None,
    disp_res: bool = False,
    knockdown: bool = False,
    knockdown_factor: float = 0.9,
    maximum_battery_power_w: float | None = None,
) -> tuple[CurveFit, CurveFit]:
    """
    Return the throttled thrust and flight-time quadratic fits.

    Each fit has the form:

        (a, b, c)

    where:

        output(V) = a*V**2 + b*V + c

    Velocity V is measured in m/s.
    """

    if mission not in (1, 2, 3):
        raise ValueError(
            "mission must be 1, 2, or 3."
        )

    if prop_database is None:
        prop_database = (
            load_default_continuous_prop_database()
        )

    if velocities_mps is None:
        fit_velocities_mps = (
            DEFAULT_VELOCITIES_MPS.copy()
        )
    else:
        fit_velocities_mps = np.asarray(
            velocities_mps,
            dtype=np.float64,
        ).reshape(-1)

    if fit_velocities_mps.size < 3:
        raise ValueError(
            "At least three velocity samples are required "
            "for a quadratic fit."
        )

    if not np.all(
        np.isfinite(fit_velocities_mps)
    ):
        raise ValueError(
            "Fit velocities must all be finite."
        )

    if np.any(fit_velocities_mps < 0.0):
        raise ValueError(
            "Fit velocities cannot be negative."
        )

    # Missions 1 and 2 fly one propeller; Mission 3 flies its own.
    diameter_in, pitch_in = design_vector.propeller_for_mission(mission)

    if mission in (1, 2):
        cruise_throttle = float(
            _get_value(
                design_vector,
                "cruise_throttle",
                1.0,
            )
        )
    else:
        cruise_throttle = float(
            _get_value(
                design_vector,
                "mission3_cruise_throttle",
                1.0,
            )
        )

    motor = make_motor_from_design(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
    )

    battery = make_battery_from_design(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
    )

    result = solve_cruise_samples(
        diameter_in=diameter_in,
        pitch_in=pitch_in,
        velocities_mps=fit_velocities_mps,
        motor=motor,
        battery=battery,
        max_current_a=min(motor.max_current, battery.get_max_current()),
        cruise_throttle=cruise_throttle,
        prop_database=prop_database,
        min_rpm=3000,
        max_rpm=20000,
        rpm_step=100,
        knockdown=knockdown,
        knockdown_factor=knockdown_factor,
        maximum_battery_power_w=maximum_battery_power_w,
    )

    thrust_samples_n = (
        result.thrust_samples_n.copy()
    )

    flight_time_samples_s = (
        result.flight_time_samples_s.copy()
    )


    # ---------------------------------------------------------
    # OPTIONAL OLD MATLAB BEHAVIOR
    # ---------------------------------------------------------
    # Uncomment this block if the team later decides that once
    # one velocity has zero thrust, every later velocity should
    # also be forced to zero.
    #
    # for index in range(
    #     1,
    #     len(thrust_samples_n),
    # ):
    #     if thrust_samples_n[index - 1] == 0.0:
    #         thrust_samples_n[index] = 0.0
    #         flight_time_samples_s[index] = 0.0
    # ---------------------------------------------------------

    # Reject the entire propulsion curve if no valid RPM exists
    # at any one of the required velocity points.
    curve_failed = np.any(result.failed_mask)

    if curve_failed:
        # Zero the displayed/stored sample values.
        thrust_samples_n.fill(0.0)
        flight_time_samples_s.fill(0.0)

        # Return exact zero polynomial coefficients.
        thrust_fit = np.zeros(
            3,
            dtype=np.float64,
        )

        flight_time_fit = np.zeros(
            3,
            dtype=np.float64,
        )

    else:
        thrust_fit = np.polyfit(
            fit_velocities_mps,
            thrust_samples_n,
            2,
        )

        flight_time_fit = np.polyfit(
            fit_velocities_mps,
            flight_time_samples_s,
            2,
        )

    if disp_res:
        print()
        print(
            f"Propeller: "
            f"{diameter_in:g}x{pitch_in:g}"
        )

        print(
            f"Cruise throttle: "
            f"{cruise_throttle:.3f}"
        )

        print()
        print(
            "Velocity [m/s] | RPM | Thrust [N] | "
            "Current [A] | Throttle | Valid RPMs"
        )

        for index, velocity_mps in enumerate(
            result.velocities_mps
        ):
            print(
                f"{velocity_mps:14.4f} | "
                f"{result.selected_rpm[index]:5.0f} | "
                f"{thrust_samples_n[index]:10.4f} | "
                f"{result.selected_current_a[index]:11.4f} | "
                f"{result.selected_throttle[index]:8.4f} | "
                f"{result.valid_rpm_count[index]:10d}"
            )

        print()
        print(
            f"Failed velocities: "
            f"{np.count_nonzero(result.failed_mask)}"
        )

    return (thrust_fit, flight_time_fit)
