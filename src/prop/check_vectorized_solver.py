from __future__ import annotations

from time import perf_counter

import numpy as np

from src.vectors import (
    DesignVector,
    ParameterVector,
)
from src.prop.prop_classes import (
    DEFAULT_VELOCITIES_MPS,
)
from src.prop.continuous_prop_database import (
    load_default_continuous_prop_database,
)
from src.prop.main_prop import prop_main
from src.prop.prop_cruise_values import (
    solve_cruise_samples,
)
from src.prop.prop_helper_functions import (
    make_battery_from_design,
    make_motor_from_design,
)


def main() -> None:
    design_vector = DesignVector(
        batt_capacity=3.0,
        prop_diameter_in=14.0,
        prop_pitch_in=10.0,
        motor_kv=520.0,
        motor_max_power=2000.0,
        cruise_throttle=0.90,
        mission3_cruise_throttle=0.90,
    )

    parameter_vector = ParameterVector()

    # Load the database before starting the timer.
    database = (
        load_default_continuous_prop_database()
    )

    motor = make_motor_from_design(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
    )

    battery = make_battery_from_design(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
    )

    start_time = perf_counter()

    result = solve_cruise_samples(
        diameter_in=design_vector.prop_diameter_in,
        pitch_in=design_vector.prop_pitch_in,
        velocities_mps=DEFAULT_VELOCITIES_MPS,
        motor=motor,
        battery=battery,
        max_current_a=motor.max_current,
        cruise_throttle=(
            design_vector.cruise_throttle
        ),
        prop_database=database,
        min_rpm=3000,
        max_rpm=16000,
        rpm_step=100,
    )

    solver_runtime_s = (
        perf_counter() - start_time
    )

    print()
    print("Vectorized solver result:")
    print()

    print(
        "Velocity [m/s] | Selected RPM | "
        "Thrust [N] | Current [A] | "
        "Throttle | Power [W] | Valid RPMs"
    )

    for index, velocity_mps in enumerate(
        result.velocities_mps
    ):
        print(
            f"{velocity_mps:14.4f} | "
            f"{result.selected_rpm[index]:12.0f} | "
            f"{result.thrust_samples_n[index]:10.4f} | "
            f"{result.selected_current_a[index]:11.4f} | "
            f"{result.selected_throttle[index]:8.4f} | "
            f"{result.selected_power_w[index]:9.2f} | "
            f"{result.valid_rpm_count[index]:10d}"
        )

    # Confirm every selected RPM lies on the 100-RPM grid.
    successful_mask = ~result.failed_mask

    if np.any(successful_mask):
        selected_rpms = result.selected_rpm[
            successful_mask
        ]

        rpm_grid_error = np.mod(
            selected_rpms - 3000.0,
            100.0,
        )

        if not np.allclose(
            rpm_grid_error,
            0.0,
            atol=1.0e-10,
        ):
            raise AssertionError(
                "A selected RPM is not on the 100-RPM grid."
            )

    if np.any(
        ~np.isfinite(result.thrust_samples_n)
    ):
        raise AssertionError(
            "Thrust samples contain non-finite values."
        )

    if np.any(
        ~np.isfinite(
            result.flight_time_samples_s
        )
    ):
        raise AssertionError(
            "Flight-time samples contain non-finite values."
        )

    thrust_fit, time_fit = prop_main(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
        mission=1,
        prop_database=database,
    )

    print()
    print(
        f"Solver runtime only: "
        f"{solver_runtime_s:.6f} s"
    )

    print(
        f"Failed velocities: "
        f"{np.count_nonzero(result.failed_mask)}"
    )

    print()
    print(
        f"Throttled thrust fit: {thrust_fit}"
    )

    print(
        f"Throttled time fit:   {time_fit}"
    )


if __name__ == "__main__":
    main()