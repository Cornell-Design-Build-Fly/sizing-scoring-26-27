from types import SimpleNamespace

import numpy as np

from src.prop.main_prop import (
    load_default_prop_database,
    prop_main,
)

from src.prop.plotting import evaluate_curve


def main():
    print("=== Loading prop database ===")
    prop_database = load_default_prop_database()

    design_vector = SimpleNamespace(
        batt_capacity=4.5,
        prop_diameter_in=14.0,
        prop_pitch_in=10.0,
        motor_kv=335.0,
        motor_max_power=2200.0,
        cruise_throttle=0.85,
        mission3_cruise_throttle=0.75,
    )

    parameter_vector = SimpleNamespace(
        voltage=22.2,
        max_current=100.0,
        usable_battery_fraction=0.85,
        num_battery_cells=6,
    )

    print("\n=== Mission 1 ===")
    throttled_fit_m1, max_fit_m1 = prop_main(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
        mission=1,
        prop_database=prop_database,
        disp_res=False,
    )

    print("Throttled thrust fit:")
    print(throttled_fit_m1)

    print("\nMax thrust fit:")
    print(max_fit_m1)

    print("\n=== Mission 3 ===")
    throttled_fit_m3, max_fit_m3 = prop_main(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
        mission=3,
        prop_database=prop_database,
        disp_res=False,
    )

    print("Throttled thrust fit:")
    print(throttled_fit_m3)

    print("\nMax thrust fit:")
    print(max_fit_m3)

    print("\n=== Evaluating curves ===")

    test_speeds = np.array([5.0, 10.0, 15.0, 20.0])

    throttled_eval = evaluate_curve(throttled_fit_m1, test_speeds)
    max_eval = evaluate_curve(max_fit_m1, test_speeds)

    for V, T_cruise, T_max in zip(test_speeds, throttled_eval, max_eval):
        print(
            f"{V:5.1f} m/s | "
            f"Cruise = {T_cruise:8.3f} N | "
            f"Max = {T_max:8.3f} N"
        )

    print("\n=== Sanity checks ===")

    assert throttled_fit_m1.shape == (3,)
    assert max_fit_m1.shape == (3,)
    assert throttled_fit_m3.shape == (3,)
    assert max_fit_m3.shape == (3,)

    assert np.all(np.isfinite(throttled_fit_m1))
    assert np.all(np.isfinite(max_fit_m1))
    assert np.all(np.isfinite(throttled_fit_m3))
    assert np.all(np.isfinite(max_fit_m3))

    # Max throttle should generally produce at least as much thrust
    assert np.all(max_eval + 1e-6 >= throttled_eval)

    print("\nPassed interface sanity checks.")


if __name__ == "__main__":
    main()