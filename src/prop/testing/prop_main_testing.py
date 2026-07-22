from types import SimpleNamespace

import numpy as np

from src.prop.main_prop import (
    load_default_prop_database,
    prop_main,
    prop_main_interp,
)

from src.prop.plotting import evaluate_curve

def main():
    print("=== Loading prop database ===")
    prop_database = load_default_prop_database()

    print("\n=== Creating test design/parameters ===")

    # Using SimpleNamespace avoids constructor/name issues while your
    # DesignVector and ParameterVector are still being updated.
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

    print("Design vector:", design_vector)
    print("Parameter vector:", parameter_vector)

    print("\n=== Running prop_main for Mission 1 ===")
    result_m1 = prop_main(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
        mission=1,
        prop_database=prop_database,
        disp_res=False,
    )

    print("\nMission 1 polynomial fits:")
    print("throttled_thrust:", result_m1.throttled_thrust)
    print("max_thrust:", result_m1.max_thrust)
    print("throttled_time:", result_m1.throttled_time)
    print("max_time:", result_m1.max_time)

    print("\nMission 1 sample velocities [m/s]:")
    print(result_m1.sample_velocities_mps)

    print("\nMission 1 throttled thrust samples [N]:")
    print(result_m1.throttled_thrust_samples)

    print("\nMission 1 max thrust samples [N]:")
    print(result_m1.max_thrust_samples)

    print("\nMission 1 throttled time samples [s]:")
    print(result_m1.throttled_time_samples)

    print("\nMission 1 max time samples [s]:")
    print(result_m1.max_time_samples)

    print("\n=== Running prop_main for Mission 3 ===")
    result_m3 = prop_main(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
        mission=3,
        prop_database=prop_database,
        disp_res=False,
    )

    print("\nMission 3 polynomial fits:")
    print("throttled_thrust:", result_m3.throttled_thrust)
    print("max_thrust:", result_m3.max_thrust)
    print("throttled_time:", result_m3.throttled_time)
    print("max_time:", result_m3.max_time)

    print("\nMission 3 throttled thrust samples [N]:")
    print(result_m3.throttled_thrust_samples)

    print("\n=== Testing MATLAB-style wrapper ===")
    p_throttled_thrust, p_max_thrust, p_throttled_t, p_max_t = prop_main_interp(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
        mission=1,
        prop_database=prop_database,
        disp_res=False,
    )

    print("p_throttled_thrust:", p_throttled_thrust)
    print("p_max_thrust:", p_max_thrust)
    print("p_throttled_t:", p_throttled_t)
    print("p_max_t:", p_max_t)

    print("\n=== Evaluating thrust curves at a few speeds ===")
    test_speeds_mps = np.array([5.0, 10.0, 15.0, 20.0])

    throttled_thrust_eval = evaluate_curve(
        result_m1.throttled_thrust,
        test_speeds_mps,
    )

    max_thrust_eval = evaluate_curve(
        result_m1.max_thrust,
        test_speeds_mps,
    )

    for speed, thrust_cruise, thrust_max in zip(
        test_speeds_mps,
        throttled_thrust_eval,
        max_thrust_eval,
    ):
        print(
            f"V = {speed:5.1f} m/s | "
            f"cruise thrust = {thrust_cruise:9.3f} N | "
            f"max thrust = {thrust_max:9.3f} N"
        )

    print("\n=== Sanity checks ===")

    assert np.all(np.isfinite(result_m1.throttled_thrust))
    assert np.all(np.isfinite(result_m1.max_thrust))
    assert np.all(np.isfinite(result_m1.throttled_time))
    assert np.all(np.isfinite(result_m1.max_time))

    assert np.all(np.isfinite(result_m1.throttled_thrust_samples))
    assert np.all(np.isfinite(result_m1.max_thrust_samples))
    assert np.all(np.isfinite(result_m1.throttled_time_samples))
    assert np.all(np.isfinite(result_m1.max_time_samples))

    assert np.any(result_m1.max_thrust_samples > 0.0)

    # Max throttle should usually produce at least as much thrust as cruise throttle.
    assert np.all(
        result_m1.max_thrust_samples + 1e-6 >= result_m1.throttled_thrust_samples
    )

    # Mission 3 uses a lower throttle in this test, so throttled thrust should
    # usually be less than or equal to Mission 1 throttled thrust.
    assert np.all(
        result_m1.throttled_thrust_samples + 1e-6 >= result_m3.throttled_thrust_samples
    )

    print("Passed prop_main sanity checks.")


if __name__ == "__main__":
    main()