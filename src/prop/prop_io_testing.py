from types import SimpleNamespace

import numpy as np

from src.prop.main_prop import (
    load_default_prop_database,
    prop_main,
    prop_main_interp,
)


# ============================================================
# EDIT INPUTS HERE
# ============================================================

MISSION = 1
# Use 1, 2, or 3

# DesignVector-like inputs
BATT_CAPACITY_AH = 4.5
BATT_CAPACITY_AH_2 = 4.2

PROP_DIAMETER_IN = 14.8
PROP_PITCH_IN = 10.2
PROP_DIAMETER_IN_2 = 15.6
PROP_PITCH_IN_2 = 9.7

MOTOR_KV = 335.0
MOTOR_MAX_POWER_W = 2200.0
MOTOR_KV_2 = 350.0
MOTOR_MAX_POWER_W_2 = 2300.0

CRUISE_THROTTLE = 0.85
MISSION3_CRUISE_THROTTLE = 0.75
CRUISE_THROTTLE_2 = 0.80
MISSION3_CRUISE_THROTTLE_2 = 0.70

# ParameterVector-like inputs
BATTERY_VOLTAGE_V = 22.2
NUM_BATTERY_CELLS = 6
MAX_CURRENT_A = 100.0
USABLE_BATTERY_FRACTION = 0.85
BATTERY_VOLTAGE_V_2 = 22.2
NUM_BATTERY_CELLS_2 = 6
MAX_CURRENT_A_2 = 100.0
USABLE_BATTERY_FRACTION_2 = 0.90


# Velocity sample points used to build the curve fits
# These are in m/s.
VELOCITIES_MPS = np.linspace(0.001, 25.0, 8)

# Speeds where you want to evaluate the final curve fits
# These are also in m/s.
EVALUATION_SPEEDS_MPS = np.array([5.0, 10.0, 15.0, 20.0])


# ============================================================
# TEST SCRIPT
# ============================================================

def main():
    print("=== Prop Section Input/Output Test ===")

    print("\n=== Loading prop database ===")
    prop_database = load_default_prop_database()

    design_vector = SimpleNamespace(
        batt_capacity=BATT_CAPACITY_AH,
        prop_diameter_in=PROP_DIAMETER_IN,
        prop_pitch_in=PROP_PITCH_IN,
        motor_kv=MOTOR_KV,
        motor_max_power=MOTOR_MAX_POWER_W,
        cruise_throttle=CRUISE_THROTTLE,
        mission3_cruise_throttle=MISSION3_CRUISE_THROTTLE,
    )

    design_vector_2 = SimpleNamespace(
        batt_capacity=BATT_CAPACITY_AH_2,
        prop_diameter_in=PROP_DIAMETER_IN_2,
        prop_pitch_in=PROP_PITCH_IN_2,
        motor_kv=MOTOR_KV_2,
        motor_max_power=MOTOR_MAX_POWER_W_2,
        cruise_throttle=CRUISE_THROTTLE_2,
        mission3_cruise_throttle=MISSION3_CRUISE_THROTTLE_2,
    )

    parameter_vector = SimpleNamespace(
        voltage=BATTERY_VOLTAGE_V,
        num_battery_cells=NUM_BATTERY_CELLS,
        max_current=MAX_CURRENT_A,
        usable_battery_fraction=USABLE_BATTERY_FRACTION,
    )

    parameter_vector_2 = SimpleNamespace(
        voltage=BATTERY_VOLTAGE_V_2,
        num_battery_cells=NUM_BATTERY_CELLS_2,
        max_current=MAX_CURRENT_A_2,
        usable_battery_fraction=USABLE_BATTERY_FRACTION_2,
    )

    print("\n=== Inputs ===")
    print(f"Mission: {MISSION}")
    print(f"Battery capacity: {BATT_CAPACITY_AH} Ah")
    print(f"Battery voltage: {BATTERY_VOLTAGE_V} V")
    print(f"Battery cells: {NUM_BATTERY_CELLS}")
    print(f"Max current: {MAX_CURRENT_A} A")
    print(f"Usable battery fraction: {USABLE_BATTERY_FRACTION}")

    print(f"\nProp diameter: {PROP_DIAMETER_IN} in")
    print(f"Prop pitch: {PROP_PITCH_IN} in")

    print(f"\nMotor Kv: {MOTOR_KV}")
    print(f"Motor max power: {MOTOR_MAX_POWER_W} W")

    print(f"\nCruise throttle: {CRUISE_THROTTLE}")
    print(f"Mission 3 cruise throttle: {MISSION3_CRUISE_THROTTLE}")

    print("\nVelocity samples used for fitting [m/s]:")
    print(VELOCITIES_MPS)

    print("\n=== Running prop_main ===")
    result = prop_main(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
        mission=MISSION,
        prop_database=prop_database,
        velocities_mps=VELOCITIES_MPS,
        disp_res=False,
    )

    print("\n=== Prop Section Outputs: Full Python Result ===")

    print("\nPolynomial fits:")
    print("result.throttled_thrust:")
    print(result.throttled_thrust)

    print("\nresult.max_thrust:")
    print(result.max_thrust)

    print("\nresult.throttled_time:")
    print(result.throttled_time)

    print("\nresult.max_time:")
    print(result.max_time)

    print("\nRaw sample outputs used to create fits:")

    print("\nSample velocities [m/s]:")
    print(result.sample_velocities_mps)

    print("\nThrottled thrust samples [N]:")
    print(result.throttled_thrust_samples)

    print("\nMax thrust samples [N]:")
    print(result.max_thrust_samples)

    print("\nThrottled flight time samples [s]:")
    print(result.throttled_time_samples)

    print("\nMax flight time samples [s]:")
    print(result.max_time_samples)

    print("\n=== MATLAB-style prop_main_interp outputs ===")
    p_throttled_thrust, p_max_thrust, p_throttled_t, p_max_t = prop_main_interp(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
        mission=MISSION,
        prop_database=prop_database,
        velocities_mps=VELOCITIES_MPS,
        disp_res=False,
    )

    print("\np_throttled_thrust:")
    print(p_throttled_thrust)

    print("\np_max_thrust:")
    print(p_max_thrust)

    print("\np_throttled_t:")
    print(p_throttled_t)

    print("\np_max_t:")
    print(p_max_t)

    print("\n=== Evaluating curve fits at chosen speeds ===")

    throttled_thrust_eval = np.polyval(
        result.throttled_thrust,
        EVALUATION_SPEEDS_MPS,
    )

    max_thrust_eval = np.polyval(
        result.max_thrust,
        EVALUATION_SPEEDS_MPS,
    )

    throttled_time_eval = np.polyval(
        result.throttled_time,
        EVALUATION_SPEEDS_MPS,
    )

    max_time_eval = np.polyval(
        result.max_time,
        EVALUATION_SPEEDS_MPS,
    )

    print(
        "\n"
        "Speed [m/s] | Cruise thrust [N] | Max thrust [N] | "
        "Cruise time [s] | Max time [s]"
    )

    for speed, cruise_thrust, max_thrust, cruise_time, max_time in zip(
        EVALUATION_SPEEDS_MPS,
        throttled_thrust_eval,
        max_thrust_eval,
        throttled_time_eval,
        max_time_eval,
    ):
        print(
            f"{speed:10.3f} | "
            f"{cruise_thrust:17.3f} | "
            f"{max_thrust:14.3f} | "
            f"{cruise_time:15.3f} | "
            f"{max_time:12.3f}"
        )

    print("\n=== Basic sanity checks ===")

    assert np.all(np.isfinite(result.throttled_thrust))
    assert np.all(np.isfinite(result.max_thrust))
    assert np.all(np.isfinite(result.throttled_time))
    assert np.all(np.isfinite(result.max_time))

    assert np.all(np.isfinite(result.throttled_thrust_samples))
    assert np.all(np.isfinite(result.max_thrust_samples))
    assert np.all(np.isfinite(result.throttled_time_samples))
    assert np.all(np.isfinite(result.max_time_samples))

    assert np.any(result.max_thrust_samples > 0.0)

    print("Passed basic prop input/output sanity checks.")

    print("\n=== Inputs ===")
    print(f"Mission: {MISSION}")
    print(f"Battery capacity: {BATT_CAPACITY_AH} Ah")
    print(f"Battery voltage: {BATTERY_VOLTAGE_V} V")
    print(f"Battery cells: {NUM_BATTERY_CELLS}")
    print(f"Max current: {MAX_CURRENT_A} A")
    print(f"Usable battery fraction: {USABLE_BATTERY_FRACTION}")

    print(f"\nProp diameter: {PROP_DIAMETER_IN} in")
    print(f"Prop pitch: {PROP_PITCH_IN} in")

    print(f"\nMotor Kv: {MOTOR_KV}")
    print(f"Motor max power: {MOTOR_MAX_POWER_W} W")

    print(f"\nCruise throttle: {CRUISE_THROTTLE}")
    print(f"Mission 3 cruise throttle: {MISSION3_CRUISE_THROTTLE}")

    print("\nVelocity samples used for fitting [m/s]:")
    print(VELOCITIES_MPS)

    print("\n=== Running prop_main ===")
    result_2 = prop_main(
        design_vector=design_vector_2,
        parameter_vector=parameter_vector_2,
        mission=MISSION,
        prop_database=prop_database,
        velocities_mps=VELOCITIES_MPS,
        disp_res=False,
    )

    print("\n=== Prop Section Outputs: Full Python Result ===")

    print("\nPolynomial fits:")
    print("result_2.throttled_thrust:")
    print(result_2.throttled_thrust)

    print("\nresult_2.max_thrust:")
    print(result_2.max_thrust)

    print("\nresult_2.throttled_time:")
    print(result_2.throttled_time)

    print("\nresult_2.max_time:")
    print(result_2.max_time)

    print("\nRaw sample outputs used to create fits:")

    print("\nSample velocities [m/s]:")
    print(result_2.sample_velocities_mps)

    print("\nThrottled thrust samples [N]:")
    print(result_2.throttled_thrust_samples)

    print("\nMax thrust samples [N]:")
    print(result_2.max_thrust_samples)

    print("\nThrottled flight time samples [s]:")
    print(result_2.throttled_time_samples)

    print("\nMax flight time samples [s]:")
    print(result_2.max_time_samples)

    print("\n=== MATLAB-style prop_main_interp outputs ===")
    p_throttled_thrust, p_max_thrust, p_throttled_t, p_max_t = prop_main_interp(
        design_vector=design_vector_2,
        parameter_vector=parameter_vector_2,
        mission=MISSION,
        prop_database=prop_database,
        velocities_mps=VELOCITIES_MPS,
        disp_res=False,
    )

    print("\np_throttled_thrust:")
    print(p_throttled_thrust)

    print("\np_max_thrust:")
    print(p_max_thrust)

    print("\np_throttled_t:")
    print(p_throttled_t)

    print("\np_max_t:")
    print(p_max_t)

    print("\n=== Evaluating curve fits at chosen speeds ===")

    throttled_thrust_eval = np.polyval(
        result_2.throttled_thrust,
        EVALUATION_SPEEDS_MPS,
    )

    max_thrust_eval = np.polyval(
        result_2.max_thrust,
        EVALUATION_SPEEDS_MPS,
    )

    throttled_time_eval = np.polyval(
        result_2.throttled_time,
        EVALUATION_SPEEDS_MPS,
    )

    max_time_eval = np.polyval(
        result_2.max_time,
        EVALUATION_SPEEDS_MPS,
    )

    print(
        "\n"
        "Speed [m/s] | Cruise thrust [N] | Max thrust [N] | "
        "Cruise time [s] | Max time [s]"
    )

    for speed, cruise_thrust, max_thrust, cruise_time, max_time in zip(
        EVALUATION_SPEEDS_MPS,
        throttled_thrust_eval,
        max_thrust_eval,
        throttled_time_eval,
        max_time_eval,
    ):
        print(
            f"{speed:10.3f} | "
            f"{cruise_thrust:17.3f} | "
            f"{max_thrust:14.3f} | "
            f"{cruise_time:15.3f} | "
            f"{max_time:12.3f}"
        )

    print("\n=== Basic sanity checks ===")

    assert np.all(np.isfinite(result_2.throttled_thrust))
    assert np.all(np.isfinite(result_2.max_thrust))
    assert np.all(np.isfinite(result_2.throttled_time))
    assert np.all(np.isfinite(result_2.max_time))

    assert np.all(np.isfinite(result_2.throttled_thrust_samples))
    assert np.all(np.isfinite(result_2.max_thrust_samples))
    assert np.all(np.isfinite(result_2.throttled_time_samples))
    assert np.all(np.isfinite(result_2.max_time_samples))

    assert np.any(result_2.max_thrust_samples > 0.0)

    print("Passed basic prop input/output sanity checks.")

if __name__ == "__main__":
    main()