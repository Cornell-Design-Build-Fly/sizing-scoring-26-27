from __future__ import annotations

import time

import numpy as np
from types import SimpleNamespace

from src.prop.main_prop import prop_main
from src.prop.prop_database import load_default_prop_database
from src.prop.fixed_velocity_prop_database import (
    FIXED_FIT_VELOCITIES_MPS,
    load_fixed_velocity_3d_prop_database,
)


TEST_PROPS = [
    ("13x8", 13.0, 8.0),
    ("14x8", 14.0, 8.0),
    ("15x10", 15.0, 10.0),
    ("16x12", 16.0, 12.0),
    ("15.6x12.3", 15.6, 12.3),
]


def make_design_vector(diameter_in: float, pitch_in: float):
    return SimpleNamespace(
        prop_diameter_in=diameter_in,
        prop_pitch_in=pitch_in,
        batt_capacity=4.2,
        motor_kv=520.0,
        motor_max_power=2000.0,
        cruise_throttle=0.85,
        mission3_cruise_throttle=0.85,
    )


def make_parameter_vector():
    return SimpleNamespace(
        voltage=22.2,
        num_battery_cells=6,
        max_current=100.0,
        usable_battery_fraction=0.85,
    )


def percent_error(new, old):
    old = np.asarray(old, dtype=float)
    new = np.asarray(new, dtype=float)

    denom = np.maximum(np.abs(old), 1e-9)
    return 100.0 * np.abs(new - old) / denom


def main():
    print("Loading original 4D database...")
    db_4d = load_default_prop_database()

    print("Loading fixed-velocity 3D database...")
    db_3d = load_fixed_velocity_3d_prop_database()

    print(f"\nFixed velocities [m/s]: {FIXED_FIT_VELOCITIES_MPS}\n")

    pv = make_parameter_vector()

    for name, diameter, pitch in TEST_PROPS:
        print("=" * 80)
        print(f"{name} ({diameter}x{pitch})")
        print("=" * 80)

        dv = make_design_vector(diameter, pitch)

        t0 = time.perf_counter()
        out_4d = prop_main(
            dv,
            pv,
            mission=1,
            prop_database=db_4d,
            velocities_mps=FIXED_FIT_VELOCITIES_MPS,
        )
        t_4d = time.perf_counter() - t0

        t0 = time.perf_counter()
        out_3d = prop_main(
            dv,
            pv,
            mission=1,
            prop_database=db_3d,
            velocities_mps=FIXED_FIT_VELOCITIES_MPS,
        )
        t_3d = time.perf_counter() - t0

        throttled_4d = np.asarray(out_4d[0], dtype=float)
        max_4d = np.asarray(out_4d[1], dtype=float)

        throttled_3d = np.asarray(out_3d[0], dtype=float)
        max_3d = np.asarray(out_3d[1], dtype=float)

        print(f"4D runtime: {t_4d:.9f} s")
        print(f"3D runtime: {t_3d:.9f} s")

        print("\nThrottled 4D:", throttled_4d)
        print("Throttled 3D:", throttled_3d)
        print("Throttle % error:", percent_error(throttled_3d, throttled_4d))

        print("\nMax 4D:", max_4d)
        print("Max 3D:", max_3d)
        print("Max % error:", percent_error(max_3d, max_4d))

        print()


if __name__ == "__main__":
    main()