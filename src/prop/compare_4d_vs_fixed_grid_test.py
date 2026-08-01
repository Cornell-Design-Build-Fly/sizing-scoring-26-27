from __future__ import annotations

import time
import numpy as np

from src.prop.main_prop import prop_main
from src.prop.prop_database import load_default_prop_database
from src.prop.fixed_grid_prop_database import (
    FIXED_GRID_VELOCITIES_MPS,
    load_fixed_velocity_rpm_grid_database,
)
from src.prop.prop_batch_comparison_test import (
    DESIGN_CASES,
    PARAMETER_VECTOR,
    MISSION,
    KNOCKDOWN,
    make_design_vector,
)


PLOT_VELOCITIES_MPS = np.linspace(
    float(FIXED_GRID_VELOCITIES_MPS[0]),
    float(FIXED_GRID_VELOCITIES_MPS[-1]),
    300,
)


def eval_quad(coefficients, velocities_mps):
    a, b, c = np.asarray(coefficients, dtype=float)
    return a * velocities_mps**2 + b * velocities_mps + c


def percent_error(new, old):
    new = np.asarray(new, dtype=float)
    old = np.asarray(old, dtype=float)
    return 100.0 * np.abs(new - old) / np.maximum(np.abs(old), 1e-9)


def summarize_curve_error(name: str, fast_values, trusted_values) -> None:
    abs_error = np.abs(fast_values - trusted_values)
    pct_error = percent_error(fast_values, trusted_values)

    print(f"{name} curve error:")
    print(f"  mean abs error: {np.mean(abs_error):.6f} N")
    print(f"  max abs error:  {np.max(abs_error):.6f} N")
    print(f"  mean % error:   {np.mean(pct_error):.3f}%")
    print(f"  max % error:    {np.max(pct_error):.3f}%")


def main() -> None:
    print("Loading trusted 4D prop database...")
    database_4d = load_default_prop_database()

    print("Loading fixed-grid prop database...")
    database_grid = load_fixed_velocity_rpm_grid_database()

    print(f"\nFixed fit velocities [m/s]: {FIXED_GRID_VELOCITIES_MPS}")

    all_throttled_max_pct_errors = []
    all_max_max_pct_errors = []
    all_speedups = []

    for case in DESIGN_CASES:
        print("\n" + "=" * 90)
        print(f"{case.label} ({case.prop_diameter_in:g}x{case.prop_pitch_in:g} in)")
        print("=" * 90)

        design_vector = make_design_vector(case)

        t0 = time.perf_counter()
        output_4d = prop_main(
            design_vector=design_vector,
            parameter_vector=PARAMETER_VECTOR,
            mission=MISSION,
            prop_database=database_4d,
            velocities_mps=FIXED_GRID_VELOCITIES_MPS,
            knockdown=KNOCKDOWN,
        )
        runtime_4d = time.perf_counter() - t0

        t0 = time.perf_counter()
        output_grid = prop_main(
            design_vector=design_vector,
            parameter_vector=PARAMETER_VECTOR,
            mission=MISSION,
            prop_database=database_grid,
            velocities_mps=FIXED_GRID_VELOCITIES_MPS,
            knockdown=KNOCKDOWN,
        )
        runtime_grid = time.perf_counter() - t0

        throttled_4d = np.asarray(output_4d[0], dtype=float)
        max_4d = np.asarray(output_4d[1], dtype=float)

        throttled_grid = np.asarray(output_grid[0], dtype=float)
        max_grid = np.asarray(output_grid[1], dtype=float)

        throttled_values_4d = eval_quad(throttled_4d, PLOT_VELOCITIES_MPS)
        throttled_values_grid = eval_quad(throttled_grid, PLOT_VELOCITIES_MPS)

        max_values_4d = eval_quad(max_4d, PLOT_VELOCITIES_MPS)
        max_values_grid = eval_quad(max_grid, PLOT_VELOCITIES_MPS)

        throttled_pct_error = percent_error(
            throttled_values_grid,
            throttled_values_4d,
        )
        max_pct_error = percent_error(
            max_values_grid,
            max_values_4d,
        )

        all_throttled_max_pct_errors.append(float(np.max(throttled_pct_error)))
        all_max_max_pct_errors.append(float(np.max(max_pct_error)))

        if runtime_grid > 0.0:
            all_speedups.append(runtime_4d / runtime_grid)

        print(f"4D runtime:         {runtime_4d:.9f} s")
        print(f"fixed-grid runtime: {runtime_grid:.9f} s")
        if runtime_grid > 0.0:
            print(f"speedup:            {runtime_4d / runtime_grid:.2f}x")

        print("\n4D throttled coeffs:        ", throttled_4d)
        print("fixed-grid throttled coeffs:", throttled_grid)
        print("throttled coeff % error:    ", percent_error(throttled_grid, throttled_4d))

        print("\n4D max coeffs:              ", max_4d)
        print("fixed-grid max coeffs:      ", max_grid)
        print("max coeff % error:          ", percent_error(max_grid, max_4d))

        print()
        summarize_curve_error(
            "Throttled",
            throttled_values_grid,
            throttled_values_4d,
        )
        summarize_curve_error(
            "Max",
            max_values_grid,
            max_values_4d,
        )

    print("\n" + "=" * 90)
    print("OVERALL SUMMARY")
    print("=" * 90)

    print(
        "Worst throttled curve max % error over all cases: "
        f"{max(all_throttled_max_pct_errors):.3f}%"
    )
    print(
        "Worst max curve max % error over all cases: "
        f"{max(all_max_max_pct_errors):.3f}%"
    )

    if all_speedups:
        print(f"Mean speedup:   {np.mean(all_speedups):.2f}x")
        print(f"Median speedup: {np.median(all_speedups):.2f}x")


if __name__ == "__main__":
    main()