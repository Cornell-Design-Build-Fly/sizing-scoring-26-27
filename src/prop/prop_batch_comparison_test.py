"""
Batch test and comparison script for src.prop.main_prop.prop_main().

Run from the repository root with:

    python -m src.prop.prop_batch_comparison_test

What this script does:
1. Loads the prop database once, outside all timing measurements.
2. Runs prop_main() for 10 editable design vectors.
3. Times only the prop_main() call for each design.
4. Prints the two direct coefficient-array outputs from every run.
5. Verifies that each output is a finite three-element array [a, b, c].
6. Creates one thrust-vs-velocity graph for every design.
7. Creates combined throttled-thrust and max-thrust comparison graphs.
8. Prints a compact comparison table and saves a detailed CSV summary.

The quadratic arrays represent:

    thrust(V) = a * V**2 + b * V + c

where velocity V is in m/s and thrust is in N.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.prop.main_prop import prop_main
from src.prop.prop_database import load_default_prop_database
from src.vectors import DesignVector, ParameterVector


# =============================================================================
# EDIT INPUTS HERE
# =============================================================================

MISSION = 1
KNOCKDOWN = False

# Points passed into prop_main() to generate each quadratic fit.
# prop_main() requires at least three points.
FIT_VELOCITIES_MPS = np.linspace(0.001, 25.0, 8)

# Denser points used only to draw and compare the returned quadratic curves.
# These do not affect the prop_main() output or its measured runtime.
PLOT_VELOCITIES_MPS = np.linspace(0.001, 25.0, 300)

# Speeds included as columns in the printed/CSV comparison.
COMPARISON_SPEEDS_MPS = np.array([0.001, 5.0, 10.0, 15.0, 20.0, 25.0])

# Main speed used for the printed ranking.
RANKING_SPEED_MPS = 15.0

SHOW_PLOTS = True
SAVE_PLOTS = True
SAVE_SUMMARY_CSV = False

OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent / "prop_batch_comparison_outputs"
)

# Parameters shared by every design vector.
# ParameterVector currently uses class attributes, so the additional propulsion
# parameters are assigned directly to this one instance.
PARAMETER_VECTOR = ParameterVector()
PARAMETER_VECTOR.voltage = 22.2
PARAMETER_VECTOR.num_battery_cells = 6
PARAMETER_VECTOR.max_current = 100.0
PARAMETER_VECTOR.usable_battery_fraction = 0.85

# Propulsion settings held constant so differences are caused by the propellers.
COMMON_BATTERY_CAPACITY_AH = 4.5
COMMON_MOTOR_KV = 335.0
COMMON_MOTOR_MAX_POWER_W = 2200.0
COMMON_CRUISE_THROTTLE = 0.85
COMMON_MISSION3_CRUISE_THROTTLE = 0.75


@dataclass(frozen=True)
class DesignCase:
    """Editable definition of one test design."""

    label: str
    prop_diameter_in: float
    prop_pitch_in: float

    # Optional per-design overrides. Leave as None to use the common values.
    battery_capacity_ah: float | None = None
    motor_kv: float | None = None
    motor_max_power_w: float | None = None
    cruise_throttle: float | None = None
    mission3_cruise_throttle: float | None = None


# Ten default propeller cases.
# Edit, replace, add, or remove rows freely; the rest of the script adapts.
DESIGN_CASES = [
    DesignCase("13x8", 13.0, 8.0),
    DesignCase("13.5x10.2", 13.5, 10.2),
    DesignCase("14x8", 14.0, 8.0),
    DesignCase("14.9x10.6", 14.9, 10.6),
    DesignCase("14x12", 14.0, 12.0),
    DesignCase("15.7x8.3", 15.7, 8.3),
    DesignCase("15x10", 15.0, 10.0),
    DesignCase("15.6x12.3", 15.6, 12.3),
    DesignCase("16x10", 16.0, 10.0),
    DesignCase("16x12", 16.0, 12.0),
]


# =============================================================================
# TEST IMPLEMENTATION
# =============================================================================

@dataclass
class RunResult:
    """Stored result and derived metrics for one prop_main() call."""

    case: DesignCase
    design_vector: DesignVector
    runtime_seconds: float
    throttled_coefficients: np.ndarray
    max_coefficients: np.ndarray
    throttled_plot_values: np.ndarray
    max_plot_values: np.ndarray


def override_or_common(override: float | None, common: float) -> float:
    return common if override is None else override


def make_design_vector(case: DesignCase) -> DesignVector:
    """Creates the actual DesignVector passed into prop_main()."""

    return DesignVector(
        batt_capacity=override_or_common(
            case.battery_capacity_ah,
            COMMON_BATTERY_CAPACITY_AH,
        ),
        prop_diameter_in=case.prop_diameter_in,
        prop_pitch_in=case.prop_pitch_in,
        motor_kv=override_or_common(case.motor_kv, COMMON_MOTOR_KV),
        motor_max_power=override_or_common(
            case.motor_max_power_w,
            COMMON_MOTOR_MAX_POWER_W,
        ),
        cruise_throttle=override_or_common(
            case.cruise_throttle,
            COMMON_CRUISE_THROTTLE,
        ),
        mission3_cruise_throttle=override_or_common(
            case.mission3_cruise_throttle,
            COMMON_MISSION3_CRUISE_THROTTLE,
        ),
    )


def validate_input_arrays() -> None:
    """Catches common input mistakes before the expensive runs begin."""

    fit_velocities = np.asarray(FIT_VELOCITIES_MPS, dtype=float).reshape(-1)
    plot_velocities = np.asarray(PLOT_VELOCITIES_MPS, dtype=float).reshape(-1)
    comparison_velocities = np.asarray(
        COMPARISON_SPEEDS_MPS,
        dtype=float,
    ).reshape(-1)

    if len(DESIGN_CASES) == 0:
        raise ValueError("DESIGN_CASES must contain at least one design.")

    if len(fit_velocities) < 3:
        raise ValueError(
            "FIT_VELOCITIES_MPS needs at least three points for a quadratic fit."
        )

    for name, values in (
        ("FIT_VELOCITIES_MPS", fit_velocities),
        ("PLOT_VELOCITIES_MPS", plot_velocities),
        ("COMPARISON_SPEEDS_MPS", comparison_velocities),
    ):
        if values.size == 0:
            raise ValueError(f"{name} cannot be empty.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        if np.any(values < 0.0):
            raise ValueError(f"{name} cannot contain negative velocities.")

    labels = [case.label for case in DESIGN_CASES]
    if len(labels) != len(set(labels)):
        raise ValueError("Every DesignCase label must be unique.")


def warn_about_database_bounds(prop_database) -> None:
    """Reports any prop definitions outside the interpolation database bounds."""

    diameter_low, diameter_high = prop_database.diameter_bounds_in
    pitch_low, pitch_high = prop_database.pitch_bounds_in

    print("\n=== Prop database bounds ===")
    print(f"Diameter: {diameter_low:.3f} to {diameter_high:.3f} in")
    print(f"Pitch:    {pitch_low:.3f} to {pitch_high:.3f} in")

    for case in DESIGN_CASES:
        outside_diameter = not (
            diameter_low <= case.prop_diameter_in <= diameter_high
        )
        outside_pitch = not (pitch_low <= case.prop_pitch_in <= pitch_high)

        if outside_diameter or outside_pitch:
            print(
                f"WARNING: {case.label} is outside a database bound. "
                "The database may use nearest-neighbor fallback, so interpret "
                "that result cautiously."
            )


def validate_prop_main_output(
    raw_output,
    case_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Strictly verifies the interface expected by the rest of the team.

    This deliberately fails if prop_main() stops returning exactly two arrays.
    """

    if not isinstance(raw_output, tuple) or len(raw_output) != 2:
        raise TypeError(
            f"{case_label}: prop_main() returned {type(raw_output).__name__}, "
            "but this test expects exactly "
            "(throttled_thrust_coefficients, max_thrust_coefficients)."
        )

    throttled_coefficients = np.asarray(raw_output[0], dtype=float).reshape(-1)
    max_coefficients = np.asarray(raw_output[1], dtype=float).reshape(-1)

    for output_name, coefficients in (
        ("throttled", throttled_coefficients),
        ("max", max_coefficients),
    ):
        if coefficients.shape != (3,):
            raise ValueError(
                f"{case_label}: {output_name} output has shape "
                f"{coefficients.shape}; expected exactly (3,) for [a, b, c]."
            )

        if not np.all(np.isfinite(coefficients)):
            raise ValueError(
                f"{case_label}: {output_name} coefficients contain "
                "NaN or infinity."
            )

    return throttled_coefficients, max_coefficients


def format_coefficients(coefficients: np.ndarray) -> str:
    """Readable representation that preserves enough digits for inspection."""

    return np.array2string(
        coefficients,
        precision=10,
        separator=", ",
        suppress_small=False,
    )


def polynomial_equation(coefficients: np.ndarray) -> str:
    a, b, c = coefficients
    return f"T(V) = {a:.8g} V^2 + {b:.8g} V + {c:.8g}"


def create_individual_plot(result: RunResult, run_number: int) -> None:
    """Creates one graph containing max and throttled curves for one design."""

    plt.figure(figsize=(9, 6))
    plt.plot(
        PLOT_VELOCITIES_MPS,
        result.throttled_plot_values,
        label="Throttled thrust",
        linewidth=2,
    )
    plt.plot(
        PLOT_VELOCITIES_MPS,
        result.max_plot_values,
        label="Max thrust",
        linewidth=2,
    )
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Velocity [m/s]")
    plt.ylabel("Thrust [N]")
    plt.title(
        f"Run {run_number}: {result.case.label} "
        f"({result.case.prop_diameter_in:g}x"
        f"{result.case.prop_pitch_in:g} in)\n"
        f"prop_main runtime: {result.runtime_seconds:.6f} s"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if SAVE_PLOTS:
        filename = (
            f"run_{run_number:02d}_"
            f"{safe_filename(result.case.label)}_thrust_curves.png"
        )
        plt.savefig(OUTPUT_DIRECTORY / filename, dpi=180)


def create_combined_plots(results: list[RunResult]) -> None:
    """Creates the two requested all-propeller comparison figures."""

    plt.figure(figsize=(11, 7))
    for result in results:
        plt.plot(
            PLOT_VELOCITIES_MPS,
            result.throttled_plot_values,
            label=result.case.label,
            linewidth=1.8,
        )

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Velocity [m/s]")
    plt.ylabel("Throttled thrust [N]")
    plt.title("All throttled-thrust quadratic curves")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(
            OUTPUT_DIRECTORY / "combined_throttled_thrust.png",
            dpi=180,
        )

    plt.figure(figsize=(11, 7))
    for result in results:
        plt.plot(
            PLOT_VELOCITIES_MPS,
            result.max_plot_values,
            label=result.case.label,
            linewidth=1.8,
        )

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Velocity [m/s]")
    plt.ylabel("Maximum thrust [N]")
    plt.title("All maximum-thrust quadratic curves")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(
            OUTPUT_DIRECTORY / "combined_max_thrust.png",
            dpi=180,
        )


def safe_filename(text: str) -> str:
    """Converts a display label into a filesystem-safe filename fragment."""

    return "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in text
    ).strip("_")


def trapz_mean(y_values: np.ndarray, x_values: np.ndarray) -> float:
    """
    Mean curve value across the plotted velocity interval.

    Uses trapezoidal integration divided by interval length, rather than merely
    averaging array entries.
    """

    interval = float(x_values[-1] - x_values[0])
    if interval <= 0.0:
        return float(y_values[0])

    # np.trapezoid is preferred in newer NumPy; np.trapz supports older versions.
    trapezoid_function = getattr(np, "trapezoid", np.trapezoid)
    return float(trapezoid_function(y_values, x_values) / interval)


def first_zero_crossing_in_plot_range(
    coefficients: np.ndarray,
) -> float | None:
    """
    Returns the first real positive polynomial root inside the plot range.

    This is a property of the quadratic fit, not necessarily an exact physical
    zero-thrust speed.
    """

    roots = np.roots(coefficients)
    minimum_velocity = float(np.min(PLOT_VELOCITIES_MPS))
    maximum_velocity = float(np.max(PLOT_VELOCITIES_MPS))

    valid_roots = sorted(
        float(root.real)
        for root in roots
        if abs(root.imag) < 1e-9
        and minimum_velocity <= root.real <= maximum_velocity
    )

    return valid_roots[0] if valid_roots else None


def comparison_row(result: RunResult) -> dict[str, float | str | bool]:
    throttled_comparison = np.polyval(
        result.throttled_coefficients,
        COMPARISON_SPEEDS_MPS,
    )
    max_comparison = np.polyval(
        result.max_coefficients,
        COMPARISON_SPEEDS_MPS,
    )

    margin = result.max_plot_values - result.throttled_plot_values
    max_below_throttled = bool(np.any(margin < -1e-6))

    throttled_zero = first_zero_crossing_in_plot_range(
        result.throttled_coefficients
    )
    max_zero = first_zero_crossing_in_plot_range(result.max_coefficients)

    row: dict[str, float | str | bool] = {
        "label": result.case.label,
        "prop_diameter_in": result.case.prop_diameter_in,
        "prop_pitch_in": result.case.prop_pitch_in,
        "runtime_seconds": result.runtime_seconds,
        "throttled_a": result.throttled_coefficients[0],
        "throttled_b": result.throttled_coefficients[1],
        "throttled_c": result.throttled_coefficients[2],
        "max_a": result.max_coefficients[0],
        "max_b": result.max_coefficients[1],
        "max_c": result.max_coefficients[2],
        "mean_throttled_thrust_n": trapz_mean(
            result.throttled_plot_values,
            PLOT_VELOCITIES_MPS,
        ),
        "mean_max_thrust_n": trapz_mean(
            result.max_plot_values,
            PLOT_VELOCITIES_MPS,
        ),
        "minimum_max_minus_throttled_n": float(np.min(margin)),
        "max_curve_below_throttled_curve": max_below_throttled,
        "throttled_zero_crossing_mps": (
            math.nan if throttled_zero is None else throttled_zero
        ),
        "max_zero_crossing_mps": math.nan if max_zero is None else max_zero,
    }

    for speed, throttled_value, max_value in zip(
        COMPARISON_SPEEDS_MPS,
        throttled_comparison,
        max_comparison,
    ):
        speed_key = f"{speed:g}".replace(".", "p")
        row[f"throttled_at_{speed_key}_mps_n"] = float(throttled_value)
        row[f"max_at_{speed_key}_mps_n"] = float(max_value)

    return row


def print_comparison_table(
    results: list[RunResult],
    rows: list[dict[str, float | str | bool]],
) -> None:
    """Prints a compact table plus useful rankings and warnings."""

    print("\n" + "=" * 104)
    print("COMPACT PROPELLER COMPARISON")
    print("=" * 104)

    header = (
        f"{'Prop':<12}"
        f"{'Runtime [s]':>13}"
        f"{'Thr @ rank V':>15}"
        f"{'Max @ rank V':>15}"
        f"{'Mean thr':>13}"
        f"{'Mean max':>13}"
        f"{'Min(max-thr)':>15}"
    )
    print(f"Ranking velocity: {RANKING_SPEED_MPS:g} m/s")
    print(header)
    print("-" * len(header))

    for result, row in zip(results, rows):
        throttled_at_ranking = float(
            np.polyval(result.throttled_coefficients, RANKING_SPEED_MPS)
        )
        max_at_ranking = float(
            np.polyval(result.max_coefficients, RANKING_SPEED_MPS)
        )

        print(
            f"{result.case.label:<12}"
            f"{result.runtime_seconds:>13.6f}"
            f"{throttled_at_ranking:>15.3f}"
            f"{max_at_ranking:>15.3f}"
            f"{float(row['mean_throttled_thrust_n']):>13.3f}"
            f"{float(row['mean_max_thrust_n']):>13.3f}"
            f"{float(row['minimum_max_minus_throttled_n']):>15.3f}"
        )

    print("\nRanking by throttled thrust at "
          f"{RANKING_SPEED_MPS:g} m/s:")
    throttled_ranking = sorted(
        results,
        key=lambda result: float(
            np.polyval(result.throttled_coefficients, RANKING_SPEED_MPS)
        ),
        reverse=True,
    )
    for rank, result in enumerate(throttled_ranking, start=1):
        value = float(
            np.polyval(result.throttled_coefficients, RANKING_SPEED_MPS)
        )
        print(f"  {rank:2d}. {result.case.label:<12} {value:10.3f} N")

    print("\nRanking by max thrust at "
          f"{RANKING_SPEED_MPS:g} m/s:")
    max_ranking = sorted(
        results,
        key=lambda result: float(
            np.polyval(result.max_coefficients, RANKING_SPEED_MPS)
        ),
        reverse=True,
    )
    for rank, result in enumerate(max_ranking, start=1):
        value = float(np.polyval(result.max_coefficients, RANKING_SPEED_MPS))
        print(f"  {rank:2d}. {result.case.label:<12} {value:10.3f} N")

    problem_rows = [
        row
        for row in rows
        if bool(row["max_curve_below_throttled_curve"])
    ]
    if problem_rows:
        print(
            "\nWARNING: The fitted max curve drops below the fitted throttled "
            "curve somewhere in the plotted range for:"
        )
        for row in problem_rows:
            print(
                f"  - {row['label']} "
                f"(minimum max-throttled margin = "
                f"{float(row['minimum_max_minus_throttled_n']):.3f} N)"
            )
        print(
            "This may be a quadratic-fit artifact even if the original sampled "
            "max values were always greater."
        )
    else:
        print(
            "\nSanity check passed: every fitted max curve stays at or above "
            "its fitted throttled curve over the plotted velocity range."
        )


def save_summary_csv(
    rows: list[dict[str, float | str | bool]],
) -> Path:
    output_path = OUTPUT_DIRECTORY / "prop_comparison_summary.csv"

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main() -> None:
    validate_input_arrays()

    if SAVE_PLOTS or SAVE_SUMMARY_CSV:
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("=== Loading prop database outside the timed section ===")
    database_load_start = time.perf_counter()
    prop_database = load_default_prop_database()
    database_load_seconds = time.perf_counter() - database_load_start
    print(f"Database load time (not counted in run timings): "
          f"{database_load_seconds:.6f} s")

    warn_about_database_bounds(prop_database)

    print("\n=== Shared test inputs ===")
    print(f"Mission: {MISSION}")
    print(f"Knockdown enabled: {KNOCKDOWN}")
    print(f"Battery voltage: {PARAMETER_VECTOR.voltage:g} V")
    print(f"Battery cells: {PARAMETER_VECTOR.num_battery_cells}")
    print(f"Max current: {PARAMETER_VECTOR.max_current:g} A")
    print(
        "Usable battery fraction: "
        f"{PARAMETER_VECTOR.usable_battery_fraction:g}"
    )
    print(f"Fit velocities [m/s]: {FIT_VELOCITIES_MPS}")
    print(
        f"Plot range [m/s]: {PLOT_VELOCITIES_MPS[0]:g} to "
        f"{PLOT_VELOCITIES_MPS[-1]:g}"
    )

    results: list[RunResult] = []

    for run_number, case in enumerate(DESIGN_CASES, start=1):
        design_vector = make_design_vector(case)

        print("\n" + "=" * 80)
        print(
            f"RUN {run_number}/{len(DESIGN_CASES)}: {case.label} "
            f"({case.prop_diameter_in:g}x{case.prop_pitch_in:g} in)"
        )
        print("=" * 80)

        # TIME ONLY prop_main().
        # No printing, curve evaluation, graphing, or file writing occurs
        # between these two perf_counter() calls.
        call_start = time.perf_counter()
        raw_output = prop_main(
            design_vector=design_vector,
            parameter_vector=PARAMETER_VECTOR,
            mission=MISSION,
            prop_database=prop_database,
            velocities_mps=FIT_VELOCITIES_MPS,
            disp_res=False,
            knockdown=KNOCKDOWN,
        )
        call_end = time.perf_counter()

        runtime_seconds = call_end - call_start

        throttled_coefficients, max_coefficients = validate_prop_main_output(
            raw_output,
            case.label,
        )

        # These calculations happen after timing has stopped.
        throttled_plot_values = np.polyval(
            throttled_coefficients,
            PLOT_VELOCITIES_MPS,
        )
        max_plot_values = np.polyval(
            max_coefficients,
            PLOT_VELOCITIES_MPS,
        )

        result = RunResult(
            case=case,
            design_vector=design_vector,
            runtime_seconds=runtime_seconds,
            throttled_coefficients=throttled_coefficients,
            max_coefficients=max_coefficients,
            throttled_plot_values=throttled_plot_values,
            max_plot_values=max_plot_values,
        )
        results.append(result)

        print(f"prop_main() runtime only: {runtime_seconds:.9f} s")
        print("\nDirect prop_main() output:")
        print(f"  Type: {type(raw_output).__name__}")
        print(f"  Tuple length: {len(raw_output)}")
        print(
            "  throttled_thrust [a, b, c] = "
            f"{format_coefficients(throttled_coefficients)}"
        )
        print(
            "  max_thrust       [a, b, c] = "
            f"{format_coefficients(max_coefficients)}"
        )
        print(f"  Throttled equation: {polynomial_equation(throttled_coefficients)}")
        print(f"  Max equation:       {polynomial_equation(max_coefficients)}")

        negative_throttled = bool(np.any(throttled_plot_values < 0.0))
        negative_max = bool(np.any(max_plot_values < 0.0))
        if negative_throttled or negative_max:
            print(
                "  NOTE: At least one quadratic evaluates below zero within "
                "the plotted range. The graph intentionally shows the direct "
                "polynomial output without clipping it."
            )

        create_individual_plot(result, run_number)

    create_combined_plots(results)

    rows = [comparison_row(result) for result in results]
    print_comparison_table(results, rows)

    print("\n=== Timing summary ===")
    runtimes = np.array(
        [result.runtime_seconds for result in results],
        dtype=float,
    )
    for run_number, result in enumerate(results, start=1):
        print(
            f"Run {run_number:2d} | {result.case.label:<12} | "
            f"{result.runtime_seconds:.9f} s"
        )

    print(f"\nMean prop_main() runtime:   {np.mean(runtimes):.9f} s")
    print(f"Median prop_main() runtime: {np.median(runtimes):.9f} s")
    print(f"Minimum prop_main() runtime:{np.min(runtimes):.9f} s")
    print(f"Maximum prop_main() runtime:{np.max(runtimes):.9f} s")
    print(f"Total of timed calls only:  {np.sum(runtimes):.9f} s")

    if SAVE_SUMMARY_CSV:
        summary_path = save_summary_csv(rows)
        print(f"\nDetailed comparison CSV saved to:\n  {summary_path}")

    if SAVE_PLOTS:
        print(f"\nPlots saved to:\n  {OUTPUT_DIRECTORY}")

    if SHOW_PLOTS:
        # Display happens after all timing measurements are finished.
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
