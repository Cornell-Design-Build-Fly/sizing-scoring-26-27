from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from src.prop.continuous_prop_database import (
    load_default_continuous_prop_database,
)
from src.prop.main_prop import prop_main
from src.vectors import DesignVector, ParameterVector


# ============================================================
# EDITABLE INPUTS
# ============================================================

# Label, diameter [in], pitch [in]
#
# The label is only used for printing and graph legends.
# Diameter and pitch are the actual values passed into prop_main().
PROPELLERS = [
    ("8x6E", 8.0, 6.0),
    ("10.22x6.93E", 10.0, 6.0),
    ("12.7x8.5E", 12.7, 8.5),
    ("13x4.5EP", 13.0, 4.5),
    ("13x8E", 13.0, 8.0),
    ("14.4x10.2E", 14.4, 10.2),
    ("15x8E", 15.0, 8.0),
    ("15x8.9E", 15.0, 8.9),
    ("15x10E", 15.0, 10.0),
    ("26x15E", 26.0, 15.0),
]


# These are the actual velocities used by prop_main
# to calculate and fit the thrust and time curves.

FIT_VELOCITIES_MPS = np.linspace(0.001, 25.0, 4)
# FIT_VELOCITIES_MPS = np.array(
#     [0.01, 9.5, 19.0, 28.35],
#     dtype=float,
# )


# These velocities are only used to draw smooth curves.
# They do not affect prop_main's calculations.
PLOT_VELOCITIES_MPS = np.linspace(
    FIT_VELOCITIES_MPS.min(),
    FIT_VELOCITIES_MPS.max(),
    300,
)


# Mission 1 and 2 use CRUISE_THROTTLE.
# Mission 3 uses MISSION_3_CRUISE_THROTTLE.
MISSION = 1


# Motor inputs.
MOTOR_KV = 520.0
MOTOR_MAX_POWER_W = 2000.0
MAX_CURRENT_A = 100.0


# Battery inputs.
BATTERY_CAPACITY_AH = 4.5
BATTERY_NOMINAL_V = 22.2
BATTERY_CELLS = 6
USABLE_BATTERY_FRACTION = 0.85


# Throttle inputs.
CRUISE_THROTTLE = 0.90
MISSION_3_CRUISE_THROTTLE = 0.85


# Optional thrust knockdown.
APPLY_KNOCKDOWN = False
KNOCKDOWN_FACTOR = 0.90


# When True, prop_main prints its selected RPM, thrust,
# current, throttle, valid RPM count, and failure count.
PRINT_OPERATING_POINTS = True


# Graph options.
SHOW_INDIVIDUAL_THRUST_GRAPHS = True
SHOW_COMBINED_TIME_GRAPH = True


# Results are saved in this folder inside src/prop.
SAVE_RESULTS = False
OUTPUT_FOLDER_NAME = "prop_main_10_prop_results"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def evaluate_curve(
    fit: tuple[float, float, float],
    velocities_mps: np.ndarray,
) -> np.ndarray:
    """
    Evaluate:

        y = a*V^2 + b*V + c
    """

    return np.polyval(
        np.asarray(
            fit,
            dtype=float,
        ),
        velocities_mps,
    )


def fit_is_exactly_zero(
    fit: tuple[float, float, float],
) -> bool:
    """
    Check whether prop_main rejected the propulsion curve
    and returned exactly (0, 0, 0).
    """

    return bool(
        np.all(
            np.asarray(
                fit,
                dtype=float,
            ) == 0.0
        )
    )


def save_summary_csv(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Save all coefficients and runtimes to one CSV."""

    if not rows:
        raise ValueError(
            "No propeller results were generated."
        )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN TEST
# ============================================================

def main() -> None:
    script_dir = Path(__file__).resolve().parent

    output_dir = (
        script_dir
        / OUTPUT_FOLDER_NAME
    )

    if SAVE_RESULTS:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


    # --------------------------------------------------------
    # Basic input checks
    # --------------------------------------------------------

    if len(PROPELLERS) != 10:
        raise ValueError(
            "PROPELLERS must contain exactly 10 entries."
        )

    if MISSION not in (1, 2, 3):
        raise ValueError(
            "MISSION must be 1, 2, or 3."
        )

    if FIT_VELOCITIES_MPS.size < 3:
        raise ValueError(
            "At least three fit velocities are required."
        )


    # --------------------------------------------------------
    # Create the ParameterVector
    # --------------------------------------------------------

    # ParameterVector currently defines voltage as a class
    # attribute, so set the class value before constructing
    # the DesignVectors.
    ParameterVector.voltage = BATTERY_NOMINAL_V

    parameter_vector = ParameterVector()

    # These are the explicit parameter values used by the
    # prop helper functions.
    parameter_vector.voltage = (
        BATTERY_NOMINAL_V
    )

    parameter_vector.max_current = (
        MAX_CURRENT_A
    )

    parameter_vector.num_battery_cells = (
        BATTERY_CELLS
    )

    parameter_vector.usable_battery_fraction = (
        USABLE_BATTERY_FRACTION
    )


    # --------------------------------------------------------
    # Load the database once
    # --------------------------------------------------------

    print(
        "Loading continuous propeller database..."
    )

    database_start = perf_counter()

    prop_database = (
        load_default_continuous_prop_database()
    )

    database_startup_s = (
        perf_counter()
        - database_start
    )

    print(
        f"Database startup time: "
        f"{database_startup_s:.6f} s"
    )


    # --------------------------------------------------------
    # Result storage
    # --------------------------------------------------------

    summary_rows: list[
        dict[str, object]
    ] = []

    thrust_results: list[
        tuple[
            str,
            tuple[float, float, float],
        ]
    ] = []

    time_results: list[
        tuple[
            str,
            tuple[float, float, float],
        ]
    ] = []


    # --------------------------------------------------------
    # Run prop_main for all 10 propellers
    # --------------------------------------------------------

    for index, (
        prop_name,
        diameter_in,
        pitch_in,
    ) in enumerate(
        PROPELLERS,
        start=1,
    ):

        # Create a real DesignVector for this propeller.
        #
        # The unrelated aero fields retain their normal values,
        # but they are not used by prop_main.
        design_vector = DesignVector(
            batt_capacity=BATTERY_CAPACITY_AH,
            prop_diameter_in=diameter_in,
            prop_pitch_in=pitch_in,
            motor_kv=MOTOR_KV,
            motor_max_power=MOTOR_MAX_POWER_W,
            cruise_throttle=CRUISE_THROTTLE,
            mission3_cruise_throttle=(
                MISSION_3_CRUISE_THROTTLE
            ),
        )

        print()
        print("=" * 72)

        print(
            f"PROP {index}/10: "
            f"{prop_name} "
            f"({diameter_in:g} x "
            f"{pitch_in:g} in)"
        )

        print("=" * 72)


        # Time prop_main only.
        #
        # Database startup and graph generation are not included.
        run_start = perf_counter()

        thrust_fit, time_fit = prop_main(
            design_vector=design_vector,
            parameter_vector=parameter_vector,
            mission=MISSION,
            prop_database=prop_database,
            velocities_mps=FIT_VELOCITIES_MPS,
            disp_res=PRINT_OPERATING_POINTS,
            knockdown=APPLY_KNOCKDOWN,
            knockdown_factor=KNOCKDOWN_FACTOR,
        )

        runtime_s = (
            perf_counter()
            - run_start
        )


        # A zero thrust fit means your production failure rule
        # rejected the whole propulsion curve.
        curve_rejected = (
            fit_is_exactly_zero(
                thrust_fit
            )
        )

        thrust_results.append(
            (
                prop_name,
                thrust_fit,
            )
        )

        time_results.append(
            (
                prop_name,
                time_fit,
            )
        )


        # Store one summary row.
        summary_rows.append(
            {
                "prop_name": prop_name,
                "diameter_in": diameter_in,
                "pitch_in": pitch_in,
                "runtime_s": runtime_s,
                "curve_rejected": (
                    curve_rejected
                ),
                "thrust_a": thrust_fit[0],
                "thrust_b": thrust_fit[1],
                "thrust_c": thrust_fit[2],
                "time_a": time_fit[0],
                "time_b": time_fit[1],
                "time_c": time_fit[2],
            }
        )


        # ----------------------------------------------------
        # Print this propeller's direct outputs
        # ----------------------------------------------------

        print()

        print(
            "Thrust fit [a, b, c]: "
            f"[{thrust_fit[0]: .10e}, "
            f"{thrust_fit[1]: .10e}, "
            f"{thrust_fit[2]: .10e}]"
        )

        print(
            "Time fit   [a, b, c]: "
            f"[{time_fit[0]: .10e}, "
            f"{time_fit[1]: .10e}, "
            f"{time_fit[2]: .10e}]"
        )

        print(
            f"prop_main runtime: "
            f"{runtime_s:.6f} s"
        )

        if curve_rejected:
            print(
                "Status: REJECTED / ZERO CURVE"
            )
        else:
            print(
                "Status: VALID CURVE"
            )


        # ----------------------------------------------------
        # Create this propeller's individual thrust graph
        # ----------------------------------------------------

        if SHOW_INDIVIDUAL_THRUST_GRAPHS:

            fitted_thrust_n = evaluate_curve(
                thrust_fit,
                PLOT_VELOCITIES_MPS,
            )

            plt.figure(
                num=f"{prop_name} thrust",
                figsize=(8, 5),
            )

            plt.plot(
                PLOT_VELOCITIES_MPS,
                fitted_thrust_n,
                label=f"{prop_name} thrust fit",
            )

            plt.axhline(
                0.0,
                linewidth=1.0,
            )

            plt.xlabel(
                "Velocity [m/s]"
            )

            plt.ylabel(
                "Throttled thrust [N]"
            )

            plt.title(
                f"{prop_name}: "
                f"throttled thrust curve"
            )

            plt.grid(True)
            plt.legend()
            plt.tight_layout()


            # Save the same graph as a PNG.
            safe_name = (
                prop_name
                .replace("/", "_")
                .replace("\\", "_")
            )

            if SAVE_RESULTS:
                plt.savefig(
                    output_dir
                    / f"{safe_name}_thrust_curve.png",
                    dpi=160,
                )


    # ========================================================
    # SAVE THE NUMERICAL SUMMARY
    # ========================================================

    if SAVE_RESULTS:
        save_summary_csv(
            output_dir
            / "prop_main_summary.csv",
            summary_rows,
        )


    # ========================================================
    # FINAL GRAPH: ALL THRUST CURVES
    # ========================================================

    plt.figure(
        num="All propeller thrust curves",
        figsize=(10, 7),
    )

    for prop_name, thrust_fit in thrust_results:

        fitted_thrust_n = evaluate_curve(
            thrust_fit,
            PLOT_VELOCITIES_MPS,
        )

        plt.plot(
            PLOT_VELOCITIES_MPS,
            fitted_thrust_n,
            label=prop_name,
        )

    plt.axhline(
        0.0,
        linewidth=1.0,
    )

    plt.xlabel(
        "Velocity [m/s]"
    )

    plt.ylabel(
        "Throttled thrust [N]"
    )

    plt.title(
        "All throttled thrust curves"
    )

    plt.grid(True)

    plt.legend(
        loc="best",
    )

    plt.tight_layout()


    if SAVE_RESULTS:
        plt.savefig(
            output_dir
            / "all_thrust_curves.png",
            dpi=180,
        )


    # ========================================================
    # OPTIONAL FINAL GRAPH: ALL FLIGHT-TIME CURVES
    # ========================================================

    if SHOW_COMBINED_TIME_GRAPH:

        plt.figure(
            num="All propeller flight-time curves",
            figsize=(10, 7),
        )

        for prop_name, time_fit in time_results:

            fitted_time_s = evaluate_curve(
                time_fit,
                PLOT_VELOCITIES_MPS,
            )

            plt.plot(
                PLOT_VELOCITIES_MPS,
                fitted_time_s,
                label=prop_name,
            )

        plt.axhline(
            0.0,
            linewidth=1.0,
        )

        plt.xlabel(
            "Velocity [m/s]"
        )

        plt.ylabel(
            "Throttled flight time [s]"
        )

        plt.title(
            "All throttled flight-time curves"
        )

        plt.grid(True)

        plt.legend(
            loc="best",
        )

        plt.tight_layout()

        if SAVE_RESULTS:
            plt.savefig(
                output_dir
                / "all_time_curves.png",
                dpi=180,
            )



    # ========================================================
    # PRINT FINAL SUMMARY TABLE
    # ========================================================

    print()
    print("=" * 96)
    print("FINAL SUMMARY")
    print("=" * 96)

    print(
        f"{'Prop':<18}"
        f"{'Runtime [s]':>14}"
        f"{'Rejected':>12}"
        f"{'Thrust a':>16}"
        f"{'Thrust b':>16}"
        f"{'Thrust c':>16}"
    )

    for row in summary_rows:

        print(
            f"{str(row['prop_name']):<18}"
            f"{float(row['runtime_s']):>14.6f}"
            f"{str(row['curve_rejected']):>12}"
            f"{float(row['thrust_a']):>16.6e}"
            f"{float(row['thrust_b']):>16.6e}"
            f"{float(row['thrust_c']):>16.6e}"
        )


    print()
    if SAVE_RESULTS:
        print(
            f"Results saved to:\n"
            f"{output_dir}"
        )
    else:
        print(
            "Results were not saved."
        )

    print()
    print(
        "Close the graph windows when you are finished."
    )


    # This opens all individual graphs and the final
    # combined graphs automatically.
    plt.show()


if __name__ == "__main__":
    main()