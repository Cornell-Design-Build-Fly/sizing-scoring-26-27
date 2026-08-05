from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from src.prop.continuous_prop_database import (
    load_default_continuous_prop_database,
)
from src.prop.prop_classes import Battery, Motor
from src.prop.prop_cruise_values import solve_cruise_samples


# ============================================================
# EDITABLE INPUTS
# Keep these values matched with the MATLAB script.
# ============================================================

PROPELLERS = [
    ("8x6E", 8.0, 6.0),
    ("10.22x6.93E", 10.0, 6.0),
    ("12.7x8.5E", 12.7, 8.5),
    ("13x4.5EP", 13.0, 4.5),
    ("13x8E", 13.0, 8.0),
    ("14.4x10.2E", 14.4, 10.2),
    ("15x10E", 15.0, 10.0),
    ("17.4x10.5E", 17.4, 10.5),
    ("18x10E", 18.0, 10.0),
    ("26x15E", 26.0, 15.0),
]

FIT_VELOCITIES_MPS = np.linspace(
    0.001,
    25.0,
    4,
)

MIN_RPM = 3000
MAX_RPM = 16000
RPM_STEP = 100

MOTOR_KV = 520.0
MOTOR_MAX_POWER_W = 2000.0
MAX_CURRENT_A = 100.0

BATTERY_CAPACITY_AH = 4.5
BATTERY_CELLS = 6
BATTERY_NOMINAL_V = 22.2
USABLE_BATTERY_FRACTION = 0.85

CRUISE_THROTTLE = 0.90
MAX_THROTTLE = 1.00

APPLY_KNOCKDOWN = False
KNOCKDOWN_FACTOR = 0.90

PLOT_VELOCITIES_MPS = np.linspace(
    0.0,
    25.0,
    201,
)

OUTPUT_FOLDER_NAME = "python_prop_comparison_output"


# ============================================================
# HELPERS
# ============================================================

def fit_tuple(values: np.ndarray) -> tuple[float, float, float]:
    coefficients = np.polyfit(
        FIT_VELOCITIES_MPS,
        values,
        2,
    )

    return (
        float(coefficients[0]),
        float(coefficients[1]),
        float(coefficients[2]),
    )


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows were generated for {path.name}."
        )

    with path.open(
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


def plot_fit_comparison(
    summaries: list[dict[str, object]],
    prefix: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 6))

    for summary in summaries:
        coefficients = np.array(
            [
                summary[f"{prefix}_a"],
                summary[f"{prefix}_b"],
                summary[f"{prefix}_c"],
            ],
            dtype=float,
        )

        values = np.polyval(
            coefficients,
            PLOT_VELOCITIES_MPS,
        )

        plt.plot(
            PLOT_VELOCITIES_MPS,
            values,
            label=str(summary["prop_name"]),
        )

    plt.xlabel("Velocity [m/s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=160,
    )
    plt.close()


# ============================================================
# MAIN TEST
# ============================================================

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    output_dir = (
        repo_root / OUTPUT_FOLDER_NAME
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    motor = Motor(
        kv=MOTOR_KV,
        max_power=MOTOR_MAX_POWER_W,
        max_current=MAX_CURRENT_A,
    )

    battery = Battery(
        vnom=BATTERY_NOMINAL_V,
        cells=BATTERY_CELLS,
        Crat=0.0,
        capacity=BATTERY_CAPACITY_AH,
        useable_fraction=USABLE_BATTERY_FRACTION,
    )

    print("Loading continuous propeller database...")

    database_start = perf_counter()

    database = (
        load_default_continuous_prop_database()
    )

    database_startup_s = (
        perf_counter() - database_start
    )

    print(
        f"Database startup time: "
        f"{database_startup_s:.6f} s"
    )

    summaries: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    rpm_plot_results: list[
        tuple[str, np.ndarray, np.ndarray]
    ] = []

    kt_nm_per_a = float(motor.get_kt())
    no_load_current_a = float(motor.get_I0())
    battery_resistance_ohm = float(
        battery.get_Rb()
    )

    for prop_index, (
        prop_name,
        diameter_in,
        pitch_in,
    ) in enumerate(
        PROPELLERS,
        start=1,
    ):
        solver_start = perf_counter()

        cruise_result = solve_cruise_samples(
            diameter_in=diameter_in,
            pitch_in=pitch_in,
            velocities_mps=FIT_VELOCITIES_MPS,
            motor=motor,
            battery=battery,
            max_current_a=MAX_CURRENT_A,
            cruise_throttle=CRUISE_THROTTLE,
            prop_database=database,
            min_rpm=MIN_RPM,
            max_rpm=MAX_RPM,
            rpm_step=RPM_STEP,
            knockdown=APPLY_KNOCKDOWN,
            knockdown_factor=KNOCKDOWN_FACTOR,
        )

        max_result = solve_cruise_samples(
            diameter_in=diameter_in,
            pitch_in=pitch_in,
            velocities_mps=FIT_VELOCITIES_MPS,
            motor=motor,
            battery=battery,
            max_current_a=MAX_CURRENT_A,
            cruise_throttle=MAX_THROTTLE,
            prop_database=database,
            min_rpm=MIN_RPM,
            max_rpm=MAX_RPM,
            rpm_step=RPM_STEP,
            knockdown=APPLY_KNOCKDOWN,
            knockdown_factor=KNOCKDOWN_FACTOR,
        )

        solver_runtime_s = (
            perf_counter() - solver_start
        )

        cruise_thrust_fit = fit_tuple(
            cruise_result.thrust_samples_n
        )

        max_thrust_fit = fit_tuple(
            max_result.thrust_samples_n
        )

        cruise_time_fit = fit_tuple(
            cruise_result.flight_time_samples_s
        )

        max_time_fit = fit_tuple(
            max_result.flight_time_samples_s
        )

        summaries.append(
            {
                "prop_name": prop_name,
                "diameter_in": diameter_in,
                "pitch_in": pitch_in,
                "solver_runtime_s": solver_runtime_s,
                "cruise_thrust_a": cruise_thrust_fit[0],
                "cruise_thrust_b": cruise_thrust_fit[1],
                "cruise_thrust_c": cruise_thrust_fit[2],
                "max_thrust_a": max_thrust_fit[0],
                "max_thrust_b": max_thrust_fit[1],
                "max_thrust_c": max_thrust_fit[2],
                "cruise_time_a": cruise_time_fit[0],
                "cruise_time_b": cruise_time_fit[1],
                "cruise_time_c": cruise_time_fit[2],
                "max_time_a": max_time_fit[0],
                "max_time_b": max_time_fit[1],
                "max_time_c": max_time_fit[2],
            }
        )

        rpm_plot_results.append(
            (
                prop_name,
                FIT_VELOCITIES_MPS.copy(),
                cruise_result.selected_rpm.copy(),
            )
        )

        for velocity_index, velocity_mps in enumerate(
            FIT_VELOCITIES_MPS
        ):
            cruise_current_a = float(
                cruise_result.selected_current_a[
                    velocity_index
                ]
            )

            max_current_a = float(
                max_result.selected_current_a[
                    velocity_index
                ]
            )

            cruise_torque_nm = (
                cruise_current_a - no_load_current_a
            ) * kt_nm_per_a

            max_torque_nm = (
                max_current_a - no_load_current_a
            ) * kt_nm_per_a

            cruise_voltage_sag_v = (
                BATTERY_NOMINAL_V
                - cruise_current_a
                * battery_resistance_ohm
            )

            max_voltage_sag_v = (
                BATTERY_NOMINAL_V
                - max_current_a
                * battery_resistance_ohm
            )

            cruise_voltage_required_v = (
                cruise_result.selected_throttle[
                    velocity_index
                ]
                * cruise_voltage_sag_v
            )

            max_voltage_required_v = (
                max_result.selected_throttle[
                    velocity_index
                ]
                * max_voltage_sag_v
            )

            sample_rows.append(
                {
                    "prop_name": prop_name,
                    "diameter_in": diameter_in,
                    "pitch_in": pitch_in,
                    "velocity_mps": float(velocity_mps),
                    "cruise_selected_rpm": float(
                        cruise_result.selected_rpm[
                            velocity_index
                        ]
                    ),
                    "cruise_thrust_n": float(
                        cruise_result.thrust_samples_n[
                            velocity_index
                        ]
                    ),
                    "cruise_torque_nm": float(
                        cruise_torque_nm
                    ),
                    "cruise_current_a": cruise_current_a,
                    "cruise_voltage_sag_v": float(
                        cruise_voltage_sag_v
                    ),
                    "cruise_voltage_required_v": float(
                        cruise_voltage_required_v
                    ),
                    "cruise_throttle": float(
                        cruise_result.selected_throttle[
                            velocity_index
                        ]
                    ),
                    "cruise_power_w": float(
                        cruise_result.selected_power_w[
                            velocity_index
                        ]
                    ),
                    "cruise_flight_time_s": float(
                        cruise_result.flight_time_samples_s[
                            velocity_index
                        ]
                    ),
                    "cruise_valid_rpm_count": int(
                        cruise_result.valid_rpm_count[
                            velocity_index
                        ]
                    ),
                    "cruise_failed": bool(
                        cruise_result.failed_mask[
                            velocity_index
                        ]
                    ),
                    "max_selected_rpm": float(
                        max_result.selected_rpm[
                            velocity_index
                        ]
                    ),
                    "max_thrust_n": float(
                        max_result.thrust_samples_n[
                            velocity_index
                        ]
                    ),
                    "max_torque_nm": float(
                        max_torque_nm
                    ),
                    "max_current_a": max_current_a,
                    "max_voltage_sag_v": float(
                        max_voltage_sag_v
                    ),
                    "max_voltage_required_v": float(
                        max_voltage_required_v
                    ),
                    "max_throttle": float(
                        max_result.selected_throttle[
                            velocity_index
                        ]
                    ),
                    "max_power_w": float(
                        max_result.selected_power_w[
                            velocity_index
                        ]
                    ),
                    "max_flight_time_s": float(
                        max_result.flight_time_samples_s[
                            velocity_index
                        ]
                    ),
                    "max_valid_rpm_count": int(
                        max_result.valid_rpm_count[
                            velocity_index
                        ]
                    ),
                    "max_failed": bool(
                        max_result.failed_mask[
                            velocity_index
                        ]
                    ),
                }
            )

        print()
        print("=" * 65)
        print(
            f"PROP {prop_index}/{len(PROPELLERS)}: "
            f"{prop_name}"
        )
        print(
            f"Geometry: {diameter_in:g} x "
            f"{pitch_in:g} in"
        )
        print(
            f"Solver runtime: "
            f"{solver_runtime_s:.6f} s"
        )
        print(
            "Cruise thrust fit: "
            f"[{cruise_thrust_fit[0]: .10e}, "
            f"{cruise_thrust_fit[1]: .10e}, "
            f"{cruise_thrust_fit[2]: .10e}]"
        )
        print(
            "Max thrust fit:    "
            f"[{max_thrust_fit[0]: .10e}, "
            f"{max_thrust_fit[1]: .10e}, "
            f"{max_thrust_fit[2]: .10e}]"
        )

        print()
        print(
            "Velocity | Cruise RPM | Cruise T | "
            "Max RPM | Max T"
        )

        for velocity_index, velocity_mps in enumerate(
            FIT_VELOCITIES_MPS
        ):
            print(
                f"{velocity_mps:8.3f} | "
                f"{cruise_result.selected_rpm[velocity_index]:10.0f} | "
                f"{cruise_result.thrust_samples_n[velocity_index]:8.3f} | "
                f"{max_result.selected_rpm[velocity_index]:7.0f} | "
                f"{max_result.thrust_samples_n[velocity_index]:8.3f}"
            )

    write_csv(
        output_dir / "python_prop_summary.csv",
        summaries,
    )

    write_csv(
        output_dir / "python_prop_samples.csv",
        sample_rows,
    )

    plot_fit_comparison(
        summaries=summaries,
        prefix="cruise_thrust",
        title="Cruise Thrust Comparison",
        ylabel="Thrust [N]",
        output_path=(
            output_dir / "python_cruise_thrust.png"
        ),
    )

    plot_fit_comparison(
        summaries=summaries,
        prefix="max_thrust",
        title="Maximum Thrust Comparison",
        ylabel="Thrust [N]",
        output_path=(
            output_dir / "python_max_thrust.png"
        ),
    )

    plot_fit_comparison(
        summaries=summaries,
        prefix="cruise_time",
        title="Cruise Flight-Time Comparison",
        ylabel="Flight time [s]",
        output_path=(
            output_dir / "python_cruise_time.png"
        ),
    )

    plt.figure(figsize=(9, 6))

    for (
        prop_name,
        velocities_mps,
        selected_rpm,
    ) in rpm_plot_results:
        plt.plot(
            velocities_mps,
            selected_rpm,
            marker="o",
            label=prop_name,
        )

    plt.xlabel("Velocity [m/s]")
    plt.ylabel("Selected cruise RPM")
    plt.title("Selected Cruise RPM Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "python_cruise_rpm.png",
        dpi=160,
    )
    plt.close()

    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)

    print(
        f"{'Prop':<12}"
        f"{'Runtime [s]':>14}"
        f"{'Cruise a':>16}"
        f"{'Cruise b':>16}"
        f"{'Cruise c':>16}"
    )

    for summary in summaries:
        print(
            f"{str(summary['prop_name']):<12}"
            f"{float(summary['solver_runtime_s']):>14.6f}"
            f"{float(summary['cruise_thrust_a']):>16.6e}"
            f"{float(summary['cruise_thrust_b']):>16.6e}"
            f"{float(summary['cruise_thrust_c']):>16.6e}"
        )

    print()
    print(f"Files written to:\n{output_dir}")


if __name__ == "__main__":
    main()