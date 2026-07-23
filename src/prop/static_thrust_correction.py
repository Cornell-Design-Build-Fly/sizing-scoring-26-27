from __future__ import annotations

"""
Build static-thrust correction factors from static thrust test data.

Expected input CSV:
    src/prop/data/static_thrust_tests_for_correction_filtered.csv

Expected output:
    src/prop/data/static_thrust_correction_factors.csv
    src/prop/data/static_thrust_correction_factors.json

Important assumption:
    RPM was not measured during the static thrust tests. This script estimates RPM by
    finding the model RPM whose calculated throttle is closest to the commanded ESC /
    controller throttle in the test data.

The correction factor is:

    correction_factor = measured_static_thrust_n / predicted_static_thrust_n

where predicted_static_thrust_n comes from the existing prop database at V = 0 mph.

This script corrects THRUST only. It does not change torque/power.
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
from typing import Any

import numpy as np

from src.prop.main_prop import motor_check
from src.prop.prop_classes import Battery, Motor

# Prefer the separated prop database file. Fall back to main_prop for compatibility
# with older branches where the database code still lived in main_prop.py.
try:
    from src.prop.prop_database import (
        ContinuousPropDatabase,
        load_default_prop_database,
    )
except ImportError:
    from src.prop.main_prop import (  # type: ignore
        ContinuousPropDatabase,
        load_default_prop_database,
    )


DATA_DIR = Path(__file__).resolve().parent / "data"

INPUT_CSV_PATH = DATA_DIR / "static_thrust_tests_for_correction_filtered.csv"
OUTPUT_CSV_PATH = DATA_DIR / "static_thrust_correction_factors.csv"
OUTPUT_JSON_PATH = DATA_DIR / "static_thrust_correction_factors.json"

G_TO_NEWTONS = 0.00980665

DEFAULT_MIN_RPM = 1000
DEFAULT_RPM_STEP = 50


@dataclass(frozen=True, slots=True)
class StaticTestRow:
    test_name: str
    sample_index: int

    prop_diameter_in: float
    prop_pitch_in: float

    battery_capacity_mah: float
    battery_capacity_ah: float
    battery_cells: int
    battery_voltage_v: float

    motor_kv: float
    motor_max_current_a: float
    motor_max_power_w: float
    motor_no_load_current_a: float
    motor_resistance_ohm: float

    commanded_throttle: float
    commanded_throttle_percent: float

    raw_load_cell_g: float
    zero_throttle_baseline_g: float
    measured_thrust_g: float
    measured_thrust_n: float


def normalize_throttle(value: float) -> float:
    """
    Accepts either 0.65 or 65 for 65% throttle.
    Returns a 0-to-1 throttle fraction.
    """
    throttle = float(value)

    if throttle > 1.0:
        throttle /= 100.0

    if throttle <= 0.0 or throttle > 1.0:
        raise ValueError(f"Commanded throttle must be in (0, 1] or (0, 100]. Got {value}")

    return throttle


def get_float(row: dict[str, str], key: str, default: float | None = None) -> float:
    raw_value = row.get(key, "")

    if raw_value == "" or raw_value is None:
        if default is None:
            raise ValueError(f"Missing required column/value: {key}")
        return float(default)

    return float(raw_value)


def get_int(row: dict[str, str], key: str, default: int | None = None) -> int:
    raw_value = row.get(key, "")

    if raw_value == "" or raw_value is None:
        if default is None:
            raise ValueError(f"Missing required column/value: {key}")
        return int(default)

    # Allows cells to appear as "6.0" from spreadsheets.
    return int(float(raw_value))


def measured_thrust_to_newtons(row: dict[str, str]) -> float:
    """
    Reads measured thrust from the CSV.

    Preferred:
        measured_thrust + measured_thrust_units

    Fallback:
        measured_thrust_g converted to N
    """
    units = row.get("measured_thrust_units", "").strip().lower()

    if row.get("measured_thrust", "") != "":
        measured_value = float(row["measured_thrust"])

        if units in ("n", "newton", "newtons", ""):
            return measured_value

        if units in ("g", "gram", "grams"):
            return measured_value * G_TO_NEWTONS

        raise ValueError(f"Unsupported measured_thrust_units: {units}")

    measured_thrust_g = get_float(row, "measured_thrust_g")
    return measured_thrust_g * G_TO_NEWTONS


def read_static_test_rows(path: Path = INPUT_CSV_PATH) -> list[StaticTestRow]:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find static thrust CSV: {path}\n"
            "Place static_thrust_tests_for_correction_filtered.csv in src/prop/data/."
        )

    rows: list[StaticTestRow] = []

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for raw in reader:
            commanded_throttle = normalize_throttle(
                get_float(raw, "commanded_throttle", default=get_float(raw, "commanded_throttle_percent", 0.0))
            )

            battery_capacity_mah = get_float(raw, "battery_capacity_mah")
            battery_capacity_ah = get_float(
                raw,
                "battery_capacity_ah",
                default=battery_capacity_mah / 1000.0,
            )

            measured_thrust_n = measured_thrust_to_newtons(raw)

            # Skip bad/non-lifting points defensively. The filtered CSV should already
            # remove these, but this keeps the script robust if someone uses raw data.
            if measured_thrust_n <= 0.0:
                continue

            rows.append(
                StaticTestRow(
                    test_name=raw.get("test_name", "unnamed_test"),
                    sample_index=get_int(raw, "sample_index", default=len(rows) + 1),
                    prop_diameter_in=get_float(raw, "prop_diameter_in"),
                    prop_pitch_in=get_float(raw, "prop_pitch_in"),
                    battery_capacity_mah=battery_capacity_mah,
                    battery_capacity_ah=battery_capacity_ah,
                    battery_cells=get_int(raw, "battery_cells", default=6),
                    battery_voltage_v=get_float(raw, "battery_voltage_v", default=22.2),
                    motor_kv=get_float(raw, "motor_kv", default=520.0),
                    motor_max_current_a=get_float(raw, "motor_max_current_a", default=100.0),
                    motor_max_power_w=get_float(raw, "motor_max_power_w", default=2000.0),
                    motor_no_load_current_a=get_float(raw, "motor_no_load_current_a", default=1.4),
                    motor_resistance_ohm=get_float(raw, "motor_resistance_ohm", default=0.016),
                    commanded_throttle=commanded_throttle,
                    commanded_throttle_percent=get_float(
                        raw,
                        "commanded_throttle_percent",
                        default=100.0 * commanded_throttle,
                    ),
                    raw_load_cell_g=get_float(raw, "raw_load_cell_g", default=0.0),
                    zero_throttle_baseline_g=get_float(raw, "zero_throttle_baseline_g", default=0.0),
                    measured_thrust_g=get_float(
                        raw,
                        "measured_thrust_g",
                        default=measured_thrust_n / G_TO_NEWTONS,
                    ),
                    measured_thrust_n=measured_thrust_n,
                )
            )

    return rows


def battery_resistance_ohm(capacity_ah: float, cells: int) -> float:
    """
    Matches the simple battery resistance model used by the prop code/MATLAB conversion.
    """
    if capacity_ah <= 0:
        raise ValueError("Battery capacity must be positive.")

    if cells <= 0:
        raise ValueError("Battery cell count must be positive.")

    return (0.013 / capacity_ah) * cells


def make_motor(row: StaticTestRow) -> Motor:
    """
    Uses the measured/known motor constants directly instead of estimating them from Kv/power.
    """
    return Motor(
        kv=row.motor_kv,
        Rm=row.motor_resistance_ohm,
        max_power=row.motor_max_power_w,
        I0=row.motor_no_load_current_a,
        max_current=row.motor_max_current_a,
    )


def make_battery(row: StaticTestRow) -> Battery:
    """
    For static test calibration, use the listed pack capacity directly.

    The usable-battery fraction is not relevant to estimating RPM or static thrust;
    it only affects predicted flight time.
    """
    return Battery(
        vnom=row.battery_voltage_v,
        cells=row.battery_cells,
        Rb=battery_resistance_ohm(row.battery_capacity_ah, row.battery_cells),
        Crat=0.0,
        capacity=row.battery_capacity_ah,
        useable_fraction=1.0,
    )


def estimate_max_search_rpm(row: StaticTestRow, prop_database: ContinuousPropDatabase) -> int:
    """
    Picks a reasonable upper RPM bound.

    No-load RPM is approximately Kv * voltage. Loaded RPM should be below this,
    but a small margin avoids clipping edge cases.
    """
    no_load_rpm_estimate = row.motor_kv * row.battery_voltage_v

    database_max_rpm = getattr(prop_database, "rpm_bounds", (DEFAULT_MIN_RPM, 20000))[1]

    return int(
        max(
            DEFAULT_MIN_RPM + 500,
            min(
                1.10 * no_load_rpm_estimate,
                database_max_rpm,
                20000,
            ),
        )
    )


def throttle_error_at_rpm(
    rpm: float,
    row: StaticTestRow,
    motor: Motor,
    battery: Battery,
    prop_database: ContinuousPropDatabase,
) -> tuple[float, Any]:
    torque_nm = prop_database.torque(
        row.prop_diameter_in,
        row.prop_pitch_in,
        0.0,
        rpm,
    )

    if not math.isfinite(torque_nm):
        return math.inf, None

    check = motor_check(
        torque=torque_nm,
        rpm=rpm,
        motor=motor,
        battery=battery,
    )

    if not math.isfinite(check.throttle):
        return math.inf, check

    return abs(check.throttle - row.commanded_throttle), check


def estimate_static_rpm_for_commanded_throttle(
    row: StaticTestRow,
    prop_database: ContinuousPropDatabase,
    min_rpm: int = DEFAULT_MIN_RPM,
    rpm_step: int = DEFAULT_RPM_STEP,
) -> tuple[float, Any]:
    """
    Estimates test RPM by matching commanded throttle to model-calculated throttle.

    This assumes commanded ESC/controller throttle is close enough to the model's
    voltage-based throttle estimate to use as an RPM proxy.
    """
    motor = make_motor(row)
    battery = make_battery(row)

    max_rpm = estimate_max_search_rpm(row, prop_database)

    low = float(min_rpm)
    high = float(max_rpm)

    best_rpm = low
    best_check = None
    best_error = math.inf

    while (high - low) > rpm_step:
        mid = 0.5 * (low + high)

        error, check = throttle_error_at_rpm(
            rpm=mid,
            row=row,
            motor=motor,
            battery=battery,
            prop_database=prop_database,
        )

        if error < best_error:
            best_error = error
            best_rpm = mid
            best_check = check

        if check is None or not math.isfinite(check.throttle):
            high = mid
            continue

        if check.throttle < row.commanded_throttle:
            low = mid
        else:
            high = mid

    # Small local sweep around the binary-search answer to avoid midpoint artifacts.
    local_low = max(min_rpm, int(best_rpm - 3 * rpm_step))
    local_high = min(max_rpm, int(best_rpm + 3 * rpm_step))

    for rpm in range(local_low, local_high + rpm_step, rpm_step):
        error, check = throttle_error_at_rpm(
            rpm=float(rpm),
            row=row,
            motor=motor,
            battery=battery,
            prop_database=prop_database,
        )

        if error < best_error:
            best_error = error
            best_rpm = float(rpm)
            best_check = check

    if best_check is None:
        raise RuntimeError(
            f"Could not estimate RPM for {row.test_name}, "
            f"{row.prop_diameter_in}x{row.prop_pitch_in}, "
            f"commanded throttle {row.commanded_throttle:.2f}."
        )

    return float(best_rpm), best_check


def calculate_correction_factor(
    row: StaticTestRow,
    prop_database: ContinuousPropDatabase,
    rpm_cache: dict[tuple[Any, ...], tuple[float, Any]],
) -> dict[str, Any]:
    """
    Calculates one correction-factor row.

    Uses a cache because many rows share the same setup and commanded throttle.
    """
    rpm_cache_key = (
        row.prop_diameter_in,
        row.prop_pitch_in,
        row.battery_capacity_ah,
        row.battery_cells,
        row.battery_voltage_v,
        row.motor_kv,
        row.motor_resistance_ohm,
        row.motor_no_load_current_a,
        row.commanded_throttle,
    )

    if rpm_cache_key in rpm_cache:
        estimated_rpm, check = rpm_cache[rpm_cache_key]
    else:
        estimated_rpm, check = estimate_static_rpm_for_commanded_throttle(
            row=row,
            prop_database=prop_database,
        )
        rpm_cache[rpm_cache_key] = (estimated_rpm, check)

    predicted_static_thrust_n = prop_database.thrust(
        row.prop_diameter_in,
        row.prop_pitch_in,
        0.0,
        estimated_rpm,
    )

    if predicted_static_thrust_n <= 0.0 or not math.isfinite(predicted_static_thrust_n):
        correction_factor = 0.0
    else:
        correction_factor = row.measured_thrust_n / predicted_static_thrust_n

    return {
        "test_name": row.test_name,
        "sample_index": row.sample_index,
        "prop_diameter_in": row.prop_diameter_in,
        "prop_pitch_in": row.prop_pitch_in,
        "battery_capacity_mah": row.battery_capacity_mah,
        "battery_capacity_ah": row.battery_capacity_ah,
        "battery_cells": row.battery_cells,
        "battery_voltage_v": row.battery_voltage_v,
        "motor_kv": row.motor_kv,
        "motor_max_current_a": row.motor_max_current_a,
        "motor_max_power_w": row.motor_max_power_w,
        "motor_no_load_current_a": row.motor_no_load_current_a,
        "motor_resistance_ohm": row.motor_resistance_ohm,
        "commanded_throttle": row.commanded_throttle,
        "commanded_throttle_percent": row.commanded_throttle_percent,
        "estimated_rpm": estimated_rpm,
        "measured_static_thrust_g": row.measured_thrust_g,
        "measured_static_thrust_n": row.measured_thrust_n,
        "predicted_static_thrust_n": float(predicted_static_thrust_n),
        "correction_factor": float(correction_factor),
        "model_calculated_throttle": float(check.throttle),
        "model_current_a": float(check.current_a),
        "model_power_w": float(check.power_w),
        "model_voltage_required_v": float(check.voltage_required_v),
        "model_voltage_sag_v": float(check.voltage_sag_v),
        "raw_load_cell_g": row.raw_load_cell_g,
        "zero_throttle_baseline_g": row.zero_throttle_baseline_g,
    }


def save_results(results: list[dict[str, Any]]) -> None:
    if not results:
        raise ValueError("No correction-factor results to save.")

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    payload = {
        "description": "Static thrust correction factors for prop database knockdown.",
        "warning": (
            "RPM was not measured. RPM was estimated by matching commanded ESC/controller "
            "throttle to the motor/battery model's calculated throttle. Correction factors "
            "should be treated as approximate."
        ),
        "input_csv": str(INPUT_CSV_PATH),
        "output_csv": str(OUTPUT_CSV_PATH),
        "results": results,
    }

    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


def print_summary(results: list[dict[str, Any]]) -> None:
    print()
    print("Static thrust correction factors")
    print("--------------------------------")
    print(
        f"{'Test':<20} | {'Prop':>7} | {'Thr%':>5} | {'RPM':>7} | "
        f"{'Meas N':>8} | {'Pred N':>8} | {'Factor':>7}"
    )
    print("-" * 90)

    for result in results:
        prop_name = f"{result['prop_diameter_in']:.1f}x{result['prop_pitch_in']:.1f}"

        print(
            f"{result['test_name']:<20} | "
            f"{prop_name:>7} | "
            f"{result['commanded_throttle_percent']:5.0f} | "
            f"{result['estimated_rpm']:7.0f} | "
            f"{result['measured_static_thrust_n']:8.2f} | "
            f"{result['predicted_static_thrust_n']:8.2f} | "
            f"{result['correction_factor']:7.3f}"
        )

    factors = np.array([row["correction_factor"] for row in results], dtype=float)
    factors = factors[np.isfinite(factors)]

    print()
    print(f"Rows processed: {len(results)}")
    print(f"Mean correction factor:   {float(np.mean(factors)):.3f}")
    print(f"Median correction factor: {float(np.median(factors)):.3f}")
    print(f"Min correction factor:    {float(np.min(factors)):.3f}")
    print(f"Max correction factor:    {float(np.max(factors)):.3f}")
    print()
    print(f"Saved CSV:  {OUTPUT_CSV_PATH}")
    print(f"Saved JSON: {OUTPUT_JSON_PATH}")


def main() -> None:
    print("Loading prop database...")
    prop_database = load_default_prop_database()

    # Optional first-use warmup. This prevents the first real correction row from
    # paying all of SciPy's first-interpolation overhead.
    prop_database.thrust(14.0, 8.5, 0.0, 8000.0)
    prop_database.torque(14.0, 8.5, 0.0, 8000.0)

    print(f"Reading static thrust data: {INPUT_CSV_PATH}")
    test_rows = read_static_test_rows(INPUT_CSV_PATH)

    print(f"Calculating correction factors for {len(test_rows)} rows...")
    rpm_cache: dict[tuple[Any, ...], tuple[float, Any]] = {}

    results = [
        calculate_correction_factor(
            row=row,
            prop_database=prop_database,
            rpm_cache=rpm_cache,
        )
        for row in test_rows
    ]

    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()
