"""Run matched top-line optimizations at fixed battery cell counts.

The battery model belongs to ``src.prop``; this runner keeps the two optimizer
runs paired by using the same seed and differential-evolution settings.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.mech.main_mech import evaluate_mechanical_module
from src.opt.topline_opt import ToplineConfig, _objective, run_topline_optimization
from src.prop.prop_classes import normalize_battery_cell_count
from src.prop.prop_helper_functions import make_battery_from_design
from src.vectors import DesignVector, ParameterVector


DEFAULT_OUTPUT_DIR = Path("data_dump") / "prop" / "battery_cell_comparison"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer.")
    return parsed


def _battery_mass_kg(report: dict[str, Any]) -> float:
    items = report["mechanical"]["missions"]["M1"]["items"]
    return float(next(item["mass_kg"] for item in items if item["name"] == "Battery"))


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    report = json.loads(
        (run_dir / "best_design_report.json").read_text(encoding="utf-8")
    )
    run_summary = json.loads(
        (run_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    vector = report["resolved_vector"]
    battery = report["battery"]
    missions = report["mechanical"]["missions"]
    aero = report["aero"]

    row: dict[str, Any] = {
        "battery_cell_count": int(battery["cell_count"]),
        "run_dir": str(run_dir),
        "score": float(report["score"]),
        "propulsion_feasible": bool(report.get("propulsion_feasible", False)),
        "propulsion_penalty": float(report["penalties"].get("propulsion", 0.0)),
        "ground_score": float(report["breakdown"]["ground"]),
        "m1_score": float(report["breakdown"]["m1"]),
        "m2_score": float(report["breakdown"]["m2"]),
        "m3_score": float(report["breakdown"]["m3"]),
        "nominal_voltage_v": float(battery["nominal_voltage_v"]),
        "battery_capacity_ah": float(battery["capacity_ah"]),
        "nominal_energy_wh": float(battery["nominal_energy_wh"]),
        "usable_energy_wh": float(battery["usable_energy_wh"]),
        "battery_internal_resistance_ohm": float(
            battery["internal_resistance_ohm"]
        ),
        "battery_mass_kg": _battery_mass_kg(report),
        "motor_kv_rpm_per_v": float(vector["motor_kv"]),
        "motor_max_power_w": float(vector["motor_max_power"]),
        "prop_diameter_in": float(vector["prop_diameter_in"]),
        "prop_pitch_in": float(vector["prop_pitch_in"]),
        "cruise_throttle": float(vector["cruise_throttle"]),
        "mission3_cruise_throttle": float(vector["mission3_cruise_throttle"]),
        "extra_shipping_containers": int(
            round(float(vector["extra_shipping_containers"]))
        ),
        "sensor_length_m": float(vector["sensor_length_m"]),
        "sensor_weight_kg": float(vector["sensor_weight_kg"]),
        "mission3_sensor_weight_kg": float(
            vector["mission3_sensor_weight_kg"]
        ),
        "mission3_sensor_length_m": float(
            vector["mission3_sensor_length_m"]
        ),
        "wing_span_m": float(vector["wing_span"]),
        "wing_chord_m": float(vector["wing_chord"]),
        "m1_mass_kg": float(missions["M1"]["total_mass_kg"]),
        "m2_mass_kg": float(missions["M2"]["total_mass_kg"]),
        "m3_mass_kg": float(missions["M3"]["total_mass_kg"]),
        "m1_lap_time_s": float(aero["M1"]["lap_time"]),
        "m2_lap_time_s": float(aero["M2"]["lap_time"]),
        "m3_lap_time_s": float(aero["M3"]["lap_time"]),
        "nfev": int(run_summary["nfev"]),
        "nit": int(run_summary["nit"]),
        "success": bool(run_summary["success"]),
    }
    return row


def _sanity_row(cell_count: int, capacity_ah: float = 3.0) -> dict[str, float | int]:
    """Evaluate pack invariants on the same baseline aircraft."""

    design = DesignVector(
        batt_capacity=capacity_ah,
        battery_cell_count=cell_count,
    )
    parameters = ParameterVector()
    battery = make_battery_from_design(design, parameters)
    mechanical = evaluate_mechanical_module(design, parameter_vector=parameters)
    battery_item = next(
        item for item in mechanical.for_mission("M1").items if item.name == "Battery"
    )
    return {
        "battery_cell_count": battery.cells,
        "capacity_ah": battery.capacity,
        "nominal_voltage_v": battery.vnom,
        "nominal_energy_wh": design.batt_energy,
        "usable_energy_wh": design.batt_energy * battery.useable_fraction,
        "internal_resistance_ohm": battery.get_Rb(),
        "battery_mass_kg": battery_item.mass_kg,
        "m1_mass_kg": mechanical.for_mission("M1").total_mass_kg,
    }


def _numeric_deltas(
    baseline: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for name in baseline.keys() & comparison.keys():
        if name in {"battery_cell_count", "nfev", "nit"}:
            continue
        left = baseline[name]
        right = comparison[name]
        if isinstance(left, bool) or isinstance(right, bool):
            continue
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            deltas[f"{name}_delta"] = float(right) - float(left)
            if float(left) != 0.0:
                deltas[f"{name}_percent_change"] = (
                    100.0 * (float(right) - float(left)) / abs(float(left))
                )
    return deltas


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cross_evaluate_fixed_designs(
    run_dirs: dict[int, Path],
    cell_counts: list[int],
) -> list[dict[str, float | int]]:
    """Score every fixed-run winner at every requested battery cell count."""

    rows: list[dict[str, float | int]] = []
    for source_cell_count, run_dir in run_dirs.items():
        report = json.loads(
            (run_dir / "best_design_report.json").read_text(encoding="utf-8")
        )
        vector = report["optimizer_vector"]
        optimizer_array = np.asarray(
            [vector[name] for name in DesignVector.opt_names()],
            dtype=float,
        )
        for evaluation_cell_count in cell_counts:
            objective = _objective(
                optimizer_array,
                config=ToplineConfig(
                    battery_cell_count=evaluation_cell_count,
                    optimize_battery_cell_count=False,
                ),
            )
            rows.append(
                {
                    "source_cell_count": source_cell_count,
                    "evaluation_cell_count": evaluation_cell_count,
                    "score": -float(objective),
                }
            )
    return rows


def run_comparison(args: argparse.Namespace) -> Path:
    cell_counts = [normalize_battery_cell_count(value) for value in args.cells]
    if len(set(cell_counts)) != len(cell_counts):
        raise ValueError("Each requested battery cell count must be unique.")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    comparison_dir = args.output_dir / f"comparison_{stamp}"
    comparison_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    joint_row: dict[str, Any] | None = None
    fixed_run_dirs: dict[int, Path] = {}
    cross_evaluations: list[dict[str, float | int]] = []
    sanity_rows = [_sanity_row(cell_count) for cell_count in cell_counts]
    if args.mode in {"fixed", "both"}:
        for cell_count in cell_counts:
            config = ToplineConfig(
                workers=args.workers,
                popsize=args.popsize,
                maxiter=args.maxiter,
                seed=args.seed,
                output_dir=comparison_dir / f"{cell_count}S",
                battery_cell_count=cell_count,
                optimize_battery_cell_count=False,
                save_best_visualization=not args.no_visualization,
            )
            result = run_topline_optimization(config)
            run_dir = Path(result["output_dir"])
            fixed_run_dirs[cell_count] = run_dir
            rows.append(_summarize_run(run_dir))
        _write_csv(comparison_dir / "comparison_summary.csv", rows)
        cross_evaluations = _cross_evaluate_fixed_designs(
            fixed_run_dirs,
            cell_counts,
        )
        _write_csv(comparison_dir / "cross_evaluation.csv", cross_evaluations)

    if args.mode in {"joint", "both"}:
        bounds = (min(cell_counts), max(cell_counts))
        config = ToplineConfig(
            workers=args.workers,
            popsize=args.popsize,
            maxiter=args.maxiter,
            seed=args.seed,
            output_dir=comparison_dir / "joint",
            optimize_battery_cell_count=True,
            battery_cell_count_bounds=bounds,
            battery_cell_count_choices=tuple(cell_counts),
            save_best_visualization=not args.no_visualization,
        )
        result = run_topline_optimization(config)
        joint_row = _summarize_run(Path(result["output_dir"]))
        _write_csv(comparison_dir / "joint_summary.csv", [joint_row])

    payload: dict[str, Any] = {
        "settings": {
            "mode": args.mode,
            "cell_counts": cell_counts,
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "workers": args.workers,
            "seed": args.seed,
        },
        "baseline_pack_sanity": sanity_rows,
        "runs": rows,
        "joint_run": joint_row,
        "cross_evaluations": cross_evaluations,
    }
    if rows:
        payload["best_fixed_run"] = max(rows, key=lambda row: row["score"])
    if len(rows) == 2:
        payload["second_minus_first"] = _numeric_deltas(rows[0], rows[1])
        payload["sanity_second_minus_first"] = _numeric_deltas(
            sanity_rows[0], sanity_rows[1]
        )
    (comparison_dir / "comparison_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nBattery-cell comparison")
    for row in rows:
        print(
            f"  {row['battery_cell_count']}S: score={row['score']:.6g}, "
            f"battery={row['battery_mass_kg']:.3f} kg, "
            f"energy={row['nominal_energy_wh']:.1f} Wh"
        )
    if rows:
        best_fixed = max(rows, key=lambda row: row["score"])
        print(
            f"  best fixed case: {best_fixed['battery_cell_count']}S "
            f"(score={best_fixed['score']:.6g})"
        )
    if joint_row is not None:
        print(
            f"  joint optimum: {joint_row['battery_cell_count']}S, "
            f"score={joint_row['score']:.6g}"
        )
    if rows:
        print(f"  summary: {comparison_dir / 'comparison_summary.csv'}")
    if joint_row is not None:
        print(f"  joint summary: {comparison_dir / 'joint_summary.csv'}")
    print(f"  details: {comparison_dir / 'comparison_summary.json'}")
    return comparison_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run matched fixed-cell-count top-line optimizations."
    )
    parser.add_argument("--cells", nargs="+", type=_positive_int, default=[6, 8])
    parser.add_argument(
        "--mode",
        choices=("fixed", "joint", "both"),
        default="fixed",
        help=(
            "Run fixed cases, one joint discrete optimization, or both. Joint "
            "mode allows exactly the values supplied through --cells."
        ),
    )
    parser.add_argument("--maxiter", type=_nonnegative_int, default=300)
    parser.add_argument("--popsize", type=_positive_int, default=25)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-visualization", action="store_true")
    return parser


def main() -> None:
    run_comparison(_parser().parse_args())


if __name__ == "__main__":
    main()
