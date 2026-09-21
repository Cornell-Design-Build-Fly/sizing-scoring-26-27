"""Exhaustive catalog search; run from the repository root with -m."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from src.main import main
from src.mech.main_mech import evaluate_mechanical_module
from src.motor_search.catalog import InterpolationOnlyDatabase
from src.opt.score import total_score
from src.prop.continuous_prop_database import load_default_continuous_prop_database
from src.prop.mission_performance import DEFAULT_PROPULSION_REQUIREMENTS
from src.vectors import DesignVector, ParameterVector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CSV = ROOT / "src/prop/data/motor_data - motor_data.csv.csv"
DB = None
BASE = None
PROPS = None


def initialize(airframe):
    global DB, BASE, PROPS
    with redirect_stdout(io.StringIO()):
        DB = InterpolationOnlyDatabase(load_default_continuous_prop_database().catalog)
    BASE = DesignVector(**airframe)
    PROPS = sorted(DB.catalog.surfaces, key=lambda s: (s.diameter_in, s.pitch_in, s.key))


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_motors(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"Manufacturer", "Motor", "Kv", "Watts", "Io", "Rm", "Mass"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Motor CSV must contain {sorted(required)}")
    result = []
    for number, row in enumerate(rows):
        motor = {"row_index": number, "source_id": row.get("", str(number)),
                 "manufacturer": row["Manufacturer"], "motor": row["Motor"]}
        try:
            for column, name in (("Kv", "motor_kv"), ("Watts", "motor_max_power"),
                                 ("Io", "motor_no_load_current_a"),
                                 ("Rm", "motor_resistance_ohm"), ("Mass", "motor_mass_kg")):
                value = float(row[column]) / (1000 if column == "Mass" else 1)
                if not math.isfinite(value) or value < 0 or (column in {"Kv", "Watts", "Mass"} and value == 0):
                    raise ValueError(f"Invalid {column}: {row[column]!r}")
                motor[name] = value
        except ValueError as exc:
            motor["input_error"] = str(exc)
        result.append(motor)
    return result


def evaluate(design, first, third):
    dv = replace(design, prop_diameter_in=first.diameter_in, prop_pitch_in=first.pitch_in,
                 mission3_prop_diameter_in=third.diameter_in, mission3_prop_pitch_in=third.pitch_in)
    score, breakdown, details = main(dv, ParameterVector(), prop_database=DB, return_details=True)
    details["score"] = score
    details["breakdown"] = breakdown
    return details


def summarize(details, first, third):
    propulsion = details["propulsion"]
    result = {
        "m12_prop": first.key, "m3_prop": third.key,
        "m12_diameter_in": first.diameter_in, "m12_pitch_in": first.pitch_in,
        "m3_diameter_in": third.diameter_in, "m3_pitch_in": third.pitch_in,
        "feasible": bool(details["propulsion_feasible"] and details["penalty_total"] <= 1e-9
                         and details["breakdown"][2] > 0 and details["breakdown"][3] > 0),
        "propulsion_feasible": bool(details["propulsion_feasible"]),
        "score": float(details["score"]), "raw_score": sum(details["breakdown"]),
        "penalty_total": float(details["penalty_total"]),
        "penalty_mechanical": float(details["penalty_mechanical"]),
        "penalty_aero_m2": float(details["penalty_aero_m2"]),
        "penalty_aero_m3": float(details["penalty_aero_m3"]),
        "penalty_propulsion": float(details["penalty_propulsion"]),
        "penalty_overweight": float(details["penalty_overweight"]),
        "max_takeoff_mass_kg": float(details["max_takeoff_mass_kg"]),
    }
    for mission in (1, 2, 3):
        perf = propulsion.get(f"M{mission}", {})
        result[f"m{mission}_lap_time_s"] = float(details[f"M{mission}"].lap_time)
        result[f"m{mission}_feasible"] = bool(perf.get("feasible", False))
        result[f"m{mission}_limiting_constraint"] = perf.get("limiting_constraint", "cruise_trim")
        for field in ("cruise_speed_mps", "takeoff_distance_m", "climb_rate_mps",
                      "required_energy_wh", "energy_margin_wh", "maximum_propeller_rpm"):
            result[f"m{mission}_{field}"] = perf.get(field)
    return result


def rank_key(row):
    # Feasibility first; official net score next. Faster laps break score ties.
    # JSON checkpoints encode nonfinite (untrimmable) lap times as null.
    lap_times = [math.inf if row[f"m{mission}_lap_time_s"] is None
                 else row[f"m{mission}_lap_time_s"] for mission in (3, 2, 1)]
    return (bool(row["feasible"]), bool(row["propulsion_feasible"]), row["score"],
            *(-lap for lap in lap_times))


def combine(first_details, third_details, design, payload_mass):
    """Combine independent M1/M2 performance with a same-diameter M3 evaluation.

    M2 determines installation placement. Its propeller DIAMETER affects mass,
    hence M3 CG; its PITCH does not. M3 must therefore be solved once per M1/M2
    diameter, rather than incorrectly optimized independently of M1/M2 mass.
    """
    details = dict(third_details)
    details["M1"] = first_details["M1"]
    details["M2"] = first_details["M2"]
    details["propulsion"] = {**third_details["propulsion"],
                             **{key: value for key, value in first_details["propulsion"].items()
                                if key in ("M1", "M2")}}
    perfs = details["propulsion"]
    details["propulsion_feasible"] = (set(perfs) == {"M1", "M2", "M3"}
                                      and all(p["feasible"] for p in perfs.values()))
    details["penalty_aero_m2"] = first_details["penalty_aero_m2"]
    details["penalty_propulsion"] = max((p["penalty"] for p in perfs.values()), default=0.0)
    if not details["propulsion_feasible"]:
        details["penalty_propulsion"] = max(10.0, details["penalty_propulsion"])
    details["penalty_total"] = sum(details[key] for key in (
        "penalty_mechanical", "penalty_aero_m2", "penalty_aero_m3",
        "penalty_propulsion", "penalty_overweight"))
    score, breakdown = total_score(design, details["M1"].lap_time, details["M2"].lap_time,
                                   details["M3"].lap_time, payload_mass)
    details["score"] = score - details["penalty_total"]
    details["breakdown"] = breakdown
    return details


def search_motor(motor, single_prop=False):
    started = time.perf_counter()
    if "input_error" in motor:
        return {"motor": motor, "status": "invalid_input", "error": motor["input_error"]}
    design = replace(BASE, **{key: value for key, value in motor.items() if key.startswith("motor_")})
    common = []
    details_by_key = {}
    errors = []
    evaluations = 0
    for prop in PROPS:
        try:
            details = evaluate(design, prop, prop)
            evaluations += 1
            details_by_key[prop.key] = details
            common.append(summarize(details, prop, prop))
        except Exception as exc:
            errors.append({"m12_prop": prop.key, "m3_prop": prop.key,
                           "error": f"{type(exc).__name__}: {exc}"})
    if not common:
        return {"motor": motor, "status": "evaluation_error", "errors": errors}
    best_single = max(common, key=rank_key)
    best = best_single
    best_by_first = {row["m12_prop"]: row for row in common}
    eligible = [prop for prop in PROPS if prop.key in details_by_key
                and all(details_by_key[prop.key]["propulsion"].get(mission, {}).get("feasible", False)
                        for mission in ("M1", "M2"))]
    # No M3 propeller can repair an M1/M2 propulsion failure. Prune those
    # first-prop choices while retaining every common-prop diagnostic.
    considered_pairs = len(PROPS)
    if not single_prop and eligible:
        mechanical = evaluate_mechanical_module(design, parameter_vector=ParameterVector())
        payload_mass = sum(item.mass_kg for item in mechanical.for_mission("M2").items
                           if item.category == "mission_2_payload")
        by_diameter = {}
        for prop in eligible:
            by_diameter.setdefault(prop.diameter_in, []).append(prop)
        for choices in by_diameter.values():
            representative = choices[0]
            for third in PROPS:
                try:
                    if representative.key == third.key:
                        details = details_by_key[representative.key]
                    else:
                        details = evaluate(design, representative, third)
                        evaluations += 1
                    for first in choices:
                        if first.key == third.key:
                            continue
                        combined = combine(details_by_key[first.key], details, design, payload_mass)
                        row = summarize(combined, first, third)
                        considered_pairs += 1
                        if rank_key(row) > rank_key(best_by_first[first.key]):
                            best_by_first[first.key] = row
                        if rank_key(row) > rank_key(best):
                            best = row
                except Exception as exc:
                    errors.append({"m12_prop": representative.key, "m3_prop": third.key,
                                   "error": f"{type(exc).__name__}: {exc}"})
    first = next(prop for prop in PROPS if prop.key == best["m12_prop"])
    third = next(prop for prop in PROPS if prop.key == best["m3_prop"])
    verified = evaluate(design, first, third)
    verified_row = summarize(verified, first, third)
    for key in ("score", "penalty_total", "m1_lap_time_s", "m2_lap_time_s", "m3_lap_time_s"):
        if not np.isclose(verified_row[key], best[key], rtol=1e-10, atol=1e-10):
            raise AssertionError(f"Combined versus full evaluation mismatch: {key}: {verified_row[key]} != {best[key]}; {first.key}/{third.key}")
    if verified_row["feasible"] != best["feasible"]:
        raise AssertionError("Combined versus full feasibility mismatch")
    best_design = replace(design, prop_diameter_in=first.diameter_in, prop_pitch_in=first.pitch_in,
                          mission3_prop_diameter_in=third.diameter_in, mission3_prop_pitch_in=third.pitch_in)
    return {
        "motor": motor, "status": "evaluation_error" if errors else "complete",
        "best": verified_row, "best_single": best_single,
        "best_by_m12_prop": list(best_by_first.values()), "single_prop_results": common,
        "best_design": asdict(best_design),
        "best_details": {key: (asdict(value) if key in ("M1", "M2", "M3") else value)
                         for key, value in verified.items()},
        "eligible_m12_props": len(eligible), "catalog_props": len(PROPS),
        "considered_pairs": considered_pairs,
        "pruned_m12_pairs": 0 if single_prop else (len(PROPS) - len(eligible)) * (len(PROPS) - 1),
        "full_evaluations": evaluations + 1, "errors": errors,
        "runtime_s": time.perf_counter() - started,
    }


def export_results(output, results, manifest):
    ranked = []
    singles = []
    audit = []
    by_first = []
    for result in results:
        identity = result["motor"]
        if "best" in result:
            ranked.append({**identity, "status": result["status"], **result["best"],
                           "eligible_m12_props": result["eligible_m12_props"],
                           "considered_pairs": result["considered_pairs"],
                           "pruned_m12_pairs": result["pruned_m12_pairs"],
                           "full_evaluations": result["full_evaluations"]})
            singles.append({**identity, **result["best_single"]})
            audit.extend({**identity, **row} for row in result["single_prop_results"])
            by_first.extend({**identity, **row} for row in result["best_by_m12_prop"])
    ranked.sort(key=rank_key, reverse=True)
    singles.sort(key=rank_key, reverse=True)
    for rows in (ranked, singles):
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
    write_csv(output / "ranked_motors.csv", ranked)
    write_csv(output / "ranked_single_prop.csv", singles)
    write_csv(output / "all_single_prop_combinations.csv", audit)
    write_csv(output / "best_m3_for_each_m12.csv", by_first)
    write_csv(output / "errors.csv", [{**r["motor"], **error} for r in results
                                       for error in r.get("errors", [])])
    write_csv(output / "motor_status.csv", [{**r["motor"], "status": r["status"],
                                            "error": r.get("error", ""),
                                            "feasible": r.get("best", {}).get("feasible", False)}
                                           for r in results])
    manifest.update(completed_motors=len(results), feasible_motors=sum(r["feasible"] for r in ranked),
                    errored_motors=sum(r["status"] != "complete" for r in results))
    write_json(output / "manifest.json", manifest)
    lines = ["# Motor and propeller search — 8S 3300 mAh", "",
             f"Completed {len(results)} / {manifest['motor_count']} motor rows; "
             f"{manifest['feasible_motors']} have a feasible combination; "
             f"{manifest['errored_motors']} have input or evaluation errors.", "",
             "Ranked by feasibility, then the sizing model's official mission score minus penalties. "
             "Faster M3, M2 and M1 lap times break ties. Infeasible rows are diagnostics, not recommendations.", "",
             "| Rank | Manufacturer | Motor | Kv | Mass (g) | M1/M2 prop | M3 prop | Score | Feasible |",
             "|---:|---|---|---:|---:|---|---|---:|---|"]
    for row in ranked[:20]:
        lines.append(f"| {row['rank']} | {row['manufacturer']} | {row['motor']} | {row['motor_kv']:g} | "
                     f"{row['motor_mass_kg'] * 1000:g} | {row['m12_prop']} | {row['m3_prop']} | "
                     f"{row['score']:.6f} | {row['feasible']} |")
    lines.extend(["", "See `ranked_motors.csv` for all motors and `ranked_single_prop.csv` for "
                  "the exhaustive one-prop comparison. Per-motor JSON files contain full winning "
                  "designs, mission diagnostics and search coverage.", "",
                  "Assumptions: CSV Watts is treated as the model's electrical input power limit; "
                  "Rm is ohms, Io is amps, and Mass is grams. Battery: 29.6 V nominal, 97.68 Wh, "
                  "85% usable, 25C (82.5 A), 100 A ESC ceiling. CSV does not specify motor-specific "
                  "current, maximum voltage, or thermal-duration limits, so hardware 8S compatibility "
                  "is unverified. Tail dimensions and disabled fuselage/sensor drag follow airframe.json. "
                  "The existing M3 downward tow-load surrogate remains active.", "",
                  "Search covers every two-blade, unique-geometry propeller retained by the repository "
                  "catalog, without a ground-clearance diameter restriction. Existing source-hull, "
                  "100 RPM grid, takeoff, climb, mission energy and RPM/tip-Mach checks are retained. "
                  "Optimum means best feasible catalog combination under this discrete sizing model, "
                  "not a hardware validation. When no feasible pair exists, the retained least-penalized "
                  "diagnostic is not guaranteed optimal among pruned infeasible pairs."])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--airframe", type=Path, default=HERE / "airframe.json")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--single-prop", action="store_true")
    parser.add_argument("--motor-rows", type=int, nargs="+", help="Zero-based CSV rows for a subset/smoke run")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    airframe = json.loads(args.airframe.read_text(encoding="utf-8"))
    motors = read_motors(args.csv)
    if args.motor_rows is not None:
        selected = set(args.motor_rows)
        if not selected.issubset({m['row_index'] for m in motors}):
            parser.error("motor row outside CSV")
        motors = [motor for motor in motors if motor["row_index"] in selected]
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / "motors"
    checkpoints.mkdir(exist_ok=True)
    # Resume only byte-identical inputs AND analysis source files.
    source_paths = sorted((ROOT / "src").rglob("*.py"))
    fingerprint = hashlib.sha256(args.csv.read_bytes() + args.airframe.read_bytes()
                                 + str(args.single_prop).encode()
                                 + json.dumps(args.motor_rows).encode()
                                 + b"".join(path.read_bytes() for path in source_paths)
                                 + (ROOT / "src/prop/data/prop_data.json").read_bytes()).hexdigest()
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise ValueError("Output contains a different run. Choose a new --output directory.")
    manifest = {"fingerprint": fingerprint, "started_utc": datetime.now(timezone.utc).isoformat(),
                "motor_count": len(motors), "single_prop": args.single_prop, "airframe": airframe,
                "csv": str(args.csv), "workers": args.workers,
                "requirements": asdict(DEFAULT_PROPULSION_REQUIREMENTS)}
    write_json(manifest_path, manifest)
    initialize(airframe)
    write_csv(args.output / "propeller_catalog.csv", [dict(key=s.key, diameter_in=s.diameter_in,
                                                         pitch_in=s.pitch_in) for s in PROPS])
    results = []
    pending = []
    for motor in motors:
        path = checkpoints / f"{motor['row_index']:04d}.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append(motor)
    print(f"Motors: {len(motors)}, catalog props: {len(PROPS)}, resumed: {len(results)}", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(airframe,)) as pool:
        futures = {pool.submit(search_motor, motor, args.single_prop): motor for motor in pending}
        for future in as_completed(futures):
            motor = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"motor": motor, "status": "evaluation_error", "error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            write_json(checkpoints / f"{motor['row_index']:04d}.json", result)
            best = result.get("best", {})
            print(f"[{len(results)}/{len(motors)}] {motor['manufacturer']} {motor['motor']}: "
                  f"{result['status']}, feasible={best.get('feasible')}, "
                  f"score={best.get('score')}, elapsed={time.perf_counter() - started:.0f}s", flush=True)
    results.sort(key=lambda result: result["motor"]["row_index"])
    manifest["elapsed_s"] = time.perf_counter() - started
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    export_results(args.output, results, manifest)
    print(f"Wrote {args.output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    cli()
