"""Benchmark mech, prop, and aero across the randomized design set."""

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import src.main as main_module
from src.prop.continuous_prop_database import load_default_continuous_prop_database
from src.vectors import DesignVector, ParameterVector


DATASET = Path("data_dump/accuracy_designs/randomized_design_vectors.json")
OUTPUT_DIR = Path("data_dump/accuracy_results/module_runtimes")
MAX_CASES: int | None = None


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    designs = json.loads(DATASET.read_text(encoding="utf-8"))
    if MAX_CASES is not None: designs = designs[:MAX_CASES]
    pv = ParameterVector()
    load_start = perf_counter(); database = load_default_continuous_prop_database(); database_load = perf_counter() - load_start

    original_mech, original_prop, original_aero = main_module.evaluate_mechanical_module, main_module.prop_main, main_module.aero_main
    current: dict[str, float] = {}

    def timed_mech(*args, **kwargs):
        start = perf_counter()
        try: return original_mech(*args, **kwargs)
        finally: current["mech_seconds"] = current.get("mech_seconds", 0.0) + perf_counter() - start

    def timed_prop(*args, **kwargs):
        mission = kwargs.get("mission", args[2] if len(args) > 2 else 0); start = perf_counter()
        try: return original_prop(*args, **kwargs)
        finally: current[f"prop_m{mission}_seconds"] = perf_counter() - start

    def timed_aero(*args, **kwargs):
        mission = kwargs.get("mission", args[3] if len(args) > 3 else 0); start = perf_counter()
        try: return original_aero(*args, **kwargs)
        finally: current[f"aero_m{mission}_seconds"] = perf_counter() - start

    main_module.evaluate_mechanical_module, main_module.prop_main, main_module.aero_main = timed_mech, timed_prop, timed_aero
    # Exclude one-time solver/library initialization from repeated-design timing.
    try: main_module.main(DesignVector(), pv, disp_res=False, prop_database=database)
    except Exception: pass

    rows = []
    for index, values in enumerate(designs):
        current = {}; row = {"case_id": index, **{f"design_{key}": value for key, value in values.items()}}
        start = perf_counter()
        try:
            main_module.main(DesignVector(**values), pv, disp_res=False, prop_database=database)
            row["status"], row["error"] = "ok", ""
        except Exception as exc:
            row["status"], row["error"] = "error", f"{type(exc).__name__}: {exc}"
        row["main_seconds"] = perf_counter() - start; row.update(current)
        row["prop_total_seconds"] = sum(current.get(f"prop_m{mission}_seconds", 0.0) for mission in (1, 2, 3))
        row["aero_total_seconds"] = sum(current.get(f"aero_m{mission}_seconds", 0.0) for mission in (1, 2, 3))
        row["module_total_seconds"] = row.get("mech_seconds", 0.0) + row["prop_total_seconds"] + row["aero_total_seconds"]
        row["other_seconds"] = row["main_seconds"] - row["module_total_seconds"]
        rows.append(row); print(f"[{index + 1}/{len(designs)}] {row['status']} ({row['main_seconds']:.4f} s)", flush=True)

    write_csv(OUTPUT_DIR / "per_design_runtimes.csv", rows)
    successful = [row for row in rows if row["status"] == "ok"]
    fields = ("mech_seconds", "prop_m1_seconds", "prop_m2_seconds", "prop_m3_seconds", "prop_total_seconds",
              "aero_m1_seconds", "aero_m2_seconds", "aero_m3_seconds", "aero_total_seconds",
              "other_seconds", "main_seconds")
    summary = []
    for field in fields:
        values = np.array([row[field] for row in successful if field in row], dtype=float)
        summary.append({"calculation": field.removesuffix("_seconds"), "successful_designs": len(values),
                        "mean_seconds": np.mean(values), "median_seconds": np.median(values),
                        "p95_seconds": np.percentile(values, 95), "min_seconds": np.min(values), "max_seconds": np.max(values)})
    summary.append({"calculation": "prop_database_load_once", "successful_designs": 1, "mean_seconds": database_load,
                    "median_seconds": database_load, "p95_seconds": database_load, "min_seconds": database_load, "max_seconds": database_load})
    write_csv(OUTPUT_DIR / "summary.csv", summary)
    print(f"Saved module runtimes to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
