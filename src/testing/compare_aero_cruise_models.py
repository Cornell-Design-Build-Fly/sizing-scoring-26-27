"""Compare coarse and full aero_main trim outputs."""

import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.aero.main_aero as aero_module
from src.testing.coarse_accuracy_batch import _constant_aero_inputs
from src.vectors import DesignVector, ParameterVector


DATASET = Path("data_dump/accuracy_designs/randomized_design_vectors.json")
OUTPUT_DIR = Path("data_dump/accuracy_results/aero_cruise_coarse_full")
MISSION = 1
MAX_CASES: int | None = None
QUANTITIES = ("velocity", "alpha", "elevator_deflection")
COARSE_CRUISE, FULL_CRUISE = aero_module.cruise_analysis_coarse, aero_module.cruise_analysis
COARSE_STABILITY, FULL_STABILITY = aero_module.stability_analysis_coarse, aero_module.stability_analysis


def run(design, pv, inputs, model):
    cruise = COARSE_CRUISE if model == "coarse" else FULL_CRUISE
    stability = COARSE_STABILITY if model == "coarse" else FULL_STABILITY
    captured = []

    def capture(*args, **kwargs):
        result = cruise(*args, **kwargs); captured.append(result); return result

    aero_module.cruise_analysis_coarse = capture
    aero_module.stability_analysis_coarse = stability
    start = perf_counter()
    try:
        with redirect_stdout(io.StringIO()):
            aero_module.aero_main(
                design, pv, tuple(inputs["thrust_velocity"]), tuple(inputs["flight_time_fit"]), MISSION,
                tuple(inputs["cg"]), np.asarray(inputs["inertia_matrix"]),
                float(inputs["mass"]), False, debug=False,
            )
    finally:
        aero_module.cruise_analysis_coarse = COARSE_CRUISE
        aero_module.stability_analysis_coarse = COARSE_STABILITY
    if not captured or not captured[0].converged:
        raise RuntimeError("Cruise trim did not converge.")
    condition = captured[0]
    return {
        "velocity": float(condition.operating_point.velocity),
        "alpha": float(condition.operating_point.alpha),
        "elevator_deflection": float(condition.elevator_deflection),
        "elapsed_seconds": perf_counter() - start,
    }


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    values = json.loads(DATASET.read_text(encoding="utf-8"))
    if MAX_CASES is not None: values = values[:MAX_CASES]
    pv = ParameterVector(); inputs = _constant_aero_inputs(pv); rows = []
    for index, design_values in enumerate(values):
        row = {"case_id": index, **{f"design_{k}": v for k, v in design_values.items()}}
        for model in ("coarse", "full"):
            try:
                result = run(DesignVector(**design_values), pv, inputs, model)
                row.update({f"{model}_status": "ok", **{f"{model}_{k}": v for k, v in result.items()}})
            except Exception as exc:
                row.update({f"{model}_status": "error", f"{model}_error": f"{type(exc).__name__}: {exc}"})
        rows.append(row); print(f"[{index + 1}/{len(values)}] coarse={row['coarse_status']} full={row['full_status']}", flush=True)

    paired = [row for row in rows if row["coarse_status"] == row["full_status"] == "ok"]
    summary = []
    for quantity in QUANTITIES:
        coarse = np.array([row[f"coarse_{quantity}"] for row in paired]); full = np.array([row[f"full_{quantity}"] for row in paired]); error = coarse - full
        summary.append({"quantity": quantity, "paired_count": len(paired), "mae": np.mean(abs(error)),
                        "median_absolute_error": np.median(abs(error)), "bias": np.mean(error),
                        "correlation": np.corrcoef(coarse, full)[0, 1]})
    write_csv(OUTPUT_DIR / "raw_results.csv", rows); write_csv(OUTPUT_DIR / "summary.csv", summary)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, quantity in zip(axes, QUANTITIES):
        coarse = np.array([row[f"coarse_{quantity}"] for row in paired]); full = np.array([row[f"full_{quantity}"] for row in paired])
        ax.scatter(full, coarse, s=15, alpha=.55); low, high = min(full.min(), coarse.min()), max(full.max(), coarse.max())
        ax.plot([low, high], [low, high], "k--"); ax.set(title=quantity, xlabel="Full", ylabel="Coarse"); ax.grid(True)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "trim_comparison.png", dpi=180); plt.close(fig)
    print(f"Saved cruise comparison to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
