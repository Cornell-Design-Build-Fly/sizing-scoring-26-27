"""Compare coarse and full models on the randomized design set."""

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
from src.main import main as run_main
from src.prop.prop_database import load_default_prop_database
from src.vectors import DesignVector, ParameterVector


DATASET = Path("data_dump/accuracy_designs/randomized_design_vectors.json")
OUTPUT_DIR = Path("data_dump/accuracy_results/randomized_coarse_full")
MAX_CASES: int | None = None
MISSIONS = ("M1", "M2", "M3")
PENALTIES = (
    "penalty_static_margin", "penalty_longitudinal",
    "penalty_directional", "penalty_spiral",
)

COARSE_CRUISE = aero_module.cruise_analysis_coarse
FULL_CRUISE = aero_module.cruise_analysis
COARSE_STABILITY = aero_module.stability_analysis_coarse
FULL_STABILITY = aero_module.stability_analysis


def select_model(model: str) -> None:
    aero_module.cruise_analysis_coarse = COARSE_CRUISE if model == "coarse" else FULL_CRUISE
    aero_module.stability_analysis_coarse = COARSE_STABILITY if model == "coarse" else FULL_STABILITY


def evaluate(design: DesignVector, pv: ParameterVector, database, model: str) -> tuple[dict, float]:
    select_model(model)
    start = perf_counter()
    with redirect_stdout(io.StringIO()):
        score, breakdown, missions = run_main(
            design, pv, disp_res=False, prop_database=database, return_details=True
        )
    return {
        "total_score": float(score),
        "breakdown": [float(value) for value in breakdown],
        "missions": missions,
    }, perf_counter() - start


def number(value) -> float:
    return np.nan if value is None else float(value)


def flatten(prefix: str, result: dict, elapsed: float) -> dict:
    row = {f"{prefix}_elapsed_seconds": elapsed, f"{prefix}_total_score": result["total_score"]}
    row.update({f"{prefix}_breakdown_{i}": value for i, value in enumerate(result["breakdown"])})
    for mission, score in result["missions"].items():
        row.update({
            f"{prefix}_{mission}_can_fly": bool(score.can_fly),
            f"{prefix}_{mission}_lap_time": number(score.lap_time),
            f"{prefix}_{mission}_penalty": number(score.penalty),
            **{f"{prefix}_{mission}_{name}": number(getattr(score, name)) for name in PENALTIES},
        })
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def observations(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        if row["coarse_status"] != "ok" or row["full_status"] != "ok":
            continue
        for mission in MISSIONS:
            result.append({
                **row,
                "mission": mission,
                "coarse_lap": row[f"coarse_{mission}_lap_time"],
                "full_lap": row[f"full_{mission}_lap_time"],
                "coarse_fly": row[f"coarse_{mission}_can_fly"],
                "full_fly": row[f"full_{mission}_can_fly"],
                "coarse_penalty": row[f"coarse_{mission}_penalty"],
                "full_penalty": row[f"full_{mission}_penalty"],
            })
    return result


def summary_rows(rows: list[dict], obs: list[dict]) -> list[dict]:
    summaries = []
    for label in (*MISSIONS, "ALL"):
        subset = obs if label == "ALL" else [item for item in obs if item["mission"] == label]
        valid = [item for item in subset if item["coarse_lap"] < 1e6 and item["full_lap"] < 1e6]
        errors = np.array([item["coarse_lap"] - item["full_lap"] for item in valid], dtype=float)
        percentages = np.array([100 * error / item["full_lap"] for error, item in zip(errors, valid)], dtype=float)
        coarse_times = np.array([row["coarse_elapsed_seconds"] for row in rows if row["coarse_status"] == "ok"])
        full_times = np.array([row["full_elapsed_seconds"] for row in rows if row["full_status"] == "ok"])
        summaries.append({
            "mission": label,
            "design_count": len(rows),
            "paired_success_count": len(subset),
            "coarse_exception_count": sum(row["coarse_status"] != "ok" for row in rows),
            "full_exception_count": sum(row["full_status"] != "ok" for row in rows),
            "valid_lap_pair_count": len(valid),
            "valid_lap_agreement_rate": np.mean([(x["coarse_lap"] < 1e6) == (x["full_lap"] < 1e6) for x in subset]) if subset else np.nan,
            "coarse_valid_full_invalid_count": sum(x["coarse_lap"] < 1e6 <= x["full_lap"] for x in subset),
            "coarse_invalid_full_valid_count": sum(x["full_lap"] < 1e6 <= x["coarse_lap"] for x in subset),
            "flyability_agreement_rate": np.mean([x["coarse_fly"] == x["full_fly"] for x in subset]) if subset else np.nan,
            "flyability_false_pass_count": sum(x["coarse_fly"] and not x["full_fly"] for x in subset),
            "flyability_false_rejection_count": sum(not x["coarse_fly"] and x["full_fly"] for x in subset),
            "mean_signed_lap_error_seconds": np.mean(errors) if len(errors) else np.nan,
            "mean_absolute_lap_error_seconds": np.mean(np.abs(errors)) if len(errors) else np.nan,
            "median_percentage_lap_error": np.median(percentages) if len(percentages) else np.nan,
            "mean_absolute_percentage_lap_error": np.mean(np.abs(percentages)) if len(percentages) else np.nan,
            "p90_absolute_percentage_lap_error": np.percentile(np.abs(percentages), 90) if len(percentages) else np.nan,
            "p95_absolute_percentage_lap_error": np.percentile(np.abs(percentages), 95) if len(percentages) else np.nan,
            "coarse_median_runtime_seconds": np.median(coarse_times) if len(coarse_times) else np.nan,
            "full_median_runtime_seconds": np.median(full_times) if len(full_times) else np.nan,
            "median_speedup": np.median(full_times / coarse_times) if len(coarse_times) == len(full_times) and len(full_times) else np.nan,
        })
    return summaries


def confusion(items: list[dict], coarse_key: str, full_key: str) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=int)
    for item in items:
        matrix[int(bool(item[full_key])), int(bool(item[coarse_key]))] += 1
    return matrix


def draw_matrix(matrix: np.ndarray, title: str, labels: tuple[str, str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, matrix[row, column], ha="center", va="center", fontsize=14)
    ax.set(xticks=(0, 1), yticks=(0, 1), xticklabels=labels, yticklabels=labels,
           xlabel="Coarse", ylabel="Full", title=title)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def create_plots(rows: list[dict], obs: list[dict]) -> None:
    colors = {"M1": "tab:blue", "M2": "tab:orange", "M3": "tab:green"}
    valid = [x for x in obs if x["coarse_lap"] < 1e6 and x["full_lap"] < 1e6]

    fig, ax = plt.subplots()
    for mission in MISSIONS:
        points = [x for x in valid if x["mission"] == mission]
        ax.scatter([x["full_lap"] for x in points], [x["coarse_lap"] for x in points], s=18, alpha=.65, label=mission)
    limits = ax.get_xlim(); low, high = min(limits[0], ax.get_ylim()[0]), max(limits[1], ax.get_ylim()[1])
    ax.plot([low, high], [low, high], "k--", label="Perfect agreement")
    ax.set(xlabel="Full lap time [s]", ylabel="Coarse lap time [s]", title="Coarse versus full lap time"); ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "lap_time_scatter.png", dpi=180); plt.close(fig)

    relative = np.array([100 * (x["coarse_lap"] - x["full_lap"]) / x["full_lap"] for x in valid])
    fig, ax = plt.subplots(); ax.hist(relative, bins=30)
    ax.axvline(np.median(relative), color="k", linestyle="--", label=f"Median {np.median(relative):.1f}%")
    ax.set(xlabel="Relative lap-time error [%]", ylabel="Count", title="Lap-time error distribution"); ax.legend(); ax.grid(True)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "lap_time_error_histogram.png", dpi=180); plt.close(fig)

    draw_matrix(confusion(obs, "coarse_fly", "full_fly"), "Flyability classification", ("Not flyable", "Flyable"), OUTPUT_DIR / "flyability_confusion.png")
    invalid_obs = [{**x, "coarse_invalid": x["coarse_lap"] >= 1e6, "full_invalid": x["full_lap"] >= 1e6} for x in obs]
    draw_matrix(confusion(invalid_obs, "coarse_invalid", "full_invalid"), "Invalid-lap classification", ("Valid lap", "Invalid lap"), OUTPUT_DIR / "invalid_lap_confusion.png")

    fig, ax = plt.subplots(); ax.scatter([x["full_penalty"] for x in obs], [x["coarse_penalty"] for x in obs], s=16, alpha=.6)
    ax.plot([0, 10], [0, 10], "k--"); ax.set(xlabel="Full penalty", ylabel="Coarse penalty", title="Total penalty comparison"); ax.grid(True)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "penalty_comparison.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, penalty in zip(axes.flat, PENALTIES):
        full = [x[f"full_{x['mission']}_{penalty}"] for x in obs]
        coarse = [x[f"coarse_{x['mission']}_{penalty}"] for x in obs]
        ax.scatter(full, coarse, s=12, alpha=.5); ax.plot([0, 10], [0, 10], "k--"); ax.set(title=penalty, xlabel="Full", ylabel="Coarse"); ax.grid(True)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "penalty_components.png", dpi=180); plt.close(fig)

    geometry = ("wing_span", "wing_chord", "tail_arm", "nose_length", "ducks_num", "pucks_num", "banner_length", "batt_capacity")
    absolute_error = np.abs(relative)
    fig, axes = plt.subplots(4, 2, figsize=(11, 14))
    for ax, name in zip(axes.flat, geometry):
        ax.scatter([x[f"design_{name}"] for x in valid], absolute_error, c=[colors[x["mission"]] for x in valid], s=12, alpha=.5)
        ax.set(xlabel=name, ylabel="Absolute lap error [%]"); ax.grid(True)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "error_vs_geometry.png", dpi=180); plt.close(fig)

    correlations = [np.corrcoef([x[f"design_{name}"] for x in valid], absolute_error)[0, 1] for name in geometry]
    fig, ax = plt.subplots(figsize=(9, 4)); ax.bar(geometry, correlations); ax.axhline(0, color="k", linewidth=.8)
    ax.set(ylabel="Correlation with absolute error", title="Geometry sensitivity of coarse-model error"); ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "error_correlations.png", dpi=180); plt.close(fig)

    coarse_time = np.array([x["coarse_elapsed_seconds"] for x in rows if x["coarse_status"] == "ok"])
    full_time = np.array([x["full_elapsed_seconds"] for x in rows if x["full_status"] == "ok"])
    speedup = full_time / coarse_time
    fig, axes = plt.subplots(1, 2, figsize=(10, 4)); axes[0].boxplot([coarse_time, full_time], tick_labels=("Coarse", "Full")); axes[0].set(ylabel="Runtime [s]", title="Runtime distributions"); axes[0].grid(True)
    axes[1].hist(speedup, bins=25); axes[1].set(xlabel="Full/coarse speedup", ylabel="Count", title="Speedup distribution"); axes[1].grid(True)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "runtime_comparison.png", dpi=180); plt.close(fig)

    speedup_by_case = {row["case_id"]: row["full_elapsed_seconds"] / row["coarse_elapsed_seconds"] for row in rows if row["coarse_status"] == row["full_status"] == "ok"}
    fig, ax = plt.subplots(); ax.scatter([speedup_by_case[x["case_id"]] for x in valid], absolute_error, s=14, alpha=.55)
    ax.set(xlabel="Full/coarse speedup", ylabel="Absolute lap error [%]", title="Accuracy versus speedup"); ax.grid(True)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "error_vs_speedup.png", dpi=180); plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    values = json.loads(DATASET.read_text(encoding="utf-8"))
    if MAX_CASES is not None:
        values = values[:MAX_CASES]
    pv, database = ParameterVector(), load_default_prop_database()

    print("Warming propulsion interpolators and both aero models...", flush=True)
    baseline = DesignVector()
    database.thrust(baseline.prop_diameter_in, baseline.prop_pitch_in, 0.0, 8000.0)
    database.torque(baseline.prop_diameter_in, baseline.prop_pitch_in, 0.0, 8000.0)
    for model in ("coarse", "full"):
        try: evaluate(baseline, pv, database, model)
        except Exception: pass

    rows = []
    for index, design_values in enumerate(values):
        row = {"case_id": index, **{f"design_{key}": value for key, value in design_values.items()}}
        design = DesignVector(**design_values)
        for model in ("coarse", "full"):
            try:
                result, elapsed = evaluate(design, pv, database, model)
                row.update({f"{model}_status": "ok", f"{model}_error": "", **flatten(model, result, elapsed)})
            except Exception as exc:
                row.update({f"{model}_status": "error", f"{model}_error": f"{type(exc).__name__}: {exc}", f"{model}_elapsed_seconds": np.nan})
        rows.append(row)
        print(f"[{index + 1}/{len(values)}] coarse={row['coarse_status']} full={row['full_status']}", flush=True)

    select_model("coarse")
    write_csv(OUTPUT_DIR / "raw_results.csv", rows)
    obs = observations(rows)
    summaries = summary_rows(rows, obs)
    write_csv(OUTPUT_DIR / "summary.csv", summaries)
    create_plots(rows, obs)
    print(f"Results and plots saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
