"""Diagnose coarse stability estimates against full AeroBuildup derivatives."""

import csv
import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import aerosandbox as asb
import matplotlib
import numpy as np
from aerosandbox.dynamics.flight_dynamics.airplane import get_modes

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.aero.aero_score import SPIRAL_DOUBLING_TIME_MIN_S
from src.aero.stability_criteria import time_to_double_s
from src.aero.cruise_analysis import cruise_analysis
from src.aero.stability_analysis_coarse import (
    estimate_stability_derivatives,
    stability_analysis_coarse,
)
from src.aero.utils import require_scalar
from src.mech.main_mech import evaluate_mechanical_module
from src.prop.main_prop import prop_main
from src.prop.prop_database import load_default_prop_database
from src.vectors import ASBDesignVector, DesignVector, ParameterVector


DATASET = "randomized"              # "randomized" or "warmed"
MISSION = 1
MAX_CASES: int | None = None
DATA_DIR = Path("data_dump/accuracy_designs")
OUTPUT_DIR = Path("data_dump/accuracy_results/stability_diagnostics_after_derivative_correction")
DERIVATIVES = ("CLa", "Cma", "Cmq", "CYb", "CYr", "Clb", "Clp", "Clr", "Cnb", "Cnr")
MODES = ("phugoid", "short_period", "dutch_roll", "spiral", "roll_subsidence")
GEOMETRY = ("wing_span", "wing_chord", "tail_arm", "nose_length", "ducks_num", "pucks_num", "banner_length", "batt_capacity")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def mass_properties(properties) -> asb.MassProperties:
    inertia = properties.inertia_tensor_kg_m2
    cg = properties.cg_m
    return asb.MassProperties(
        mass=properties.total_mass_kg, x_cg=cg[0], y_cg=cg[1], z_cg=cg[2],
        Ixx=inertia[0, 0], Iyy=inertia[1, 1], Izz=inertia[2, 2],
        Ixy=inertia[0, 1], Iyz=inertia[1, 2], Ixz=inertia[0, 2],
    )


def gates(cma: float, cnb: float, margin: float, spiral: float) -> dict[str, bool]:
    values = {
        "longitudinal": cma < 0,
        "directional": cnb > 0,
        "static_margin": margin > 0,
        # Spiral is judged on time to double bank angle; the raw eigenvalue
        # is not comparable across models (see src/aero/stability_criteria.py).
        "spiral": time_to_double_s(spiral) >= SPIRAL_DOUBLING_TIME_MIN_S,
    }
    return {**values, "overall": all(values.values())}


def evaluate(values: dict, pv: ParameterVector, database) -> dict:
    design = DesignVector(**values)
    mech = evaluate_mechanical_module(design, parameter_vector=pv)
    design = replace(
        design,
        fuselage_width=mech.resolved_fuselage_width_m,
        fuselage_height=mech.resolved_fuselage_height_m,
    )
    properties = mech.for_mission(f"M{MISSION}")
    props = mass_properties(properties)
    thrust, _ = prop_main(design, pv, mission=MISSION, prop_database=database)

    with redirect_stdout(io.StringIO()):
        cruise = cruise_analysis(design, pv, thrust, properties.cg_m, properties.total_mass_kg, MISSION)
    if not cruise.converged:
        raise RuntimeError("Full cruise did not converge.")

    start = perf_counter()
    coarse_aero = estimate_stability_derivatives(design, cruise, props)
    coarse_result = stability_analysis_coarse(design, cruise, props)
    coarse_time = perf_counter() - start

    airplane = ASBDesignVector.from_design_vector(design).make_airplane(
        elevator_deflection=cruise.elevator_deflection,
        tail_incidence=cruise.tail_incidence,
    )
    start = perf_counter()
    full_aero = asb.AeroBuildup(
        airplane=airplane,
        op_point=cruise.operating_point,
        xyz_ref=properties.cg_m,
        include_wave_drag=False,
    ).run_with_stability_derivatives()
    full_modes = get_modes(airplane, cruise.operating_point, props, full_aero)
    full_time = perf_counter() - start

    full_margin = (require_scalar(full_aero["x_np"]) - properties.cg_m[0]) / design.wing_chord
    row = {
        **{f"design_{key}": value for key, value in values.items()},
        "cruise_velocity": require_scalar(cruise.operating_point.velocity),
        "cruise_alpha": require_scalar(cruise.operating_point.alpha),
        "coarse_runtime_seconds": coarse_time,
        "full_runtime_seconds": full_time,
        "coarse_static_margin": coarse_result.static_margin,
        "full_static_margin": full_margin,
    }
    for derivative in DERIVATIVES:
        row[f"coarse_{derivative}"] = require_scalar(coarse_aero[derivative])
        row[f"full_{derivative}"] = require_scalar(full_aero[derivative])
    for mode in MODES:
        coarse_mode = getattr(coarse_result, mode)
        for field in ("eigenvalue_real", "eigenvalue_imag", "damping_ratio"):
            row[f"coarse_{mode}_{field}"] = float(getattr(coarse_mode, field))
            row[f"full_{mode}_{field}"] = require_scalar(full_modes[mode][field])

    coarse_gates = gates(row["coarse_Cma"], row["coarse_Cnb"], row["coarse_static_margin"], row["coarse_spiral_eigenvalue_real"])
    full_gates = gates(row["full_Cma"], row["full_Cnb"], row["full_static_margin"], row["full_spiral_eigenvalue_real"])
    row.update({f"coarse_gate_{key}": value for key, value in coarse_gates.items()})
    row.update({f"full_gate_{key}": value for key, value in full_gates.items()})
    return row


def confusion(rows: list[dict], gate: str) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=int)
    for row in rows:
        matrix[int(row[f"full_gate_{gate}"]), int(row[f"coarse_gate_{gate}"])] += 1
    return matrix


def correlation(x, y) -> float:
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan


def summarize(rows: list[dict], failures: int) -> list[dict]:
    summary = []
    for derivative in (*DERIVATIVES, "static_margin", "spiral_eigenvalue_real"):
        coarse = np.array([row[f"coarse_{derivative}"] for row in rows])
        full = np.array([row[f"full_{derivative}"] for row in rows])
        error = coarse - full
        summary.append({
            "category": "quantity", "name": derivative, "successful_designs": len(rows), "failed_designs": failures,
            "mean_signed_error": np.mean(error), "mean_absolute_error": np.mean(np.abs(error)),
            "median_absolute_error": np.median(np.abs(error)), "correlation": correlation(coarse, full),
        })
    for gate in ("longitudinal", "directional", "static_margin", "spiral", "overall"):
        matrix = confusion(rows, gate)
        summary.append({
            "category": "gate", "name": gate, "successful_designs": len(rows), "failed_designs": failures,
            "agreement_rate": np.trace(matrix) / matrix.sum(), "false_pass_count": matrix[0, 1],
            "false_rejection_count": matrix[1, 0], "both_fail_count": matrix[0, 0], "both_pass_count": matrix[1, 1],
        })
    summary.append({
        "category": "runtime", "name": "stability_analysis", "successful_designs": len(rows), "failed_designs": failures,
        "coarse_median_seconds": np.median([row["coarse_runtime_seconds"] for row in rows]),
        "full_median_seconds": np.median([row["full_runtime_seconds"] for row in rows]),
        "median_speedup": np.median([row["full_runtime_seconds"] / row["coarse_runtime_seconds"] for row in rows]),
    })
    return summary


def draw_confusions(rows: list[dict]) -> None:
    gates_to_plot = ("longitudinal", "directional", "static_margin", "spiral", "overall")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, gate in zip(axes.flat, gates_to_plot):
        matrix = confusion(rows, gate); ax.imshow(matrix, cmap="Blues")
        for y in range(2):
            for x in range(2): ax.text(x, y, matrix[y, x], ha="center", va="center", fontsize=13)
        ax.set(title=gate, xlabel="Coarse", ylabel="Full", xticks=(0, 1), yticks=(0, 1), xticklabels=("Fail", "Pass"), yticklabels=("Fail", "Pass"))
    axes.flat[-1].axis("off"); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "gate_confusion_matrices.png", dpi=180); plt.close(fig)


def create_plots(rows: list[dict]) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(12, 15))
    for ax, derivative in zip(axes.flat, DERIVATIVES):
        full = np.array([row[f"full_{derivative}"] for row in rows]); coarse = np.array([row[f"coarse_{derivative}"] for row in rows])
        ax.scatter(full, coarse, s=14, alpha=.55); low, high = min(full.min(), coarse.min()), max(full.max(), coarse.max())
        ax.plot([low, high], [low, high], "k--"); ax.set(title=derivative, xlabel="Full", ylabel="Coarse"); ax.grid(True)
    for ax in axes.flat[len(DERIVATIVES):]: ax.axis("off")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "derivative_scatter.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, quantity in zip(axes, ("static_margin", "spiral_eigenvalue_real")):
        full = np.array([row[f"full_{quantity}"] for row in rows]); coarse = np.array([row[f"coarse_{quantity}"] for row in rows])
        ax.scatter(full, coarse, s=16, alpha=.55); low, high = min(full.min(), coarse.min()), max(full.max(), coarse.max())
        ax.plot([low, high], [low, high], "k--"); ax.set(title=quantity, xlabel="Full", ylabel="Coarse"); ax.grid(True)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "scoring_quantities.png", dpi=180); plt.close(fig)

    draw_confusions(rows)
    false_passes = [row for row in rows if row["coarse_gate_overall"] and not row["full_gate_overall"]]
    causes = {gate: sum(not row[f"full_gate_{gate}"] for row in false_passes) for gate in ("longitudinal", "directional", "static_margin", "spiral")}
    fig, ax = plt.subplots(); ax.bar(list(causes), list(causes.values())); ax.set(ylabel="False-pass designs", title="Full-model gates missed by coarse stability"); ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "false_pass_causes.png", dpi=180); plt.close(fig)

    errors = np.array([[abs(row[f"coarse_{d}"] - row[f"full_{d}"]) for g in GEOMETRY] for row in rows for d in DERIVATIVES]).reshape(len(rows), len(DERIVATIVES), len(GEOMETRY))
    correlations = np.empty((len(DERIVATIVES), len(GEOMETRY)))
    for i, derivative in enumerate(DERIVATIVES):
        for j, geometry in enumerate(GEOMETRY):
            correlations[i, j] = correlation([row[f"design_{geometry}"] for row in rows], errors[:, i, j])
    fig, ax = plt.subplots(figsize=(10, 7)); image = ax.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set(xticks=range(len(GEOMETRY)), xticklabels=GEOMETRY, yticks=range(len(DERIVATIVES)), yticklabels=DERIVATIVES, title="Correlation of geometry with absolute derivative error"); ax.tick_params(axis="x", rotation=40); fig.colorbar(image, ax=ax)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "error_geometry_correlations.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, mode in zip(axes.flat, MODES):
        full = np.array([row[f"full_{mode}_eigenvalue_real"] for row in rows]); coarse = np.array([row[f"coarse_{mode}_eigenvalue_real"] for row in rows])
        ax.scatter(full, coarse, s=14, alpha=.55); low, high = min(full.min(), coarse.min()), max(full.max(), coarse.max())
        ax.plot([low, high], [low, high], "k--"); ax.set(title=mode, xlabel="Full real eigenvalue", ylabel="Coarse"); ax.grid(True)
    axes.flat[-1].axis("off"); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "mode_eigenvalues.png", dpi=180); plt.close(fig)

    coarse = [row["coarse_runtime_seconds"] for row in rows]; full = [row["full_runtime_seconds"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4)); axes[0].boxplot([coarse, full], tick_labels=("Coarse", "Full")); axes[0].set(ylabel="Seconds", title="Stability runtime"); axes[0].grid(True)
    axes[1].hist(np.array(full) / np.array(coarse), bins=25); axes[1].set(xlabel="Full/coarse speedup", ylabel="Count", title="Speedup"); axes[1].grid(True)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "runtime_comparison.png", dpi=180); plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    designs = json.loads((DATA_DIR / f"{DATASET}_design_vectors.json").read_text(encoding="utf-8"))
    if MAX_CASES is not None: designs = designs[:MAX_CASES]
    pv, database = ParameterVector(), load_default_prop_database()
    baseline = DesignVector(); database.thrust(baseline.prop_diameter_in, baseline.prop_pitch_in, 0, 8000); database.torque(baseline.prop_diameter_in, baseline.prop_pitch_in, 0, 8000)

    rows, failures = [], []
    for index, values in enumerate(designs):
        try:
            row = {"case_id": index, **evaluate(values, pv, database), "status": "ok", "error": ""}; rows.append(row)
        except Exception as exc:
            failures.append({"case_id": index, **{f"design_{key}": value for key, value in values.items()}, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
        print(f"[{index + 1}/{len(designs)}] {'ok' if len(rows) + len(failures) == index + 1 and (not failures or failures[-1]['case_id'] != index) else 'error'}", flush=True)

    write_csv(OUTPUT_DIR / "raw_results.csv", [*rows, *failures])
    write_csv(OUTPUT_DIR / "summary.csv", summarize(rows, len(failures)))
    false_passes = [row for row in rows if row["coarse_gate_overall"] and not row["full_gate_overall"]]
    false_passes.sort(key=lambda row: sum(not row[f"full_gate_{gate}"] for gate in ("longitudinal", "directional", "static_margin", "spiral")), reverse=True)
    write_csv(OUTPUT_DIR / "false_passes.csv", false_passes)
    create_plots(rows)
    print(f"Saved stability diagnostics to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
