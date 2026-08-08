"""Compare coarse and AeroBuildup drag on randomized designs."""

import csv
import json
import argparse
from pathlib import Path

import aerosandbox as asb
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.aero.cruise_analysis import cruise_analysis
from src.aero.cruise_analysis_coarse import cruise_analysis_coarse
from src.aero.drag_model import drag_coefficients
from src.vectors import ASBDesignVector, DesignVector, ParameterVector


DATA_DIR = Path("data_dump/accuracy_designs")
INPUTS = Path("data_dump/accuracy_designs/aero_main_constant_inputs_m1.json")
RESULTS_DIR = Path("data_dump/accuracy_results")
MISSION = 1


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _coarse_drag(design, velocity, alpha, elevator, rho):
    parameters = ParameterVector()
    s, ar = design.wing_area, design.wing_span**2 / design.wing_area
    st = design.hstab_area
    art = design.hstab_span**2 / st
    aw, at = 2 * np.pi / (1 + 2 / ar), 2 * np.pi / (1 + 2 / art)
    wing_cl = 0.064247 + aw * np.radians(alpha + 2) * (0.910703 + 0.443763 / ar)
    tail_cl = at * np.radians(0.929043 * alpha + 0.815490 * elevator)
    cds = drag_coefficients(design, parameters, velocity, wing_cl, tail_cl)
    q = 0.5 * rho * velocity**2
    return {"CD": sum(cds.values()), "D": q * s * sum(cds.values()), **{f"CD_{k}": v for k, v in cds.items()}, **{f"D_{k}": q * s * v for k, v in cds.items()}}


def _full_drag(design, velocity, alpha, elevator, cg):
    airplane = ASBDesignVector.from_design_vector(design).make_airplane(elevator_deflection=elevator)
    aero = asb.AeroBuildup(
        airplane, asb.OperatingPoint(velocity=velocity, alpha=alpha),
        xyz_ref=np.asarray(cg), include_wave_drag=False,
    ).run()
    wings, bodies = aero["wing_aero_components"], aero["fuselage_aero_components"]
    wing_d = _scalar(wings[0].D)
    tail_d = sum(_scalar(wing.D) for wing in wings[1:])
    body_d = sum(_scalar(body.D) for body in bodies)
    return {
        "CD": _scalar(aero["CD"]), "D": _scalar(aero["D"]),
        "D_profile": _scalar(aero["D_profile"]), "D_induced": _scalar(aero["D_induced"]),
        "D_wing": wing_d, "D_tail": tail_d, "D_body": body_d,
    }


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main(dataset="randomized"):
    dataset_path = DATA_DIR / f"{dataset}_design_vectors.json"
    output_dir = RESULTS_DIR / f"drag_coarse_full_{dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)
    designs = json.loads(dataset_path.read_text(encoding="utf-8"))
    inputs = json.loads(INPUTS.read_text(encoding="utf-8"))
    pv, rows = ParameterVector(), []
    common = (pv, tuple(inputs["thrust_velocity"]), tuple(inputs["cg"]), float(inputs["mass"]), MISSION, False)
    for index, values in enumerate(designs):
        design = DesignVector(**values)
        row = {"case_id": index, **{f"design_{k}": v for k, v in values.items()}}
        try:
            coarse_trim = cruise_analysis_coarse(design, *common)
            full_trim = cruise_analysis(design, *common)
            if not coarse_trim.converged or not full_trim.converged:
                raise RuntimeError("trim did not converge")
            velocity = float(full_trim.operating_point.velocity)
            alpha = float(full_trim.operating_point.alpha)
            elevator = float(full_trim.elevator_deflection)
            coarse = _coarse_drag(design, velocity, alpha, elevator, pv.rho)
            full = _full_drag(design, velocity, alpha, elevator, inputs["cg"])
            row.update(status="ok", full_trim_velocity=velocity, full_trim_alpha=alpha,
                       full_trim_elevator=elevator,
                       coarse_trim_velocity=float(coarse_trim.operating_point.velocity),
                       coarse_trim_alpha=float(coarse_trim.operating_point.alpha),
                       coarse_trim_elevator=float(coarse_trim.elevator_deflection),
                       **{f"coarse_{k}": v for k, v in coarse.items()},
                       **{f"full_{k}": v for k, v in full.items()})
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        print(f"[{index + 1}/{len(designs)}] {row['status']}", flush=True)

    valid = [row for row in rows if row["status"] == "ok"]
    summary = []
    for quantity in ("D", "CD"):
        full = np.array([r[f"full_{quantity}"] for r in valid])
        coarse = np.array([r[f"coarse_{quantity}"] for r in valid])
        error, relative = coarse - full, (coarse - full) / full
        summary.append({"quantity": quantity, "count": len(valid), "full_mean": full.mean(),
                        "coarse_mean": coarse.mean(), "bias": error.mean(), "mae": abs(error).mean(),
                        "mean_relative_error_percent": 100 * relative.mean(),
                        "mean_absolute_percentage_error": 100 * abs(relative).mean(),
                        "p95_absolute_percentage_error": 100 * np.percentile(abs(relative), 95),
                        "correlation": np.corrcoef(full, coarse)[0, 1]})
    _write_csv(output_dir / "raw_results.csv", rows)
    _write_csv(output_dir / "summary.csv", summary)

    component_specs = (
        ("profile", "full_D_profile", ("coarse_D_wing_profile", "coarse_D_tail_profile", "coarse_D_body")),
        ("induced", "full_D_induced", ("coarse_D_wing_induced", "coarse_D_tail_induced", "coarse_D_interaction")),
        ("wing_profile", "full_D_wing", ("coarse_D_wing_profile",)),
        ("tail_profile", "full_D_tail", ("coarse_D_tail_profile",)),
        ("body_profile", "full_D_body", ("coarse_D_body",)),
    )
    component_summary = []
    for name, full_key, coarse_keys in component_specs:
        full = np.array([r[full_key] for r in valid])
        coarse = np.array([sum(r[key] for key in coarse_keys) for r in valid])
        error = coarse - full
        component_summary.append({"component": name, "full_mean_D": full.mean(), "coarse_mean_D": coarse.mean(),
                                  "bias_D": error.mean(), "mae_D": abs(error).mean(),
                                  "mean_absolute_percentage_error": 100 * np.mean(abs(error / full)),
                                  "correlation": np.corrcoef(full, coarse)[0, 1]})
    _write_csv(output_dir / "component_summary.csv", component_summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, quantity in zip(axes, ("D", "CD")):
        x = np.array([r[f"full_{quantity}"] for r in valid]); y = np.array([r[f"coarse_{quantity}"] for r in valid])
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        ax.scatter(x, y, s=18, alpha=.55); ax.plot([low, high], [low, high], "k--")
        ax.set(xlabel=f"Full {quantity}", ylabel=f"Coarse {quantity}", title=f"{quantity} at full trim"); ax.grid(True)
    fig.tight_layout(); fig.savefig(output_dir / "total_drag_parity.png", dpi=180); plt.close(fig)

    relative = 100 * np.array([(r["coarse_D"] - r["full_D"]) / r["full_D"] for r in valid])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(relative, bins=24, edgecolor="black"); axes[0].axvline(0, color="k", ls="--")
    axes[0].set(xlabel="Coarse drag error (%)", ylabel="Designs", title="Total-drag error")
    axes[1].scatter([r["full_trim_velocity"] for r in valid], relative, s=18, alpha=.55)
    axes[1].axhline(0, color="k", ls="--"); axes[1].set(xlabel="Full trim velocity (m/s)", ylabel="Drag error (%)", title="Error vs velocity")
    for ax in axes: ax.grid(True)
    fig.tight_layout(); fig.savefig(output_dir / "drag_error.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (name, full_key, coarse_keys) in zip(axes.flat, component_specs):
        x = np.array([r[full_key] for r in valid])
        y = np.array([sum(r[key] for key in coarse_keys) for r in valid])
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        ax.scatter(x, y, s=18, alpha=.55); ax.plot([low, high], [low, high], "k--")
        ax.set(xlabel="Full drag (N)", ylabel="Coarse drag (N)", title=name.replace("_", " ").title()); ax.grid(True)
    axes.flat[-1].axis("off")
    fig.tight_layout(); fig.savefig(output_dir / "component_drag_parity.png", dpi=180); plt.close(fig)

    features = {"Velocity": "full_trim_velocity", "Alpha": "full_trim_alpha", "Tail arm": "design_tail_arm",
                "Wing span": "design_wing_span", "Wing chord": "design_wing_chord", "Elevator": "full_trim_elevator"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (label, key) in zip(axes.flat, features.items()):
        values = np.array([r[key] for r in valid])
        ax.scatter(values, relative, s=18, alpha=.55); ax.axhline(0, color="k", ls="--")
        ax.set(xlabel=label, ylabel="Drag error (%)"); ax.grid(True)
    fig.tight_layout(); fig.savefig(output_dir / "drag_error_sensitivity.png", dpi=180); plt.close(fig)
    print(f"Saved {len(valid)} valid comparisons to {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("randomized", "warmed"), default="randomized")
    main(parser.parse_args().dataset)
