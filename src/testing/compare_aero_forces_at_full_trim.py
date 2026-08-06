"""Compare coarse and full aerodynamics at the full-model trim point."""

import csv
from pathlib import Path

import aerosandbox as asb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.testing.coarse_accuracy_batch import _constant_aero_inputs
from src.vectors import ASBDesignVector, DesignVector, ParameterVector


INPUT = Path("data_dump/accuracy_results/aero_cruise_coarse_full/raw_results.csv")
OUTPUT_DIR = Path("data_dump/accuracy_results/aero_force_diagnostics")
QUANTITIES = ("CL", "CD", "Cm", "L", "D", "m_b")
GEOMETRY = ("wing_span", "wing_chord", "tail_arm", "nose_length", "ducks_num", "pucks_num", "banner_length", "batt_capacity")


def scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def coarse(design, alpha, elevator, velocity, cg, rho):
    s, c, b = design.wing_area, design.wing_chord, design.wing_span
    st, sv = design.hstab_area, design.vstab_area
    ar, art = b**2 / s, design.hstab_span**2 / st
    aw, at = 2 * np.pi / (1 + 2 / ar), 2 * np.pi / (1 + 2 / art)
    wing_cl_est = aw * np.radians(alpha + 2.0)
    wing_cl = 0.064247 + wing_cl_est * (0.910703 + 0.443763 / ar)
    tail_cl = at * np.radians(0.929043 * alpha + 0.815490 * elevator)
    ratio = st / s
    wing_cm_ac = -0.047069 - 0.021599 * np.radians(alpha) + 0.067132 / ar - 0.044650 * c
    wing_cm = wing_cm_ac + wing_cl * (cg[0] - 0.25 * c) / c
    tail_cm = -ratio * tail_cl * (design.tail_arm + 0.25 * design.hstab_chord - cg[0]) / c
    fuselage_length = design.nose_length + design.tail_arm + max(design.hstab_chord, design.vstab_chord)
    body_cm = design.fuselage_height * fuselage_length**2 / (s * c) * (
        0.002201 + 0.059479 * np.radians(alpha) + 0.000757 * design.nose_length / fuselage_length
        - 0.057975 * cg[0] / fuselage_length
    )
    reynolds = rho * velocity * c / 1.81e-5
    wing_cd = 0.001870 + 3.66232 / np.sqrt(reynolds) + wing_cl**2 / (np.pi * ar)
    tail_cd = 0.68 * (0.012 * (st + sv) / s + ratio * tail_cl**2 / (np.pi * 0.80 * art))
    body_cd = 0.126 * design.fuselage_width * design.fuselage_height / s
    cl, cd, cm = wing_cl + ratio * tail_cl, wing_cd + tail_cd + body_cd, wing_cm + tail_cm + body_cm
    q = 0.5 * rho * velocity**2
    return {"CL": cl, "CD": cd, "Cm": cm, "L": q*s*cl, "D": q*s*cd, "m_b": q*s*c*cm,
            "wing_L": q*s*wing_cl, "tail_L": q*s*ratio*tail_cl,
            "wing_D": q*s*wing_cd, "tail_D": q*s*tail_cd, "body_D": q*s*body_cd,
            "wing_m_b": q*s*c*wing_cm, "tail_m_b": q*s*c*tail_cm, "body_m_b": q*s*c*body_cm}


def full(design, alpha, elevator, velocity, cg):
    airplane = ASBDesignVector.from_design_vector(design).make_airplane(elevator_deflection=elevator)
    aero = asb.AeroBuildup(airplane, asb.OperatingPoint(velocity=velocity, alpha=alpha),
                           xyz_ref=np.asarray(cg), include_wave_drag=False).run()
    wings, bodies = aero["wing_aero_components"], aero["fuselage_aero_components"]
    body = lambda key: sum(scalar(getattr(item, key)) for item in bodies)
    return {**{key: scalar(aero[key]) for key in QUANTITIES},
            "D_induced": scalar(aero["D_induced"]), "D_profile": scalar(aero["D_profile"]),
            "wing_L": scalar(wings[0].L), "tail_L": sum(scalar(wing.L) for wing in wings[1:]),
            "wing_D": scalar(wings[0].D), "tail_D": sum(scalar(wing.D) for wing in wings[1:]), "body_D": body("D"),
            "wing_m_b": scalar(wings[0].m_b), "tail_m_b": sum(scalar(wing.m_b) for wing in wings[1:]), "body_m_b": body("m_b")}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT); data = data[(data.coarse_status == "ok") & (data.full_status == "ok")]
    inputs, rho = _constant_aero_inputs(ParameterVector()), ParameterVector().rho
    rows = []
    for count, (_, source) in enumerate(data.iterrows(), 1):
        design = DesignVector(**{name: source[f"design_{name}"] for name in GEOMETRY})
        alpha, elevator, velocity = source.full_alpha, source.full_elevator_deflection, source.full_velocity
        c = coarse(design, alpha, elevator, velocity, inputs["cg"], rho)
        f = full(design, alpha, elevator, velocity, inputs["cg"])
        row = {"case_id": int(source.case_id), **{f"design_{name}": getattr(design, name) for name in GEOMETRY},
               "wing_area": design.wing_area, "aspect_ratio": design.wing_span / design.wing_chord,
               "full_trim_velocity": velocity, "full_trim_alpha": alpha, "full_trim_elevator": elevator}
        row.update({f"coarse_{key}": value for key, value in c.items()}); row.update({f"full_{key}": value for key, value in f.items()})
        rows.append(row); print(f"[{count}/{len(data)}] ok", flush=True)
    write_csv(OUTPUT_DIR / "raw_results.csv", rows)

    summary = []
    sensitivity = []
    features = (*GEOMETRY, "wing_area", "aspect_ratio")
    for quantity in QUANTITIES:
        error = np.array([row[f"coarse_{quantity}"] - row[f"full_{quantity}"] for row in rows])
        summary.append({"quantity": quantity, "mean_error": error.mean(), "mae": abs(error).mean(),
                        "p95_absolute_error": np.percentile(abs(error), 95),
                        "correlation": np.corrcoef([row[f"coarse_{quantity}"] for row in rows], [row[f"full_{quantity}"] for row in rows])[0, 1]})
        for feature in features:
            values = [row[feature if feature in ("wing_area", "aspect_ratio") else f"design_{feature}"] for row in rows]
            sensitivity.append({"quantity": quantity, "feature": feature,
                                "signed_error_correlation": np.corrcoef(values, error)[0, 1],
                                "absolute_error_correlation": np.corrcoef(values, abs(error))[0, 1]})
    write_csv(OUTPUT_DIR / "summary.csv", summary); write_csv(OUTPUT_DIR / "sensitivity.csv", sensitivity)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, quantity in zip(axes.flat, QUANTITIES):
        x = np.array([row[f"full_{quantity}"] for row in rows]); y = np.array([row[f"coarse_{quantity}"] for row in rows])
        ax.scatter(x, y, s=14, alpha=.55); low, high = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([low, high], [low, high], "k--"); ax.set(title=quantity, xlabel="Full", ylabel="Coarse"); ax.grid(True)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "force_moment_comparison.png", dpi=180); plt.close(fig)

    components = ("wing_L", "tail_L", "wing_D", "tail_D", "body_D", "wing_m_b", "tail_m_b")
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for ax, quantity in zip(axes.flat, components):
        x = np.array([row[f"full_{quantity}"] for row in rows]); y = np.array([row[f"coarse_{quantity}"] for row in rows])
        ax.scatter(x, y, s=12, alpha=.5); low, high = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([low, high], [low, high], "k--"); ax.set(title=quantity, xlabel="Full", ylabel="Coarse"); ax.grid(True)
    for ax in axes.flat[len(components):]: ax.axis("off")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "component_comparison.png", dpi=180); plt.close(fig)
    print(f"Saved force diagnostics to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
