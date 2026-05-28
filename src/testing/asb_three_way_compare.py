from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aero import (
    run_aerobuildup_on_design_vector,
    run_lifting_line_on_design_vector,
    run_vlm_on_design_vector,
)
from src.vectors import ASBDesignVector


DATA_DUMP_DIR = PROJECT_ROOT / "data_dump"
CSV_OUTPUT_PATH = DATA_DUMP_DIR / "asb_three_way_compare.csv"
PLOT_OUTPUT_PATH = DATA_DUMP_DIR / "asb_three_way_compare.png"


def save_rows(rows: list[dict[str, float]]) -> None:
    DATA_DUMP_DIR.mkdir(exist_ok=True)
    with CSV_OUTPUT_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


dv = ASBDesignVector(
    wing_span=1.181354,
    wing_chord=0.307086,
    tail_arm=0.845058,
    nose_length=0.20,
)
airplane = dv.make_airplane()

print(f"Wing span: {dv.wing_span:.3f} m")
print(f"Wing chord: {dv.wing_chord:.3f} m")
print(f"Tail arm: {dv.tail_arm:.3f} m")
print(f"Nose length: {dv.nose_length:.3f} m")
print(f"H-stab span: {dv.hstab_span:.3f} m")
print(f"V-stab span: {dv.vstab_span:.3f} m")
print(f"Fuselage section: {dv.fuselage_width:.2f} m x {dv.fuselage_height:.2f} m")
print(f"Built ASB airplane: {airplane.name}")

alphas = np.arange(-10.0, 20.0 + 0.25, 0.5)

vlm_cl_values: list[float] = []
vlm_cd_values: list[float] = []
vlm_cm_values: list[float] = []
vlm_runtime_values: list[float] = []

ll_cl_values: list[float] = []
ll_cd_values: list[float] = []
ll_cm_values: list[float] = []
ll_runtime_values: list[float] = []

ab_cl_values: list[float] = []
ab_cd_values: list[float] = []
ab_cm_values: list[float] = []
ab_runtime_values: list[float] = []
ab_d_induced_values: list[float] = []
ab_d_profile_values: list[float] = []

rows: list[dict[str, float]] = []

sweep_start_time = perf_counter()
for alpha in alphas:
    vlm_result = run_vlm_on_design_vector(dv, alpha=float(alpha))
    ll_result = run_lifting_line_on_design_vector(dv, alpha=float(alpha))
    ab_result = run_aerobuildup_on_design_vector(dv, alpha=float(alpha))

    vlm_cl_values.append(vlm_result.CL)
    vlm_cd_values.append(vlm_result.CD)
    vlm_cm_values.append(vlm_result.Cm)
    vlm_runtime_values.append(vlm_result.runtime_seconds)

    ll_cl_values.append(ll_result.CL)
    ll_cd_values.append(ll_result.CD)
    ll_cm_values.append(ll_result.Cm)
    ll_runtime_values.append(ll_result.runtime_seconds)

    ab_cl_values.append(ab_result.CL)
    ab_cd_values.append(ab_result.CD)
    ab_cm_values.append(ab_result.Cm)
    ab_runtime_values.append(ab_result.runtime_seconds)
    ab_d_induced_values.append(ab_result.D_induced if ab_result.D_induced is not None else np.nan)
    ab_d_profile_values.append(ab_result.D_profile if ab_result.D_profile is not None else np.nan)

    rows.append(
        {
            "alpha_deg": float(alpha),
            "vlm_CL": vlm_result.CL,
            "vlm_CD": vlm_result.CD,
            "vlm_Cm": vlm_result.Cm,
            "vlm_runtime_s": vlm_result.runtime_seconds,
            "ll_CL": ll_result.CL,
            "ll_CD": ll_result.CD,
            "ll_Cm": ll_result.Cm,
            "ll_runtime_s": ll_result.runtime_seconds,
            "ab_CL": ab_result.CL,
            "ab_CD": ab_result.CD,
            "ab_Cm": ab_result.Cm,
            "ab_runtime_s": ab_result.runtime_seconds,
            "ab_D_induced_N": ab_d_induced_values[-1],
            "ab_D_profile_N": ab_d_profile_values[-1],
        }
    )

    print(
        f"alpha={alpha:5.1f} deg | "
        f"VLM t={vlm_result.runtime_seconds:6.3f} s | "
        f"LL t={ll_result.runtime_seconds:6.3f} s | "
        f"AB t={ab_result.runtime_seconds:6.3f} s"
    )
total_runtime = perf_counter() - sweep_start_time

save_rows(rows)

print(
    f"Sweep complete: {len(alphas)} cases in {total_runtime:.3f} s | "
    f"avg VLM={np.mean(vlm_runtime_values):.3f} s, "
    f"avg LL={np.mean(ll_runtime_values):.3f} s, "
    f"avg AB={np.mean(ab_runtime_values):.3f} s"
)
print(f"Saved comparison data to: {CSV_OUTPUT_PATH}")

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle("ASB Three-Way Comparison: VLM vs LiftingLine vs AeroBuildup")

axes[0, 0].plot(alphas, vlm_cl_values, label="VLM", color="navy", linewidth=2)
axes[0, 0].plot(alphas, ll_cl_values, label="LiftingLine", color="teal", linewidth=2)
axes[0, 0].plot(alphas, ab_cl_values, label="AeroBuildup", color="darkorange", linewidth=2)
axes[0, 0].set_title("CL vs Alpha")
axes[0, 0].set_xlabel("Alpha [deg]")
axes[0, 0].set_ylabel("CL [-]")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(alphas, vlm_cd_values, label="VLM", color="firebrick", linewidth=2)
axes[0, 1].plot(alphas, ll_cd_values, label="LiftingLine", color="purple", linewidth=2)
axes[0, 1].plot(alphas, ab_cd_values, label="AeroBuildup", color="black", linewidth=2)
axes[0, 1].set_title("CD vs Alpha")
axes[0, 1].set_xlabel("Alpha [deg]")
axes[0, 1].set_ylabel("CD [-]")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

axes[1, 0].plot(alphas, vlm_cm_values, label="VLM", color="darkgreen", linewidth=2)
axes[1, 0].plot(alphas, ll_cm_values, label="LiftingLine", color="magenta", linewidth=2)
axes[1, 0].plot(alphas, ab_cm_values, label="AeroBuildup", color="saddlebrown", linewidth=2)
axes[1, 0].set_title("Cm vs Alpha")
axes[1, 0].set_xlabel("Alpha [deg]")
axes[1, 0].set_ylabel("Cm [-]")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

axes[1, 1].plot(alphas, vlm_runtime_values, label="VLM runtime [s]", color="navy", linewidth=2)
axes[1, 1].plot(alphas, ll_runtime_values, label="LiftingLine runtime [s]", color="teal", linewidth=2)
axes[1, 1].plot(alphas, ab_runtime_values, label="AeroBuildup runtime [s]", color="darkorange", linewidth=2)
axes[1, 1].set_title("Runtime vs Alpha")
axes[1, 1].set_xlabel("Alpha [deg]")
axes[1, 1].set_ylabel("Runtime [s]")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

axes[0, 2].plot(alphas, np.array(vlm_cl_values) / np.array(vlm_cd_values), label="VLM", color="firebrick", linewidth=2)
axes[0, 2].plot(alphas, np.array(ll_cl_values) / np.array(ll_cd_values), label="LiftingLine", color="purple", linewidth=2)
axes[0, 2].plot(alphas, np.array(ab_cl_values) / np.array(ab_cd_values), label="AeroBuildup", color="black", linewidth=2)
axes[0, 2].set_title("CL/CD vs Alpha")
axes[0, 2].set_xlabel("Alpha [deg]")
axes[0, 2].set_ylabel("CL/CD [-]")
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].legend()

fig.tight_layout()
DATA_DUMP_DIR.mkdir(exist_ok=True)
fig.savefig(PLOT_OUTPUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved comparison plot to: {PLOT_OUTPUT_PATH}")
plt.show()
