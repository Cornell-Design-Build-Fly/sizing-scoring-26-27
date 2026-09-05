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
    run_nonlinear_lifting_line_on_design_vector,
    run_vlm_on_design_vector,
)
from src.vectors import ASBDesignVector


DATA_DUMP_DIR = PROJECT_ROOT / "data_dump"
CSV_OUTPUT_PATH = DATA_DUMP_DIR / "vector_test_alpha_sweep.csv"
PLOT_OUTPUT_PATH = DATA_DUMP_DIR / "vector_test_alpha_sweep.png"


def save_sweep_csv(rows: list[dict[str, float]]) -> None:
    """Saves the collected sweep data for later inspection."""
    DATA_DUMP_DIR.mkdir(exist_ok=True)
    fieldnames = list(rows[0].keys())
    with CSV_OUTPUT_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
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
# print(f"ASB refs: S={s_ref:.3f} m^2, c={c_ref:.3f} m, b={b_ref:.3f} m")
print(f"Built ASB airplane: {airplane.name}")

alphas = np.arange(-10.0, 20.0 + 0.25, 0.5)

vlm_cl_values: list[float] = []
vlm_cd_values: list[float] = []
vlm_cm_values: list[float] = []
vlm_ld_values: list[float] = []
vlm_runtime_values: list[float] = []

nll_cl_values: list[float] = []
nll_cd_values: list[float] = []
nll_cm_values: list[float] = []
nll_ld_values: list[float] = []
nll_runtime_values: list[float] = []

rows: list[dict[str, float]] = []

sweep_start_time = perf_counter()
for alpha in alphas:
    vlm_result = run_vlm_on_design_vector(dv, alpha=float(alpha))
    nll_result = run_nonlinear_lifting_line_on_design_vector(
        dv,
        alpha=float(alpha),
        spanwise_resolution=3,
    )

    vlm_cl_values.append(vlm_result.CL)
    vlm_cd_values.append(vlm_result.CD)
    vlm_cm_values.append(vlm_result.Cm)
    vlm_ld_values.append(vlm_result.CL / vlm_result.CD if vlm_result.CD != 0.0 else np.nan)
    vlm_runtime_values.append(vlm_result.runtime_seconds)

    nll_cl_values.append(nll_result.CL)
    nll_cd_values.append(nll_result.CD)
    nll_cm_values.append(nll_result.Cm)
    nll_ld_values.append(nll_result.CL / nll_result.CD if nll_result.CD != 0.0 else np.nan)
    nll_runtime_values.append(nll_result.runtime_seconds)

    rows.append(
        {
            "alpha_deg": float(alpha),
            "vlm_CL": vlm_result.CL,
            "vlm_CD": vlm_result.CD,
            "vlm_Cm": vlm_result.Cm,
            "vlm_L_over_D": vlm_ld_values[-1],
            "vlm_runtime_s": vlm_result.runtime_seconds,
            "vlm_converged": float(vlm_result.converged),
            "nll_CL": nll_result.CL,
            "nll_CD": nll_result.CD,
            "nll_Cm": nll_result.Cm,
            "nll_L_over_D": nll_ld_values[-1],
            "nll_runtime_s": nll_result.runtime_seconds,
            "nll_converged": float(nll_result.converged),
        }
    )

    print(
        f"alpha={alpha:5.1f} deg | "
        f"VLM CL={vlm_result.CL:7.3f}, CD={vlm_result.CD:8.4f}, Cm={vlm_result.Cm:7.3f}, "
        f"t={vlm_result.runtime_seconds:6.3f} s | "
        f"NLL CL={nll_result.CL:7.3f}, CD={nll_result.CD:8.4f}, Cm={nll_result.Cm:7.3f}, "
        f"t={nll_result.runtime_seconds:6.3f} s, conv={nll_result.converged}"
    )
total_runtime = perf_counter() - sweep_start_time

save_sweep_csv(rows)

print(
    f"Sweep complete: {len(alphas)} cases in {total_runtime:.3f} s | "
    f"avg VLM={np.mean(vlm_runtime_values):.3f} s, "
    f"avg NLL={np.mean(nll_runtime_values):.3f} s"
)
print(f"Saved sweep data to: {CSV_OUTPUT_PATH}")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("Design Vector Whole-Plane Aero Comparison")

axes[0, 0].plot(alphas, vlm_cl_values, color="navy", linewidth=2, label="VLM")
axes[0, 0].plot(alphas, nll_cl_values, color="teal", linewidth=2, label="Nonlinear LL")
axes[0, 0].set_title("CL vs Alpha")
axes[0, 0].set_xlabel("Alpha [deg]")
axes[0, 0].set_ylabel("CL [-]")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(alphas, vlm_cd_values, color="firebrick", linewidth=2, label="VLM")
axes[0, 1].plot(alphas, nll_cd_values, color="darkorange", linewidth=2, label="Nonlinear LL")
axes[0, 1].set_title("CD vs Alpha")
axes[0, 1].set_xlabel("Alpha [deg]")
axes[0, 1].set_ylabel("CD [-]")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

axes[1, 0].plot(alphas, vlm_cm_values, color="darkgreen", linewidth=2, label="VLM")
axes[1, 0].plot(alphas, nll_cm_values, color="purple", linewidth=2, label="Nonlinear LL")
axes[1, 0].set_title("Cm vs Alpha")
axes[1, 0].set_xlabel("Alpha [deg]")
axes[1, 0].set_ylabel("Cm [-]")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

axes[1, 1].plot(alphas, vlm_ld_values, color="goldenrod", linewidth=2, label="VLM L/D")
axes[1, 1].plot(alphas, nll_ld_values, color="saddlebrown", linewidth=2, label="NLL L/D")
axes[1, 1].plot(alphas, vlm_runtime_values, color="magenta", linewidth=1.5, label="VLM runtime [s]")
axes[1, 1].plot(alphas, nll_runtime_values, color="black", linewidth=1.5, label="NLL runtime [s]")
axes[1, 1].set_title("Performance and Runtime")
axes[1, 1].set_xlabel("Alpha [deg]")
axes[1, 1].set_ylabel("Value")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

fig.tight_layout()
DATA_DUMP_DIR.mkdir(exist_ok=True)
fig.savefig(PLOT_OUTPUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved plot to: {PLOT_OUTPUT_PATH}")
plt.show()
