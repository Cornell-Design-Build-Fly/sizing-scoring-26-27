from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from time import perf_counter

import aerosandbox as asb
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vectors import ASBDesignVector
from src.testing.asb_design_vector_airplane import (
    make_streamline_seed_points,
    plot_downwash_slice,
)


DATA_DUMP_DIR = PROJECT_ROOT / "data_dump"
SUMMARY_OUTPUT_PATH = DATA_DUMP_DIR / "geometry_flow_showcase_summary.json"
WAKE_HTML_OUTPUT_PATH = DATA_DUMP_DIR / "geometry_flow_showcase_wake.html"
WIREFRAME_OUTPUT_PATH = DATA_DUMP_DIR / "geometry_flow_showcase_wireframe.png"
LOADING_OUTPUT_PATH = DATA_DUMP_DIR / "geometry_flow_showcase_loading.png"
DOWNWASH_OUTPUT_PATH = DATA_DUMP_DIR / "geometry_flow_showcase_downwash.png"


def build_design_vector() -> ASBDesignVector:
    return ASBDesignVector(
        wing_span=1.181354,
        wing_chord=0.307086,
        tail_arm=0.845058,
        nose_length=0.20,
    )


def make_vlm_analysis(airplane: asb.Airplane) -> asb.VortexLatticeMethod:
    return asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(
            velocity=18.0,
            alpha=8.0,
            beta=0.0,
            p=0.0,
            q=0.0,
            r=0.0,
        ),
        spanwise_resolution=8,
        chordwise_resolution=6,
        align_trailing_vortices_with_wind=True,
        verbose=False,
    )


def save_spanwise_loading_plot(vlm: asb.VortexLatticeMethod) -> None:
    front_left = vlm.front_left_vertices
    front_right = vlm.front_right_vertices
    back_left = vlm.back_left_vertices
    back_right = vlm.back_right_vertices

    panel_centers = 0.25 * (front_left + front_right + back_left + back_right)
    span_positions = panel_centers[:, 1]
    vertical_positions = panel_centers[:, 2]
    strengths = vlm.vortex_strengths

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    scatter = axes[0].scatter(
        span_positions,
        strengths,
        c=vertical_positions,
        cmap="viridis",
        s=26,
        alpha=0.9,
    )
    axes[0].axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[0].set_title("Panel Vortex Strength vs Span")
    axes[0].set_xlabel("y [m]")
    axes[0].set_ylabel("Vortex strength")
    axes[0].grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=axes[0], label="Panel z [m]")

    order = np.argsort(span_positions)
    axes[1].plot(span_positions[order], strengths[order], color="darkorange", linewidth=2)
    axes[1].fill_between(
        span_positions[order],
        strengths[order],
        alpha=0.2,
        color="darkorange",
    )
    axes[1].set_title("Spanwise Loading Trace")
    axes[1].set_xlabel("y [m]")
    axes[1].set_ylabel("Vortex strength")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Geometry Flow Showcase: Loading View")
    fig.tight_layout()
    fig.savefig(LOADING_OUTPUT_PATH, dpi=200, bbox_inches="tight")


def main() -> None:
    DATA_DUMP_DIR.mkdir(exist_ok=True)
    headless = os.environ.get("ASB_HEADLESS", "0") == "1"

    design_vector = build_design_vector()
    airplane, s_ref, c_ref, b_ref = design_vector.make_airplane(
        name="Geometry Showcase Plane"
    )
    vlm = make_vlm_analysis(airplane)

    solve_start = perf_counter()
    aero = vlm.run()
    solve_runtime = perf_counter() - solve_start

    seed_points = make_streamline_seed_points(vlm)
    vlm.calculate_streamlines(
        seed_points=seed_points,
        n_steps=220,
        length=c_ref * 9.0,
    )

    print(f"Built airplane: {airplane.name}")
    print(
        "Showcase VLM results: "
        f"CL={float(aero['CL']):.3f}, "
        f"CD={float(aero['CD']):.4f}, "
        f"Cm={float(aero['Cm']):.3f}, "
        f"L={float(aero['L']):.2f} N, "
        f"D={float(aero['D']):.2f} N"
    )
    print(f"Solve runtime: {solve_runtime:.3f} s")

    airplane.draw_wireframe(
        color="navy",
        thin_linewidth=0.8,
        thick_linewidth=1.6,
        show=False,
    )
    plt.title("Geometry Showcase Wireframe")
    plt.savefig(WIREFRAME_OUTPUT_PATH, dpi=200, bbox_inches="tight")

    wake_fig = vlm.draw(
        c=vlm.vortex_strengths,
        cmap="plasma",
        colorbar_label="Vortex strength",
        draw_streamlines=True,
        recalculate_streamlines=False,
        backend="plotly",
        show=False,
        show_kwargs={"title": "Geometry Flow Showcase Wake"},
    )
    wake_fig.write_html(WAKE_HTML_OUTPUT_PATH)

    plot_downwash_slice(
        vlm=vlm,
        x_cut=design_vector.tail_arm + c_ref * 1.8,
        y_limit=b_ref * 0.6,
        z_min=-0.35,
        z_max=max(design_vector.vstab_span * 1.2, 0.9),
    )
    plt.savefig(DOWNWASH_OUTPUT_PATH, dpi=200, bbox_inches="tight")

    save_spanwise_loading_plot(vlm)

    summary = {
        "velocity_mps": 18.0,
        "alpha_deg": 8.0,
        "spanwise_resolution": 8,
        "chordwise_resolution": 6,
        "wing_area_m2": s_ref,
        "reference_span_m": b_ref,
        "reference_chord_m": c_ref,
        "CL": float(aero["CL"]),
        "CD": float(aero["CD"]),
        "Cm": float(aero["Cm"]),
        "L_N": float(aero["L"]),
        "D_N": float(aero["D"]),
        "solve_runtime_s": solve_runtime,
        "panel_count": int(len(vlm.vortex_strengths)),
        "streamline_seed_count": int(seed_points.shape[0]),
    }
    SUMMARY_OUTPUT_PATH.write_text(json.dumps(summary, indent=2))

    print(f"Saved summary to: {SUMMARY_OUTPUT_PATH}")
    print(f"Saved wake view to: {WAKE_HTML_OUTPUT_PATH}")
    print(f"Saved wireframe snapshot to: {WIREFRAME_OUTPUT_PATH}")
    print(f"Saved downwash slice to: {DOWNWASH_OUTPUT_PATH}")
    print(f"Saved loading plot to: {LOADING_OUTPUT_PATH}")

    if not headless:
        wake_fig.show()
        plt.show()


if __name__ == "__main__":
    main()
