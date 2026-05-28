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


DATA_DUMP_DIR = PROJECT_ROOT / "data_dump"
SUMMARY_OUTPUT_PATH = DATA_DUMP_DIR / "nll_viewer_summary.json"
HTML_OUTPUT_PATH = DATA_DUMP_DIR / "nll_viewer_plotly.html"
WIRE_OUTPUT_PATH = DATA_DUMP_DIR / "nll_viewer_wireframe.png"


def make_streamline_seed_points(analysis: asb.NonlinearLiftingLine) -> np.ndarray:
    """Uses one midpoint seed per trailing-edge panel for a lightweight wake view."""
    return 0.5 * (analysis.back_left_vertices + analysis.back_right_vertices)


def main() -> None:
    DATA_DUMP_DIR.mkdir(exist_ok=True)
    headless = os.environ.get("ASB_HEADLESS", "0") == "1"

    design_vector = ASBDesignVector(
        wing_span=1.181354,
        wing_chord=0.307086,
        tail_arm=0.845058,
        nose_length=0.20,
    )
    airplane, s_ref, c_ref, b_ref = design_vector.make_airplane(
        name="Design Vector Plane"
    )

    alpha_deg = 10.0
    analysis = asb.NonlinearLiftingLine(
        airplane=airplane,
        op_point=asb.OperatingPoint(
            velocity=18.0,
            alpha=alpha_deg,
            beta=0.0,
            p=0.0,
            q=0.0,
            r=0.0,
        ),
        spanwise_resolution=15,
        align_trailing_vortices_with_wind=True,
        verbose=True,
    )

    start_time = perf_counter()
    aero = analysis.run()
    solve_runtime = perf_counter() - start_time

    print(f"Built airplane: {airplane.name}")
    print(f"Alpha: {alpha_deg:.1f} deg")
    print(f"Reference wing area: {s_ref:.3f} m^2")
    print(f"Reference span: {b_ref:.3f} m")
    print(f"Reference chord: {c_ref:.3f} m")
    print(
        "NLL results: "
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
    plt.title("Design Vector Geometry for NLL Case")
    plt.savefig(WIRE_OUTPUT_PATH, dpi=200, bbox_inches="tight")

    seed_points = make_streamline_seed_points(analysis)
    analysis.calculate_streamlines(
        seed_points=seed_points,
        n_steps=80,
        length=c_ref * 4.0,
    )

    nll_plot = analysis.draw(
        c=analysis.vortex_strengths,
        cmap="plasma",
        colorbar_label="Vortex strength",
        draw_streamlines=True,
        recalculate_streamlines=False,
        backend="plotly",
        show=False,
        show_kwargs={"title": "Nonlinear Lifting Line Wake View"},
    )
    nll_plot.write_html(HTML_OUTPUT_PATH)

    summary = {
        "alpha_deg": alpha_deg,
        "velocity_mps": 18.0,
        "spanwise_resolution": 3,
        "wing_area_m2": s_ref,
        "reference_span_m": b_ref,
        "reference_chord_m": c_ref,
        "CL": float(aero["CL"]),
        "CD": float(aero["CD"]),
        "Cm": float(aero["Cm"]),
        "L_N": float(aero["L"]),
        "D_N": float(aero["D"]),
        "solve_runtime_s": solve_runtime,
        "streamline_seed_count": int(seed_points.shape[0]),
        "streamline_steps": 80,
        "panel_count": int(len(analysis.front_left_vertices)),
    }
    SUMMARY_OUTPUT_PATH.write_text(json.dumps(summary, indent=2))

    print(f"Saved summary to: {SUMMARY_OUTPUT_PATH}")
    print(f"Saved interactive NLL view to: {HTML_OUTPUT_PATH}")
    print(f"Saved wireframe snapshot to: {WIRE_OUTPUT_PATH}")

    if not headless:
        nll_plot.show()
        plt.show()


if __name__ == "__main__":
    main()
