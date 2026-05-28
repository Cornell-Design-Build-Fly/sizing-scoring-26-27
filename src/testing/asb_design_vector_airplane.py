"""
Design-vector-driven AeroSandbox geometry and VLM demo.

This version uses the sizing design vector as the source of truth, assumes all
design-vector inputs are already in meters, builds the lifting surfaces in
AeroSandbox, and defers every visualization until the analysis work is already
complete.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

import aerosandbox as asb
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vectors import ASBDesignVector


DESIGN_VECTOR_METERS: Final[dict[str, float]] = {
    "wing_span": 1.181354,
    "wing_chord": 0.307086,
    "tail_arm": 0.845058,
    "nose_length": 0.20,
}


def make_design_vector() -> ASBDesignVector:
    """Creates the metric design vector used by this demo."""
    return ASBDesignVector(**DESIGN_VECTOR_METERS)


def make_vlm_analysis(airplane: asb.Airplane) -> asb.VortexLatticeMethod:
    """Sets up the same style of moderate-angle VLM case as the ASB tutorial."""
    op_point = asb.OperatingPoint(
        velocity=18.0,
        alpha=6.0,
        beta=0.0,
        p=0.0,
        q=0.0,
        r=0.0,
    )

    return asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=6,
        chordwise_resolution=6,
        align_trailing_vortices_with_wind=True,
        verbose=False,
    )


def make_streamline_seed_points(vlm: asb.VortexLatticeMethod) -> np.ndarray:
    """Builds a richer set of wake seeds across each trailing-edge panel."""
    trailing_edge_mask = vlm.is_trailing_edge.astype(bool)
    left_trailing_edge = vlm.back_left_vertices[trailing_edge_mask]
    right_trailing_edge = vlm.back_right_vertices[trailing_edge_mask]

    blend_fractions: Final[list[float]] = [0.15, 0.35, 0.55, 0.75, 0.90]
    seed_sets = [
        fraction * left_trailing_edge + (1.0 - fraction) * right_trailing_edge
        for fraction in blend_fractions
    ]
    return np.concatenate(seed_sets, axis=0)


def plot_downwash_slice(
    vlm: asb.VortexLatticeMethod,
    x_cut: float,
    y_limit: float,
    z_min: float,
    z_max: float,
) -> None:
    """Plots a Y-Z slice of the induced wake velocity behind the airplane."""
    y = np.linspace(-y_limit, y_limit, 49)
    z = np.linspace(z_min, z_max, 37)
    yy, zz = np.meshgrid(y, z)
    xx = np.full_like(yy, x_cut)

    sample_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    velocity = vlm.get_velocity_at_points(sample_points)

    vy = velocity[:, 1].reshape(yy.shape)
    vz = velocity[:, 2].reshape(yy.shape)

    fig, ax = plt.subplots(figsize=(9, 5))
    contour = ax.contourf(yy, zz, vz, levels=31, cmap="coolwarm")
    ax.quiver(
        yy[::2, ::2],
        zz[::2, ::2],
        vy[::2, ::2],
        vz[::2, ::2],
        color="black",
        alpha=0.55,
        scale=50,
    )
    fig.colorbar(contour, ax=ax, label="Vertical velocity Vz [m/s]")
    ax.set_title(f"Design Vector Wake Slice at x = {x_cut:.2f} m")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)


def main() -> None:
    design_vector = make_design_vector()
    airplane = design_vector.make_airplane(
        name="Design Vector Plane"
    )
    vlm = make_vlm_analysis(airplane)

    print(f"Built airplane: {airplane.name}")
    print(f"Wing span: {design_vector.wing_span:.3f} m")
    print(f"Wing chord: {design_vector.wing_chord:.3f} m")
    print(f"Tail arm: {design_vector.tail_arm:.3f} m")
    print(f"Nose length: {design_vector.nose_length:.3f} m")
    print(f"Horizontal tail span: {design_vector.hstab_span:.3f} m")
    print(f"Vertical tail span: {design_vector.vstab_span:.3f} m")
    print(
        f"Fuselage section: {design_vector.fuselage_width:.2f} m x "
        f"{design_vector.fuselage_height:.2f} m"
    )
    # print(f"Reference wing area: {s_ref:.3f} m^2")
    # print(f"Reference span: {b_ref:.3f} m")
    # print(f"Reference chord: {c_ref:.3f} m")

    airplane.draw_wireframe(
        color="navy",
        thin_linewidth=0.8,
        thick_linewidth=1.6,
        show=False,
    )
    plt.title("AeroSandbox Design Vector Geometry")

    aero = vlm.run()
    print(
        "VLM results: "
        f"CL={float(aero['CL']):.3f}, "
        f"CD={float(aero['CD']):.4f}, "
        f"Cm={float(aero['Cm']):.3f}, "
        f"L={float(aero['L']):.2f} N, "
        f"D={float(aero['D']):.2f} N"
    )

    seed_points = make_streamline_seed_points(vlm)
    vlm.calculate_streamlines(
        seed_points=seed_points,
        n_steps=260,
        length=c_ref * 10.0,
    )

    wake_view = None
    wake_backend = None
    try:
        wake_view = vlm.draw(
            c=vlm.vortex_strengths,
            cmap="plasma",
            colorbar_label="Vortex strength",
            draw_streamlines=True,
            recalculate_streamlines=False,
            backend="plotly",
            show=False,
            show_kwargs={"title": "Design Vector VLM Wake"},
        )
        wake_backend = "plotly"
    except Exception as exc:
        print(f"Plotly VLM view failed ({exc}). Falling back to PyVista.")
        wake_view = vlm.draw(
            c=vlm.vortex_strengths,
            cmap="plasma",
            colorbar_label="Vortex strength",
            draw_streamlines=True,
            recalculate_streamlines=False,
            backend="pyvista",
            show=False,
        )
        wake_backend = "pyvista"

    plot_downwash_slice(
        vlm=vlm,
        x_cut=design_vector.tail_arm + c_ref * 1.5,
        y_limit=b_ref * 0.55,
        z_min=-0.3,
        z_max=max(design_vector.vstab_span * 1.15, 0.8),
    )

    print("Showing saved visualizations now that the analysis is finished...")
    if wake_backend == "plotly" and wake_view is not None:
        wake_view.show()
    elif wake_backend == "pyvista" and wake_view is not None:
        wake_view.show()

    plt.show()


if __name__ == "__main__":
    main()
