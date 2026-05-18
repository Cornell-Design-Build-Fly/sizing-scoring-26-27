"""
Basic AeroSandbox airplane geometry tutorial.

This example is based on the official AeroSandbox geometry docs/tutorial references:
https://aerosandbox.readthedocs.io/en/master/autoapi/aerosandbox/geometry/airplane/index.html
https://aerosandbox.readthedocs.io/en/develop/autoapi/aerosandbox/geometry/wing/
https://aerosandbox.readthedocs.io/en/develop/autoapi/aerosandbox/geometry/fuselage/
https://aerosandbox.readthedocs.io/en/master/autoapi/aerosandbox/aerodynamics/aero_3D/vortex_lattice_method/index.html
https://github.com/peterdsharpe/AeroSandbox

What this script does:
1. Builds a simple airplane out of a fuselage, main wing, horizontal tail, and vertical tail.
2. Uses AeroSandbox's built-in geometry tools to show the airplane in 3D.
3. Runs a small VLM analysis and visualizes trailing-edge streamlines.
4. Plots a downwash slice behind the airplane for an extra look at the flowfield.

Run with:
    python src/tutorials/asb_basic_3d_airplane.py
"""

from typing import Final

import matplotlib.pyplot as plt
import aerosandbox as asb
import numpy as np


def require_scalar(value: float | list[float], label: str) -> float:
    """
    Converts an AeroSandbox scalar-like return value into a plain float.

    Some geometry methods are typed as `float | list[float]` because they can return
    sectional data in other modes. For airplane reference values, we want a single scalar.
    """
    if isinstance(value, list):
        raise TypeError(f"{label} must be a scalar, but AeroSandbox returned sectional data.")
    return float(value)


def make_example_airplane() -> tuple[asb.Airplane, float, float, float]:
    """Creates a small tutorial airplane in SI units."""
    main_wing = asb.Wing(
        name="Main Wing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(
                xyz_le=[0.0, 0.0, 0.0],
                chord=1.60,
                twist=2.0,
                airfoil=asb.Airfoil("naca2412"),
            ),
            asb.WingXSec(
                xyz_le=[0.35, 3.00, 0.20],
                chord=0.85,
                twist=0.0,
                airfoil=asb.Airfoil("naca2412"),
            ),
        ],
    )

    horizontal_tail = asb.Wing(
        name="Horizontal Tail",
        symmetric=True,
        xsecs=[
            asb.WingXSec(
                xyz_le=[4.40, 0.0, 0.25],
                chord=0.75,
                twist=0.0,
                airfoil=asb.Airfoil("naca0012"),
            ),
            asb.WingXSec(
                xyz_le=[4.65, 1.25, 0.25],
                chord=0.40,
                twist=0.0,
                airfoil=asb.Airfoil("naca0012"),
            ),
        ],
    )

    vertical_tail = asb.Wing(
        name="Vertical Tail",
        symmetric=False,
        xsecs=[
            asb.WingXSec(
                xyz_le=[4.50, 0.0, 0.15],
                chord=0.80,
                twist=0.0,
                airfoil=asb.Airfoil("naca0012"),
            ),
            asb.WingXSec(
                xyz_le=[4.90, 0.0, 1.00],
                chord=0.35,
                twist=0.0,
                airfoil=asb.Airfoil("naca0012"),
            ),
        ],
    )

    fuselage = asb.Fuselage(
        name="Fuselage",
        xsecs=[
            asb.FuselageXSec(xyz_c=[-2.00, 0.0, 0.0], radius=0.02),
            asb.FuselageXSec(xyz_c=[0.35, 0.0, 0.0], radius=0.18),
            asb.FuselageXSec(xyz_c=[1.30, 0.0, 0.0], radius=0.32),
            asb.FuselageXSec(xyz_c=[3.60, 0.0, 0.05], radius=0.27),
            asb.FuselageXSec(xyz_c=[4.90, 0.0, 0.12], radius=0.14),
            asb.FuselageXSec(xyz_c=[5.45, 0.0, 0.15], radius=0.03),
        ],
    )

    s_ref = require_scalar(main_wing.area(), "s_ref")
    c_ref = float(main_wing.mean_aerodynamic_chord())
    b_ref = require_scalar(main_wing.span(), "b_ref")

    airplane = asb.Airplane(
        name="Tutorial Plane",
        xyz_ref=[0.0, 0.0, 0.0],
        wings=[main_wing, horizontal_tail, vertical_tail],
        fuselages=[fuselage],
        s_ref=s_ref,
        c_ref=c_ref,
        b_ref=b_ref,
    )

    return airplane, s_ref, c_ref, b_ref


def make_vlm_analysis(airplane: asb.Airplane) -> asb.VortexLatticeMethod:
    """
    Sets up a moderate-angle VLM case that makes the wake shape easy to see.

    Note: this is a lifting-surface analysis, so the streamlines are a wing/tail wake
    visualization rather than a high-fidelity viscous fuselage CFD wake.
    """
    op_point = asb.OperatingPoint(
        velocity=28.0,
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
    """
    Builds a denser set of seeds along each trailing-edge panel so the wake looks richer.
    """
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
    """
    Plots a Y-Z slice of vertical velocity behind the airplane.
    """
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
        scale=70,
    )
    plt.colorbar(contour, ax=ax, label="Vertical velocity Vz [m/s]")
    ax.set_title(f"Wake Slice at x = {x_cut:.2f} m")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)


def main() -> None:
    airplane, s_ref, c_ref, b_ref = make_example_airplane()
    vlm = make_vlm_analysis(airplane)

    print(f"Built airplane: {airplane.name}")
    print(f"Reference wing area: {s_ref:.3f} m^2")
    print(f"Reference span: {b_ref:.3f} m")
    print(f"Reference chord: {c_ref:.3f} m")

    # For a standard orthographic layout, try: airplane.draw_three_view(show=True)
    airplane.draw_wireframe(
        color="navy",
        thin_linewidth=0.8,
        thick_linewidth=1.6,
        show=False,
    )
    plt.title("AeroSandbox 3D Airplane Tutorial")
    plt.show()

    aero = vlm.run()
    print(
        "VLM results: "
        f"CL={float(aero['CL']):.3f}, "
        f"CD={float(aero['CD']):.4f}, "
        f"Cm={float(aero['Cm']):.3f}, "
        f"L={float(aero['L']):.1f} N, "
        f"D={float(aero['D']):.1f} N"
    )

    seed_points = make_streamline_seed_points(vlm)
    vlm.calculate_streamlines(
        seed_points=seed_points,
        n_steps=260,
        length=c_ref * 10.0,
    )

    print("Opening VLM wake visualization with streamline trails...")
    try:
        vlm.draw(
            c=vlm.vortex_strengths,
            cmap="plasma",
            colorbar_label="Vortex strength",
            draw_streamlines=True,
            recalculate_streamlines=False,
            backend="plotly",
            show=True,
        )
    except Exception as exc:
        print(f"Plotly VLM view failed ({exc}). Falling back to PyVista.")
        vlm.draw(
            c=vlm.vortex_strengths,
            cmap="plasma",
            colorbar_label="Vortex strength",
            draw_streamlines=True,
            recalculate_streamlines=False,
            backend="pyvista",
            show=True,
        )

    plot_downwash_slice(
        vlm=vlm,
        x_cut=6.5,
        y_limit=b_ref * 0.55,
        z_min=-0.8,
        z_max=1.8,
    )
    plt.show()


if __name__ == "__main__":
    main()
