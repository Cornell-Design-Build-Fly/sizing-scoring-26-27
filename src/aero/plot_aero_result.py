from __future__ import annotations

from pathlib import Path

import aerosandbox as asb
import matplotlib.pyplot as plt
import numpy as np

from src.aero.cruise_analysis import eval_thrust
from src.aero.custom_classes import CruiseCondition
from src.aero.utils import require_scalar
from src.vectors import ASBDesignVector, DesignVector, ParameterVector

def plot_aero_result(
    design_vector: DesignVector,
    cruise_condition: CruiseCondition,
    thrust_velocity: tuple[float, float, float],
    cg: tuple[float, float, float],
    parameter_vector: ParameterVector,
    mass: float,
    num_samples: int = 31,
    mission: int | None = None,
    output_directory: str | Path = "data_dump/aero_plots",
) -> tuple[Path, ...]:
    """
    Plot aircraft geometry and aerodynamic sweeps about the trim point.

    Every figure is saved as a PNG before the interactive windows are shown.
    """

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    filename_prefix = f"mission_{mission}_" if mission is not None else ""
    saved_paths: list[Path] = []

    cruise_velocity = require_scalar(
        cruise_condition.operating_point.velocity
    )
    cruise_alpha = require_scalar(
        cruise_condition.operating_point.alpha
    )
    cruise_elevator = float(
        cruise_condition.elevator_deflection
    )
    cruise_tail_incidence = float(
        cruise_condition.tail_incidence
    )

    if cruise_condition.stall_speed is None:
        raise ValueError("Stall speed is unavailable.")

    stall_speed = float(cruise_condition.stall_speed)

    weight = mass * parameter_vector.gravity

    # ------------------------------------------------------------
    # Geometry views
    # ------------------------------------------------------------

    trimmed_airplane = (
        ASBDesignVector
        .from_design_vector(design_vector)
        .make_airplane(
            elevator_deflection=cruise_elevator,
            tail_incidence=cruise_tail_incidence,
        )
    )

    trimmed_airplane.draw_three_view(style="shaded")
    geometry_path = output_path / f"{filename_prefix}geometry.png"
    plt.gcf().savefig(geometry_path, dpi=200, bbox_inches="tight")
    saved_paths.append(geometry_path.resolve())

    # ------------------------------------------------------------
    # Sweep ranges
    # ------------------------------------------------------------

    velocities = np.linspace(
        max(0.5 * cruise_velocity, 0.1),
        1.5 * cruise_velocity,
        num_samples,
    )

    # Ensure cruise and stall speeds are included exactly.
    velocities = np.unique(
        np.append(
            velocities,
            [cruise_velocity, stall_speed],
        )
    )

    # ------------------------------------------------------------
    # Run velocity sweep
    # ------------------------------------------------------------

    velocity_lift = []
    velocity_drag = []
    velocity_thrust = []
    velocity_moment = []

    for velocity in velocities:
        airplane = (
            ASBDesignVector
            .from_design_vector(design_vector)
            .make_airplane(
                elevator_deflection=cruise_elevator,
                tail_incidence=cruise_tail_incidence,
            )
        )

        aero = asb.AeroBuildup(
            airplane=airplane,
            op_point=asb.OperatingPoint(
                velocity=float(velocity),
                alpha=cruise_alpha,
                beta=0.0,
                p=0.0,
                q=0.0,
                r=0.0,
            ),
            xyz_ref=np.asarray(cg),
        ).run()

        velocity_lift.append(require_scalar(aero["L"]))
        velocity_drag.append(require_scalar(aero["D"]))
        velocity_moment.append(require_scalar(aero["m_b"]))
        velocity_thrust.append(
            eval_thrust(
                float(velocity),
                thrust_velocity,
            )
        )

    cruise_thrust = eval_thrust(
        cruise_velocity,
        thrust_velocity,
    )

    # ------------------------------------------------------------
    # Get exact trim-point values
    # ------------------------------------------------------------

    trim_aero = asb.AeroBuildup(
        airplane=trimmed_airplane,
        op_point=cruise_condition.operating_point,
        xyz_ref=np.asarray(cg),
    ).run()

    trim_lift = require_scalar(trim_aero["L"])
    trim_drag = require_scalar(trim_aero["D"])
    trim_moment = require_scalar(trim_aero["m_b"])

    # ------------------------------------------------------------
    # Plotting helper
    # ------------------------------------------------------------

    def plot_sweep(
        x_values: np.ndarray,
        lift_values: list[float],
        drag_values: list[float],
        thrust_values: list[float],
        moment_values: list[float],
        equilibrium_x: float,
        xlabel: str,
        title: str,
        filename: str,
        moment_title: str,
        moment_filename: str,
        mark_stall_speed: bool = False,
    ) -> None:

        figure, force_axis = plt.subplots()

        force_axis.plot(
            x_values,
            lift_values,
            label="Lift",
        )
        force_axis.plot(
            x_values,
            drag_values,
            label="Drag",
        )
        force_axis.plot(
            x_values,
            thrust_values,
            label="Thrust",
        )
        force_axis.axhline(
            weight,
            linestyle="--",
            label="Weight",
        )

        force_axis.scatter(
            equilibrium_x,
            trim_lift,
            marker="x",
            s=90,
            zorder=5,
            label="Trimmed lift",
        )
        force_axis.scatter(
            equilibrium_x,
            trim_drag,
            marker="x",
            s=90,
            zorder=5,
            label="Trimmed drag",
        )
        force_axis.scatter(
            equilibrium_x,
            cruise_thrust,
            marker="x",
            s=90,
            zorder=5,
            label="Trimmed thrust",
        )

        force_axis.axvline(
            equilibrium_x,
            linestyle="--",
            label="Equilibrium point",
        )

        if mark_stall_speed:
            force_axis.axvline(
                stall_speed,
                linestyle=":",
                label=(
                    f"Stall speed: "
                    f"{stall_speed:.2f} m/s"
                ),
            )

        force_axis.set_xlabel(xlabel)
        force_axis.set_ylabel("Force [N]")
        force_axis.grid(True)
        force_axis.legend(loc="best")

        figure.suptitle(title)
        figure.tight_layout()
        figure_path = output_path / f"{filename_prefix}{filename}.png"
        figure.savefig(figure_path, dpi=200, bbox_inches="tight")
        saved_paths.append(figure_path.resolve())

        moment_figure, moment_axis = plt.subplots()

        moment_axis.plot(
            x_values,
            moment_values,
            linestyle="-.",
            label="Pitching moment",
        )
        moment_axis.axhline(
            0.0,
            linestyle=":",
            label="Zero moment",
        )
        moment_axis.scatter(
            equilibrium_x,
            trim_moment,
            marker="o",
            s=55,
            zorder=5,
            label="Trimmed moment",
        )

        moment_axis.set_ylabel(
            "Pitching moment [N·m]"
        )

        moment_axis.set_xlabel(xlabel)
        moment_axis.axvline(
            equilibrium_x,
            linestyle="--",
            label="Equilibrium point",
        )
        if mark_stall_speed:
            moment_axis.axvline(
                stall_speed,
                linestyle=":",
                label=f"Stall speed: {stall_speed:.2f} m/s",
            )
        moment_axis.grid(True)
        moment_axis.legend(loc="best")
        moment_figure.suptitle(moment_title)
        moment_figure.tight_layout()
        moment_path = output_path / f"{filename_prefix}{moment_filename}.png"
        moment_figure.savefig(moment_path, dpi=200, bbox_inches="tight")
        saved_paths.append(moment_path.resolve())

    # ------------------------------------------------------------
    # Create velocity force and moment plots
    # ------------------------------------------------------------

    plot_sweep(
        x_values=velocities,
        lift_values=velocity_lift,
        drag_values=velocity_drag,
        thrust_values=velocity_thrust,
        moment_values=velocity_moment,
        equilibrium_x=cruise_velocity,
        xlabel="Velocity [m/s]",
        title="Aerodynamic forces versus velocity",
        filename="velocity_sweep",
        moment_title="Pitching moment versus velocity",
        moment_filename="velocity_moment",
        mark_stall_speed=True,
    )

    print(f"[aero] Saved plots to {output_path.resolve()}", flush=True)
    plt.show()
    return tuple(saved_paths)
