"""Steady tension-only tow-line loads for Mission 3 cruise trim.

This is the SI-unit, optimizer-facing portion of the supplied dynamic rope
model. The complete transient model is useful for sizing peak catch loads, but
is too expensive and path-dependent to place inside every cruise root solve.
In steady flight the sensor force balance is exact: the airplane receives the
sensor aerodynamic drag backward and its weight downward through the rope.
"""

from dataclasses import dataclass

import aerosandbox.numpy as np

from src.aero.drag_model import sensor_drag_force
from src.vectors import DesignVector, ParameterVector


LBF_PER_FT_TO_N_PER_M = 4.4482216152605 / 0.3048
DEFAULT_ROPE_REST_LENGTH_M = 9.0 * 0.3048
DEFAULT_ROPE_STIFFNESS_N_M = 1000.0 * LBF_PER_FT_TO_N_PER_M
DEFAULT_ROPE_DAMPING_RATIO = 0.03


@dataclass(frozen=True)
class SteadyTowLineLoad:
    """Resolved steady force and geometry of the taut tow line."""

    backward_force_n: float
    downward_force_n: float
    tension_n: float
    extension_m: float
    length_m: float
    angle_from_vertical_rad: float


def tow_line_force_components(
    design: DesignVector,
    parameters: ParameterVector,
    velocity,
):
    """Return backward force, downward force, and tension on the airplane.

    The sensor is in steady equilibrium and the massless rope carries tension
    only. Both components are positive magnitudes.
    """
    backward_force = sensor_drag_force(design, parameters, velocity)
    downward_force = design.sensor_weight_kg * parameters.gravity
    tension = np.sqrt(backward_force**2 + downward_force**2)
    return backward_force, downward_force, tension


def steady_tow_line_load(
    design: DesignVector,
    parameters: ParameterVector,
    velocity: float,
    *,
    rope_rest_length_m: float = DEFAULT_ROPE_REST_LENGTH_M,
    rope_stiffness_n_m: float = DEFAULT_ROPE_STIFFNESS_N_M,
) -> SteadyTowLineLoad:
    """Resolve a numerical steady tow-line state for reporting/checking."""
    if rope_rest_length_m <= 0.0:
        raise ValueError("Rope rest length must be positive.")
    if rope_stiffness_n_m <= 0.0:
        raise ValueError("Rope stiffness must be positive.")
    backward, downward, tension = tow_line_force_components(
        design, parameters, velocity
    )
    backward = float(backward)
    downward = float(downward)
    tension = float(tension)
    extension = tension / rope_stiffness_n_m
    return SteadyTowLineLoad(
        backward_force_n=backward,
        downward_force_n=downward,
        tension_n=tension,
        extension_m=extension,
        length_m=rope_rest_length_m + extension,
        angle_from_vertical_rad=float(np.arctan2(backward, downward)),
    )


__all__ = [
    "DEFAULT_ROPE_DAMPING_RATIO",
    "DEFAULT_ROPE_REST_LENGTH_M",
    "DEFAULT_ROPE_STIFFNESS_N_M",
    "SteadyTowLineLoad",
    "steady_tow_line_load",
    "tow_line_force_components",
]
