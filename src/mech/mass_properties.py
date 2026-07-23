"""Geometry stations, neutral point, CG, and inertia calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.mech.models import MassItem, NeutralPointConfig
from src.vectors import DesignVector


@dataclass(frozen=True)
class GeometryStations:
    wing_le_x_m: float
    wing_ac_x_m: float
    wing_center_x_m: float
    wing_te_x_m: float
    horizontal_tail_le_x_m: float
    horizontal_tail_ac_x_m: float
    horizontal_tail_center_x_m: float
    horizontal_tail_te_x_m: float
    vertical_tail_le_x_m: float
    vertical_tail_ac_x_m: float
    vertical_tail_center_x_m: float
    vertical_tail_te_x_m: float
    tail_te_x_m: float
    nose_tip_x_m: float
    fuselage_length_m: float


def geometry_stations(design_vector: DesignVector) -> GeometryStations:
    """Build longitudinal stations using an LE-to-LE ``tail_arm``.

    ``DesignVector.tail_arm`` is the distance from the main-wing leading edge
    to the common horizontal/vertical-tail leading-edge station.  Aerodynamic
    centers are derived from those physical leading edges and each chord.
    """

    wing_le = 0.0
    wing_ac = wing_le + 0.25 * design_vector.wing_chord
    wing_center = wing_le + 0.50 * design_vector.wing_chord
    wing_te = wing_le + design_vector.wing_chord

    htail_le = wing_le + design_vector.tail_arm
    htail_ac = htail_le + 0.25 * design_vector.hstab_chord
    htail_center = htail_le + 0.50 * design_vector.hstab_chord
    htail_te = htail_le + design_vector.hstab_chord

    vtail_le = wing_le + design_vector.tail_arm
    vtail_ac = vtail_le + 0.25 * design_vector.vstab_chord
    vtail_center = vtail_le + 0.50 * design_vector.vstab_chord
    vtail_te = vtail_le + design_vector.vstab_chord

    tail_te = max(htail_te, vtail_te)
    nose_tip = -design_vector.nose_length

    return GeometryStations(
        wing_le_x_m=wing_le,
        wing_ac_x_m=wing_ac,
        wing_center_x_m=wing_center,
        wing_te_x_m=wing_te,
        horizontal_tail_le_x_m=htail_le,
        horizontal_tail_ac_x_m=htail_ac,
        horizontal_tail_center_x_m=htail_center,
        horizontal_tail_te_x_m=htail_te,
        vertical_tail_le_x_m=vtail_le,
        vertical_tail_ac_x_m=vtail_ac,
        vertical_tail_center_x_m=vtail_center,
        vertical_tail_te_x_m=vtail_te,
        tail_te_x_m=tail_te,
        nose_tip_x_m=nose_tip,
        fuselage_length_m=tail_te - nose_tip,
    )


def finite_wing_lift_curve_slope(
    aspect_ratio: float,
    oswald_efficiency: float,
    two_dimensional_slope_per_rad: float,
) -> float:
    """Finite-wing lift-curve slope using a standard lifting-line correction."""
    if aspect_ratio <= 0:
        raise ValueError("Aspect ratio must be positive.")
    return two_dimensional_slope_per_rad / (
        1.0
        + two_dimensional_slope_per_rad
        / (np.pi * oswald_efficiency * aspect_ratio)
    )


def estimate_aerodynamic_center_x(
    design_vector: DesignVector,
    stations: GeometryStations | None = None,
) -> float:
    """Estimate the aircraft aerodynamic center, used as the neutral point."""

    stations = stations or geometry_stations(design_vector)
    wing_span = float(design_vector.wing_span)
    wing_chord = float(design_vector.wing_chord)
    tail_span = float(design_vector.hstab_span)
    tail_chord = float(design_vector.hstab_chord)

    dimensions = {
        "wing_span": wing_span,
        "wing_chord": wing_chord,
        "hstab_span": tail_span,
        "hstab_chord": tail_chord,
    }
    for name, value in dimensions.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    wing_area = wing_span * wing_chord
    tail_area = tail_span * tail_chord
    wing_ar = wing_span**2 / wing_area
    tail_ar = tail_span**2 / tail_area

    if wing_ar <= 2.0:
        raise ValueError(
            "Wing aspect ratio must be greater than 2 for the "
            "formula-sheet horizontal-tail correction."
        )

    wing_slope = 2.0 * np.pi / (1.0 + 2.0 / wing_ar)
    tail_slope = 2.0 * np.pi / (1.0 + 2.0 / tail_ar)
    tail_correction = (wing_ar - 2.0) / (wing_ar + 2.0)

    wing_weight = wing_area * wing_slope
    tail_weight = tail_area * tail_correction * tail_slope
    total_weight = wing_weight + tail_weight
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("Combined aerodynamic-center weight must be positive.")

    aerodynamic_center_x = (
        stations.wing_ac_x_m * wing_weight
        + stations.horizontal_tail_ac_x_m * tail_weight
    ) / total_weight
    return float(aerodynamic_center_x)


def estimate_neutral_point_x(
    design_vector: DesignVector,
    config: NeutralPointConfig | None = None,
    stations: GeometryStations | None = None,
) -> float:
    """Compatibility wrapper; the aircraft aerodynamic center is the neutral point."""

    del config
    return estimate_aerodynamic_center_x(design_vector, stations=stations)


def total_mass(items: Iterable[MassItem]) -> float:
    return float(sum(item.mass_kg for item in items))


def center_of_gravity(items: Iterable[MassItem]) -> np.ndarray:
    items = tuple(items)
    mass = total_mass(items)
    if mass <= 0:
        raise ValueError("At least one positive-mass item is required to compute CG.")
    moment = sum((item.mass_kg * item.position_m for item in items), start=np.zeros(3))
    return moment / mass


def inertia_tensor_about_point(
    items: Iterable[MassItem],
    reference_point_m: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Combine intrinsic tensors with the full 3-D parallel-axis theorem."""
    reference = np.asarray(reference_point_m, dtype=float)
    if reference.shape != (3,):
        raise ValueError("reference_point_m must contain exactly three values.")

    tensor = np.zeros((3, 3), dtype=float)
    identity = np.eye(3)
    for item in items:
        displacement = item.position_m - reference
        parallel_axis = item.mass_kg * (
            np.dot(displacement, displacement) * identity
            - np.outer(displacement, displacement)
        )
        tensor += item.intrinsic_inertia() + parallel_axis

    # Remove numerical asymmetry without changing the physical result.
    return 0.5 * (tensor + tensor.T)


def inertia_tensor_about_cg(items: Iterable[MassItem]) -> tuple[np.ndarray, np.ndarray]:
    items = tuple(items)
    cg = center_of_gravity(items)
    return cg, inertia_tensor_about_point(items, cg)


def static_margin(
    neutral_point_x_m: float,
    cg_x_m: float,
    mean_aerodynamic_chord_m: float,
) -> float:
    if mean_aerodynamic_chord_m <= 0:
        raise ValueError("Mean aerodynamic chord must be positive.")
    # x is positive aft, so a CG forward of the neutral point is positive margin.
    return float((neutral_point_x_m - cg_x_m) / mean_aerodynamic_chord_m)
