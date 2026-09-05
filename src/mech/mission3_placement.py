"""Mission 3 banner-and-mechanism placement."""

from __future__ import annotations

import numpy as np

from src.mech.electronics import ElectronicsLayout
from src.mech.mass_properties import geometry_stations
from src.mech.models import MassItem, MechanicalModuleConfig
from src.vectors import DesignVector


def place_mission3_payload(
    design_vector: DesignVector,
    base_items: tuple[MassItem, ...],
    electronics_layout: ElectronicsLayout,
    neutral_point_x_m: float,
    config: MechanicalModuleConfig,
    warnings: list[str],
) -> tuple[MassItem, ...]:
    """Place the fixed-spacing Mission 3 banner system."""

    mission3 = config.mission3
    stations = geometry_stations(design_vector)
    banner_mass = mission3.resolved_banner_mass_kg(design_vector.banner_length)
    front_mass = mission3.forward_mechanism_mass_kg
    aft_mass = mission3.aft_mechanism_mass_kg
    total_payload_mass = banner_mass + front_mass + aft_mass

    if total_payload_mass <= 0:
        warnings.append(
            "Mission-3 masses are all zero. Set mechanism masses and either a banner "
            "mass, banner areal density, or banner linear density in Mission3Config "
            "before using M3 results."
        )
        return ()

    forward_distance = mission3.forward_mechanism_distance_m
    aft_distance = mission3.aft_mechanism_distance_m
    offset_moment = -front_mass * forward_distance + aft_mass * aft_distance

    if mission3.banner_center_x_m is None:
        base_mass = sum(item.mass_kg for item in base_items)
        base_moment = sum(item.mass_kg * item.position_m[0] for item in base_items)
        target_cg_x = (
            neutral_point_x_m
            - config.static_margin.target * design_vector.wing_chord
        )
        center_x = (
            target_cg_x * (base_mass + total_payload_mass)
            - base_moment
            - offset_moment
        ) / total_payload_mass
    else:
        center_x = mission3.banner_center_x_m

    payload_min_x = electronics_layout.back_edge_x_m
    payload_max_x = min(
        stations.horizontal_tail_le_x_m, stations.vertical_tail_le_x_m
    )
    forward_half_length = 0.5 * mission3.forward_mechanism_dimensions_m[0]
    banner_half_length = 0.5 * mission3.banner_packed_dimensions_m[0]
    aft_half_length = 0.5 * mission3.aft_mechanism_dimensions_m[0]

    physical_center_bounds = (
        max(
            payload_min_x + forward_distance + forward_half_length,
            payload_min_x + banner_half_length,
            payload_min_x - aft_distance + aft_half_length,
        ),
        min(
            payload_max_x + forward_distance - forward_half_length,
            payload_max_x - banner_half_length,
            payload_max_x - aft_distance - aft_half_length,
        ),
    )
    if mission3.banner_center_x_bounds_m is None:
        center_bounds = physical_center_bounds
    else:
        center_bounds = (
            max(
                physical_center_bounds[0],
                mission3.banner_center_x_bounds_m[0],
            ),
            min(
                physical_center_bounds[1],
                mission3.banner_center_x_bounds_m[1],
            ),
        )

    if center_bounds[0] > center_bounds[1]:
        raise ValueError(
            "Mission-3 fixed mechanism distances leave no valid banner-center "
            "range between the electronics and tail."
        )
    clipped_center_x = float(np.clip(center_x, *center_bounds))
    if not np.isclose(clipped_center_x, center_x):
        warnings.append(
            f"Mission-3 banner center x={center_x:.4f} m was outside its allowed "
            f"range and was clipped to x={clipped_center_x:.4f} m."
        )
    center_x = clipped_center_x
    center_y = mission3.banner_center_y_m
    center_z = mission3.banner_center_z_m

    return (
        MassItem(
            name="M3 forward mechanism",
            mass_kg=front_mass,
            position_m=(center_x - forward_distance, center_y, center_z),
            dimensions_m=mission3.forward_mechanism_dimensions_m,
            missions=frozenset({"M3"}),
            category="mission_3_payload",
        ),
        MassItem(
            name="M3 banner",
            mass_kg=banner_mass,
            position_m=(center_x, center_y, center_z),
            dimensions_m=mission3.banner_packed_dimensions_m,
            missions=frozenset({"M3"}),
            category="mission_3_payload",
        ),
        MassItem(
            name="M3 aft mechanism",
            mass_kg=aft_mass,
            position_m=(center_x + aft_distance, center_y, center_z),
            dimensions_m=mission3.aft_mechanism_dimensions_m,
            missions=frozenset({"M3"}),
            category="mission_3_payload",
        ),
    )


__all__ = ["place_mission3_payload"]
