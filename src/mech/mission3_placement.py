"""Mission-3 sensor placement."""

from __future__ import annotations

import numpy as np

from src.mech.electronics import ElectronicsLayout
from src.mech.models import MassItem, MechanicalModuleConfig
from src.vectors import DesignVector


def place_mission3_payload(
    design_vector: DesignVector,
    base_items: tuple[MassItem, ...],
    mission2_payload: tuple[MassItem, ...],
    electronics_layout: ElectronicsLayout,
    neutral_point_x_m: float,
    config: MechanicalModuleConfig,
    warnings: list[str],
) -> tuple[MassItem, ...]:
    """Place the sensor inside the former primary-container volume."""

    del base_items, electronics_layout, neutral_point_x_m, warnings
    if not mission2_payload:
        raise ValueError("Mission 3 requires the primary M2 container location.")
    primary_container = mission2_payload[0]
    position = primary_container.position_m + np.asarray(
        config.mission3.center_offset_m,
        dtype=float,
    )

    sensor_mass = float(design_vector.sensor_weight_kg)
    sensor_length = config.sensor.length_m(sensor_mass)
    radius = 0.5 * config.sensor.diameter_m
    sensor_dimensions = np.asarray(
        (sensor_length, config.sensor.diameter_m, config.sensor.diameter_m),
        dtype=float,
    )
    center_delta = np.abs(position - primary_container.position_m)
    if np.any(
        center_delta + 0.5 * sensor_dimensions
        > 0.5 * primary_container.dimensions_m + 1e-12
    ):
        raise ValueError(
            "Mission3Config.center_offset_m places the sensor outside the "
            "former primary-container volume."
        )
    axial_inertia = 0.5 * sensor_mass * radius**2
    transverse_inertia = (
        sensor_mass * (3.0 * radius**2 + sensor_length**2) / 12.0
    )
    intrinsic_inertia = np.diag(
        (axial_inertia, transverse_inertia, transverse_inertia)
    )

    return (
        MassItem(
            name="M3 sensor",
            mass_kg=sensor_mass,
            position_m=position,
            dimensions_m=sensor_dimensions,
            missions=frozenset({"M3"}),
            category="mission_3_payload",
            intrinsic_inertia_kg_m2=intrinsic_inertia,
            notes=(
                "Solid 3-inch-diameter steel rod centered inside the volume "
                "formerly occupied by the primary M2 sensor container, with "
                "exact cylindrical intrinsic inertia."
            ),
        ),
    )


__all__ = ["place_mission3_payload"]
