"""Mission-3 sensor placement."""

from __future__ import annotations

import numpy as np

from src.mech.electronics import ElectronicsLayout
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
    """Place the solid cylindrical sensor at the installed airplane CG."""

    del electronics_layout, neutral_point_x_m, warnings
    base_mass = sum(item.mass_kg for item in base_items)
    if base_mass <= 0:
        raise ValueError("Mission-3 base airplane mass must be positive.")
    base_cg = sum(
        (item.mass_kg * item.position_m for item in base_items),
        start=np.zeros(3),
    ) / base_mass
    position = base_cg + np.asarray(config.mission3.center_offset_m, dtype=float)

    sensor_mass = float(design_vector.sensor_weight_kg)
    sensor_length = config.sensor.length_m(sensor_mass)
    radius = 0.5 * config.sensor.diameter_m
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
            dimensions_m=(
                sensor_length,
                config.sensor.diameter_m,
                config.sensor.diameter_m,
            ),
            missions=frozenset({"M3"}),
            category="mission_3_payload",
            intrinsic_inertia_kg_m2=intrinsic_inertia,
            notes=(
                "Solid 3-inch-diameter steel rod placed at the installed M1 "
                "airplane CG, with exact cylindrical intrinsic inertia."
            ),
        ),
    )


__all__ = ["place_mission3_payload"]
