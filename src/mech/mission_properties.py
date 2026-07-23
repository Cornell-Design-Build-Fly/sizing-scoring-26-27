"""Mission-level mass, CG, inertia, and static-margin calculations."""

from __future__ import annotations

from src.mech.mass_properties import inertia_tensor_about_cg, static_margin
from src.mech.models import (
    MassItem,
    MechanicalModuleConfig,
    MissionMassProperties,
)
from src.vectors import DesignVector, ParameterVector


def calculate_mission_properties(
    *,
    mission: str,
    items: tuple[MassItem, ...],
    design_vector: DesignVector,
    neutral_point_x_m: float,
    config: MechanicalModuleConfig,
    placement_feasible: bool = True,
    warnings: tuple[str, ...] = (),
) -> MissionMassProperties:
    """Calculate the complete mass-properties record for one mission."""

    cg, inertia = inertia_tensor_about_cg(items)
    mass = float(sum(item.mass_kg for item in items))
    margin = static_margin(neutral_point_x_m, cg[0], design_vector.wing_chord)
    margin_config = config.static_margin
    tolerance = 1e-12
    return MissionMassProperties(
        mission=mission,
        items=items,
        total_mass_kg=mass,
        weight_n=mass * ParameterVector.gravity,
        cg_m=cg,
        inertia_tensor_kg_m2=inertia,
        static_margin=margin,
        static_margin_feasible=(
            margin_config.minimum - tolerance
            <= margin
            <= margin_config.maximum + tolerance
        ),
        placement_feasible=placement_feasible,
        warnings=warnings,
    )


__all__ = ["calculate_mission_properties"]
