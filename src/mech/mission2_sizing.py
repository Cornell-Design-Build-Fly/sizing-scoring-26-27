"""Mission 2 payload resolution, fuselage-width sizing, and acceptance."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.mech.airframe_assembly import (
    build_local_fuselage_assembly,
    translate_electronics_layout_x,
    translate_mass_items_x,
)
from src.mech.electronics import ElectronicsLayout
from src.mech.mass_properties import GeometryStations
from src.mech.mission_properties import calculate_mission_properties
from src.mech.models import (
    MassItem,
    MechanicalModuleConfig,
    MissionMassProperties,
)
from src.mech.payload_placement import PayloadPlacementError
from src.vectors import DesignVector, ParameterVector


@dataclass(frozen=True)
class Mission2Selection:
    """Accepted fuselage assembly and its M1/M2 mass properties."""

    base_items: tuple[MassItem, ...]
    payload_items: tuple[MassItem, ...]
    electronics_layout: ElectronicsLayout
    mission1: MissionMassProperties
    mission2: MissionMassProperties
    fuselage_width_m: float
    width_increases: int
    target_cg_x_m: float


def resolve_payload_count(value: float, name: str, warnings: list[str]) -> int:
    """Validate and round a discrete Mission 2 payload count."""

    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    rounded = int(round(float(value)))
    if not np.isclose(value, rounded, atol=1e-9):
        warnings.append(
            f"{name}={value:.6g} is not an integer; the mechanical module rounded it "
            f"to {rounded}. The optimizer should eventually treat payload counts as integers."
        )
    return rounded


def select_mission2_fuselage(
    design_vector: DesignVector,
    parameter_vector: ParameterVector,
    config: MechanicalModuleConfig,
    stations: GeometryStations,
    neutral_point_x_m: float,
    fixed_items: tuple[MassItem, ...],
    warnings: list[str],
) -> Mission2Selection:
    """Try permitted fuselage widths and return the first accepted assembly."""

    duck_count = resolve_payload_count(
        design_vector.ducks_num, "ducks_num", warnings
    )
    puck_count = resolve_payload_count(
        design_vector.pucks_num, "pucks_num", warnings
    )
    target_cg_x = (
        neutral_point_x_m
        - config.mission2.target_static_margin * design_vector.wing_chord
    )
    fixed_mass = sum(item.mass_kg for item in fixed_items)
    fixed_x_moment = sum(item.mass_kg * item.position_m[0] for item in fixed_items)

    mission2_config = config.mission2
    width_increment = mission2_config.duck.dimensions_m[1]
    attempt_failures: list[str] = []

    for width_increases in range(mission2_config.maximum_width_increases + 1):
        fuselage_width = float(
            design_vector.fuselage_width + width_increases * width_increment
        )
        local_m1_group, local_m2_payload, local_layout = (
            build_local_fuselage_assembly(
                design_vector,
                battery_nominal_voltage_v=float(parameter_vector.voltage),
                fuselage_width_m=fuselage_width,
                duck_count=duck_count,
                puck_count=puck_count,
                config=config,
            )
        )

        local_m2_group = local_m1_group + local_m2_payload
        group_mass = sum(item.mass_kg for item in local_m2_group)
        group_x_moment = sum(
            item.mass_kg * item.position_m[0] for item in local_m2_group
        )
        translation_x = float(
            (
                target_cg_x * (fixed_mass + group_mass)
                - fixed_x_moment
                - group_x_moment
            )
            / group_mass
        )
        base_items = fixed_items + translate_mass_items_x(
            local_m1_group, translation_x
        )
        payload_items = translate_mass_items_x(local_m2_payload, translation_x)
        electronics_layout = translate_electronics_layout_x(
            local_layout, translation_x
        )

        fuselage = next(
            item for item in base_items if item.name == "Fuselage structure"
        )
        fuselage_back_x = float(
            fuselage.position_m[0] + 0.5 * fuselage.dimensions_m[0]
        )
        tail_front_x = float(
            min(stations.horizontal_tail_le_x_m, stations.vertical_tail_le_x_m)
        )
        permitted_fuselage_back_x = (
            tail_front_x - mission2_config.tail_leading_edge_clearance_m
        )
        if fuselage_back_x >= permitted_fuselage_back_x - 1e-12:
            attempt_failures.append(
                f"width {fuselage_width:.4f} m puts the fuselage back at "
                f"x={fuselage_back_x:.4f} m, at or behind the permitted "
                f"tail-front limit x={permitted_fuselage_back_x:.4f} m"
            )
            continue

        electronics_bounds = config.airframe.electronics_x_bounds_m
        if electronics_bounds is not None and not (
            electronics_bounds[0]
            <= electronics_layout.cg_x_m
            <= electronics_bounds[1]
        ):
            raise ValueError(
                "The exact M2 placement requires electronics CM "
                f"x={electronics_layout.cg_x_m:.4f} m outside the configured "
                f"bounds [{electronics_bounds[0]:.4f}, "
                f"{electronics_bounds[1]:.4f}] m."
            )

        mission2 = calculate_mission_properties(
            mission="M2",
            items=base_items + payload_items,
            design_vector=design_vector,
            neutral_point_x_m=neutral_point_x_m,
            config=config,
        )
        if not np.isclose(
            mission2.static_margin,
            config.mission2.target_static_margin,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Translating the completed fuselage did not achieve the exact "
                "Mission-2 static-margin target."
            )
        mission2 = replace(mission2, static_margin_feasible=True)

        mission1 = calculate_mission_properties(
            mission="M1",
            items=base_items,
            design_vector=design_vector,
            neutral_point_x_m=neutral_point_x_m,
            config=config,
        )
        mission1_is_acceptable = (
            mission1.static_margin <= config.static_margin.maximum + 1e-12
        )
        mission1 = replace(
            mission1, static_margin_feasible=mission1_is_acceptable
        )
        if not mission1_is_acceptable:
            attempt_failures.append(
                f"width {fuselage_width:.4f} m gives M1 static margin "
                f"{100 * mission1.static_margin:.2f}%"
            )
            continue

        return Mission2Selection(
            base_items=base_items,
            payload_items=payload_items,
            electronics_layout=electronics_layout,
            mission1=mission1,
            mission2=mission2,
            fuselage_width_m=fuselage_width,
            width_increases=width_increases,
            target_cg_x_m=target_cg_x,
        )

    detail = "; ".join(attempt_failures)
    raise PayloadPlacementError(
        "No fuselage width kept the fuselage ahead of the tail, produced "
        "exact M2 static margin, and kept M1 at or below "
        f"{100 * config.static_margin.maximum:.1f}% after "
        f"{mission2_config.maximum_width_increases} permitted width increases. "
        f"Attempts: {detail}"
    )


__all__ = [
    "Mission2Selection",
    "resolve_payload_count",
    "select_mission2_fuselage",
]
