"""Mission-2 fuselage construction, installation, and acceptance."""

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
from src.mech.models import MassItem, MechanicalModuleConfig, MissionMassProperties
from src.mech.payload_placement import (
    PayloadPlacementError,
    resolve_extra_container_count,
)
from src.vectors import DesignVector, ParameterVector

# Keeps the clamped fuselage strictly ahead of the tail leading edge rather than
# exactly touching it, so downstream geometry builders that require a strict
# inequality (ASBDesignVector.make_fuselage) still succeed.
PLACEMENT_EPSILON_M = 1.0e-6


@dataclass(frozen=True)
class Mission2Selection:
    """Installed fuselage assembly and its M1/M2 mass properties."""

    base_items: tuple[MassItem, ...]
    payload_items: tuple[MassItem, ...]
    electronics_layout: ElectronicsLayout
    mission1: MissionMassProperties
    mission2: MissionMassProperties
    fuselage_width_m: float
    fuselage_height_m: float
    width_increases: int
    target_cg_x_m: float
    placement_penalty: float = 0.0


def select_mission2_fuselage(
    design_vector: DesignVector,
    parameter_vector: ParameterVector,
    config: MechanicalModuleConfig,
    stations: GeometryStations,
    neutral_point_x_m: float,
    fixed_items: tuple[MassItem, ...],
    warnings: list[str],
) -> Mission2Selection:
    """Build the local M2 fuselage, then install it at the target static margin."""

    extra_count = resolve_extra_container_count(
        design_vector.extra_shipping_containers,
        config.mission2,
        warnings,
    )
    total_count = 1 + extra_count
    local_base, local_payload, local_layout = build_local_fuselage_assembly(
        design_vector,
        battery_nominal_voltage_v=design_vector.battery_nominal_voltage_v,
        total_container_count=total_count,
        config=config,
    )

    target_cg_x = (
        neutral_point_x_m
        - config.mission2.target_static_margin * design_vector.wing_chord
    )
    local_group = local_base + local_payload
    group_mass = sum(item.mass_kg for item in local_group)
    group_x_moment = sum(item.mass_kg * item.position_m[0] for item in local_group)
    fixed_mass = sum(item.mass_kg for item in fixed_items)
    fixed_x_moment = sum(item.mass_kg * item.position_m[0] for item in fixed_items)
    translation_x = float(
        (
            target_cg_x * (fixed_mass + group_mass)
            - fixed_x_moment
            - group_x_moment
        )
        / group_mass
    )

    # Clamp the placement so a block that would reach the tail is still scored,
    # with a penalty proportional to the static-margin error it is forced into,
    # instead of being rejected outright. A hard rejection here turned a smooth
    # trade into a cliff: the optimizer saw BAD_OBJECTIVE with no direction back
    # toward feasibility, and neighbouring container counts alternated between
    # accepted and rejected.
    local_fuselage = next(
        item for item in local_base if item.name == "Fuselage structure"
    )
    local_back_x = local_fuselage.position_m[0] + 0.5 * local_fuselage.dimensions_m[0]
    permitted_back_x = min(
        stations.horizontal_tail_le_x_m,
        stations.vertical_tail_le_x_m,
    ) - config.mission2.tail_leading_edge_clearance_m
    maximum_translation_x = permitted_back_x - local_back_x - PLACEMENT_EPSILON_M
    requested_translation_x = translation_x
    translation_x = min(translation_x, maximum_translation_x)
    placement_clamped = translation_x < requested_translation_x - 1e-12

    installed_base = fixed_items + translate_mass_items_x(local_base, translation_x)
    installed_payload = translate_mass_items_x(local_payload, translation_x)
    electronics_layout = translate_electronics_layout_x(local_layout, translation_x)
    fuselage = next(item for item in installed_base if item.name == "Fuselage structure")
    fuselage_back_x = fuselage.position_m[0] + 0.5 * fuselage.dimensions_m[0]
    if fuselage_back_x >= permitted_back_x:
        # The block is longer than the space between the nose and the tail, so
        # no translation can fit it. This is a genuine geometric impossibility
        # rather than a trim shortfall.
        raise PayloadPlacementError(
            "The Mission-2 fuselage cannot fit ahead of the tail at any "
            f"placement: back edge x={fuselage_back_x:.4f} m, permitted limit "
            f"x={permitted_back_x:.4f} m."
        )

    mission2 = calculate_mission_properties(
        mission="M2",
        items=installed_base + installed_payload,
        design_vector=design_vector,
        neutral_point_x_m=neutral_point_x_m,
        config=config,
    )
    if not placement_clamped and not np.isclose(
        mission2.static_margin,
        config.mission2.target_static_margin,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Installing the completed fuselage did not achieve the exact "
            "Mission-2 static-margin target."
        )
    mission2 = replace(
        mission2,
        static_margin_feasible=(
            config.static_margin.minimum - 1e-12
            <= mission2.static_margin
            <= config.static_margin.maximum + 1e-12
        ),
    )

    mission1 = calculate_mission_properties(
        mission="M1",
        items=installed_base,
        design_vector=design_vector,
        neutral_point_x_m=neutral_point_x_m,
        config=config,
    )
    mission1 = replace(
        mission1,
        static_margin_feasible=(
            mission1.static_margin <= config.static_margin.maximum + 1e-12
        ),
    )
    # A successful M2 flight also satisfies M1 under the rules, so M1's
    # payload-free static margin is diagnostic here. The static-margin penalty
    # itself is applied centrally in mechanical_evaluation across the missions
    # named by StaticMarginConfig.penalized_missions.
    if not mission1.static_margin_feasible:
        warnings.append(
            f"M1 static margin is {100 * mission1.static_margin:.2f}%, above "
            f"the configured {100 * config.static_margin.maximum:.2f}% limit; "
            "no penalty is applied because a successful M2 flight satisfies M1."
        )

    # Penalize how far the clamped placement missed its static-margin target,
    # on the same log scale used for the static-margin band itself.
    penalty = 0.0
    if placement_clamped:
        target_error = abs(
            mission2.static_margin - config.mission2.target_static_margin
        )
        penalty = float(
            min(
                10.0,
                10.0
                * np.log2(
                    1.0
                    + target_error / config.static_margin.optimizer_penalty_scale
                ),
            )
        )
        warnings.append(
            "The Mission-2 fuselage placement was clamped to keep the body "
            f"ahead of the tail: requested translation "
            f"{requested_translation_x:.4f} m, applied {translation_x:.4f} m. "
            f"Static margin is {100 * mission2.static_margin:.2f}% against a "
            f"{100 * config.mission2.target_static_margin:.2f}% target; "
            f"penalty {penalty:.3f}."
        )

    return Mission2Selection(
        base_items=installed_base,
        payload_items=installed_payload,
        electronics_layout=electronics_layout,
        mission1=mission1,
        mission2=mission2,
        fuselage_width_m=float(fuselage.dimensions_m[1]),
        fuselage_height_m=float(fuselage.dimensions_m[2]),
        width_increases=0,
        target_cg_x_m=target_cg_x,
        placement_penalty=penalty,
    )


__all__ = ["Mission2Selection", "select_mission2_fuselage"]
