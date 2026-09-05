"""Mission 2 payload resolution, fuselage-width sizing, and acceptance."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.mech.airframe_assembly import (
    build_local_fuselage_assembly,
    translate_electronics_layout_x,
    translate_mass_items_x,
)
from src.mech.electronics import ElectronicsLayout, resolve_electronics_layout
from src.mech.mass_properties import GeometryStations
from src.mech.mission_properties import calculate_mission_properties
from src.mech.models import (
    MassItem,
    MechanicalModuleConfig,
    MissionMassProperties,
)
from src.mech.payload_placement import (
    PayloadPlacementError,
    summarize_mission2_payload,
)
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
    static_margin_penalty: float = 0.0


@dataclass(frozen=True)
class _Mission2Candidate:
    """Lightweight width candidate used before creating a full mass ledger."""

    fuselage_width_m: float
    width_increases: int
    translation_x_m: float
    fuselage_back_x_m: float
    electronics_cg_x_m: float
    mission1_static_margin: float


def _buffered_static_margin_penalty(
    static_margin: float,
    config: MechanicalModuleConfig,
) -> float:
    """Return a finite 0-10 penalty outside the buffered acceptable SM range."""

    margin_config = config.static_margin
    lower = margin_config.minimum - margin_config.optimizer_penalty_buffer
    upper = margin_config.maximum + margin_config.optimizer_penalty_buffer
    violation = max(lower - static_margin, static_margin - upper, 0.0)
    if violation == 0.0:
        return 0.0
    return min(
        10.0,
        10.0 * np.log2(1.0 + violation / margin_config.optimizer_penalty_scale),
    )


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


def _summarize_candidate(
    *,
    design_vector: DesignVector,
    parameter_vector: ParameterVector,
    config: MechanicalModuleConfig,
    duck_count: int,
    puck_count: int,
    fuselage_width_m: float,
    width_increases: int,
    target_cg_x_m: float,
    neutral_point_x_m: float,
    fixed_mass_kg: float,
    fixed_x_moment_kg_m: float,
) -> _Mission2Candidate:
    """Evaluate one width using scalar packing totals and no mass-item creation."""

    airframe = config.airframe
    packaging = airframe.electronics_packaging
    local_layout = resolve_electronics_layout(
        cg_x_m=packaging.skinny_cg_from_front_m,
        fuselage_width_m=fuselage_width_m,
        fuselage_height_m=design_vector.fuselage_height,
        config=packaging,
        cg_y_m=airframe.electronics_y_m,
    )
    if not np.isclose(local_layout.front_edge_x_m, 0.0, atol=1e-12):
        local_layout = resolve_electronics_layout(
            cg_x_m=local_layout.cg_from_front_m,
            fuselage_width_m=fuselage_width_m,
            fuselage_height_m=design_vector.fuselage_height,
            config=packaging,
            cg_y_m=airframe.electronics_y_m,
        )

    component_masses = airframe.electronics_component_masses_kg(
        design_vector.batt_capacity,
        float(parameter_vector.voltage),
        design_vector.motor_kv,
        design_vector.motor_max_power,
        design_vector.prop_diameter_in,
    )
    electronics_mass = float(sum(mass for _, mass in component_masses))
    if electronics_mass <= 0.0:
        raise ValueError("Permanent electronics mass must be positive.")

    payload = summarize_mission2_payload(
        duck_count=duck_count,
        puck_count=puck_count,
        config=config.mission2,
        electronics_back_x_m=(
            local_layout.back_edge_x_m
            + config.mission2.electronics_aft_clearance_m
        ),
        fuselage_width_m=fuselage_width_m,
        z_bounds_m=(-design_vector.fuselage_height, 0.0),
    )
    local_fuselage_back_x = max(
        local_layout.back_edge_x_m,
        payload.back_edge_x_m
        if payload.back_edge_x_m is not None
        else local_layout.back_edge_x_m,
    )
    fuselage_length = float(
        local_fuselage_back_x - local_layout.front_edge_x_m
    )
    if fuselage_length <= 0.0:
        raise RuntimeError("The locally packed fuselage length must be positive.")
    fuselage_mass = float(
        airframe.fuselage_shell_areal_density_kg_m2
        * fuselage_length
        * 2.0
        * (fuselage_width_m + design_vector.fuselage_height)
    )
    fuselage_center_x = 0.5 * (
        local_layout.front_edge_x_m + local_fuselage_back_x
    )
    local_m1_mass = fuselage_mass + electronics_mass
    local_m1_x_moment = (
        fuselage_mass * fuselage_center_x
        + electronics_mass * local_layout.cg_x_m
    )
    group_mass = local_m1_mass + payload.total_mass_kg
    group_x_moment = local_m1_x_moment + payload.x_moment_kg_m
    translation_x = float(
        (
            target_cg_x_m * (fixed_mass_kg + group_mass)
            - fixed_x_moment_kg_m
            - group_x_moment
        )
        / group_mass
    )
    mission1_cg_x = (
        fixed_x_moment_kg_m
        + local_m1_x_moment
        + translation_x * local_m1_mass
    ) / (fixed_mass_kg + local_m1_mass)
    mission1_static_margin = float(
        (neutral_point_x_m - mission1_cg_x) / design_vector.wing_chord
    )
    return _Mission2Candidate(
        fuselage_width_m=fuselage_width_m,
        width_increases=width_increases,
        translation_x_m=translation_x,
        fuselage_back_x_m=float(local_fuselage_back_x + translation_x),
        electronics_cg_x_m=float(local_layout.cg_x_m + translation_x),
        mission1_static_margin=mission1_static_margin,
    )


def _materialize_candidate(
    *,
    candidate: _Mission2Candidate,
    design_vector: DesignVector,
    parameter_vector: ParameterVector,
    config: MechanicalModuleConfig,
    duck_count: int,
    puck_count: int,
    neutral_point_x_m: float,
    fixed_items: tuple[MassItem, ...],
    target_cg_x_m: float,
    static_margin_penalty: float = 0.0,
) -> Mission2Selection:
    """Build items and inertia once for the candidate selected by scalar checks."""

    local_m1_group, local_m2_payload, local_layout = build_local_fuselage_assembly(
        design_vector,
        battery_nominal_voltage_v=float(parameter_vector.voltage),
        fuselage_width_m=candidate.fuselage_width_m,
        duck_count=duck_count,
        puck_count=puck_count,
        config=config,
    )
    local_m2_group = local_m1_group + local_m2_payload
    group_mass = sum(item.mass_kg for item in local_m2_group)
    group_x_moment = sum(
        item.mass_kg * item.position_m[0] for item in local_m2_group
    )
    fixed_mass = sum(item.mass_kg for item in fixed_items)
    fixed_x_moment = sum(
        item.mass_kg * item.position_m[0] for item in fixed_items
    )
    translation_x = float(
        (
            target_cg_x_m * (fixed_mass + group_mass)
            - fixed_x_moment
            - group_x_moment
        )
        / group_mass
    )
    base_items = fixed_items + translate_mass_items_x(
        local_m1_group, translation_x
    )
    payload_items = translate_mass_items_x(
        local_m2_payload, translation_x
    )
    electronics_layout = translate_electronics_layout_x(
        local_layout, translation_x
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
    if not np.isclose(
        mission1.static_margin,
        candidate.mission1_static_margin,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Lightweight width screening disagreed with the completed M1 "
            "static-margin calculation."
        )
    mission1 = replace(
        mission1,
        static_margin_feasible=(
            mission1.static_margin <= config.static_margin.maximum + 1e-12
        ),
    )
    return Mission2Selection(
        base_items=base_items,
        payload_items=payload_items,
        electronics_layout=electronics_layout,
        mission1=mission1,
        mission2=mission2,
        fuselage_width_m=candidate.fuselage_width_m,
        width_increases=candidate.width_increases,
        target_cg_x_m=target_cg_x_m,
        static_margin_penalty=static_margin_penalty,
    )


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
    last_sm_rejected_candidate: _Mission2Candidate | None = None
    selected_candidate: _Mission2Candidate | None = None
    tail_front_x = float(
        min(stations.horizontal_tail_le_x_m, stations.vertical_tail_le_x_m)
    )
    permitted_fuselage_back_x = (
        tail_front_x - mission2_config.tail_leading_edge_clearance_m
    )

    for width_increases in range(mission2_config.maximum_width_increases + 1):
        fuselage_width = float(
            design_vector.fuselage_width + width_increases * width_increment
        )
        candidate = _summarize_candidate(
            design_vector=design_vector,
            parameter_vector=parameter_vector,
            config=config,
            duck_count=duck_count,
            puck_count=puck_count,
            fuselage_width_m=fuselage_width,
            width_increases=width_increases,
            target_cg_x_m=target_cg_x,
            neutral_point_x_m=neutral_point_x_m,
            fixed_mass_kg=fixed_mass,
            fixed_x_moment_kg_m=fixed_x_moment,
        )
        if candidate.fuselage_back_x_m >= permitted_fuselage_back_x - 1e-12:
            attempt_failures.append(
                f"width {fuselage_width:.4f} m puts the fuselage back at "
                f"x={candidate.fuselage_back_x_m:.4f} m, at or behind the permitted "
                f"tail-front limit x={permitted_fuselage_back_x:.4f} m"
            )
            continue

        electronics_bounds = config.airframe.electronics_x_bounds_m
        if electronics_bounds is not None and not (
            electronics_bounds[0]
            <= candidate.electronics_cg_x_m
            <= electronics_bounds[1]
        ):
            raise ValueError(
                "The exact M2 placement requires electronics CM "
                f"x={candidate.electronics_cg_x_m:.4f} m outside the configured "
                f"bounds [{electronics_bounds[0]:.4f}, "
                f"{electronics_bounds[1]:.4f}] m."
            )

        mission1_is_acceptable = (
            candidate.mission1_static_margin
            <= config.static_margin.maximum + 1e-12
        )
        if not mission1_is_acceptable:
            last_sm_rejected_candidate = candidate
            attempt_failures.append(
                f"width {fuselage_width:.4f} m gives M1 static margin "
                f"{100 * candidate.mission1_static_margin:.2f}%"
            )
            continue

        selected_candidate = candidate
        break

    if selected_candidate is not None:
        return _materialize_candidate(
            candidate=selected_candidate,
            design_vector=design_vector,
            parameter_vector=parameter_vector,
            config=config,
            duck_count=duck_count,
            puck_count=puck_count,
            neutral_point_x_m=neutral_point_x_m,
            fixed_items=fixed_items,
            target_cg_x_m=target_cg_x,
        )

    if last_sm_rejected_candidate is not None:
        rejected_selection = _materialize_candidate(
            candidate=last_sm_rejected_candidate,
            design_vector=design_vector,
            parameter_vector=parameter_vector,
            config=config,
            duck_count=duck_count,
            puck_count=puck_count,
            neutral_point_x_m=neutral_point_x_m,
            fixed_items=fixed_items,
            target_cg_x_m=target_cg_x,
        )
        penalty = _buffered_static_margin_penalty(
            rejected_selection.mission1.static_margin,
            config,
        )
        buffered_lower = (
            config.static_margin.minimum
            - config.static_margin.optimizer_penalty_buffer
        )
        buffered_upper = (
            config.static_margin.maximum
            + config.static_margin.optimizer_penalty_buffer
        )
        warnings.append(
            "No fuselage-width attempt met the ordinary M1 static-margin "
            f"limit; using the last completed placement at "
            f"{100 * rejected_selection.mission1.static_margin:.2f}% "
            f"with optimizer penalty {penalty:.4f}. The penalty-free buffered "
            f"range is {100 * buffered_lower:.2f}% to "
            f"{100 * buffered_upper:.2f}%."
        )
        return replace(
            rejected_selection,
            static_margin_penalty=penalty,
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
