"""High-level mechanical evaluation assembled from focused domain functions."""

from __future__ import annotations

from dataclasses import replace

from src.mech.airframe_assembly import build_fixed_airframe_items
from src.mech.mass_properties import (
    buffered_static_margin_penalty,
    estimate_aerodynamic_center_x,
    geometry_stations,
)
from src.mech.mission2_sizing import select_mission2_fuselage
from src.mech.mission3_placement import place_mission3_payload
from src.mech.mission_properties import calculate_mission_properties
from src.mech.models import MechanicalModuleConfig, MechanicalResult
from src.vectors import DesignVector, ParameterVector


def evaluate_mechanical_design(
    design_vector: DesignVector,
    config: MechanicalModuleConfig | None = None,
    parameter_vector: ParameterVector | None = None,
) -> MechanicalResult:
    """Evaluate M1, M2, and M3 by coordinating the focused mech modules."""

    config = config or MechanicalModuleConfig()
    parameter_vector = parameter_vector or ParameterVector()
    warnings: list[str] = []

    stations = geometry_stations(design_vector)
    neutral_point_x = estimate_aerodynamic_center_x(design_vector, stations)
    fixed_items = build_fixed_airframe_items(design_vector, config)

    selection = select_mission2_fuselage(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
        config=config,
        stations=stations,
        neutral_point_x_m=neutral_point_x,
        fixed_items=fixed_items,
        warnings=warnings,
    )

    mission1 = replace(selection.mission1, warnings=tuple(warnings))
    mission2 = replace(selection.mission2, warnings=tuple(warnings))

    mission3_payload = place_mission3_payload(
        design_vector=design_vector,
        base_items=selection.base_items,
        mission2_payload=selection.payload_items,
        electronics_layout=selection.electronics_layout,
        neutral_point_x_m=neutral_point_x,
        config=config,
        warnings=warnings,
    )
    mission3 = calculate_mission_properties(
        mission="M3",
        items=selection.base_items + mission3_payload,
        design_vector=design_vector,
        neutral_point_x_m=neutral_point_x,
        config=config,
    )
    mission3_warnings = list(warnings)
    if not mission3.static_margin_feasible:
        warning = (
            f"M3 static margin is {100 * mission3.static_margin:.2f}%, "
            "outside the configured range."
        )
        warnings.append(warning)
        mission3_warnings.append(warning)
    mission3 = replace(
        mission3,
        warnings=tuple(dict.fromkeys(mission3_warnings)),
    )

    acceptable_cg_range = (
        neutral_point_x - config.static_margin.maximum * design_vector.wing_chord,
        neutral_point_x - config.static_margin.minimum * design_vector.wing_chord,
    )
    all_items = (
        selection.base_items + selection.payload_items + mission3_payload
    )

    # Static-margin feedback for the optimizer. Before the 2026-27 rules update
    # this was driven solely by M1; the M1 waiver made that inappropriate, but
    # zeroing it removed the only static-margin signal in the whole stack. It is
    # now driven by the missions actually flown for score (see
    # StaticMarginConfig.penalized_missions). The worst offending mission sets
    # the penalty so the total stays on the same 0-10 scale as before rather
    # than summing to 30 and swamping the mission scores.
    evaluated = {"M1": mission1, "M2": mission2, "M3": mission3}
    per_mission_penalties = {
        name: buffered_static_margin_penalty(
            evaluated[name].static_margin, config.static_margin
        )
        for name in config.static_margin.penalized_missions
    }
    static_margin_penalty = max(per_mission_penalties.values(), default=0.0)
    for name, value in sorted(per_mission_penalties.items()):
        if value > 0.0:
            warnings.append(
                f"{name} static margin is "
                f"{100 * evaluated[name].static_margin:.2f}%, outside the "
                f"buffered optimizer band; penalty {value:.3f}."
            )

    return MechanicalResult(
        neutral_point_x_m=neutral_point_x,
        wing_aerodynamic_center_x_m=stations.wing_ac_x_m,
        horizontal_tail_aerodynamic_center_x_m=(
            stations.horizontal_tail_ac_x_m
        ),
        target_cg_x_m=selection.target_cg_x_m,
        acceptable_cg_x_range_m=acceptable_cg_range,
        electronics_position_m=selection.electronics_layout.position_m,
        electronics_layout=selection.electronics_layout,
        input_fuselage_width_m=float(design_vector.fuselage_width),
        input_fuselage_height_m=float(design_vector.fuselage_height),
        resolved_fuselage_width_m=selection.fuselage_width_m,
        resolved_fuselage_height_m=selection.fuselage_height_m,
        fuselage_width_increases=selection.width_increases,
        all_items=all_items,
        missions={"M1": mission1, "M2": mission2, "M3": mission3},
        warnings=tuple(dict.fromkeys(warnings)),
        penalty=selection.placement_penalty + static_margin_penalty,
        penalty_static_margin=static_margin_penalty,
        penalty_placement=selection.placement_penalty,
        penalty_static_margin_by_mission=dict(per_mission_penalties),
    )


__all__ = ["evaluate_mechanical_design"]
