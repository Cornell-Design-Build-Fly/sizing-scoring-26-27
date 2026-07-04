"""Executable regression test for the mechanical module.

Run from the repository root with:
    python -m src.testing.mech_test
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np

from src.mech import (
    MechanicalModuleConfig,
    Mission3Config,
    PlacementRules,
    RelativePayloadRules,
    evaluate_mechanical_module,
)
from src.mech.mass_properties import geometry_stations
from src.vectors import DesignVector


def _assert_no_payload_overlap(items, clearance_m: float) -> None:
    payloads = [item for item in items if item.category == "mission_2_payload"]
    for first, second in combinations(payloads, 2):
        required_separation = (
            0.5 * first.dimensions_m + 0.5 * second.dimensions_m + clearance_m
        )
        overlap_in_every_axis = np.all(
            np.abs(first.position_m - second.position_m)
            < required_separation - 1e-12
        )
        assert not overlap_in_every_axis, f"{first.name} overlaps {second.name}"


def main() -> None:
    design = DesignVector()
    config = MechanicalModuleConfig()
    result = evaluate_mechanical_module(design, config)

    assert set(result.missions) == {"M1", "M2", "M3"}
    assert result.for_mission("M2").total_mass_kg > result.for_mission("M1").total_mass_kg
    assert result.for_mission("M3").total_mass_kg > result.for_mission("M1").total_mass_kg

    # Supplied current-year mass and geometry inputs are the module defaults.
    assert config.mission2.duck.dimensions_m == (0.053, 0.053, 0.053)
    assert np.isclose(config.airframe.tail_linear_density_kg_m, 0.049 / 0.259)
    assert np.isclose(config.airframe.tail_servo_mass_kg, 0.021)
    assert np.isclose(config.mission3.forward_mechanism_mass_kg, 0.100)
    assert np.isclose(config.mission3.aft_mechanism_mass_kg, 0.100)
    assert np.isclose(config.mission3.banner_areal_density_kg_m2, 0.233 / 2.9)

    m1_items = {item.name: item for item in result.for_mission("M1").items}
    assert np.isclose(
        m1_items["Horizontal tail structure"].mass_kg,
        (0.049 / 0.259) * design.hstab_span,
    )
    assert np.isclose(
        m1_items["Vertical tail structure"].mass_kg,
        (0.049 / 0.259) * design.vstab_span,
    )
    assert np.isclose(m1_items["Horizontal tail servo"].mass_kg, 0.021)
    assert np.isclose(m1_items["Vertical tail servo"].mass_kg, 0.021)

    m3_items = {item.name: item for item in result.for_mission("M3").items}
    expected_banner_mass = (
        (0.233 / 2.9) * design.banner_length * config.mission3.banner_height_m
    )
    assert np.isclose(m3_items["M3 banner"].mass_kg, expected_banner_mass)
    assert np.isclose(m3_items["M3 forward mechanism"].mass_kg, 0.100)
    assert np.isclose(m3_items["M3 aft mechanism"].mass_kg, 0.100)

    # Preserve support for callers that already have a linear banner-density
    # model rather than an areal-density model.
    legacy_m3 = Mission3Config(banner_linear_density_kg_m=0.020)
    assert np.isclose(
        legacy_m3.resolved_banner_mass_kg(design.banner_length),
        0.020 * design.banner_length,
    )

    for mission in ("M1", "M2", "M3"):
        properties = result.for_mission(mission)
        assert properties.total_mass_kg > 0
        assert properties.weight_n > 0
        assert properties.cg_m.shape == (3,)
        assert properties.inertia_tensor_kg_m2.shape == (3, 3)
        assert np.allclose(
            properties.inertia_tensor_kg_m2,
            properties.inertia_tensor_kg_m2.T,
            atol=1e-12,
        )
        eigenvalues = np.linalg.eigvalsh(properties.inertia_tensor_kg_m2)
        assert np.all(eigenvalues >= -1e-10)

    m2_array = result.component_array("M2")
    assert np.count_nonzero(np.char.startswith(m2_array["name"], "Duck")) == round(
        design.ducks_num
    )
    assert np.count_nonzero(np.char.startswith(m2_array["name"], "Puck")) == round(
        design.pucks_num
    )
    _assert_no_payload_overlap(result.for_mission("M2").items, config.mission2.clearance_m)
    stations = geometry_stations(design)
    tail_leading_edge_x = min(stations.horizontal_tail_le_x_m, stations.vertical_tail_le_x_m)
    electronics_aft_limit_x = (
        result.electronics_position_m[0] + config.mission2.electronics_aft_clearance_m
    )
    default_payloads = [
        item
        for item in result.for_mission("M2").items
        if item.category == "mission_2_payload"
    ]
    for payload in default_payloads:
        assert payload.position_m[0] - 0.5 * payload.dimensions_m[0] >= (
            electronics_aft_limit_x - 1e-12
        )
        assert payload.position_m[0] + 0.5 * payload.dimensions_m[0] <= (
            tail_leading_edge_x + 1e-12
        )

    # Verify that the full current optimizer count bounds can be packed in the
    # default 0.15 m by 0.15 m payload envelope.
    max_payload_result = evaluate_mechanical_module(
        DesignVector(ducks_num=10, pucks_num=10), config
    )
    max_payload_items = max_payload_result.for_mission("M2").items
    _assert_no_payload_overlap(max_payload_items, config.mission2.clearance_m)
    assert max_payload_result.for_mission("M2").static_margin_feasible

    # Exercise front/aft, above/below, and no-stacking flags explicitly.
    directional_m2 = replace(
        config.mission2,
        duck=replace(
            config.mission2.duck,
            rules=PlacementRules(
                allow_forward=False,
                allow_aft=True,
                allow_above=True,
                allow_below=False,
                allow_stacking=False,
            ),
        ),
        puck=replace(
            config.mission2.puck,
            rules=PlacementRules(
                allow_forward=True,
                allow_aft=False,
                allow_above=False,
                allow_below=True,
                allow_stacking=False,
            ),
        ),
        relative_reference_x_m=0.5 * (electronics_aft_limit_x + tail_leading_edge_x),
    )
    directional_config = replace(config, mission2=directional_m2)
    directional_result = evaluate_mechanical_module(
        DesignVector(ducks_num=1, pucks_num=1), directional_config
    )
    reference_x = directional_config.mission2.relative_reference_x_m
    directional_payloads = [
        item
        for item in directional_result.for_mission("M2").items
        if item.category == "mission_2_payload"
    ]
    duck = next(item for item in directional_payloads if item.name.startswith("Duck"))
    puck = next(item for item in directional_payloads if item.name.startswith("Puck"))
    assert duck.position_m[0] - 0.5 * duck.dimensions_m[0] >= reference_x - 1e-12
    assert duck.position_m[2] - 0.5 * duck.dimensions_m[2] >= -1e-12
    assert puck.position_m[0] + 0.5 * puck.dimensions_m[0] <= reference_x + 1e-12
    assert puck.position_m[2] + 0.5 * puck.dimensions_m[2] <= 1e-12

    # Relative type-to-type rules can be combined: every puck is ahead of and
    # below every duck in this example.
    relative_config = replace(
        config,
        mission2=replace(
            config.mission2,
            relative_payload_rules=RelativePayloadRules(
                pucks_forward_of_ducks=True,
                pucks_below_ducks=True,
            ),
        ),
    )
    relative_result = evaluate_mechanical_module(
        DesignVector(ducks_num=2, pucks_num=2), relative_config
    )
    relative_payloads = [
        item
        for item in relative_result.for_mission("M2").items
        if item.category == "mission_2_payload"
    ]
    ducks = [item for item in relative_payloads if item.name.startswith("Duck")]
    pucks = [item for item in relative_payloads if item.name.startswith("Puck")]
    clearance = relative_config.mission2.clearance_m
    for relative_puck in pucks:
        for relative_duck in ducks:
            assert (
                relative_puck.position_m[0] + 0.5 * relative_puck.dimensions_m[0] + clearance
                <= relative_duck.position_m[0] - 0.5 * relative_duck.dimensions_m[0] + 1e-12
            )
            assert (
                relative_puck.position_m[2] + 0.5 * relative_puck.dimensions_m[2] + clearance
                <= relative_duck.position_m[2] - 0.5 * relative_duck.dimensions_m[2] + 1e-12
            )

    # The default battery line passes through 4.5 Ah / 0.690 kg and the origin.
    high_capacity = evaluate_mechanical_module(
        DesignVector(batt_capacity=9.0), config
    )
    expected_mass_increase = 0.690
    actual_mass_increase = (
        high_capacity.for_mission("M1").total_mass_kg
        - result.for_mission("M1").total_mass_kg
    )
    assert np.isclose(actual_mass_increase, expected_mass_increase)

    print(f"Neutral point x: {result.neutral_point_x_m:.4f} m")
    print(f"Electronics x:    {result.electronics_position_m[0]:.4f} m")
    for mission in ("M1", "M2", "M3"):
        properties = result.for_mission(mission)
        print(
            f"{mission}: mass={properties.total_mass_kg:.3f} kg, "
            f"CG={properties.cg_m}, static margin={100*properties.static_margin:.2f}%"
        )
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
