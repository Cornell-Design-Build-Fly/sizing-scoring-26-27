"""Executable regression test for the mechanical module.

Run from the repository root with:
    python -m src.testing.mech_test
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np

from src.mech import (
    ElectronicsPackagingConfig,
    LinearMassModel,
    MechanicalModuleConfig,
    PayloadPlacementError,
    evaluate_mechanical_module,
    mech_main,
    resolve_electronics_layout,
)
from src.mech.mass_properties import geometry_stations
from src.vectors import DesignVector


def _payloads(result):
    return tuple(
        item
        for item in result.for_mission("M2").items
        if item.category == "mission_2_payload"
    )


def _assert_no_payload_overlap(items) -> None:
    payloads = [item for item in items if item.category == "mission_2_payload"]
    for first, second in combinations(payloads, 2):
        required_separation = 0.5 * (
            first.dimensions_m + second.dimensions_m
        )
        overlap_in_every_axis = np.all(
            np.abs(first.position_m - second.position_m)
            < required_separation - 1e-12
        )
        assert not overlap_in_every_axis, f"{first.name} overlaps {second.name}"


def _assert_mass_properties(result) -> None:
    for mission in ("M1", "M2", "M3"):
        properties = result.for_mission(mission)
        assert properties.total_mass_kg > 0
        assert properties.weight_n > 0
        assert properties.cg_m.shape == (3,)
        assert np.isfinite(properties.static_margin)
        assert properties.inertia_tensor_kg_m2.shape == (3, 3)
        assert np.allclose(
            properties.inertia_tensor_kg_m2,
            properties.inertia_tensor_kg_m2.T,
            atol=1e-12,
        )
        assert np.all(np.linalg.eigvalsh(properties.inertia_tensor_kg_m2) >= -1e-10)


def main() -> None:
    design = DesignVector()
    config = MechanicalModuleConfig()
    result = evaluate_mechanical_module(design, config)

    assert set(result.missions) == {"M1", "M2", "M3"}
    assert result.for_mission("M2").total_mass_kg > result.for_mission("M1").total_mass_kg
    assert result.for_mission("M3").total_mass_kg > result.for_mission("M1").total_mass_kg
    _assert_mass_properties(result)
    adapter_cg, adapter_inertia, adapter_weight = mech_main(design, mission="M1")
    assert np.allclose(adapter_cg, result.for_mission("M1").cg_m)
    assert np.allclose(adapter_inertia, result.for_mission("M1").inertia_tensor_kg_m2)
    assert np.isclose(adapter_weight, result.for_mission("M1").weight_n)

    # tail_arm is now wing-LE to tail-LE in both mechanical and aero geometry.
    stations = geometry_stations(design)
    assert np.isclose(stations.wing_le_x_m, 0.0)
    assert np.isclose(stations.horizontal_tail_le_x_m, design.tail_arm)
    assert np.isclose(stations.vertical_tail_le_x_m, design.tail_arm)
    assert np.isclose(
        stations.horizontal_tail_ac_x_m,
        design.tail_arm + 0.25 * design.hstab_chord,
    )

    # The permanent electronics CM gives exact 20% M1 static margin.
    m1 = result.for_mission("M1")
    assert np.isclose(config.static_margin.target, 0.20)
    assert np.isclose(m1.static_margin, 0.20, rtol=0.0, atol=1e-12)
    assert m1.static_margin_feasible
    assert np.isclose(m1.cg_m[0], result.target_cg_x_m, rtol=0.0, atol=1e-12)
    assert np.isclose(result.electronics_position_m[2], -3.0 * 0.0254)
    assert result.electronics_layout.profile == "fat"
    assert np.isclose(result.electronics_layout.length_m, 0.228)
    assert np.isclose(result.electronics_layout.cg_from_front_m, 0.119)
    assert np.isclose(
        result.electronics_layout.back_edge_x_m
        - result.electronics_layout.front_edge_x_m,
        0.228,
    )
    for varied_design in (
        DesignVector(
            wing_span=0.8,
            wing_chord=0.12,
            tail_arm=0.9,
            nose_length=0.08,
        ),
        DesignVector(
            wing_span=1.8,
            wing_chord=0.35,
            tail_arm=0.9,
            nose_length=0.30,
            batt_capacity=9.0,
        ),
    ):
        varied_result = evaluate_mechanical_module(varied_design, config)
        assert np.isclose(
            varied_result.for_mission("M1").static_margin,
            0.20,
            rtol=0.0,
            atol=1e-12,
        )

    packaging = ElectronicsPackagingConfig()
    skinny = resolve_electronics_layout(
        cg_x_m=0.3,
        fuselage_width_m=0.126,
        fuselage_height_m=0.126,
        config=packaging,
    )
    threshold = resolve_electronics_layout(
        cg_x_m=0.3,
        fuselage_width_m=0.127,
        fuselage_height_m=0.126,
        config=packaging,
    )
    assert skinny.profile == "skinny"
    assert np.isclose(skinny.length_m, 0.254)
    assert np.isclose(skinny.cg_from_front_m, 0.135)
    assert threshold.profile == "fat"  # strict "less than 0.127 m" rule

    # The component ledger uses every supplied permanent mass explicitly.
    m1_items = {item.name: item for item in m1.items}
    assert np.isclose(m1_items["Battery"].mass_kg, 0.690)
    assert np.isclose(m1_items["Motor and propeller"].mass_kg, 0.390)
    assert np.isclose(m1_items["ESC"].mass_kg, 0.118)
    assert np.isclose(m1_items["Other electronics"].mass_kg, 0.050)
    assert np.isclose(m1_items["Wing integration"].mass_kg, 0.100)
    assert np.isclose(m1_items["Tail integration"].mass_kg, 0.025)
    assert np.isclose(m1_items["Landing gear"].mass_kg, 0.220)
    assert np.isclose(
        m1_items["Landing gear"].position_m[0], m1.cg_m[0], atol=1e-12
    )
    assert np.isclose(
        m1_items["Landing gear"].position_m[2],
        m1.cg_m[2] - 4.0 * 0.0254,
        atol=1e-12,
    )
    assert np.isclose(
        m1_items["Main wing structure"].mass_kg,
        (0.356 / 0.36258) * design.wing_area,
    )
    assert np.isclose(
        m1_items["Wing spar"].mass_kg,
        (0.202 / 1.18) * design.wing_span,
    )
    expected_boom_length = stations.tail_te_x_m - stations.wing_te_x_m
    assert np.isclose(
        m1_items["Boom spar"].mass_kg,
        (0.202 / 1.18) * expected_boom_length,
    )
    assert any("ahead of the modeled nose tip" in warning for warning in result.warnings)

    # Lightweight linear fits can drive motor and propeller mass independently.
    motor_model = LinearMassModel.from_points(
        500.0, 0.200, 1000.0, 0.400, input_name="motor power [W]"
    )
    propeller_model = LinearMassModel.from_points(
        10.0, 0.040, 20.0, 0.080, input_name="propeller diameter [in]"
    )
    assert np.isclose(motor_model.mass_kg(750.0), 0.300)
    assert np.isclose(propeller_model.mass_kg(15.0), 0.060)
    interpolated_config = replace(
        config,
        airframe=replace(
            config.airframe,
            motor_mass_model=motor_model,
            propeller_mass_model=propeller_model,
            motor_sizing_value=750.0,
            propeller_sizing_value=15.0,
        ),
    )
    interpolated = evaluate_mechanical_module(design, interpolated_config)
    interpolated_items = {
        item.name: item for item in interpolated.for_mission("M1").items
    }
    assert np.isclose(interpolated_items["Motor"].mass_kg, 0.300)
    assert np.isclose(interpolated_items["Propeller"].mass_kg, 0.060)
    assert "Motor and propeller" not in interpolated_items
    assert np.isclose(interpolated.for_mission("M1").static_margin, 0.20, atol=1e-12)

    # The default battery line passes through 4.5 Ah / 690 g and the origin.
    high_capacity = evaluate_mechanical_module(DesignVector(batt_capacity=9.0), config)
    assert np.isclose(
        high_capacity.for_mission("M1").total_mass_kg - m1.total_mass_kg,
        0.690,
    )

    # Mission 2 starts both layers at the actual M1 CG and grows center-out.
    payloads = _payloads(result)
    ducks = [item for item in payloads if item.name.startswith("Duck")]
    pucks = [item for item in payloads if item.name.startswith("Puck")]
    assert len(ducks) == round(design.ducks_num)
    assert len(pucks) == round(design.pucks_num)
    assert np.allclose(ducks[0].position_m[:2], (m1.cg_m[0], 0.0))
    assert np.allclose(pucks[0].position_m[:2], (m1.cg_m[0], 0.0))
    assert np.isclose(ducks[0].position_m[2], -3.0 * 0.0254)
    assert pucks[0].position_m[2] < ducks[0].position_m[2]
    assert np.isclose(ducks[1].position_m[0], m1.cg_m[0] - 0.053)
    assert np.isclose(ducks[2].position_m[0], m1.cg_m[0] + 0.053)
    _assert_no_payload_overlap(result.for_mission("M2").items)

    tail_front_x = min(
        stations.horizontal_tail_le_x_m, stations.vertical_tail_le_x_m
    )
    for payload in payloads:
        assert payload.position_m[0] - 0.5 * payload.dimensions_m[0] >= (
            result.electronics_layout.back_edge_x_m - 1e-12
        )
        assert payload.position_m[0] + 0.5 * payload.dimensions_m[0] <= (
            tail_front_x + 1e-12
        )
        assert payload.position_m[1] - 0.5 * payload.dimensions_m[1] >= (
            -0.5 * design.fuselage_width - 1e-12
        )
        assert payload.position_m[1] + 0.5 * payload.dimensions_m[1] <= (
            0.5 * design.fuselage_width + 1e-12
        )

    # Maximum optimizer counts fit deterministically; a blocked forward side
    # simply causes later cells to continue aft.
    max_design = DesignVector(ducks_num=10, pucks_num=10)
    first_max = evaluate_mechanical_module(max_design, config)
    second_max = evaluate_mechanical_module(max_design, config)
    first_positions = np.asarray([item.position_m for item in _payloads(first_max)])
    second_positions = np.asarray([item.position_m for item in _payloads(second_max)])
    assert np.array_equal(first_positions, second_positions)
    _assert_no_payload_overlap(first_max.for_mission("M2").items)
    assert len(_payloads(first_max)) == 20

    try:
        evaluate_mechanical_module(DesignVector(ducks_num=100, pucks_num=1), config)
    except PayloadPlacementError:
        pass
    else:
        raise AssertionError("An over-capacity M2 load must fail explicitly.")

    # Mission 3 uses explicit fixed distances from the banner center.
    m3_items = {item.name: item for item in result.for_mission("M3").items}
    banner_x = m3_items["M3 banner"].position_m[0]
    assert np.isclose(
        banner_x - m3_items["M3 forward mechanism"].position_m[0],
        config.mission3.forward_mechanism_distance_m,
    )
    assert np.isclose(
        m3_items["M3 aft mechanism"].position_m[0] - banner_x,
        config.mission3.aft_mechanism_distance_m,
    )
    for item_name in (
        "M3 forward mechanism",
        "M3 banner",
        "M3 aft mechanism",
    ):
        item = m3_items[item_name]
        assert item.position_m[0] - 0.5 * item.dimensions_m[0] >= (
            result.electronics_layout.back_edge_x_m - 1e-12
        )
        assert item.position_m[0] + 0.5 * item.dimensions_m[0] <= (
            tail_front_x + 1e-12
        )
    expected_banner_mass = (
        (0.233 / 2.9) * design.banner_length * config.mission3.banner_height_m
    )
    assert np.isclose(m3_items["M3 banner"].mass_kg, expected_banner_mass)

    m2_array = result.component_array("M2")
    assert np.count_nonzero(np.char.startswith(m2_array["name"], "Duck")) == 3
    assert np.count_nonzero(np.char.startswith(m2_array["name"], "Puck")) == 1

    print(f"Neutral point x:       {result.neutral_point_x_m:.4f} m")
    print(f"Electronics CM x:      {result.electronics_layout.cg_x_m:.4f} m")
    print(f"Electronics back edge: {result.electronics_layout.back_edge_x_m:.4f} m")
    for mission in ("M1", "M2", "M3"):
        properties = result.for_mission(mission)
        print(
            f"{mission}: mass={properties.total_mass_kg:.3f} kg, "
            f"CG={properties.cg_m}, static margin={100*properties.static_margin:.2f}%"
        )


if __name__ == "__main__":
    main()
