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
    MechanicalModuleConfig,
    PayloadPlacementError,
    PiecewiseLinearMassModel,
    evaluate_mechanical_module,
    mech_main,
    resolve_electronics_layout,
)
from src.mech.main_mech import _fixed_airframe_items
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
        required_separation = 0.5 * (first.dimensions_m + second.dimensions_m)
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


def _assert_fuselage_envelope(
    result, design: DesignVector, config: MechanicalModuleConfig
) -> None:
    fuselage = next(
        item
        for item in result.for_mission("M1").items
        if item.name == "Fuselage structure"
    )
    payloads = _payloads(result)
    front_edge = fuselage.position_m[0] - 0.5 * fuselage.dimensions_m[0]
    back_edge = fuselage.position_m[0] + 0.5 * fuselage.dimensions_m[0]
    expected_back_edge = max(
        [result.electronics_layout.back_edge_x_m]
        + [
            item.position_m[0] + 0.5 * item.dimensions_m[0]
            for item in payloads
        ]
    )
    assert np.isclose(front_edge, result.electronics_layout.front_edge_x_m)
    assert np.isclose(back_edge, expected_back_edge)
    assert np.isclose(fuselage.dimensions_m[1], result.fuselage_width_m)
    assert np.isclose(fuselage.dimensions_m[2], design.fuselage_height)
    expected_perimeter = 2.0 * (result.fuselage_width_m + design.fuselage_height)
    expected_mass = (
        config.airframe.fuselage_shell_areal_density_kg_m2
        * fuselage.dimensions_m[0]
        * expected_perimeter
    )
    assert np.isclose(fuselage.mass_kg, expected_mass)


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

    # Fixed airframe is built first and contains no fuselage or electronics.
    fixed_items = _fixed_airframe_items(design, config)
    fixed_names = {item.name for item in fixed_items}
    assert "Landing gear" in fixed_names
    assert "Fuselage structure" not in fixed_names
    assert "Battery" not in fixed_names

    # tail_arm remains wing-LE to tail-LE in mechanical geometry.
    stations = geometry_stations(design)
    assert np.isclose(stations.wing_le_x_m, 0.0)
    assert np.isclose(stations.horizontal_tail_le_x_m, design.tail_arm)
    assert np.isclose(stations.vertical_tail_le_x_m, design.tail_arm)
    landing_gear = next(item for item in fixed_items if item.name == "Landing gear")
    assert np.allclose(
        landing_gear.position_m,
        (
            stations.wing_le_x_m,
            0.0,
            -config.airframe.landing_gear_vertical_offset_m,
        ),
    )

    # The loaded fuselage is translated to exactly 12% M2 SM. M1 has only an
    # upper 20% acceptance limit; a value below 10% does not trigger widening.
    m1 = result.for_mission("M1")
    m2 = result.for_mission("M2")
    assert np.isclose(config.static_margin.target, 0.20)
    assert np.isclose(config.mission2.target_static_margin, 0.12)
    assert config.mission2.maximum_width_increases == 4
    assert np.isclose(
        m2.static_margin,
        config.mission2.target_static_margin,
        atol=1e-12,
    )
    assert m1.static_margin <= config.static_margin.maximum + 1e-12
    assert m1.static_margin_feasible
    assert m2.static_margin_feasible
    assert np.isclose(result.electronics_position_m[2], -3.0 * 0.0254)

    # The 76.2 mm start fits the 76.2 mm puck exactly. Small payloads remain at
    # that width because their resulting M1 margin is already below 20%.
    assert np.isclose(design.fuselage_width, 0.0762)
    assert result.fuselage_width_increases == 0
    assert np.isclose(result.fuselage_width_m, design.fuselage_width)
    assert not any("duck-width increase" in warning for warning in result.warnings)

    # Electronics are packed locally first; profile selection still follows
    # the skinny/fat envelope definitions.
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
    assert threshold.profile == "fat"

    # Payload rows begin at the electronics back face and against a sidewall,
    # fill laterally, then advance aft. The installed fuselage must end before
    # the leading edge of both tails.
    payloads = _payloads(result)
    ducks = [item for item in payloads if item.name.startswith("Duck")]
    pucks = [item for item in payloads if item.name.startswith("Puck")]
    assert len(ducks) == round(design.ducks_num)
    assert len(pucks) == round(design.pucks_num)
    half_width = 0.5 * result.fuselage_width_m
    for first in (ducks[0], pucks[0]):
        assert np.isclose(
            first.position_m[0] - 0.5 * first.dimensions_m[0],
            result.electronics_layout.back_edge_x_m,
        )
        assert np.isclose(
            first.position_m[1] - 0.5 * first.dimensions_m[1],
            -half_width,
        )
    assert np.isclose(ducks[0].position_m[1], ducks[1].position_m[1])
    assert np.isclose(
        ducks[1].position_m[0] - ducks[0].position_m[0],
        config.mission2.duck.dimensions_m[0],
    )
    assert pucks[0].position_m[2] < ducks[0].position_m[2]
    for payload in payloads:
        assert payload.position_m[1] - 0.5 * payload.dimensions_m[1] >= -half_width - 1e-12
        assert payload.position_m[1] + 0.5 * payload.dimensions_m[1] <= half_width + 1e-12
    _assert_no_payload_overlap(result.for_mission("M2").items)
    _assert_fuselage_envelope(result, design, config)
    fuselage = next(item for item in m1.items if item.name == "Fuselage structure")
    fuselage_back_x = fuselage.position_m[0] + 0.5 * fuselage.dimensions_m[0]
    tail_front_x = min(
        stations.horizontal_tail_le_x_m, stations.vertical_tail_le_x_m
    )
    assert fuselage_back_x < tail_front_x

    # No-payload M2 is identical to M1 and keeps the initial 76.2 mm width.
    empty = evaluate_mechanical_module(DesignVector(ducks_num=0, pucks_num=0), config)
    assert empty.fuselage_width_increases == 0
    assert np.isclose(empty.fuselage_width_m, 0.0762)
    assert np.isclose(
        empty.for_mission("M2").static_margin,
        empty.for_mission("M1").static_margin,
    )

    # For an otherwise identical empty layout, increasing only fuselage width
    # leaves length unchanged and increases structural mass with perimeter.
    wide_empty_design = DesignVector(
        ducks_num=0,
        pucks_num=0,
        fuselage_width=0.1292,
    )
    wide_empty = evaluate_mechanical_module(wide_empty_design, config)
    empty_fuselage = next(
        item for item in empty.for_mission("M1").items
        if item.name == "Fuselage structure"
    )
    wide_empty_fuselage = next(
        item for item in wide_empty.for_mission("M1").items
        if item.name == "Fuselage structure"
    )
    assert np.isclose(
        wide_empty_fuselage.dimensions_m[0], empty_fuselage.dimensions_m[0]
    )
    assert wide_empty_fuselage.mass_kg > empty_fuselage.mass_kg
    assert np.isclose(
        wide_empty_fuselage.mass_kg / empty_fuselage.mass_kg,
        (wide_empty_design.fuselage_width + wide_empty_design.fuselage_height)
        / (design.fuselage_width + design.fuselage_height),
    )

    # Large payloads widen because the narrower exact-M2 layouts fail an
    # acceptance check. With perimeter-scaled fuselage mass, this case is
    # accepted after two duck-width increases.
    large_design = DesignVector(ducks_num=10, pucks_num=9)
    large = evaluate_mechanical_module(large_design, config)
    assert large.fuselage_width_increases == 2
    assert np.isclose(
        large.fuselage_width_m,
        large_design.fuselage_width + 2 * config.mission2.duck.dimensions_m[1],
    )
    assert np.isclose(
        large.for_mission("M2").static_margin,
        config.mission2.target_static_margin,
        atol=1e-12,
    )
    assert large.for_mission("M1").static_margin <= config.static_margin.maximum

    # Tail overlap is checked before static margins. This narrow, short-tail
    # design overlaps the tail for its first two widths and succeeds at width 2.
    tail_limited_design = DesignVector(tail_arm=0.3, ducks_num=0, pucks_num=4)
    one_retry_config = replace(
        config,
        mission2=replace(config.mission2, maximum_width_increases=1),
    )
    try:
        evaluate_mechanical_module(tail_limited_design, one_retry_config)
    except PayloadPlacementError as exc:
        tail_message = str(exc)
        assert tail_message.count("puts the fuselage back") == 2
        assert "gives M1 static margin" not in tail_message
    else:
        raise AssertionError("Expected the first two widths to overlap the tail.")
    tail_limited = evaluate_mechanical_module(tail_limited_design, config)
    assert tail_limited.fuselage_width_increases == 2
    tail_fuselage = next(
        item
        for item in tail_limited.for_mission("M1").items
        if item.name == "Fuselage structure"
    )
    assert (
        tail_fuselage.position_m[0] + 0.5 * tail_fuselage.dimensions_m[0]
        < tail_limited_design.tail_arm
    )

    # If four duck-width increases still violate an acceptance condition, raise
    # the flag and report every attempted width.
    oversized_design = DesignVector(ducks_num=20, pucks_num=20)
    try:
        evaluate_mechanical_module(oversized_design, config)
    except PayloadPlacementError as exc:
        message = str(exc)
        assert "after 4 permitted width increases" in message
        assert "width 0.2882 m" in message
        assert "puts the fuselage back" in message
        assert "gives M1 static margin" in message
    else:
        raise AssertionError("Expected width-retry exhaustion to raise a flag.")

    # Component masses and lightweight interpolation hooks remain intact.
    m1_items = {item.name: item for item in m1.items}
    assert np.isclose(m1_items["Battery"].mass_kg, 0.690)
    assert np.isclose(m1_items["Motor and propeller"].mass_kg, 0.390)
    motor_model = PiecewiseLinearMassModel.from_points(
        [(500.0, 0.200), (750.0, 0.280), (1000.0, 0.400)],
        input_name="motor power [W]",
    )
    propeller_model = PiecewiseLinearMassModel.from_points(
        [(10.0, 0.040), (15.0, 0.055), (20.0, 0.080)],
        input_name="propeller diameter [in]",
    )
    interpolated_config = replace(
        config,
        airframe=replace(
            config.airframe,
            motor_mass_model=motor_model,
            propeller_mass_model=propeller_model,
            battery_model=PiecewiseLinearMassModel.from_points(
                [(4.0, 0.600), (5.0, 0.720), (6.0, 0.900)],
                input_name="battery capacity [Ah]",
            ),
        ),
    )
    interpolated_design = replace(
        design,
        batt_capacity=5.5,
        motor_max_power=875.0,
        prop_diameter_in=17.5,
    )
    interpolated = evaluate_mechanical_module(
        interpolated_design, interpolated_config
    )
    interpolated_items = {
        item.name: item for item in interpolated.for_mission("M1").items
    }
    assert np.isclose(interpolated_items["Motor"].mass_kg, 0.340)
    assert np.isclose(interpolated_items["Propeller"].mass_kg, 0.0675)
    assert np.isclose(interpolated_items["Battery"].mass_kg, 0.810)
    assert np.array_equal(
        interpolated_items["Motor"].position_m,
        interpolated_items["Propeller"].position_m,
    )
    assert np.array_equal(
        interpolated_items["Motor"].position_m,
        interpolated_items["Battery"].position_m,
    )

    # Mission 3 still uses the prior fixed-distance process after M1/M2 finish.
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

    print(f"Neutral point x:       {result.neutral_point_x_m:.4f} m")
    print(f"Selected fuselage:     {result.fuselage_width_m:.4f} m")
    print(f"Width increases:       {result.fuselage_width_increases}")
    print(f"Electronics CM x:      {result.electronics_layout.cg_x_m:.4f} m")
    for mission in ("M1", "M2", "M3"):
        properties = result.for_mission(mission)
        print(
            f"{mission}: mass={properties.total_mass_kg:.3f} kg, "
            f"CG={properties.cg_m}, static margin={100*properties.static_margin:.2f}%"
        )


if __name__ == "__main__":
    main()
