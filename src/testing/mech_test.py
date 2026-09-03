"""Executable regression test for the current mechanical mission model.

Run from the repository root with:
    python -m src.testing.mech_test
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from src.mech import MechanicalModuleConfig, evaluate_mechanical_module, mech_main
from src.vectors import ASBDesignVector, DesignVector


INCH_M = 0.0254
POUND_KG = 0.45359237


def _mission_payloads(result, mission: str):
    category = f"mission_{mission[-1]}_payload"
    return tuple(
        item for item in result.for_mission(mission).items if item.category == category
    )


def _assert_mass_properties(result) -> None:
    for mission in ("M1", "M2", "M3"):
        properties = result.for_mission(mission)
        assert properties.total_mass_kg > 0
        assert properties.weight_n > 0
        assert np.asarray(properties.cg_m).shape == (3,)
        assert np.isfinite(properties.static_margin)
        assert properties.inertia_tensor_kg_m2.shape == (3, 3)
        assert np.allclose(
            properties.inertia_tensor_kg_m2,
            properties.inertia_tensor_kg_m2.T,
            atol=1e-12,
        )
        assert np.all(
            np.linalg.eigvalsh(properties.inertia_tensor_kg_m2) >= -1e-10
        )


def _assert_no_payload_overlap(items) -> None:
    for first, second in combinations(items, 2):
        required_separation = 0.5 * (
            first.dimensions_m + second.dimensions_m
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

    assert "extra_shipping_containers" in design.opt_names()
    assert "sensor_weight_kg" not in design.opt_names()
    assert "ducks_num" not in design.opt_names()
    assert "pucks_num" not in design.opt_names()
    assert "banner_length" not in design.opt_names()
    assert DesignVector.bounds()[design.opt_names().index("extra_shipping_containers")] == (
        0,
        10,
    )
    assert np.array_equal(DesignVector.from_array(design.to_array()).to_array(), design.to_array())

    promoted = ASBDesignVector.from_design_vector(design)
    assert promoted.sensor_weight_kg == design.sensor_weight_kg
    assert promoted.extra_shipping_containers == design.extra_shipping_containers
    assert promoted.motor_max_power == design.motor_max_power
    assert promoted.prop_pitch_in == design.prop_pitch_in

    expected_sensor_length = design.sensor_weight_kg / (
        config.sensor.steel_density_kg_m3
        * np.pi
        * (0.5 * config.sensor.diameter_m) ** 2
    )
    assert np.isclose(config.sensor.length_m(design.sensor_weight_kg), expected_sensor_length)
    assert np.isclose(config.sensor.diameter_m, 3.0 * INCH_M)

    m2_payload = _mission_payloads(result, "M2")
    assert len(m2_payload) == 1
    container = m2_payload[0]
    assert container.name == "M2 sensor shipping container"
    assert np.allclose(
        container.dimensions_m,
        (expected_sensor_length + 2.0 * INCH_M, 5.0 * INCH_M, 5.0 * INCH_M),
    )
    assert np.isclose(container.mass_kg, design.sensor_weight_kg + 0.5 * POUND_KG)
    assert np.isclose(
        result.for_mission("M2").static_margin,
        config.mission2.target_static_margin,
        atol=1e-12,
    )

    release_items = tuple(
        item for item in result.all_items if item.category == "release_mechanism"
    )
    assert len(release_items) == 1
    release = release_items[0]
    assert release.missions == frozenset({"M1", "M2", "M3"})
    assert np.isclose(release.mass_kg, design.sensor_weight_kg / 20.0)
    assert np.isclose(release.position_m[0], container.position_m[0])
    assert np.isclose(release.position_m[1], container.position_m[1])
    assert np.isclose(
        release.position_m[2],
        container.position_m[2] + 0.5 * container.dimensions_m[2],
    )

    m3_payload = _mission_payloads(result, "M3")
    assert len(m3_payload) == 1
    sensor = m3_payload[0]
    assert sensor.name == "M3 sensor"
    assert np.isclose(sensor.mass_kg, design.sensor_weight_kg)
    assert np.allclose(
        sensor.dimensions_m,
        (expected_sensor_length, 3.0 * INCH_M, 3.0 * INCH_M),
    )
    assert np.allclose(sensor.position_m, result.for_mission("M1").cg_m)
    assert np.allclose(result.for_mission("M3").cg_m, result.for_mission("M1").cg_m)

    _assert_mass_properties(result)
    adapter_cg, adapter_inertia, adapter_weight = mech_main(design, mission="M3")
    assert np.allclose(adapter_cg, result.for_mission("M3").cg_m)
    assert np.allclose(adapter_inertia, result.for_mission("M3").inertia_tensor_kg_m2)
    assert np.isclose(adapter_weight, result.for_mission("M3").weight_n)

    maximum = evaluate_mechanical_module(
        DesignVector(extra_shipping_containers=10),
        config,
    )
    containers = _mission_payloads(maximum, "M2")
    assert len(containers) == 11
    assert np.isclose(maximum.resolved_fuselage_width_m, 15.0 * INCH_M)
    assert np.isclose(maximum.resolved_fuselage_height_m, 10.0 * INCH_M)
    _assert_no_payload_overlap(containers)

    # Center bay: center/left/right on the lower layer, then the same upper layer.
    assert np.isclose(containers[0].position_m[1], 0.0)
    assert containers[1].position_m[1] < 0.0
    assert containers[2].position_m[1] > 0.0
    assert np.allclose(
        [item.position_m[0] for item in containers[:6]],
        containers[0].position_m[0],
    )
    assert containers[3].position_m[2] > containers[0].position_m[2]

    # After the 3x2 center bay, matching cells alternate aft then forward.
    center_x = containers[0].position_m[0]
    assert containers[6].position_m[0] > center_x
    assert containers[7].position_m[0] < center_x
    assert containers[8].position_m[0] > center_x
    assert containers[9].position_m[0] < center_x
    assert np.isclose(
        containers[6].position_m[0] - center_x,
        center_x - containers[7].position_m[0],
    )

    rounded = evaluate_mechanical_module(
        DesignVector(extra_shipping_containers=2.4),
        config,
    )
    assert len(_mission_payloads(rounded, "M2")) == 3
    assert any("rounded it to 2" in warning for warning in rounded.warnings)

    print(f"Sensor length:          {expected_sensor_length:.4f} m")
    print(f"Default M2 containers: {len(m2_payload)}")
    print(
        "Maximum M2 envelope:   "
        f"{maximum.resolved_fuselage_width_m:.4f} m x "
        f"{maximum.resolved_fuselage_height_m:.4f} m"
    )
    for mission in ("M1", "M2", "M3"):
        properties = result.for_mission(mission)
        print(
            f"{mission}: mass={properties.total_mass_kg:.3f} kg, "
            f"CG={properties.cg_m}, static margin={100 * properties.static_margin:.2f}%"
        )


if __name__ == "__main__":
    main()
