"""Checks for measured hardware, mission isolation and exact search reuse."""
from dataclasses import replace
import json

import numpy as np
import pytest

from src.aero.drag_model import drag_coefficients, sensor_drag_force
from src.mech.main_mech import evaluate_mechanical_module
from src.motor_search import run
from src.motor_search.catalog import InterpolationOnlyDatabase
from src.prop.continuous_prop_database import load_default_continuous_prop_database
from src.prop.prop_cruise_values import solve_cruise_samples
from src.prop.prop_helper_functions import make_motor_from_design, make_battery_from_design
from src.vectors import ASBDesignVector, DesignVector, ParameterVector


@pytest.fixture(scope="module")
def design():
    run.initialize(json.loads((run.HERE / "airframe.json").read_text()))
    return replace(run.BASE, motor_kv=250, motor_max_power=2100,
                   motor_resistance_ohm=.076, motor_no_load_current_a=1,
                   motor_mass_kg=.305, prop_diameter_in=20, prop_pitch_in=15,
                   mission3_prop_diameter_in=18, mission3_prop_pitch_in=12)


def test_motor_measurements_and_tail_overrides_survive_resolution(design):
    motor = make_motor_from_design(design, ParameterVector())
    assert motor.get_rm() == .076
    assert motor.get_I0() == 1
    promoted = ASBDesignVector.from_design_vector(replace(design))
    assert promoted.hstab_area == pytest.approx(.75 * .249)
    assert promoted.vstab_area == pytest.approx(.24 * .22)
    assert ASBDesignVector.from_design_vector(design, unit_scale=2).hstab_span == 1.5
    mech = evaluate_mechanical_module(design)
    for mission in ("M1", "M2", "M3"):
        assert next(i.mass_kg for i in mech.for_mission(mission).items if i.name == "Motor") == .305


def test_uninstalled_m3_prop_cannot_change_m12_mass_or_cg(design):
    first = evaluate_mechanical_module(design)
    second = evaluate_mechanical_module(replace(design, mission3_prop_diameter_in=10))
    for mission in ("M1", "M2"):
        left, right = first.for_mission(mission), second.for_mission(mission)
        assert left.total_mass_kg == right.total_mass_kg
        np.testing.assert_array_equal(left.cg_m, right.cg_m)
        assert len([i for i in left.items if i.name.startswith("Propeller")]) == 1
    assert first.for_mission("M3").total_mass_kg > second.for_mission("M3").total_mass_kg
    assert len([i for i in first.for_mission("M3").items if i.name.startswith("Propeller")]) == 1


def test_drag_flags_preserve_other_drag(design):
    speed = np.array([20., 30.])
    off = drag_coefficients(design, ParameterVector(), speed, .5, .1)
    on = drag_coefficients(replace(design, fuselage_drag_enabled=True), ParameterVector(), speed, .5, .1)
    np.testing.assert_array_equal(off["body"], 0)
    assert np.all(on["body"] > 0)
    for key in off.keys() - {"body"}:
        np.testing.assert_array_equal(off[key], on[key])
    np.testing.assert_array_equal(sensor_drag_force(design, ParameterVector(), speed), 0)
    assert np.all(sensor_drag_force(replace(design, mission3_sensor_drag_enabled=True), ParameterVector(), speed) > 0)


def test_interpolation_only_preserves_selected_operating_points(design):
    original = load_default_continuous_prop_database()
    fast = InterpolationOnlyDatabase(original.catalog)
    kwargs = dict(diameter_in=20, pitch_in=15, velocities_mps=[.01, 15, 30, 40],
                  motor=make_motor_from_design(design), battery=make_battery_from_design(design),
                  max_current_a=82.5, cruise_throttle=1, min_rpm=3000, max_rpm=20000, rpm_step=100)
    left = solve_cruise_samples(prop_database=original, **kwargs)
    right = solve_cruise_samples(prop_database=fast, **kwargs)
    for name in ("thrust_samples_n", "selected_current_a", "selected_rpm", "failed_mask"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


@pytest.mark.parametrize("m12_key,m3_key", [("20x15E", "14x14E"), ("20x8E", "18x12E"), ("14x14E", "20x15E")])
def test_pair_recombination_matches_full_model(design, m12_key, m3_key):
    props = {p.key: p for p in run.PROPS}
    first, third = props[m12_key], props[m3_key]
    reference = next(p for p in run.PROPS if p.diameter_in == first.diameter_in)
    single = run.evaluate(design, first, first)
    mixed = run.evaluate(design, reference, third)
    mech = evaluate_mechanical_module(design)
    payload = sum(i.mass_kg for i in mech.for_mission("M2").items if i.category == "mission_2_payload")
    combined = run.combine(single, mixed, design, payload)
    actual = run.evaluate(design, first, third)
    for name in ("score", "penalty_total", "propulsion_feasible"):
        assert combined[name] == pytest.approx(actual[name], abs=1e-10)
    for mission in ("M1", "M2", "M3"):
        assert combined[mission].lap_time == pytest.approx(actual[mission].lap_time, abs=1e-10)


def test_all_csv_rows_are_parsed_without_dropping_duplicates():
    motors = run.read_motors(run.DEFAULT_CSV)
    assert len(motors) == 703
    assert len({m["row_index"] for m in motors}) == 703
    assert not any("input_error" in m for m in motors)
    assert motors[0]["motor_mass_kg"] == .421
