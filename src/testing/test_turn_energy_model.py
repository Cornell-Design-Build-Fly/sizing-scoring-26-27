"""Regression checks for propulsion-aware course turns and energy use."""

from __future__ import annotations

import math

from src.prop.continuous_prop_database import load_default_continuous_prop_database
from src.prop.mission_performance import (
    _propulsion_limited_turn,
    evaluate_mission_propulsion,
)
from src.prop.prop_cruise_values import solve_cruise_samples
from src.prop.prop_helper_functions import make_battery_from_design, make_motor_from_design
from src.vectors import DesignVector, ParameterVector


def _components():
    design = DesignVector(batt_capacity=3.0)
    parameters = ParameterVector()
    database = load_default_continuous_prop_database()
    motor = make_motor_from_design(design, parameters)
    battery = make_battery_from_design(design, parameters)
    current_limit = min(motor.max_current, battery.get_max_current())
    return design, parameters, database, motor, battery, current_limit


def test_required_thrust_mode_uses_less_current_than_maximum_thrust() -> None:
    design, _, database, motor, battery, current_limit = _components()
    maximum = solve_cruise_samples(
        design.prop_diameter_in,
        design.prop_pitch_in,
        (20.0,),
        motor,
        battery,
        current_limit,
        1.0,
        database,
        min_rpm=3000,
        max_rpm=20000,
        rpm_step=100,
    )
    required_thrust = 0.60 * maximum.thrust_samples_n[0]
    sized = solve_cruise_samples(
        design.prop_diameter_in,
        design.prop_pitch_in,
        (20.0,),
        motor,
        battery,
        current_limit,
        1.0,
        database,
        min_rpm=3000,
        max_rpm=20000,
        rpm_step=100,
        minimum_thrust_n=(required_thrust,),
    )
    assert not sized.failed_mask[0]
    assert sized.thrust_samples_n[0] >= required_thrust
    assert sized.selected_current_a[0] < maximum.selected_current_a[0]
    impossible = solve_cruise_samples(
        design.prop_diameter_in,
        design.prop_pitch_in,
        (20.0,),
        motor,
        battery,
        current_limit,
        1.0,
        database,
        min_rpm=3000,
        max_rpm=20000,
        rpm_step=100,
        minimum_thrust_n=(1.0e9,),
    )
    assert impossible.failed_mask[0]
    assert impossible.thrust_samples_n[0] == 0.0


def test_turn_search_respects_lift_structure_and_propulsion_limits() -> None:
    design, parameters, database, motor, battery, current_limit = _components()
    turn = _propulsion_limited_turn(
        design,
        parameters,
        mission=2,
        supported_weight_n=5.0 * parameters.gravity,
        cruise_speed_mps=25.0,
        stall_speed_mps=12.0,
        motor=motor,
        battery=battery,
        current_limit_a=current_limit,
        prop_database=database,
    )
    assert turn.feasible
    assert 1.0 < turn.load_factor <= 2.5
    assert 12.0 < turn.speed_mps <= 25.0
    assert turn.angular_rate_rad_s > 0.0
    assert turn.required_thrust_n > 0.0
    assert turn.battery_power_w > 0.0


def test_heavy_aircraft_turn_becomes_propulsion_limited() -> None:
    design, parameters, database, motor, battery, current_limit = _components()
    turn = _propulsion_limited_turn(
        design,
        parameters,
        mission=2,
        supported_weight_n=10.0 * parameters.gravity,
        cruise_speed_mps=30.0,
        stall_speed_mps=12.0 * math.sqrt(2.0),
        motor=motor,
        battery=battery,
        current_limit_a=current_limit,
        prop_database=database,
    )
    assert turn.feasible
    assert 1.0 < turn.load_factor < 2.5


def test_mission_energy_is_split_between_straights_and_turns() -> None:
    design, parameters, database, motor, battery, current_limit = _components()
    result = evaluate_mission_propulsion(
        design,
        parameters,
        mission=2,
        mass_kg=5.0,
        cruise_speed_mps=25.0,
        stall_speed_mps=12.0,
        lap_time_s=40.0,
        prop_database=database,
    )
    assert math.isclose(
        result.modeled_lap_time_s,
        result.straight_time_per_lap_s + result.turn_time_per_lap_s,
    )
    assert math.isclose(
        result.cruise_energy_wh,
        result.straight_energy_wh + result.turn_energy_wh,
    )
    assert result.straight_energy_wh > 0.0
    assert result.turn_energy_wh > 0.0
    cruise = solve_cruise_samples(
        design.prop_diameter_in,
        design.prop_pitch_in,
        (25.0,),
        motor,
        battery,
        current_limit,
        design.cruise_throttle,
        database,
        min_rpm=3000,
        max_rpm=20000,
        rpm_step=100,
    )
    assert math.isclose(
        result.cruise_power_w,
        cruise.selected_current_a[0] * battery.vnom,
    )
    assert result.cruise_power_w > cruise.selected_power_w[0]


def test_mission_three_energy_uses_completed_integer_laps() -> None:
    design, parameters, database, _, _, _ = _components()
    result = evaluate_mission_propulsion(
        design,
        parameters,
        mission=3,
        mass_kg=5.0,
        cruise_speed_mps=25.0,
        stall_speed_mps=12.0,
        lap_time_s=40.0,
        prop_database=database,
    )
    completed_laps = max(1, int(300.0 // result.modeled_lap_time_s))

    assert math.isclose(
        result.straight_energy_wh,
        result.cruise_power_w
        * completed_laps
        * result.straight_time_per_lap_s
        / 3600.0,
    )
    assert math.isclose(
        result.turn_energy_wh,
        result.turn_power_w
        * completed_laps
        * result.turn_time_per_lap_s
        / 3600.0,
    )


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"PASS {name}")
