"""Regression checks for propulsion-aware course turns and the energy budget."""

from __future__ import annotations

import math

from src.prop.continuous_prop_database import load_default_continuous_prop_database
from src.prop.mission_performance import (
    DEFAULT_PROPULSION_REQUIREMENTS,
    _build_turn_envelope,
    _solve_turn,
    evaluate_mission_propulsion,
)
from src.prop.prop_cruise_values import solve_cruise_samples
from src.prop.prop_helper_functions import make_battery_from_design, make_motor_from_design
from src.vectors import DesignVector, ParameterVector


def _components(**overrides):
    overrides.setdefault("batt_capacity", 3.0)
    design = DesignVector(**overrides)
    parameters = ParameterVector()
    database = load_default_continuous_prop_database()
    motor = make_motor_from_design(design, parameters)
    battery = make_battery_from_design(design, parameters)
    current_limit = min(motor.max_current, battery.get_max_current())
    return design, parameters, database, motor, battery, current_limit


def _turn(
    design,
    parameters,
    database,
    motor,
    battery,
    current_limit,
    *,
    supported_weight_n,
    cruise_speed_mps,
    stall_speed_mps,
    mission=2,
    maximum_battery_power_w=None,
):
    diameter_in, pitch_in = design.propeller_for_mission(mission)
    envelope = _build_turn_envelope(
        design,
        parameters,
        mission=mission,
        supported_weight_n=supported_weight_n,
        maximum_cruise_speed_mps=cruise_speed_mps,
        stall_speed_mps=stall_speed_mps,
        motor=motor,
        battery=battery,
        prop_database=database,
        diameter_in=diameter_in,
        pitch_in=pitch_in,
    )
    return _solve_turn(
        envelope,
        parameters,
        current_limit_a=current_limit,
        throttle_limit=1.0,
        motor_max_power_w=float(motor.max_power),
        battery_vnom_v=float(battery.vnom),
        maximum_battery_power_w=maximum_battery_power_w,
        maximum_speed_mps=cruise_speed_mps,
    )


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


def test_battery_power_cap_removes_operating_points() -> None:
    """A pack-power cap is a real limit, not a reporting field."""

    design, _, database, motor, battery, current_limit = _components()
    uncapped = solve_cruise_samples(
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
    cap = 0.5 * float(uncapped.selected_current_a[0] * battery.vnom)
    capped = solve_cruise_samples(
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
        maximum_battery_power_w=cap,
    )
    assert not capped.failed_mask[0]
    assert capped.selected_current_a[0] * battery.vnom <= cap + 1.0e-9
    assert capped.thrust_samples_n[0] < uncapped.thrust_samples_n[0]


def test_turn_search_respects_lift_structure_and_propulsion_limits() -> None:
    components = _components()
    turn = _turn(
        *components,
        supported_weight_n=5.0 * components[1].gravity,
        cruise_speed_mps=25.0,
        stall_speed_mps=12.0,
    )
    assert turn.feasible
    assert 1.0 < turn.load_factor <= 2.5
    assert 12.0 < turn.speed_mps <= 25.0
    assert turn.angular_rate_rad_s > 0.0
    assert turn.required_thrust_n > 0.0
    assert turn.battery_power_w > 0.0


def test_heavy_aircraft_turn_becomes_propulsion_limited() -> None:
    components = _components()
    turn = _turn(
        *components,
        supported_weight_n=10.0 * components[1].gravity,
        cruise_speed_mps=30.0,
        stall_speed_mps=12.0 * math.sqrt(2.0),
    )
    assert turn.feasible
    assert 1.0 < turn.load_factor < 2.5


def test_turn_under_a_power_cap_is_gentler() -> None:
    components = _components()
    kwargs = dict(
        supported_weight_n=5.0 * components[1].gravity,
        cruise_speed_mps=25.0,
        stall_speed_mps=12.0,
    )
    free = _turn(*components, **kwargs)
    capped = _turn(
        *components,
        **kwargs,
        maximum_battery_power_w=0.6 * free.battery_power_w,
    )
    assert capped.feasible
    assert capped.battery_power_w <= 0.6 * free.battery_power_w + 1.0e-9
    assert capped.load_factor < free.load_factor
    assert capped.angular_rate_rad_s < free.angular_rate_rad_s


def test_mission_energy_is_split_between_straights_and_turns() -> None:
    design, parameters, database, _, _, _ = _components()
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
    # Every phase the model charges must add up to the mission total: ground
    # roll, the flapped acceleration to climb speed, the climb to cruise
    # altitude, the straights, the turns, the turn exits, and the one
    # acceleration to cruise speed after the flaps come up.
    assert math.isclose(
        result.required_energy_wh,
        result.takeoff_energy_wh
        + result.acceleration_energy_wh
        + result.climb_energy_wh
        + result.straight_energy_wh
        + result.turn_energy_wh
        + result.reacceleration_energy_wh
        + result.flap_retraction_energy_wh,
    )
    assert result.acceleration_energy_wh > 0.0
    assert result.climb_energy_wh > 0.0


def test_mission_three_energy_uses_the_usable_window() -> None:
    """Laps stop at the window left after takeoff and landing, not 300 s."""

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
    window = DEFAULT_PROPULSION_REQUIREMENTS.usable_window_s
    assert result.usable_window_s == window
    assert window < 300.0
    assert result.completed_laps == max(1, int(window // result.modeled_lap_time_s))
    assert math.isclose(
        result.straight_energy_wh,
        result.cruise_power_w
        * result.completed_laps
        * result.straight_time_per_lap_s
        / 3600.0,
    )
    assert math.isclose(
        result.turn_energy_wh,
        result.turn_power_w
        * result.completed_laps
        * result.turn_time_per_lap_s
        / 3600.0,
    )


def test_energy_budget_slows_the_aircraft_instead_of_overdrawing_the_pack() -> None:
    """A pack too small for full-throttle cruise buys a slower lap, not a cliff."""

    design, parameters, database, _, _, _ = _components(batt_capacity=2.0)
    result = evaluate_mission_propulsion(
        design,
        parameters,
        mission=3,
        mass_kg=5.0,
        cruise_speed_mps=30.0,
        stall_speed_mps=12.0,
        lap_time_s=40.0,
        prop_database=database,
    )
    assert result.energy_limited
    assert result.required_energy_wh <= result.allowed_energy_wh + 1.0e-9
    assert math.isfinite(result.cruise_power_cap_w)
    # The cap is a real limit on the flown operating point, and the aircraft
    # never flies faster than it can trim.
    assert result.cruise_power_w <= result.cruise_power_cap_w + 1.0e-9
    assert result.cruise_speed_mps <= result.aerodynamic_cruise_speed_mps


def test_weight_costs_speed_through_the_energy_budget() -> None:
    """The MATLAB sizing result: the same watt-hours buy a heavier plane less."""

    design, parameters, database, _, _, _ = _components(batt_capacity=2.0)
    common = dict(
        mission=3,
        cruise_speed_mps=30.0,
        stall_speed_mps=12.0,
        lap_time_s=40.0,
        prop_database=database,
    )
    light = evaluate_mission_propulsion(
        design, parameters, mass_kg=4.0, **common
    )
    heavy = evaluate_mission_propulsion(
        design, parameters, mass_kg=6.0, **common
    )
    assert light.energy_limited and heavy.energy_limited
    # The budget can bite on the straights, on the turns, or both. What must
    # hold either way is that the extra weight buys a slower lap.
    assert heavy.cruise_speed_mps <= light.cruise_speed_mps
    assert heavy.modeled_lap_time_s > light.modeled_lap_time_s
    assert heavy.completed_laps < light.completed_laps
    # Neither is allowed to overdraw the pack to get there.
    assert light.required_energy_wh <= light.allowed_energy_wh + 1.0e-9
    assert heavy.required_energy_wh <= heavy.allowed_energy_wh + 1.0e-9


def test_turn_radius_guard_rail_rejects_off_course_turns() -> None:
    """Without a radius bound the budget buys laps with unflyable wide turns."""

    from dataclasses import replace as _replace

    components = _components()
    kwargs = dict(
        supported_weight_n=5.0 * components[1].gravity,
        cruise_speed_mps=25.0,
        stall_speed_mps=12.0,
    )
    free = _turn(*components, **kwargs)
    radius_m = free.speed_mps**2 / (
        components[1].gravity * math.sqrt(free.load_factor**2 - 1.0)
    )
    assert radius_m <= DEFAULT_PROPULSION_REQUIREMENTS.maximum_turn_radius_m

    design, parameters, database, motor, battery, current_limit = components
    envelope = _build_turn_envelope(
        design,
        parameters,
        mission=2,
        supported_weight_n=kwargs["supported_weight_n"],
        maximum_cruise_speed_mps=kwargs["cruise_speed_mps"],
        stall_speed_mps=kwargs["stall_speed_mps"],
        motor=motor,
        battery=battery,
        prop_database=database,
        diameter_in=design.prop_diameter_in,
        pitch_in=design.prop_pitch_in,
    )
    tight = _solve_turn(
        envelope,
        parameters,
        current_limit_a=current_limit,
        throttle_limit=1.0,
        motor_max_power_w=float(motor.max_power),
        battery_vnom_v=float(battery.vnom),
        maximum_battery_power_w=None,
        maximum_speed_mps=kwargs["cruise_speed_mps"],
        maximum_radius_m=0.5 * radius_m,
    )
    tight_radius = (
        tight.speed_mps**2
        / (parameters.gravity * math.sqrt(tight.load_factor**2 - 1.0))
        if tight.feasible
        else math.inf
    )
    assert not tight.feasible or tight_radius <= 0.5 * radius_m + 1.0e-9

    requirements = _replace(
        DEFAULT_PROPULSION_REQUIREMENTS, maximum_turn_radius_m=None
    )
    assert requirements.maximum_turn_radius_m is None


def test_each_mission_flies_its_own_propeller() -> None:
    design = DesignVector(
        batt_capacity=3.0,
        prop_diameter_in=12.0,
        prop_pitch_in=6.0,
        mission3_prop_diameter_in=18.0,
        mission3_prop_pitch_in=9.0,
    )
    parameters = ParameterVector()
    database = load_default_continuous_prop_database()
    assert design.propeller_for_mission(1) == (12.0, 6.0)
    assert design.propeller_for_mission(2) == (12.0, 6.0)
    assert design.propeller_for_mission(3) == (18.0, 9.0)
    common = dict(
        mass_kg=5.0,
        cruise_speed_mps=25.0,
        stall_speed_mps=12.0,
        lap_time_s=40.0,
        prop_database=database,
    )
    m2 = evaluate_mission_propulsion(design, parameters, mission=2, **common)
    m3 = evaluate_mission_propulsion(design, parameters, mission=3, **common)
    assert (m2.propeller_diameter_in, m2.propeller_pitch_in) == (12.0, 6.0)
    assert (m3.propeller_diameter_in, m3.propeller_pitch_in) == (18.0, 9.0)
    assert m3.static_thrust_n > m2.static_thrust_n


def test_unset_mission_three_propeller_falls_back_to_the_shared_one() -> None:
    """Archived 14-variable design vectors must still replay unchanged."""

    design = DesignVector(prop_diameter_in=15.0, prop_pitch_in=7.5)
    assert design.mission3_prop_diameter_in == 15.0
    assert design.mission3_prop_pitch_in == 7.5
    assert design.propeller_for_mission(3) == design.propeller_for_mission(2)


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"PASS {name}")
