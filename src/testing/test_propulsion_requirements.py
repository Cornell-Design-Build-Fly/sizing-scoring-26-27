import math

from src.prop.mission_performance import (
    MissionPropulsionPerformance,
    PROPULSION_INFEASIBLE_BASE_PENALTY,
    PropulsionRequirements,
    _penalty_and_limit,
    propulsion_margin_bonus,
)
from src.prop.prop_helper_functions import make_battery_from_design
from src.vectors import DesignVector, ParameterVector


def _performance(**overrides) -> MissionPropulsionPerformance:
    values = {
        "mission": 2,
        "feasible": True,
        "penalty": 0.0,
        "inertial_mass_kg": 10.0,
        "supported_mass_kg": 10.0,
        "operating_points_feasible": True,
        "static_thrust_n": 100.0,
        "static_thrust_to_weight": 0.5,
        "liftoff_speed_mps": 15.0,
        "takeoff_distance_m": 30.0,
        "optimistic_takeoff_distance_lower_bound_m": 20.0,
        "takeoff_screened_early": False,
        "takeoff_time_s": 3.0,
        "acceleration_distance_m": 25.0,
        "acceleration_time_s": 1.4,
        "acceleration_energy_wh": 0.4,
        "climb_speed_mps": 16.0,
        "climb_rate_mps": 3.0,
        "climb_gradient": 0.50,
        "climb_distance_required_m": 121.9,
        "climb_distance_allowed_m": 152.4,
        "cruise_altitude_m": 60.96,
        "climb_time_s": 20.3,
        "flap_retraction_energy_wh": 1.1,
        "maximum_propeller_tip_mach": 0.6,
        "maximum_propeller_rpm": 10_000.0,
        "manufacturer_propeller_rpm_limit": 10_714.0,
        "operating_propeller_rpm_limit": 9_642.6,
        "propeller_rpm_margin": -357.4,
        "maximum_propeller_shaft_power_w": 1_500.0,
        "maximum_propeller_disk_power_loading_w_m2": 15_000.0,
        "cruise_power_w": 700.0,
        "takeoff_energy_wh": 2.0,
        "climb_energy_wh": 3.0,
        "reacceleration_energy_wh": 4.0,
        "cruise_energy_wh": 50.0,
        "required_energy_wh": 59.0,
        "allowed_energy_wh": 80.0,
        "energy_margin_wh": 21.0,
        "takeoff_distance_margin_m": 30.0,
        "climb_rate_margin_mps": 1.0,
        "climb_distance_margin_m": 30.5,
        "limiting_constraint": "takeoff_distance",
        "aerodynamic_lap_time_s": 30.0,
        "modeled_lap_time_s": 32.0,
        "straight_time_per_lap_s": 20.0,
        "turn_time_per_lap_s": 12.0,
        "turn_speed_mps": 18.0,
        "turn_load_factor": 2.0,
        "turn_power_w": 900.0,
        "straight_energy_wh": 30.0,
        "turn_energy_wh": 20.0,
        "propeller_key": "14x10E",
        "propeller_blade_count": 2,
        "propeller_diameter_in": 14.0,
        "propeller_pitch_in": 10.0,
        "aerodynamic_cruise_speed_mps": 30.0,
        "cruise_speed_mps": 30.0,
        "cruise_power_cap_w": float("inf"),
        "energy_limited": False,
        "completed_laps": 5,
        "mission_flight_time_s": 160.0,
        "usable_window_s": 280.0,
        "clean_stall_speed_mps": 12.5,
        "takeoff_stall_speed_mps": 11.3,
        "landing_stall_speed_mps": 10.8,
        "takeoff_flap_deflection_deg": 25.0,
        "landing_flap_deflection_deg": 40.0,
    }
    values.update(overrides)
    return MissionPropulsionPerformance(**values)


def test_battery_current_limit_uses_capacity_and_c_rating() -> None:
    battery = make_battery_from_design(DesignVector(batt_capacity=3.0), ParameterVector())
    assert battery.Crat == 25.0
    assert battery.get_max_current() == 75.0


def test_propulsion_penalty_identifies_takeoff_climb_and_energy_failures() -> None:
    requirements = PropulsionRequirements()
    assert requirements.maximum_takeoff_distance_m == 75.0
    assert _penalty_and_limit(60.0, 2.0, 100.0, 152.4, 70.0, 80.0, 0.7, requirements)[0] == 0.0
    # Keep a finite runway gate, but do not invent a minimum pattern altitude.
    assert _penalty_and_limit(60.0, 2.0, 400.0, 152.4, 70.0, 80.0, 0.7, requirements)[0] == 0.0
    trade_study_requirements = PropulsionRequirements(
        maximum_takeoff_distance_m=60.0,
        cruise_altitude_m=60.96,
        climb_distance_m=152.4,
    )
    takeoff_penalty, takeoff_limit = _penalty_and_limit(
        120.0, 2.0, 100.0, 152.4, 70.0, 80.0, 0.7, trade_study_requirements
    )
    assert takeoff_penalty > PROPULSION_INFEASIBLE_BASE_PENALTY
    assert takeoff_limit == "takeoff_distance"
    assert _penalty_and_limit(60.0, 0.0, 100.0, 152.4, 70.0, 80.0, 0.7, requirements)[1] == "climb_rate"
    assert _penalty_and_limit(60.0, 2.0, 100.0, 152.4, 160.0, 80.0, 0.7, requirements)[1] == "mission_energy"
    assert _penalty_and_limit(
        60.0,
        2.0,
        100.0,
        152.4,
        70.0,
        80.0,
        0.7,
        requirements,
        operating_point_failed=True,
    )[1] == "propulsion_operating_point"

    rpm_penalty, rpm_limit = _penalty_and_limit(
        60.0,
        2.0,
        100.0,
        152.4,
        70.0,
        80.0,
        0.5,
        requirements,
        propeller_rpm=13_000.0,
        propeller_rpm_limit=12_000.0,
    )
    assert rpm_penalty > PROPULSION_INFEASIBLE_BASE_PENALTY
    assert rpm_limit == "propeller_rpm"


def test_optional_climb_to_pattern_altitude_trade_study() -> None:
    requirements = PropulsionRequirements(
        cruise_altitude_m=60.96,
        climb_distance_m=152.4,
    )
    assert requirements.cruise_altitude_m == 60.96      # 200 ft
    assert requirements.climb_distance_m == 152.4      # 500 ft
    assert abs(requirements.required_climb_gradient - 0.4) < 1e-12

    # Needs more than the available 500 ft -> flagged as the limiting constraint.
    penalty, limit = _penalty_and_limit(
        60.0, 2.0, 400.0, 152.4, 70.0, 80.0, 0.7, requirements
    )
    assert penalty > PROPULSION_INFEASIBLE_BASE_PENALTY
    assert limit == "climb_to_pattern_altitude"

    # Comfortably inside 500 ft -> no penalty at all.
    assert _penalty_and_limit(
        60.0, 2.0, 100.0, 152.4, 70.0, 80.0, 0.7, requirements
    )[0] == 0.0

    # Worse climb performance must never score better.
    penalties = [
        _penalty_and_limit(60.0, 2.0, d, 152.4, 70.0, 80.0, 0.7, requirements)[0]
        for d in (100.0, 152.4, 200.0, 400.0, 1000.0)
    ]
    assert penalties == sorted(penalties)


def test_optimizer_margin_bonus_is_small_and_requires_feasibility() -> None:
    feasible = tuple(_performance(mission=mission) for mission in (1, 2, 3))
    bonus = propulsion_margin_bonus(feasible)
    assert 0.0 < bonus <= 0.05
    assert propulsion_margin_bonus((_performance(feasible=False),)) == 0.0
    assert propulsion_margin_bonus((_performance(mission=2),)) == 0.0
    assert math.isfinite(bonus)


def test_sensor_has_fixed_diameter_and_independent_container_length_floor() -> None:
    from src.mech.models import Mission2Config
    from src.vectors import (
        INCH_M,
        MIN_SENSOR_LENGTH_M,
        MIN_SENSOR_WEIGHT_KG,
        SENSOR_DIAMETER_M,
        DesignVector,
        maximum_sensor_weight_kg,
        sensor_length_from_weight_kg,
    )

    assert "sensor_diameter_m" not in DesignVector.opt_names()
    assert "sensor_length_m" not in DesignVector.opt_names()
    assert DesignVector().sensor_diameter_m == SENSOR_DIAMETER_M == 3.0 * INCH_M
    twelve_kg_sensor = DesignVector(sensor_weight_kg=12.0)
    assert twelve_kg_sensor.sensor_length_m == sensor_length_from_weight_kg(12.0)

    # A sensor too heavy to fit within the maximum solid-steel length is rejected.
    try:
        DesignVector(sensor_length_m=0.10, sensor_weight_kg=50.0, batt_capacity=3.0)
    except ValueError:
        pass
    else:
        raise AssertionError("density bound did not reject an impossible sensor")

    # The minimum solid-steel sensor is legal, while its container remains 8 in.
    short_sensor = DesignVector(
        sensor_length_m=MIN_SENSOR_LENGTH_M,
        sensor_weight_kg=MIN_SENSOR_WEIGHT_KG,
        mission3_sensor_weight_kg=0.05,
    )
    assert math.isclose(short_sensor.sensor_length_m, MIN_SENSOR_LENGTH_M)
    container_length = Mission2Config().container_dimensions_m(
        short_sensor.sensor_length_m, short_sensor.sensor_diameter_m
    )[0]
    assert container_length == 8.0 * INCH_M
    assert maximum_sensor_weight_kg(2.0 * INCH_M) < maximum_sensor_weight_kg(
        4.0 * INCH_M
    )
