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


def test_sensor_diameter_is_a_design_variable_with_physical_coupling() -> None:
    """Freezing the diameter made "heavy" reachable only by making the sensor
    long, which is an artifact of the constant rather than physics."""
    import contextlib
    import io

    from src.mech.main_mech import evaluate_mechanical_module
    from src.vectors import (
        MAX_SENSOR_DIAMETER_M,
        MIN_SENSOR_DIAMETER_M,
        DesignVector,
        maximum_sensor_weight_kg,
    )

    assert "sensor_diameter_m" in DesignVector.opt_names()

    # The density ceiling must scale with the cross-sectional area.
    thin = maximum_sensor_weight_kg(0.30, MIN_SENSOR_DIAMETER_M)
    fat = maximum_sensor_weight_kg(0.30, MAX_SENSOR_DIAMETER_M)
    ratio = (MAX_SENSOR_DIAMETER_M / MIN_SENSOR_DIAMETER_M) ** 2
    assert abs(fat / thin - ratio) < 1e-9

    # A short fat sensor can now out-mass a much longer thin one: area scales
    # with the square of diameter, so 4x the area beats 2.7x the length.
    assert maximum_sensor_weight_kg(0.15, 6.0 * 0.0254) > maximum_sensor_weight_kg(
        0.40, 3.0 * 0.0254
    )

    # A denser-than-steel sensor is still rejected.
    try:
        DesignVector(sensor_length_m=0.10, sensor_diameter_m=0.0254,
                     sensor_weight_kg=50.0, batt_capacity=3.0)
    except ValueError:
        pass
    else:
        raise AssertionError("density bound did not reject an impossible sensor")

    # A fatter sensor must widen the container, and so the fuselage.
    def width(diameter_m: float) -> float:
        design = DesignVector(sensor_length_m=0.20, sensor_diameter_m=diameter_m,
                              sensor_weight_kg=0.5, mission3_sensor_weight_kg=0.5,
                              batt_capacity=3.0)
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluate_mechanical_module(design).resolved_fuselage_width_m

    assert width(6.0 * 0.0254) > width(3.0 * 0.0254)
