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
        "climb_speed_mps": 16.0,
        "climb_rate_mps": 3.0,
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
        "limiting_constraint": "takeoff_distance",
    }
    values.update(overrides)
    return MissionPropulsionPerformance(**values)


def test_battery_current_limit_uses_capacity_and_c_rating() -> None:
    battery = make_battery_from_design(DesignVector(batt_capacity=3.0), ParameterVector())
    assert battery.Crat == 25.0
    assert battery.get_max_current() == 75.0


def test_propulsion_penalty_identifies_takeoff_climb_and_energy_failures() -> None:
    requirements = PropulsionRequirements()
    assert _penalty_and_limit(60.0, 2.0, 70.0, 80.0, 0.7, requirements)[0] == 0.0
    takeoff_penalty, takeoff_limit = _penalty_and_limit(
        120.0, 2.0, 70.0, 80.0, 0.7, requirements
    )
    assert takeoff_penalty > PROPULSION_INFEASIBLE_BASE_PENALTY
    assert takeoff_limit == "takeoff_distance"
    assert _penalty_and_limit(60.0, 0.0, 70.0, 80.0, 0.7, requirements)[1] == "climb_rate"
    assert _penalty_and_limit(60.0, 2.0, 160.0, 80.0, 0.7, requirements)[1] == "mission_energy"
    assert _penalty_and_limit(
        60.0,
        2.0,
        70.0,
        80.0,
        0.7,
        requirements,
        operating_point_failed=True,
    )[1] == "propulsion_operating_point"


def test_optimizer_margin_bonus_is_small_and_requires_feasibility() -> None:
    feasible = tuple(_performance(mission=mission) for mission in (1, 2, 3))
    bonus = propulsion_margin_bonus(feasible)
    assert 0.0 < bonus <= 0.05
    assert propulsion_margin_bonus((_performance(feasible=False),)) == 0.0
    assert propulsion_margin_bonus((_performance(mission=2),)) == 0.0
    assert math.isfinite(bonus)
