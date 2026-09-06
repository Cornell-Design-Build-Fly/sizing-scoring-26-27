from dataclasses import replace

import numpy as np

from src.aero.flight_profile import (
    DEFAULT_FLIGHT_PROFILE_CONFIG,
    _climb_time,
    _takeoff,
    compute_flight_profile,
)
from src.vectors import DesignVector, ParameterVector


THRUST_CURVE = (-0.02, -0.5, 120.0)
ENDURANCE_CURVE = (0.0, 0.0, 1000.0)


def _profile(**kwargs):
    inputs = dict(
        cruise_speed_m_s=22.0,
        stall_speed_m_s=10.0,
        design=DesignVector(),
        parameters=ParameterVector(),
        thrust_velocity=THRUST_CURVE,
        mass_kg=4.0,
        mission=1,
        flight_time_fit=ENDURANCE_CURVE,
    )
    inputs.update(kwargs)
    return compute_flight_profile(**inputs)


def test_feasible_profile_accounts_for_potential_energy() -> None:
    result = _profile()
    expected_potential_wh = (
        4.0
        * ParameterVector.gravity
        * DEFAULT_FLIGHT_PROFILE_CONFIG.first_turn_altitude_m
        / 3600.0
    )

    assert result.feasible
    assert result.takeoff_distance_m <= 60.0
    assert (
        result.takeoff_distance_m + result.climb_horizontal_distance_m
        <= DEFAULT_FLIGHT_PROFILE_CONFIG.straight_length_m
    )
    assert np.isclose(result.potential_energy_gained_wh, expected_potential_wh)
    assert np.isclose(
        result.landing_potential_energy_dissipated_wh,
        expected_potential_wh,
    )
    assert result.mission_energy_required_wh > (
        expected_potential_wh
        / DEFAULT_FLIGHT_PROFILE_CONFIG.climb_energy_efficiency
    )


def test_takeoff_must_fit_runway() -> None:
    result = _profile(
        config=replace(DEFAULT_FLIGHT_PROFILE_CONFIG, runway_length_m=1.0)
    )
    assert not result.feasible
    assert "runway" in result.reason


def test_airplane_must_reach_altitude_before_first_turn() -> None:
    result = _profile(
        config=replace(DEFAULT_FLIGHT_PROFILE_CONFIG, straight_length_m=10.0)
    )
    assert not result.feasible
    assert "200 ft before the first turn" in result.reason


def test_m3_tow_drag_is_not_applied_before_deployment() -> None:
    design = DesignVector(sensor_weight_kg=10.0)
    parameters = ParameterVector()
    common_takeoff = (
        design,
        parameters,
        THRUST_CURVE,
        10.0,
        12.0,
    )
    assert _takeoff(*common_takeoff, 1, DEFAULT_FLIGHT_PROFILE_CONFIG) == _takeoff(
        *common_takeoff, 3, DEFAULT_FLIGHT_PROFILE_CONFIG
    )
    common_climb = (*common_takeoff, 22.0)
    assert _climb_time(*common_climb, 1, DEFAULT_FLIGHT_PROFILE_CONFIG) == _climb_time(
        *common_climb, 3, DEFAULT_FLIGHT_PROFILE_CONFIG
    )
