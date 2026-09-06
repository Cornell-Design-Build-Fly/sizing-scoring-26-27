"""Flap lift/drag increments and the configurations they apply to."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from src.aero.drag_model import drag_coefficients, fuselage_drag_geometry
from src.aero.flaps import DEFAULT_FLAPS, FlapConfig, clean_cl_max
from src.prop.continuous_prop_database import load_default_continuous_prop_database
from src.prop.mission_performance import (
    DEFAULT_PROPULSION_REQUIREMENTS,
    _drag_n,
    evaluate_mission_propulsion,
)
from src.vectors import OPT_VARS, DesignVector, ParameterVector


NO_FLAPS = FlapConfig(takeoff_deflection_deg=0.0, landing_deflection_deg=0.0)


def test_clean_cl_max_is_the_one_definition() -> None:
    """The 1.45*AR/(AR+2) estimate used to be copy-pasted in three modules."""

    for aspect_ratio in (2.0, 4.0, 8.0):
        assert clean_cl_max(aspect_ratio) == pytest.approx(
            1.45 * aspect_ratio / (aspect_ratio + 2.0)
        )
    with pytest.raises(ValueError):
        clean_cl_max(0.0)


def test_increments_grow_with_deflection_and_vanish_when_retracted() -> None:
    flaps = DEFAULT_FLAPS
    assert flaps.delta_cl_max(0.0) == 0.0
    assert flaps.delta_cd0(0.0) == 0.0
    deflections = (10.0, 20.0, 25.0, 40.0, 60.0)
    lift = [flaps.delta_cl_max(d) for d in deflections]
    drag = [flaps.delta_cd0(d) for d in deflections]
    assert lift == sorted(lift)
    assert drag == sorted(drag)
    # Plain flap, 25% chord over 60% span: the landing setting is worth a few
    # tenths of CLmax, not a whole unit.
    assert 0.30 < flaps.delta_cl_max(40.0) < 0.45
    assert 0.04 < flaps.delta_cd0(40.0) < 0.09


def test_configurations_are_kept_apart() -> None:
    """Cruise and turns must see the clean wing; only takeoff/landing are flapped."""

    flaps = DEFAULT_FLAPS
    aspect_ratio = 5.0
    clean = flaps.cl_max_for(aspect_ratio, "clean")
    takeoff = flaps.cl_max_for(aspect_ratio, "takeoff")
    landing = flaps.cl_max_for(aspect_ratio, "landing")
    assert clean == clean_cl_max(aspect_ratio)
    assert clean < takeoff < landing
    assert flaps.deflection_for("clean") == 0.0
    with pytest.raises(ValueError):
        flaps.deflection_for("cruise")

    # Stall speed rescaling is exactly the 1/sqrt(CLmax) relation.
    assert flaps.stall_speed_for(20.0, aspect_ratio, "clean") == pytest.approx(20.0)
    assert flaps.stall_speed_for(20.0, aspect_ratio, "landing") == pytest.approx(
        20.0 * math.sqrt(clean / landing)
    )


def test_flap_drag_only_appears_when_deflected() -> None:
    design = DesignVector()
    parameters = ParameterVector()
    geometry = fuselage_drag_geometry(design)
    clean = drag_coefficients(design, parameters, 20.0, 0.5, 0.0, geometry)
    flapped = drag_coefficients(
        design, parameters, 20.0, 0.5, 0.0, geometry, flap_deflection_deg=40.0
    )
    assert clean["flap_profile"] == 0.0
    assert flapped["flap_profile"] > 0.0
    # Nothing else in the build-up moves.
    for key in clean:
        if key != "flap_profile":
            assert flapped[key] == clean[key]


def _evaluate(requirements, **overrides):
    # Candidate-level takeoff deflection normally overrides this configuration.
    # Mirror the supplied configuration unless a test explicitly selects an angle.
    overrides.setdefault(
        "takeoff_flap_deflection_deg",
        requirements.flaps.takeoff_deflection_deg,
    )
    design = DesignVector(batt_capacity=3.0, **overrides)
    return evaluate_mission_propulsion(
        design,
        ParameterVector(),
        mission=2,
        mass_kg=5.0,
        cruise_speed_mps=25.0,
        stall_speed_mps=12.0,
        lap_time_s=40.0,
        prop_database=load_default_continuous_prop_database(),
        requirements=requirements,
    )


def test_flaps_shorten_the_takeoff_roll() -> None:
    without = _evaluate(replace(DEFAULT_PROPULSION_REQUIREMENTS, flaps=NO_FLAPS))
    with_flaps = _evaluate(DEFAULT_PROPULSION_REQUIREMENTS)
    assert without.takeoff_stall_speed_mps == without.clean_stall_speed_mps
    assert with_flaps.takeoff_stall_speed_mps < with_flaps.clean_stall_speed_mps
    assert with_flaps.liftoff_speed_mps < without.liftoff_speed_mps
    assert with_flaps.takeoff_distance_m < without.takeoff_distance_m


def test_takeoff_flap_deflection_is_an_optimizer_variable() -> None:
    bounds = dict(OPT_VARS)
    assert bounds["takeoff_flap_deflection_deg"] == (0.0, 40.0)

    design = DesignVector(takeoff_flap_deflection_deg=13.0)
    restored = DesignVector.from_array(design.to_array())
    assert restored.takeoff_flap_deflection_deg == 13.0


def test_design_selects_takeoff_flap_deflection() -> None:
    clean = _evaluate(
        DEFAULT_PROPULSION_REQUIREMENTS,
        takeoff_flap_deflection_deg=0.0,
    )
    flapped = _evaluate(
        DEFAULT_PROPULSION_REQUIREMENTS,
        takeoff_flap_deflection_deg=20.0,
    )
    assert clean.takeoff_flap_deflection_deg == 0.0
    assert flapped.takeoff_flap_deflection_deg == 20.0
    assert flapped.takeoff_stall_speed_mps < clean.takeoff_stall_speed_mps
    assert flapped.takeoff_distance_m < clean.takeoff_distance_m


def test_ground_effect_credit_is_conservative_and_ground_roll_only() -> None:
    without_credit = _evaluate(
        replace(
            DEFAULT_PROPULSION_REQUIREMENTS,
            ground_effect_induced_drag_factor=1.0,
        )
    )
    with_credit = _evaluate(DEFAULT_PROPULSION_REQUIREMENTS)

    assert DEFAULT_PROPULSION_REQUIREMENTS.ground_effect_induced_drag_factor == 0.90
    assert with_credit.takeoff_distance_m < without_credit.takeoff_distance_m
    # Ground effect does not change the flap lift model or airborne climb model.
    assert with_credit.takeoff_stall_speed_mps == pytest.approx(
        without_credit.takeoff_stall_speed_mps
    )
    assert with_credit.climb_rate_mps == pytest.approx(without_credit.climb_rate_mps)


def test_ground_effect_factor_reduces_only_induced_drag_terms() -> None:
    design = DesignVector()
    parameters = ParameterVector()
    speed_mps = 15.0
    lift_coefficient = 0.8
    full_drag = float(
        _drag_n(
            design,
            parameters,
            speed_mps,
            50.0,
            2,
            lift_coefficient=lift_coefficient,
            flap_deflection_deg=20.0,
            induced_drag_factor=1.0,
        )
    )
    reduced_drag = float(
        _drag_n(
            design,
            parameters,
            speed_mps,
            50.0,
            2,
            lift_coefficient=lift_coefficient,
            flap_deflection_deg=20.0,
            induced_drag_factor=0.90,
        )
    )
    coefficients = drag_coefficients(
        design,
        parameters,
        speed_mps,
        lift_coefficient,
        0.0,
        fuselage_drag_geometry(design),
        flap_deflection_deg=20.0,
    )
    induced_cd = sum(
        coefficients[name]
        for name in ("wing_induced", "tail_induced", "interaction")
    )
    expected_reduction_n = (
        0.10
        * 0.5
        * parameters.rho
        * speed_mps**2
        * design.wing_area
        * induced_cd
    )
    assert full_drag - reduced_drag == pytest.approx(expected_reduction_n)
    # Profile/body/flap drag remains, so total drag is not reduced by 10%.
    assert reduced_drag > 0.90 * full_drag


def test_flaps_do_not_touch_cruise_or_the_turn() -> None:
    """The turn envelope is built on clean CLmax, so flaps cannot fake 2.5 g."""

    without = _evaluate(replace(DEFAULT_PROPULSION_REQUIREMENTS, flaps=NO_FLAPS))
    with_flaps = _evaluate(DEFAULT_PROPULSION_REQUIREMENTS)
    assert with_flaps.clean_stall_speed_mps == without.clean_stall_speed_mps
    assert with_flaps.turn_speed_mps == pytest.approx(without.turn_speed_mps)
    assert with_flaps.turn_load_factor == pytest.approx(without.turn_load_factor)
    assert with_flaps.cruise_speed_mps == pytest.approx(without.cruise_speed_mps)


def test_climb_is_flown_flapped_and_costs_energy() -> None:
    """Flaps stay down to cruise altitude, and the climb is no longer free."""

    requirements = DEFAULT_PROPULSION_REQUIREMENTS
    assert requirements.cruise_altitude_m > 0.0
    result = _evaluate(requirements)
    # Climb speed is referenced to the flapped stall speed, not the clean one.
    assert result.climb_speed_mps == pytest.approx(
        requirements.climb_stall_speed_factor * result.takeoff_stall_speed_mps
    )
    assert result.climb_time_s > 0.0
    assert result.climb_energy_wh > 0.0
    assert result.cruise_altitude_m == requirements.cruise_altitude_m

    # Flaps cost drag in the climb, so the flapped climb rate is the lower one.
    clean_climb = _evaluate(replace(requirements, flaps=NO_FLAPS))
    assert result.climb_rate_mps < clean_climb.climb_rate_mps


def test_liftoff_to_climb_speed_is_no_longer_free() -> None:
    """The model used to jump this gap with no distance, time or energy."""

    result = _evaluate(DEFAULT_PROPULSION_REQUIREMENTS)
    assert result.climb_speed_mps > result.liftoff_speed_mps
    assert result.acceleration_distance_m > 0.0
    assert result.acceleration_time_s > 0.0
    assert result.acceleration_energy_wh > 0.0
    # It is airborne, so it does not eat into the runway allowance.
    assert result.takeoff_distance_m < result.acceleration_distance_m + result.takeoff_distance_m


def test_flap_retraction_acceleration_is_charged_once() -> None:
    """Coming up to cruise speed after the flaps retract costs kinetic energy."""

    result = _evaluate(DEFAULT_PROPULSION_REQUIREMENTS)
    assert result.cruise_speed_mps > result.climb_speed_mps
    expected = (
        0.5
        * result.inertial_mass_kg
        * (result.cruise_speed_mps**2 - result.climb_speed_mps**2)
        / DEFAULT_PROPULSION_REQUIREMENTS.reacceleration_efficiency
        / 3600.0
    )
    assert result.flap_retraction_energy_wh == pytest.approx(expected)


def test_landing_stall_speed_is_reported_but_never_constrains() -> None:
    """The team removed the landing-speed limit; the number stays informative."""

    requirements = DEFAULT_PROPULSION_REQUIREMENTS
    assert not hasattr(requirements, "maximum_landing_speed_mps")
    result = _evaluate(requirements)
    assert result.landing_stall_speed_mps < result.takeoff_stall_speed_mps
    assert result.landing_flap_deflection_deg == 40.0
    # Flaps still lower it, and nothing scores or gates on the value.
    unflapped = _evaluate(replace(requirements, flaps=NO_FLAPS))
    assert result.landing_stall_speed_mps < unflapped.landing_stall_speed_mps
    assert result.limiting_constraint != "landing_speed"


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"PASS {name}")
