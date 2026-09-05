import pytest
import numpy as np

from src.aero.aero_score import _endurance_values, aero_score


def test_m1_requires_three_laps() -> None:
    available, required, penalty = _endurance_values((0.0, 0.0, 120.0), 25.0, 40.0, 1)
    assert available == required == 120.0
    assert penalty == 0.0


@pytest.mark.parametrize("mission, expected", ((2, 200.0), (3, 300.0)))
def test_m2_requires_five_laps_and_m3_requires_five_minutes(mission: int, expected: float) -> None:
    available, required, penalty = _endurance_values((0.0, 0.0, 150.0), 25.0, 40.0, mission)
    assert available == 150.0
    assert required == expected
    assert penalty > 0.0


def test_zero_endurance_gets_full_penalty() -> None:
    assert _endurance_values((0.0, 0.0, 0.0), 25.0, 40.0, 2)[2] == 10.0


def test_invalid_mission_is_rejected() -> None:
    with pytest.raises(ValueError):
        _endurance_values((0.0, 0.0, 300.0), 25.0, 40.0, 4)


def test_legacy_endurance_fit_is_not_a_second_flyability_gate() -> None:
    import math

    import aerosandbox as asb

    from src.aero.custom_classes import CruiseCondition, ModeResult, StabilityResult
    from src.vectors import ParameterVector

    mode = ModeResult(
        eigenvalue_real=-1.0,
        eigenvalue_imag=0.0,
        damping_ratio=1.0,
    )
    cruise = CruiseCondition(
        operating_point=asb.OperatingPoint(velocity=20.0, alpha=2.0),
        stall_speed=10.0,
        converged=True,
    )
    stability = StabilityResult(
        phugoid=mode,
        short_period=mode,
        dutch_roll=mode,
        spiral=mode,
        roll_subsidence=mode,
        Cma=-1.0,
        Cnb=0.01,
        static_margin=0.20,
        spiral_time_to_double_s=math.inf,
    )
    result = aero_score(
        cruise,
        stability,
        ParameterVector(),
        (0.0, 0.0, 0.0),
        2,
    )
    assert result.can_fly
    assert result.penalty == 0.0

    cannot_turn = aero_score(
        CruiseCondition(
            operating_point=asb.OperatingPoint(velocity=10.0, alpha=2.0),
            stall_speed=10.0,
            converged=True,
        ),
        stability,
        ParameterVector(),
        (0.0, 0.0, 1.0e6),
        2,
    )
    assert not cannot_turn.can_fly
    assert cannot_turn.penalty == 10.0
