import pytest
import numpy as np

from src.aero.aero_score import _endurance_values


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
