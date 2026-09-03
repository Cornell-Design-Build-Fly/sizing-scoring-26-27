import pytest

from src.opt.score import (
    SECONDS_PER_MISSION,
    m1_optimization_score,
    m1_score,
    m2_optimization_score,
    m2_score,
    m3_optimization_score,
    m3_score,
    total_optimization_score,
    total_score,
)
from src.vectors import DesignVector


def test_relaxed_scores_equal_official_scores_at_integer_lap_capacity() -> None:
    design = DesignVector()
    for laps in (1, 3, 5, 8):
        lap_time = SECONDS_PER_MISSION / laps
        assert m2_optimization_score(design, lap_time) == pytest.approx(
            m2_score(design, lap_time)
        )
        assert m3_optimization_score(design, lap_time) == pytest.approx(
            m3_score(design, lap_time)
        )


def test_relaxed_scores_reward_speed_within_an_official_lap_bin() -> None:
    design = DesignVector()
    slow_lap_time = 70.0
    fast_lap_time = 65.0
    assert m2_score(design, fast_lap_time) == m2_score(design, slow_lap_time)
    assert m3_score(design, fast_lap_time) == m3_score(design, slow_lap_time)
    assert m2_optimization_score(design, fast_lap_time) > m2_optimization_score(
        design, slow_lap_time
    )
    assert m3_optimization_score(design, fast_lap_time) > m3_optimization_score(
        design, slow_lap_time
    )


def test_m1_relaxation_preserves_the_official_unlock_gate() -> None:
    design = DesignVector()
    failing_lap_time = 101.0
    official_total, official_breakdown = total_score(
        design, failing_lap_time, 60.0, 60.0
    )
    relaxed_total, relaxed_breakdown = total_optimization_score(
        design, failing_lap_time, 60.0, 60.0
    )
    assert m1_score(failing_lap_time) == 0.0
    assert 0.0 < m1_optimization_score(failing_lap_time) < 1.0
    assert official_breakdown[2:] == [0.0, 0.0]
    assert relaxed_breakdown[2:] == [0.0, 0.0]
    assert relaxed_total > official_total


def test_official_score_functions_remain_discrete() -> None:
    design = DesignVector()
    assert m2_score(design, 70.0) == m2_score(design, 65.0)
    assert m3_score(design, 70.0) == m3_score(design, 65.0)
