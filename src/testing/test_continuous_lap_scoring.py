from src.opt.score import (
    m1_optimization_score,
    m2_optimization_score,
    total_optimization_score,
    total_score,
)
from src.vectors import DesignVector


def test_successful_m2_waives_m1_and_unlocks_m3() -> None:
    design = DesignVector()
    payload_mass = 2.0
    official_total, official = total_score(design, 101.0, 40.0, 50.0, payload_mass)
    relaxed_total, relaxed = total_optimization_score(
        design, 101.0, 40.0, 50.0, payload_mass
    )
    assert 0.0 < m1_optimization_score(101.0) < 1.0
    assert official[1] == 1.0
    assert official[2] > 0.0
    assert official[3] > 0.0
    assert relaxed[1] == 1.0
    assert relaxed[2] > 0.0
    assert relaxed[3] > 0.0
    assert relaxed_total >= official_total


def test_relaxed_m2_rewards_progress_but_keeps_m3_locked() -> None:
    design = DesignVector()
    payload_mass = 2.0
    assert m2_optimization_score(payload_mass, 61.0) > 0.0
    _, breakdown = total_optimization_score(
        design, 101.0, 61.0, 50.0, payload_mass
    )
    assert 0.0 < breakdown[1] < 1.0
    assert breakdown[2] > 0.0
    assert breakdown[3] == 0.0
