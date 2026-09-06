from src.opt.score import (
    DEFAULT_SCORING_REFERENCES,
    POUNDS_TO_KG,
    completed_m3_laps,
    gm_score,
    m1_score,
    m2_score,
    m3_score,
)
from src.vectors import DesignVector


def test_ground_mission_uses_40_lb_at_60_in_reference() -> None:
    design = DesignVector(sensor_weight_kg=40.0 * POUNDS_TO_KG)
    assert gm_score(design) == 1.5
    assert gm_score(design, successful=False) == 0.0


def test_mission_one_counts_takeoff_and_three_laps() -> None:
    assert m1_score(90.0, takeoff_time_s=30.0) == 1.0
    assert m1_score(90.01, takeoff_time_s=30.0) == 0.0
    assert m1_score(30.0, successful_landing=False) == 0.0


def test_mission_two_uses_complete_five_lap_elapsed_time() -> None:
    refs = DEFAULT_SCORING_REFERENCES
    payload = refs.best_m2_weight_per_time_kg_s * 180.0
    assert m2_score(payload, 30.0, takeoff_time_s=30.0) == 2.0
    assert m2_score(payload, 60.0, takeoff_time_s=0.01) == 0.0
    assert m2_score(payload, 54.0, takeoff_time_s=30.0) > 0.0
    assert m2_score(payload, 54.01, takeoff_time_s=30.0) == 0.0


def test_mission_three_is_capped_at_three() -> None:
    design = DesignVector(sensor_weight_kg=40.0 * POUNDS_TO_KG)
    assert m3_score(design, lap_time_s=30.0) == 3.0
    assert m3_score(design, lap_time_s=30.0, successful_landing=False) == 0.0


def test_mission_three_overheads_reduce_completed_laps() -> None:
    assert completed_m3_laps(40.0, takeoff_time_s=20.0) == 7


def test_ubilot_normalizations_are_retained() -> None:
    refs = DEFAULT_SCORING_REFERENCES
    assert refs.best_m2_weight_per_time_kg_s / POUNDS_TO_KG == 0.182332
    assert refs.best_m3_lap_weight_kg / POUNDS_TO_KG == 252.890
