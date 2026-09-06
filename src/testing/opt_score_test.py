import pytest

from src.opt.score import (
    DEFAULT_SCORING_REFERENCES,
    GROUND_DROP_HEIGHT_IN,
    POUNDS_TO_KG,
    ScoringReferences,
    gm_score,
    m1_score,
    m2_score,
    m3_score,
    round_half_up,
)
from src.vectors import MAX_SENSOR_LENGTH_M, DesignVector


def test_ground_mission_uses_60_inches_and_35_lb_reference() -> None:
    assert GROUND_DROP_HEIGHT_IN == 60.0
    assert gm_score(
        DesignVector(
            sensor_length_m=MAX_SENSOR_LENGTH_M,
            sensor_weight_kg=35.0 * POUNDS_TO_KG,
        )
    ) == 1.5
    assert gm_score(
        DesignVector(
            sensor_length_m=MAX_SENSOR_LENGTH_M,
            sensor_weight_kg=17.5 * POUNDS_TO_KG,
        )
    ) == 1.0


def test_mission_two_uses_weight_over_exactly_five_lap_time() -> None:
    rounded_payload_kg = 44.09 * POUNDS_TO_KG
    refs = ScoringReferences(best_m2_weight_per_time_kg_s=rounded_payload_kg / 200.0)
    assert m2_score(20.0, 40.0, refs) == pytest.approx(2.0)
    # Five laps must fit the 280 s usable window, not the 300 s clock.
    assert m2_score(20.0, 56.0, refs) > 0.0
    assert m2_score(20.0, 61.0, refs) == 0.0
    assert m2_score(20.0, float("inf"), refs) == 0.0


def test_mission_three_uses_integer_laps_times_sensor_weight() -> None:
    design = DesignVector(
        sensor_length_m=12.0 * 0.0254,
        sensor_weight_kg=7.0,
        mission3_sensor_weight_kg=6.0,
    )
    refs = ScoringReferences(best_m3_lap_weight_kg=30.0)
    # Laps are counted against the window left after takeoff and landing, so a
    # 56 s lap fits five times in 280 s where a 60 s lap fits only four.
    assert refs.usable_window_s == 280.0
    assert m3_score(design, 56.0, refs) == pytest.approx(3.0, abs=1e-3)
    assert m3_score(design, 60.0, refs) == pytest.approx(2.8, abs=1e-3)
    assert m3_score(design, 281.0, refs) == 0.0


def test_ground_time_shortens_every_mission_window() -> None:
    """The 20 s reserve is the same one behind the best-team normalizers."""

    refs = DEFAULT_SCORING_REFERENCES
    assert refs.seconds_per_mission == 300.0
    assert refs.ground_time_s == 20.0
    assert refs.usable_window_s == 280.0
    # A lap that only fit under the old 300 s clock no longer scores.
    assert m1_score(280.0 / 3.0) == 1.0
    assert m1_score(100.0) == 0.0
    assert m2_score(20.0, 56.0) > 0.0
    assert m2_score(20.0, 60.0) == 0.0

    without_reserve = ScoringReferences(ground_time_s=0.0)
    assert without_reserve.usable_window_s == 300.0
    assert m1_score(100.0, without_reserve) == 1.0


def test_mission_one_accepts_exactly_the_usable_window() -> None:
    boundary = DEFAULT_SCORING_REFERENCES.usable_window_s / 3.0
    assert m1_score(boundary) == 1.0
    assert m1_score(boundary + 0.01) == 0.0


def test_rules_rounding_is_half_up() -> None:
    assert round_half_up(1.005) == 1.01
