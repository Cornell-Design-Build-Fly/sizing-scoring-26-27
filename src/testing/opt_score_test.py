from src.opt.score import gm_score, m2_score, m3_score
from src.vectors import DesignVector


def test_ground_mission_score_is_capped_at_one() -> None:
    assert gm_score(DesignVector(ducks_num=3, pucks_num=1)) == 1.0
    assert gm_score(DesignVector(ducks_num=0, pucks_num=0)) == 1.0


def test_mission_two_score_is_capped_at_two() -> None:
    design = DesignVector(ducks_num=150, pucks_num=50, batt_capacity=1.0)

    assert m2_score(design, lap_time_s=20.0) == 2.0


def test_mission_three_score_is_capped_at_three() -> None:
    design = DesignVector(banner_length=100.0)

    assert m3_score(design, lap_time_s=1.0) == 3.0
