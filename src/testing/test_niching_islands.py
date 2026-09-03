import numpy as np

from src.opt.topline_opt import _duplicate_islands
from src.vectors import DesignVector


def test_duplicate_detection_keeps_stronger_nearby_island() -> None:
    center = DesignVector().to_array()
    nearby = center.copy()
    nearby[0] += 1e-4
    distant = center.copy()
    distant[0] = DesignVector.bounds()[0][1]
    states = [
        {"best_vector": center, "best_objective": -5.0},
        {"best_vector": nearby, "best_objective": -4.9},
        {"best_vector": distant, "best_objective": -4.8},
    ]
    assert _duplicate_islands(states, radius=0.1) == {1}


def test_duplicate_detection_does_not_require_named_design_categories() -> None:
    bounds = np.asarray(DesignVector.bounds(), dtype=float)
    states = [
        {"best_vector": bounds[:, 0], "best_objective": -5.0},
        {"best_vector": bounds[:, 1], "best_objective": -4.9},
    ]
    assert _duplicate_islands(states, radius=0.1) == set()
