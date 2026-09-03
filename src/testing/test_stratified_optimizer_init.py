import numpy as np

from src.opt.topline_opt import (
    MIN_DUCKS_PER_PUCK,
    PD_MAX,
    PD_MIN,
    ToplineConfig,
    _initial_population,
)
from src.vectors import DesignVector


def test_full_range_population_is_reproducible_and_feasible() -> None:
    config = ToplineConfig(popsize=25, init="sobol", seed=123)
    first = _initial_population(config)
    second = _initial_population(config)
    assert isinstance(first, np.ndarray)
    assert np.array_equal(first, second)
    assert first.shape == (512, len(DesignVector.bounds()))

    names = DesignVector.opt_names()
    ducks = first[:, names.index("ducks_num")]
    pucks = first[:, names.index("pucks_num")]
    diameter = first[:, names.index("prop_diameter_in")]
    pitch = first[:, names.index("prop_pitch_in")]
    bounds = np.asarray(DesignVector.bounds(), dtype=float)

    assert np.all(first >= bounds[:, 0])
    assert np.all(first <= bounds[:, 1])
    assert np.all(ducks == np.rint(ducks))
    assert np.all(pucks == np.rint(pucks))
    assert np.all(ducks / pucks >= MIN_DUCKS_PER_PUCK)
    assert np.all(pitch / diameter >= PD_MIN)
    assert np.all(pitch / diameter <= PD_MAX)
    assert len(set(zip(ducks.astype(int), pucks.astype(int)))) > 100
