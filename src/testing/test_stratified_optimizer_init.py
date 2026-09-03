import numpy as np

from src.opt.topline_opt import (
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
    containers = first[:, names.index("extra_shipping_containers")]
    diameter = first[:, names.index("prop_diameter_in")]
    pitch = first[:, names.index("prop_pitch_in")]
    bounds = np.asarray(DesignVector.bounds(), dtype=float)

    assert np.all(first >= bounds[:, 0])
    assert np.all(first <= bounds[:, 1])
    assert np.all(containers == np.rint(containers))
    assert np.all(pitch / diameter >= PD_MIN)
    assert np.all(pitch / diameter <= PD_MAX)
    assert set(containers.astype(int)) == set(range(11))
