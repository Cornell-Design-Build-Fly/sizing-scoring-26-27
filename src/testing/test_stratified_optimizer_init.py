import numpy as np
from scipy.optimize import OptimizeResult

from src.opt.topline_opt import (
    PD_MAX,
    PD_MIN,
    ToplineConfig,
    _initial_population,
    _optimizer_bounds,
    _optimizer_constraints,
    _optimizer_variable_names,
    _official_population_scores,
)


def test_full_range_population_is_reproducible_and_feasible() -> None:
    config = ToplineConfig(popsize=25, init="sobol", seed=123)
    first = _initial_population(config)
    second = _initial_population(config)
    assert isinstance(first, np.ndarray)
    assert np.array_equal(first, second)
    names = _optimizer_variable_names(config)
    bounds = np.asarray(_optimizer_bounds(config), dtype=float)
    assert first.shape == (512, len(bounds))

    containers = first[:, names.index("extra_shipping_containers")]
    battery_capacity = first[:, names.index("batt_capacity")]
    diameter = first[:, names.index("prop_diameter_in")]
    pitch = first[:, names.index("prop_pitch_in")]
    m3_diameter = first[:, names.index("mission3_prop_diameter_in")]
    m3_pitch = first[:, names.index("mission3_prop_pitch_in")]
    max_sensor_weight = first[:, names.index("sensor_weight_kg")]
    m3_sensor_weight = first[:, names.index("mission3_sensor_weight_kg")]

    assert np.all(first >= bounds[:, 0])
    assert np.all(first <= bounds[:, 1])
    assert np.all(containers == np.rint(containers))
    assert "battery_cell_count" not in names
    assert np.all(
        battery_capacity * config.battery_cell_count * 3.7 <= 100.0 + 1e-12
    )
    assert np.all(pitch / diameter >= PD_MIN)
    assert np.all(pitch / diameter <= PD_MAX)
    # Mission 3 carries its own propeller and needs the same projection.
    assert np.all(m3_pitch / m3_diameter >= PD_MIN)
    assert np.all(m3_pitch / m3_diameter <= PD_MAX)
    assert not np.allclose(m3_diameter, diameter)
    assert np.all(m3_sensor_weight <= max_sensor_weight)
    assert set(containers.astype(int)) == set(range(11))

    for constraint in _optimizer_constraints(config):
        values = np.asarray([constraint.fun(candidate) for candidate in first])
        assert np.all(values >= constraint.lb)
        assert np.all(values <= constraint.ub)


def test_topline_uses_integer_laps_without_final_population_reevaluation() -> None:
    config = ToplineConfig()
    result = OptimizeResult(
        population=np.zeros((2, len(_optimizer_bounds(config)))),
        population_energies=np.array([-6.25, -5.0]),
    )

    assert not config.continuous_lap_scoring
    assert np.array_equal(
        _official_population_scores(result, config),
        np.array([6.25, 5.0]),
    )
