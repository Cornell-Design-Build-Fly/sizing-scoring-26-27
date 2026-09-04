import numpy as np

from src.opt.topline_opt import (
    PD_MAX,
    PD_MIN,
    ToplineConfig,
    _initial_population,
    _optimizer_bounds,
    _optimizer_variable_names,
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
    battery_cells = first[:, names.index("battery_cell_count")]
    battery_capacity = first[:, names.index("batt_capacity")]
    diameter = first[:, names.index("prop_diameter_in")]
    pitch = first[:, names.index("prop_pitch_in")]
    max_sensor_weight = first[:, names.index("sensor_weight_kg")]
    m3_sensor_weight = first[:, names.index("mission3_sensor_weight_kg")]

    assert np.all(first >= bounds[:, 0])
    assert np.all(first <= bounds[:, 1])
    assert np.all(containers == np.rint(containers))
    assert set(battery_cells.astype(int)) == {6, 8}
    assert np.all(battery_capacity * battery_cells * 3.7 <= 100.0 + 1e-12)
    assert np.all(pitch / diameter >= PD_MIN)
    assert np.all(pitch / diameter <= PD_MAX)
    assert np.all(m3_sensor_weight <= max_sensor_weight)
    assert set(containers.astype(int)) == set(range(11))
