import numpy as np
from dataclasses import replace

from src.tow.envelope import EnvelopeConfig, run_load_envelope
from src.tow.model import TowConfig, simulate_tow
from src.tow.surrogate import DEFAULT_M3_DOWNWARD_LOAD_SURROGATE


FAST_CONFIG = TowConfig(
    aircraft_weight_lbf=20.0,
    sensor_weight_lbf=5.0,
    airspeed_fps=50.0,
    straight_length_ft=20.0,
    rope_length_ft=9.0,
    rope_ea_lbf=3000.0,
    dt_s=0.01,
)


def test_callable_simulation_returns_finite_tension_only_loads() -> None:
    result = simulate_tow(FAST_CONFIG)
    assert len(result.time_s) == len(result.rope_tension_lbf)
    assert np.all(np.isfinite(result.rope_tension_lbf))
    assert np.all(result.rope_tension_lbf >= 0.0)
    assert result.mission_peak_tension_lbf > 0.0


def test_envelope_is_reproducible_and_ultimate_exceeds_limit() -> None:
    settings = EnvelopeConfig(monte_carlo_runs=2, seed=12, safety_factor=1.5)
    first, rows = run_load_envelope(FAST_CONFIG, settings)
    second, _ = run_load_envelope(FAST_CONFIG, settings)
    assert first == second
    assert first.cases_run == 9
    assert first.ultimate_tension_lbf == 1.5 * first.limit_tension_lbf
    assert first.limit_tension_lbf >= first.nominal_peak_tension_lbf
    assert len(rows) == first.cases_run


def test_rope_stiffness_preserves_material_ea_when_length_changes() -> None:
    longer = replace(FAST_CONFIG, rope_length_ft=18.0)
    assert longer.rope_stiffness_lbf_ft == 0.5 * FAST_CONFIG.rope_stiffness_lbf_ft


def test_default_surrogate_is_monotonic_and_applies_operational_fraction() -> None:
    model = DEFAULT_M3_DOWNWARD_LOAD_SURROGATE
    weights = np.linspace(model.minimum_weight_lbf, model.maximum_weight_lbf, 100)
    peaks = np.asarray([model.peak_downward_force_lbf(weight) for weight in weights])
    representative = np.asarray(
        [model.representative_downward_force_lbf(weight) for weight in weights]
    )
    assert np.all(np.diff(peaks) > 0.0)
    assert np.allclose(representative, 0.8 * peaks)
    assert np.all(peaks >= weights)
