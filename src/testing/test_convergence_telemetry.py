import csv

import numpy as np

from src.opt.topline_opt import _population_convergence_metrics
from src.opt.view_results import plot_generation_convergence


def test_infinite_energy_makes_scipy_convergence_unavailable() -> None:
    convergence, feasible_fraction, relative_std = (
        _population_convergence_metrics(
            np.array([-5.0, -4.0, np.inf]),
            scipy_convergence=0.0,
        )
    )

    assert np.isnan(convergence)
    assert np.isclose(feasible_fraction, 2.0 / 3.0)
    assert np.isclose(relative_std, 0.5 / 4.5)


def test_finite_population_preserves_scipy_convergence() -> None:
    convergence, feasible_fraction, relative_std = (
        _population_convergence_metrics(
            np.array([-5.0, -4.0]),
            scipy_convergence=0.25,
        )
    )

    assert convergence == 0.25
    assert feasible_fraction == 1.0
    assert np.isclose(relative_std, 0.5 / 4.5)


def test_convergence_plot_uses_global_incumbent_and_health_metrics(tmp_path) -> None:
    history = [
        {
            "generation": 1,
            "best_score": 4.0,
            "global_best_score": 4.0,
            "convergence": np.nan,
            "feasible_population_fraction": 0.75,
            "finite_energy_relative_std": 0.8,
        },
        {
            "generation": 2,
            "best_score": 3.0,
            "global_best_score": 4.0,
            "convergence": 0.2,
            "feasible_population_fraction": 1.0,
            "finite_energy_relative_std": 0.1,
        },
    ]
    output = tmp_path / "convergence.png"

    plot_generation_convergence(history, save_path=str(output), show=False)

    assert output.exists()
    assert output.stat().st_size > 0


def test_generation_health_fields_round_trip_through_csv(tmp_path) -> None:
    row = {
        "convergence": np.nan,
        "feasible_population_fraction": 0.75,
        "finite_energy_relative_std": 0.8,
        "global_best_score": 4.0,
    }
    output = tmp_path / "history.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    saved = next(csv.DictReader(output.open(encoding="utf-8")))
    assert saved["convergence"] == "nan"
    assert float(saved["feasible_population_fraction"]) == 0.75
    assert float(saved["finite_energy_relative_std"]) == 0.8
    assert float(saved["global_best_score"]) == 4.0
