"""Fast Mission-3 downward tow-load surrogate.

The detailed dynamics model is swept offline. Optimization then evaluates a
small polynomial instead of integrating the tow dynamics for every candidate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.tow.model import KG_TO_LBM, TowConfig, simulate_tow


@dataclass(frozen=True)
class DownwardLoadSurrogate:
    """Polynomial prediction of peak downward tether load versus sensor weight."""

    coefficients: tuple[float, ...]
    upward_residual_lbf: float
    minimum_weight_lbf: float
    maximum_weight_lbf: float
    representative_fraction: float = 0.8
    fit_r_squared: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if len(self.coefficients) < 2:
            raise ValueError("At least a linear polynomial is required.")
        if not 0.0 < self.minimum_weight_lbf < self.maximum_weight_lbf:
            raise ValueError("The surrogate weight range is invalid.")
        if not 0.0 <= self.representative_fraction <= 1.0:
            raise ValueError("representative_fraction must lie in [0, 1].")
        if self.upward_residual_lbf < 0.0:
            raise ValueError("upward_residual_lbf cannot be negative.")

    def peak_downward_force_lbf(self, sensor_weight_lbf: float) -> float:
        """Return the conservative fitted peak within the calibrated range."""
        weight = float(sensor_weight_lbf)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("sensor_weight_lbf must be finite and positive.")
        if weight > self.maximum_weight_lbf + 1e-12:
            raise ValueError(
                f"Sensor weight {weight:.3f} lbf exceeds the surrogate's "
                f"{self.maximum_weight_lbf:.3f} lbf calibration limit."
            )
        evaluation_weight = max(weight, self.minimum_weight_lbf)
        peak = float(np.polyval(self.coefficients, evaluation_weight))
        peak += self.upward_residual_lbf
        # Below the first sweep point, scale to the origin instead of trusting
        # a polynomial intercept that has no physical meaning.
        if weight < self.minimum_weight_lbf:
            peak *= weight / self.minimum_weight_lbf
        return max(weight, peak)

    def representative_downward_force_lbf(self, sensor_weight_lbf: float) -> float:
        return self.representative_fraction * self.peak_downward_force_lbf(
            sensor_weight_lbf
        )

    def equivalent_supported_mass_kg(self, sensor_mass_kg: float) -> float:
        sensor_weight_lbf = float(sensor_mass_kg) * KG_TO_LBM
        return self.representative_downward_force_lbf(sensor_weight_lbf) / KG_TO_LBM

    def to_dict(self) -> dict:
        return asdict(self)


# Calibrated using the default 70 ft/s, 35 degree, 9 ft rope configuration and
# 2..50 lbf sensor sweep. The quadratic residual correction makes the fitted
# PEAK no lower than any calibration point; the 0.8 operational fraction is
# applied only afterward.
DEFAULT_M3_DOWNWARD_LOAD_SURROGATE = DownwardLoadSurrogate(
    coefficients=(0.01518785698643812, 1.4591743220616114, -2.428683399644092),
    upward_residual_lbf=2.0444056470656737,
    minimum_weight_lbf=2.0,
    maximum_weight_lbf=50.0,
    representative_fraction=0.8,
    fit_r_squared=0.9978417248136906,
    description=(
        "Quadratic fit to nominal-course maximum downward tether load; "
        "70 ft/s, 35 deg bank, 1.5 s roll, 9 ft rope, 9000 lbf rope EA, "
        "12 in x 3 in sensor, dt=0.008 s."
    ),
)


def fit_downward_load_surrogate(
    sensor_weights_lbf,
    *,
    base_config: TowConfig = TowConfig(dt_s=0.008),
    polynomial_degree: int = 2,
    representative_fraction: float = 0.8,
):
    """Sweep sensor weights and fit a conservatively shifted polynomial."""
    weights = np.asarray(tuple(sensor_weights_lbf), dtype=float)
    if weights.ndim != 1 or len(weights) <= polynomial_degree:
        raise ValueError("More sweep weights than polynomial degree are required.")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Sweep weights must be finite and positive.")
    weights = np.unique(np.sort(weights))
    peaks = np.asarray(
        [
            np.max(
                simulate_tow(
                    replace(base_config, sensor_weight_lbf=float(weight))
                ).tow_force_body_lbf[:, 2]
            )
            for weight in weights
        ],
        dtype=float,
    )
    coefficients = np.polyfit(weights, peaks, polynomial_degree)
    fitted = np.polyval(coefficients, weights)
    residual = peaks - fitted
    upward_shift = max(0.0, float(np.max(residual)))
    denominator = float(np.sum((peaks - np.mean(peaks)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / denominator
    model = DownwardLoadSurrogate(
        coefficients=tuple(float(value) for value in coefficients),
        upward_residual_lbf=upward_shift,
        minimum_weight_lbf=float(weights[0]),
        maximum_weight_lbf=float(weights[-1]),
        representative_fraction=representative_fraction,
        fit_r_squared=r_squared,
        description="Generated from src.tow.model.simulate_tow.",
    )
    rows = [
        {
            "sensor_weight_lbf": float(weight),
            "simulated_peak_down_lbf": float(peak),
            "fitted_conservative_peak_down_lbf": model.peak_downward_force_lbf(
                float(weight)
            ),
            "representative_down_lbf": model.representative_downward_force_lbf(
                float(weight)
            ),
        }
        for weight, peak in zip(weights, peaks)
    ]
    return model, rows


def write_surrogate_files(
    model: DownwardLoadSurrogate,
    rows: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m3_tow_surrogate.json").write_text(
        json.dumps(model.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "m3_tow_surrogate_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_surrogate_sweep(rows: list[dict], output_path: Path) -> None:
    """Plot simulated peaks, conservative fit, and representative load."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    weights = [row["sensor_weight_lbf"] for row in rows]
    plt.figure(figsize=(8.0, 5.0))
    plt.scatter(
        weights,
        [row["simulated_peak_down_lbf"] for row in rows],
        label="Tow simulation peak",
        color="black",
    )
    plt.plot(
        weights,
        [row["fitted_conservative_peak_down_lbf"] for row in rows],
        label="Conservative quadratic peak fit",
    )
    plt.plot(
        weights,
        [row["representative_down_lbf"] for row in rows],
        label="Representative load (fraction of fitted peak)",
    )
    plt.xlabel("Sensor weight [lbf]")
    plt.ylabel("Downward tether force [lbf]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-weight", type=float, default=2.0)
    parser.add_argument("--maximum-weight", type=float, default=50.0)
    parser.add_argument("--points", type=int, default=25)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--load-fraction", type=float, default=0.8)
    parser.add_argument("--dt", type=float, default=0.008)
    parser.add_argument("--output", type=Path, default=Path("data_dump/tow_surrogate"))
    args = parser.parse_args()
    weights = np.linspace(args.minimum_weight, args.maximum_weight, args.points)
    model, rows = fit_downward_load_surrogate(
        weights,
        base_config=TowConfig(dt_s=args.dt),
        polynomial_degree=args.degree,
        representative_fraction=args.load_fraction,
    )
    write_surrogate_files(model, rows, args.output)
    plot_surrogate_sweep(rows, args.output / "m3_tow_surrogate_fit.png")
    print(json.dumps(model.to_dict(), indent=2))
    print(f"Wrote surrogate artifacts to {args.output}")


__all__ = [
    "DEFAULT_M3_DOWNWARD_LOAD_SURROGATE",
    "DownwardLoadSurrogate",
    "fit_downward_load_surrogate",
    "plot_surrogate_sweep",
    "write_surrogate_files",
]


if __name__ == "__main__":
    main()
