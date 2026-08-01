"""
Batch comparison test using the fixed velocity/RPM grid prop database.

Run from repo root:

    python -m src.prop.prop_batch_comparison_fixed_grid_test
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from src.prop import prop_batch_comparison_test as batch_test
from src.prop.fixed_grid_prop_database import (
    FIXED_GRID_VELOCITIES_MPS,
    load_fixed_velocity_rpm_grid_database,
)


batch_test.load_default_prop_database = load_fixed_velocity_rpm_grid_database

batch_test.FIT_VELOCITIES_MPS = FIXED_GRID_VELOCITIES_MPS.copy()

batch_test.PLOT_VELOCITIES_MPS = np.linspace(
    float(FIXED_GRID_VELOCITIES_MPS[0]),
    float(FIXED_GRID_VELOCITIES_MPS[-1]),
    300,
)

batch_test.COMPARISON_SPEEDS_MPS = np.array(
    [0.01, 9.5, 15.0, 19.0, 25.0, 28.35],
    dtype=float,
)

batch_test.RANKING_SPEED_MPS = 15.0

batch_test.OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent / "prop_batch_comparison_fixed_grid_outputs"
)


def main() -> None:
    print("Running fixed-grid prop batch comparison test...")
    print(f"Fixed fit velocities [m/s]: {batch_test.FIT_VELOCITIES_MPS}")
    batch_test.main()


if __name__ == "__main__":
    main()