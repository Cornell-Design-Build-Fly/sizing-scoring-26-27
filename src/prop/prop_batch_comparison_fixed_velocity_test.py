"""
Batch comparison test for the fixed-velocity 3D prop database.

Run from the repository root with:

    python -m src.prop.prop_batch_comparison_fixed_velocity_test

This intentionally reuses the existing prop_batch_comparison_test.py logic.
The only differences are:
1. It loads the fixed-velocity 3D prop database.
2. It forces the fit velocities to the fixed velocities supported by that database.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.prop import prop_batch_comparison_test as batch_test
from src.prop.fixed_velocity_prop_database import (
    FIXED_FIT_VELOCITIES_MPS,
    load_fixed_velocity_3d_prop_database,
)


# Use the new fixed-velocity database instead of the default 4D database.
batch_test.load_default_prop_database = load_fixed_velocity_3d_prop_database

# The fixed-velocity database only supports these exact fit speeds.
batch_test.FIT_VELOCITIES_MPS = FIXED_FIT_VELOCITIES_MPS.copy()

# Plot over the full fitted speed range.
batch_test.PLOT_VELOCITIES_MPS = np.linspace(
    float(FIXED_FIT_VELOCITIES_MPS[0]),
    float(FIXED_FIT_VELOCITIES_MPS[-1]),
    300,
)

# Keep useful comparison speeds inside the plotted/fitted range.
batch_test.COMPARISON_SPEEDS_MPS = np.array(
    [0.01, 9.5, 15.0, 19.0, 25.0, 28.35],
    dtype=float,
)

batch_test.RANKING_SPEED_MPS = 15.0

# Save outputs separately so they do not mix with the old test's outputs.
batch_test.OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "prop_batch_comparison_fixed_velocity_outputs"
)


def main() -> None:
    print("Running fixed-velocity 3D prop batch comparison test...")
    print(f"Fixed fit velocities [m/s]: {batch_test.FIT_VELOCITIES_MPS}")
    batch_test.main()


if __name__ == "__main__":
    main()