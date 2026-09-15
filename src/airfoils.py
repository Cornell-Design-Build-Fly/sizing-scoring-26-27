"""Repository-owned airfoil loading helpers."""

from __future__ import annotations

from pathlib import Path

import aerosandbox as asb
import numpy as np


SG6042_DATA_PATH = Path(__file__).resolve().parent / "aero" / "data" / "sg6042.dat"


def load_airfoil(name: str) -> asb.Airfoil:
    """Load an airfoil, using the checked-in SG6042 coordinates when requested."""

    if name.strip().casefold() == "sg6042":
        coordinates = np.loadtxt(SG6042_DATA_PATH, skiprows=1)
        return asb.Airfoil(name="sg6042", coordinates=coordinates)
    return asb.Airfoil(name)


__all__ = ["SG6042_DATA_PATH", "load_airfoil"]
