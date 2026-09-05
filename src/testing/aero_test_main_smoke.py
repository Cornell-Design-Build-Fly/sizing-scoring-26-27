from __future__ import annotations

import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from src.aero.main_aero import aero_main
from src.aero.aero_score import AeroScore
from src.vectors import DesignVector, ParameterVector

# Chat generated basic smoke test script with arbitrary design vector.

def make_realistic_design_vector() -> DesignVector:
    """Returns a representative DBF-sized aircraft design."""
    return DesignVector(
        wing_span=1.50,       # m
        wing_chord=0.28,      # m
        tail_arm=0.75,        # m
        nose_length=0.22,     # m
        ducks_num=5,
        pucks_num=2,
        banner_length=3.0,    # m
        batt_capacity=5.0,    # Ah
        fuselage_width=0.13,  # m
        fuselage_height=0.13, # m
    )


def make_realistic_thrust_curve() -> tuple[float, float, float]:
    """
    Returns coefficients [a, b, c] for the fixed-throttle curve

        T(V) = a V^2 + b V + c

    with thrust in newtons and velocity in m/s.

    Representative values:
        T(0)  = 38.0 N
        T(15) = 26.75 N
        T(20) = 21.0 N
        T(25) = 14.75 N
    """
    return (-0.01, -0.60, 38.0)


def make_realistic_mass_properties() -> tuple[
    float,
    tuple[float, float, float],
    np.ndarray,
]:
    """
    Returns mass, CG, and inertia matrix in SI units.

    The inertia tensor is symmetric and expressed about the CG:
        mass: kg
        CG: m
        inertia: kg m^2
    """
    mass = 7.5

    # Wing leading edge is x = 0 in the current geometry.
    # This CG is slightly aft of the wing quarter-chord.
    cg = (0.085, 0.0, 0.0)

    inertia_matrix = np.array(
        [
            [0.72, 0.00, 0.02],
            [0.00, 0.31, 0.00],
            [0.02, 0.00, 0.96],
        ],
        dtype=float,
    )

    return mass, cg, inertia_matrix


def assert_finite(value: float, name: str) -> None:
    assert math.isfinite(float(value)), f"{name} is not finite: {value}"


def test_aero_main_smoke() -> None:
    """
    End-to-end smoke test for aero_main.

    Passing behavior is either:
      1. a valid trimmed solution, or
      2. a clean, explicitly reported trim failure.

    An exception, malformed result, or partially populated failed result
    causes the test to fail.
    """
    design_vector = make_realistic_design_vector()
    parameter_vector = ParameterVector()
    thrust_velocity = make_realistic_thrust_curve()
    mass, cg, inertia_matrix = make_realistic_mass_properties()

    result = aero_main(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
        thrust_velocity=thrust_velocity,
        flight_time_fit=(0.0, 0.0, 1e6),
        mission=1,
        cg=cg,
        inertia_matrix=inertia_matrix,
        mass=mass,
        disp_res=True,
        debug=False,
    )

    assert isinstance(result, AeroScore)
    assert isinstance(result.can_fly, bool)
    assert result.lap_time > 0.0
    assert not math.isnan(float(result.lap_time))
    assert 0.0 <= result.penalty <= 10.0

    if result.can_fly:
        assert_finite(result.lap_time, "lap_time")
        assert result.penalty == 0.0


def test_input_inertia_matrix_is_physically_valid() -> None:
    """Checks the representative test inertia tensor itself."""
    _, _, inertia_matrix = make_realistic_mass_properties()

    assert inertia_matrix.shape == (3, 3)
    assert np.allclose(inertia_matrix, inertia_matrix.T)
    assert np.all(np.linalg.eigvalsh(inertia_matrix) > 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
