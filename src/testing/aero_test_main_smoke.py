from __future__ import annotations

import math

import numpy as np
import pytest

from src.aero.main_aero import aero_main
from src.vectors import DesignVector

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


def make_realistic_thrust_curve() -> list[float]:
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
    return [-0.01, -0.60, 38.0]


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
    thrust_velocity = make_realistic_thrust_curve()
    mass, cg, inertia_matrix = make_realistic_mass_properties()

    # Static margin is currently accepted by aero_main but not used.
    static_margin = 0.10

    result = aero_main(
        design_vector=design_vector,
        thrust_velocity=thrust_velocity,
        cg=cg,
        inertia_matrix=inertia_matrix,
        mass=mass,
        sm=static_margin,
    )

    assert result is not None
    assert isinstance(result.converged, bool)

    if not result.converged:
        # A design with no exact trim at this fixed throttle is allowed to
        # fail, but it must fail cleanly.
        assert (
            result.cruise_condition is None
            or result.cruise_condition.converged is False
        )
        assert result.aero_result is None
        assert result.stability_result is None
        return

    # A successful result must contain every analysis stage.
    assert result.cruise_condition is not None
    assert result.aero_result is not None
    assert result.stability_result is not None
    assert result.cruise_condition.converged is True

    op_point = result.cruise_condition.operating_point
    velocity = float(op_point.velocity)
    alpha = float(op_point.alpha)

    assert_finite(velocity, "velocity")
    assert_finite(alpha, "alpha")

    # Broad physical sanity bounds, not design requirements.
    assert 3.0 <= velocity <= 50.0
    assert -4.0 <= alpha <= 15.0

    aero = result.aero_result

    for name in (
        "CL",
        "CD",
        "CY",
        "Cl",
        "Cm",
        "Cn",
        "L",
        "D",
        "Y",
        "l_b",
        "m_b",
        "n_b",
        "runtime_seconds",
    ):
        assert_finite(getattr(aero, name), name)

    assert aero.converged
    assert aero.CL > 0.0
    assert aero.CD > 0.0
    assert aero.L > 0.0
    assert aero.D > 0.0
    assert aero.runtime_seconds >= 0.0

    # Verify approximate vertical force balance.
    weight = mass * 9.81
    relative_lift_error = abs(aero.L - weight) / weight
    assert relative_lift_error < 0.05, (
        f"Lift is not close to weight: "
        f"L={aero.L:.3f} N, W={weight:.3f} N, "
        f"relative error={relative_lift_error:.3%}"
    )

    # Verify the fixed-throttle thrust balance.
    a, b, c = thrust_velocity
    thrust = a * velocity**2 + b * velocity + c
    relative_thrust_error = abs(aero.D - thrust) / max(abs(thrust), 1.0)

    assert relative_thrust_error < 0.05, (
        f"Drag is not close to thrust: "
        f"D={aero.D:.3f} N, T={thrust:.3f} N, "
        f"relative error={relative_thrust_error:.3%}"
    )

    # Pitching moment should be close to zero at trim.
    moment_scale = max(weight * design_vector.wing_chord, 1.0)
    nondimensional_moment_error = abs(aero.m_b) / moment_scale

    assert nondimensional_moment_error < 0.01, (
        f"Pitching moment is not close to zero: "
        f"m_b={aero.m_b:.6f} N*m"
    )

    # Stability-mode checks.
    stability = result.stability_result

    for mode_name in (
        "phugoid",
        "short_period",
        "dutch_roll",
        "spiral",
        "roll_subsidence",
    ):
        mode = getattr(stability, mode_name)

        assert_finite(mode.eigenvalue_real, f"{mode_name}.eigenvalue_real")
        assert_finite(mode.eigenvalue_imag, f"{mode_name}.eigenvalue_imag")
        assert_finite(mode.damping_ratio, f"{mode_name}.damping_ratio")


def test_input_inertia_matrix_is_physically_valid() -> None:
    """Checks the representative test inertia tensor itself."""
    _, _, inertia_matrix = make_realistic_mass_properties()

    assert inertia_matrix.shape == (3, 3)
    assert np.allclose(inertia_matrix, inertia_matrix.T)
    assert np.all(np.linalg.eigvalsh(inertia_matrix) > 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])