"""Well-conditioned lateral-directional stability, used for the spiral check.

Why this exists
---------------
``aerosandbox.dynamics.flight_dynamics.airplane.get_modes`` computes the spiral
root from the classical approximation (Flight Vehicle Aerodynamics, Eq. 9.66)::

    spiral_parameter = Cnr - Cnb * Clr / Clb
    lambda_spiral    = Q*S*b**2 / (2*Izz*u0) * spiral_parameter

That expression divides by ``Clb``. This airframe is built with **zero wing
dihedral** (``ASBDesignVector.make_airplane`` puts the root and tip leading
edges at the same z), so ``Clb`` is around -0.0005 where a conventional
aircraft has -0.05 to -0.10. The approximation is therefore singular here and
returns spiral roots spanning roughly -8700 to +2000 per second, against a
physical range of about 0.01 to 1. The full AeroBuildup derivative set
reproduces the same blow-up, so this is a property of the approximation, not of
the coarse derivative estimates.

Solving the 4-state lateral system directly is well conditioned as ``Clb``
approaches zero and yields spiral roots in the physical range.

Assumptions
-----------
* States are ``[beta, p, r, phi]`` in stability axes, small-angle, wings-level
  trim, with the ``Ixz`` product of inertia neglected.
* ``Cnp`` is estimated as ``-CL / 8`` (standard straight-wing value) and
  ``CYp`` as 0, because neither is produced by the derivative models here.
  Both were checked: varying ``Cnp`` over ``-CL/4`` to ``0`` moves the spiral
  time-to-double by under 3%, so the result is not driven by that estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Yaw moment due to roll rate, as a fraction of CL. Straight-wing estimate.
CNP_PER_CL: float = -1.0 / 8.0


@dataclass(frozen=True)
class LateralModes:
    """Eigenvalues of the 4-state lateral-directional system."""

    spiral_eigenvalue_real: float
    roll_subsidence_eigenvalue_real: float
    dutch_roll_eigenvalue_real: float
    dutch_roll_eigenvalue_imag: float


def lateral_state_matrix(
    *,
    CYb: float, CYr: float,
    Clb: float, Clp: float, Clr: float,
    Cnb: float, Cnr: float,
    CL: float,
    dynamic_pressure_pa: float,
    wing_area_m2: float,
    wing_span_m: float,
    mass_kg: float,
    Ixx: float,
    Izz: float,
    velocity_m_s: float,
    gravity_m_s2: float,
) -> np.ndarray:
    """Build the ``[beta, p, r, phi]`` lateral-directional state matrix."""

    if velocity_m_s <= 0.0 or mass_kg <= 0.0 or Ixx <= 0.0 or Izz <= 0.0:
        raise ValueError("Lateral state matrix requires positive V, mass, Ixx, Izz.")

    qs = dynamic_pressure_pa * wing_area_m2
    b = wing_span_m
    u0 = velocity_m_s
    Cnp = CNP_PER_CL * CL

    y_beta = qs * CYb / mass_kg
    y_r = qs * b * CYr / (2.0 * mass_kg * u0)

    l_beta = qs * b * Clb / Ixx
    l_p = qs * b * b * Clp / (2.0 * Ixx * u0)
    l_r = qs * b * b * Clr / (2.0 * Ixx * u0)

    n_beta = qs * b * Cnb / Izz
    n_p = qs * b * b * Cnp / (2.0 * Izz * u0)
    n_r = qs * b * b * Cnr / (2.0 * Izz * u0)

    return np.array(
        [
            [y_beta / u0, 0.0, y_r / u0 - 1.0, gravity_m_s2 / u0],
            [l_beta,      l_p, l_r,            0.0],
            [n_beta,      n_p, n_r,            0.0],
            [0.0,         1.0, 0.0,            0.0],
        ],
        dtype=float,
    )


def lateral_modes(state_matrix: np.ndarray) -> LateralModes:
    """Classify the lateral eigenvalues into spiral, roll subsidence, Dutch roll."""

    eigenvalues = np.linalg.eigvals(state_matrix)
    real_roots = sorted(
        float(value.real) for value in eigenvalues if abs(value.imag) < 1e-9
    )
    oscillatory = [value for value in eigenvalues if value.imag > 1e-9]

    if real_roots:
        # The spiral is the slowest real root, roll subsidence the fastest.
        spiral = real_roots[-1]
        roll = real_roots[0]
    else:
        spiral = float("nan")
        roll = float("nan")

    if oscillatory:
        dutch = max(oscillatory, key=lambda value: value.real)
        dutch_real, dutch_imag = float(dutch.real), float(dutch.imag)
    else:
        dutch_real, dutch_imag = float("nan"), 0.0

    return LateralModes(
        spiral_eigenvalue_real=spiral,
        roll_subsidence_eigenvalue_real=roll,
        dutch_roll_eigenvalue_real=dutch_real,
        dutch_roll_eigenvalue_imag=dutch_imag,
    )


def time_to_double_s(eigenvalue_real: float) -> float:
    """Seconds for a divergent mode to double amplitude; inf if convergent."""

    if not np.isfinite(eigenvalue_real) or eigenvalue_real <= 0.0:
        return float("inf")
    return float(np.log(2.0) / eigenvalue_real)


__all__ = [
    "CNP_PER_CL",
    "LateralModes",
    "lateral_modes",
    "lateral_state_matrix",
    "time_to_double_s",
]
