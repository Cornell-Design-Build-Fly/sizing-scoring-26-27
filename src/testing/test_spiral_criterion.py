"""Regression tests for the spiral-mode criterion.

The spiral mode is judged on time to double bank angle: at or above 2.5 s is
acceptable, 10 s or better is the target. The doubling time comes from a
4-state lateral solve rather than from ``get_modes``, whose spiral
approximation (FVA Eq. 9.66) divides by ``Clb`` and is singular on this
zero-dihedral geometry.
"""

from __future__ import annotations

import math

import numpy as np

from src.aero.aero_score import (
    MAX_PENALTY,
    SPIRAL_BOUND_PENALTY,
    SPIRAL_DOUBLING_TIME_IDEAL_S,
    SPIRAL_DOUBLING_TIME_MIN_S,
    _spiral_penalty,
)
from src.aero.stability_criteria import (
    lateral_modes,
    lateral_state_matrix,
    time_to_double_s,
)

# Representative trim state from a converged optimizer candidate.
BASE = dict(
    CYb=-0.137, CYr=0.102, Clp=-0.564, Clr=0.092,
    Clb=-0.0005, Cnb=0.0073, Cnr=-0.0323, CL=0.55,
    dynamic_pressure_pa=245.0, wing_area_m2=0.673, wing_span_m=1.708,
    mass_kg=19.0, Ixx=0.243, Izz=0.889, velocity_m_s=20.0, gravity_m_s2=9.806,
)


def _spiral_t2(**overrides) -> float:
    kwargs = {**BASE, **overrides}
    modes = lateral_modes(lateral_state_matrix(**kwargs))
    return time_to_double_s(modes.spiral_eigenvalue_real)


def test_thresholds_match_the_agreed_criterion() -> None:
    assert SPIRAL_DOUBLING_TIME_MIN_S == 2.5
    assert SPIRAL_DOUBLING_TIME_IDEAL_S == 10.0


def test_penalty_is_zero_at_and_above_the_ideal() -> None:
    assert _spiral_penalty(math.inf) == 0.0
    assert _spiral_penalty(SPIRAL_DOUBLING_TIME_IDEAL_S) == 0.0
    assert _spiral_penalty(40.0) == 0.0


def test_penalty_ramps_between_the_ideal_and_the_bound() -> None:
    mid = _spiral_penalty(5.0)
    assert 0.0 < mid < SPIRAL_BOUND_PENALTY
    assert _spiral_penalty(SPIRAL_DOUBLING_TIME_MIN_S) == SPIRAL_BOUND_PENALTY
    # Monotone: shorter doubling time is always worse.
    times = [40.0, 10.0, 7.0, 5.0, 3.5, 2.5, 2.0, 1.5, 1.0, 0.3]
    penalties = [_spiral_penalty(t) for t in times]
    assert penalties == sorted(penalties)


def test_penalty_escalates_past_the_bound_and_caps() -> None:
    assert _spiral_penalty(2.0) > SPIRAL_BOUND_PENALTY
    assert _spiral_penalty(0.3) == MAX_PENALTY
    assert all(0.0 <= _spiral_penalty(t) <= MAX_PENALTY for t in (0.01, 1.0, 100.0))


def test_lateral_solve_is_well_conditioned_as_clb_approaches_zero() -> None:
    """The property the get_modes approximation lacks.

    FVA Eq. 9.66 contains Cnb * Clr / Clb, which diverges as Clb -> 0. The
    4-state solve must stay bounded and physical instead.
    """
    roots = [_spiral_t2(Clb=value) for value in (-0.05, -0.005, -0.0005, -0.00005)]
    for value in roots:
        assert value > 0.0
        # A physical spiral root is order 0.01-1 per second, i.e. T2 >= ~0.7 s.
        assert math.isinf(value) or value > 0.5

    # The singular approximation, for contrast, blows up over the same sweep.
    approximations = [
        BASE["Cnr"] - BASE["Cnb"] * BASE["Clr"] / clb
        for clb in (-0.05, -0.005, -0.0005, -0.00005)
    ]
    assert abs(approximations[-1]) > 100.0 * abs(approximations[0])


def test_stabilizing_dihedral_effect_slows_the_spiral() -> None:
    """More negative Clb (more effective dihedral) must not make it worse."""
    strong = _spiral_t2(Clb=-0.05)
    weak = _spiral_t2(Clb=-0.0005)
    assert strong >= weak or math.isinf(strong)


def test_roll_subsidence_is_convergent() -> None:
    modes = lateral_modes(lateral_state_matrix(**BASE))
    assert modes.roll_subsidence_eigenvalue_real < 0.0


def test_time_to_double_handles_convergent_and_degenerate_roots() -> None:
    assert math.isinf(time_to_double_s(-1.0))
    assert math.isinf(time_to_double_s(0.0))
    assert math.isinf(time_to_double_s(float("nan")))
    assert time_to_double_s(math.log(2.0)) == 1.0


def test_aero_score_actually_applies_the_spiral_penalty() -> None:
    """Covers the wiring, not just the helper: a fast spiral must cost score."""
    from src.aero.aero_score import W_SPIRAL, aero_score
    from src.aero.custom_classes import CruiseCondition, ModeResult, StabilityResult
    from src.vectors import ParameterVector
    import aerosandbox as asb

    op = asb.OperatingPoint(velocity=20.0, alpha=2.0)
    cruise = CruiseCondition(operating_point=op, stall_speed=10.0, converged=True)
    mode = ModeResult(eigenvalue_real=-1.0, eigenvalue_imag=0.0, damping_ratio=1.0)

    def score_for(time_to_double: float):
        stability = StabilityResult(
            phugoid=mode, short_period=mode, dutch_roll=mode, spiral=mode,
            roll_subsidence=mode,
            Cma=-1.0, Cnb=0.01, static_margin=0.20,
            spiral_time_to_double_s=time_to_double,
        )
        # Generous endurance so only the spiral term differs.
        return aero_score(cruise, stability, ParameterVector(), (0.0, 0.0, 1e6), 2)

    good = score_for(40.0)
    bad = score_for(0.5)
    assert good.can_fly
    assert good.penalty == 0.0
    assert not bad.can_fly
    assert bad.penalty > 0.0
    # The whole difference must come through the spiral weight.
    assert bad.penalty == pytest_approx(W_SPIRAL * MAX_PENALTY, absolute=1e-9)
    assert bad.penalty_spiral == MAX_PENALTY


class pytest_approx:  # noqa: N801
    def __init__(self, expected: float, absolute: float = 1e-9) -> None:
        self.expected = float(expected)
        self.absolute = absolute

    def __eq__(self, other) -> bool:
        return abs(float(other) - self.expected) <= self.absolute

    def __repr__(self) -> str:
        return f"approx({self.expected} +- {self.absolute})"


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"PASS  {name}")
    print("All spiral criterion tests passed.")
