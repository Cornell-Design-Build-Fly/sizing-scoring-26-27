"""Regression tests for the static-margin feedback path.

Covers the five defects fixed after the 2026-27 rules update:

1. the coarse stability model reported ``-Cma/CLa`` as a static margin, which
   does not preserve ``dCma/dx_cg = CLa / c`` and so implied a neutral point
   that moved with the CG;
2. the mechanical static-margin penalty was hardcoded to zero, leaving no
   static-margin feedback anywhere in the objective;
3. the buffered penalty band could extend below zero, letting a design with the
   CG behind the neutral point escape penalty;
4. a Mission-2 placement that reached the tail raised instead of being clamped
   and penalized, turning a smooth trade into a rejection cliff;
5. a design whose cruise never trimmed carried zero penalty, so it outscored a
   design that trimmed but was marginally unstable.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import replace

import aerosandbox as asb
import numpy as np

from src.aero.aero_score import MAX_PENALTY
from src.aero.cruise_analysis_fast import cruise_analysis_fast
from src.aero.stability_analysis_coarse import stability_analysis_coarse
from src.main import (
    MAX_TAKEOFF_MASS_KG,
    OVERWEIGHT_BASE_PENALTY,
    overweight_penalty,
    resolved_aerodynamic_design_vector,
)
from src.mech.main_mech import evaluate_mechanical_module
from src.mech.mass_properties import (
    buffered_static_margin_penalty,
    estimate_aerodynamic_center_x,
)
from src.mech.models import MechanicalModuleConfig, StaticMarginConfig
from src.prop.continuous_prop_database import load_default_continuous_prop_database
from src.prop.main_prop import prop_main
from src.vectors import DesignVector, ParameterVector

PV = ParameterVector()

# A design known to fly all three missions (best result of the topline run in
# data_dump/opt_topline/run_20260903_212050, score 7.23).
FLYABLE = dict(
    wing_span=1.7083958638898031,
    wing_chord=0.39419421943651445,
    tail_arm=0.7675520608196428,
    nose_length=0.28965551943374684,
    extra_shipping_containers=0,
    sensor_weight_kg=14.573582130610939,
    mission3_sensor_weight_kg=12.973052914518941,
    batt_capacity=2.6115012009206993,
    battery_cell_count=8,
    prop_diameter_in=15.717464888095260,
    prop_pitch_in=8.9767276689519560,
    motor_kv=294.36581357421875,
    motor_max_power=2314.6015625,
    cruise_throttle=0.9605384893417358,
    mission3_cruise_throttle=0.8791480660438538,
)


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _stability_at_cg(design: DesignVector, cg_x_m: float):
    """Run the coarse stability model with the CG forced to ``cg_x_m``."""
    with _quiet():
        mech = evaluate_mechanical_module(design, parameter_vector=PV)
        resolved = resolved_aerodynamic_design_vector(design, mech)
        thrust, _ = prop_main(
            resolved,
            PV,
            mission=2,
            prop_database=load_default_continuous_prop_database(),
            disp_res=False,
        )
    properties = mech.for_mission("M2")
    inertia = properties.inertia_tensor_kg_m2
    mass_props = asb.MassProperties(
        mass=properties.total_mass_kg,
        x_cg=cg_x_m,
        y_cg=properties.cg_m[1],
        z_cg=properties.cg_m[2],
        Ixx=inertia[0, 0], Iyy=inertia[1, 1], Izz=inertia[2, 2],
        Ixy=inertia[0, 1], Iyz=inertia[1, 2], Ixz=inertia[0, 2],
    )
    with _quiet():
        cruise = cruise_analysis_fast(
            resolved,
            PV,
            thrust,
            (cg_x_m, properties.cg_m[1], properties.cg_m[2]),
            properties.total_mass_kg,
            2,
            False,
        )
        assert cruise.converged
        return stability_analysis_coarse(resolved, cruise, mass_props), resolved, mech


# ---------------------------------------------------------------------------
# 1. The coarse static margin is geometric and agrees with the mech module.
# ---------------------------------------------------------------------------

def test_coarse_static_margin_is_geometric_and_matches_mech() -> None:
    design = DesignVector(**FLYABLE)
    with _quiet():
        mech = evaluate_mechanical_module(design, parameter_vector=PV)
    cg_x = mech.for_mission("M2").cg_m[0]
    stability, resolved, _ = _stability_at_cg(design, cg_x)

    expected = (mech.neutral_point_x_m - cg_x) / resolved.wing_chord
    assert stability.static_margin == pytest_approx(expected)
    assert stability.neutral_point_x_m == pytest_approx(mech.neutral_point_x_m)


def test_neutral_point_does_not_move_with_cg() -> None:
    """The defining property the -Cma/CLa formulation violated."""
    design = DesignVector(**FLYABLE)
    with _quiet():
        mech = evaluate_mechanical_module(design, parameter_vector=PV)
    base_cg = mech.for_mission("M2").cg_m[0]
    chord = design.wing_chord

    neutral_points, margins, offsets = [], [], []
    for offset in (-0.04, 0.0, 0.04):
        stability, resolved, _ = _stability_at_cg(design, base_cg + offset)
        neutral_points.append(stability.neutral_point_x_m)
        margins.append(stability.static_margin)
        offsets.append(offset)

    assert np.allclose(neutral_points, neutral_points[0], atol=1e-12)
    # d(SM)/d(x_cg) must be exactly -1 / chord.
    slope = np.diff(margins) / np.diff(offsets)
    assert np.allclose(slope, -1.0 / chord, rtol=1e-9)


def test_legacy_cma_static_margin_is_retained_as_a_diagnostic() -> None:
    design = DesignVector(**FLYABLE)
    with _quiet():
        mech = evaluate_mechanical_module(design, parameter_vector=PV)
    stability, _, _ = _stability_at_cg(design, mech.for_mission("M2").cg_m[0])
    assert stability.static_margin_from_cma is not None
    # The two genuinely differ; that difference is the bug being guarded against.
    assert stability.static_margin_from_cma != pytest_approx(
        stability.static_margin, absolute=1e-6
    )


# ---------------------------------------------------------------------------
# 2 and 3. The buffered penalty is live and floors at zero static margin.
# ---------------------------------------------------------------------------

def test_negative_static_margin_is_always_penalized() -> None:
    config = StaticMarginConfig()
    # Buffer (0.15) exceeds minimum (0.10); without the floor the unpenalized
    # band would reach -0.05 and this would return 0.0.
    assert config.minimum - config.optimizer_penalty_buffer < 0.0
    assert buffered_static_margin_penalty(-0.01, config) > 0.0
    assert buffered_static_margin_penalty(-0.20, config) > (
        buffered_static_margin_penalty(-0.01, config)
    )


def test_buffered_penalty_is_zero_inside_the_band_and_monotone_outside() -> None:
    config = StaticMarginConfig()
    assert buffered_static_margin_penalty(config.target, config) == 0.0
    assert buffered_static_margin_penalty(config.minimum, config) == 0.0
    assert buffered_static_margin_penalty(config.maximum, config) == 0.0
    upper = config.maximum + config.optimizer_penalty_buffer
    assert buffered_static_margin_penalty(upper, config) == 0.0
    escalating = [
        buffered_static_margin_penalty(upper + step, config)
        for step in (0.05, 0.20, 0.60, 2.00)
    ]
    assert escalating == sorted(escalating)
    assert all(0.0 < value <= 10.0 for value in escalating)
    assert buffered_static_margin_penalty(float("nan"), config) == 10.0


def test_mechanical_penalty_is_wired_to_the_flown_missions() -> None:
    design = DesignVector(**FLYABLE)
    config = MechanicalModuleConfig()
    assert set(config.static_margin.penalized_missions) == {"M2", "M3"}
    with _quiet():
        result = evaluate_mechanical_module(design, parameter_vector=PV)
    # The reference design sits inside the band, so it must stay penalty-free.
    assert result.penalty_static_margin == 0.0
    assert set(result.penalty_static_margin_by_mission) == {"M2", "M3"}

    # Loading extra containers drives the uncontrolled Mission-3 margin out of
    # the design band. With the buffer removed the penalty must engage, proving
    # the path is live rather than dead code.
    #
    # Note the band cannot simply be narrowed instead: MechanicalModuleConfig
    # validates that Mission2Config.target_static_margin lies inside it.
    loaded = replace(design, extra_shipping_containers=8)
    strict = MechanicalModuleConfig(
        static_margin=StaticMarginConfig(
            minimum=0.10, target=0.20, maximum=0.23,
            optimizer_penalty_buffer=0.0,
            optimizer_penalty_scale=0.15,
        )
    )
    with _quiet():
        strict_result = evaluate_mechanical_module(
            loaded, config=strict, parameter_vector=PV
        )
    assert strict_result.for_mission("M3").static_margin > 0.23
    assert strict_result.penalty_static_margin_by_mission["M3"] > 0.0
    assert strict_result.penalty_static_margin > 0.0
    assert strict_result.penalty >= strict_result.penalty_static_margin


# ---------------------------------------------------------------------------
# 4. Placement is clamped and penalized rather than raising.
# ---------------------------------------------------------------------------

def test_overlong_payload_is_clamped_and_penalized_not_rejected() -> None:
    design = DesignVector(**{**FLYABLE, "extra_shipping_containers": 7})
    with _quiet():
        result = evaluate_mechanical_module(design, parameter_vector=PV)
    assert result.penalty_placement > 0.0
    assert result.penalty >= result.penalty_placement
    assert any("clamped" in warning for warning in result.warnings)
    # A clamped placement misses the exact target, which is the point.
    assert result.for_mission("M2").static_margin != 0.12


def test_unclamped_placement_still_hits_the_exact_target() -> None:
    design = DesignVector(**FLYABLE)
    config = MechanicalModuleConfig()
    with _quiet():
        result = evaluate_mechanical_module(design, parameter_vector=PV)
    assert result.penalty_placement == 0.0
    assert result.for_mission("M2").static_margin == pytest_approx(
        config.mission2.target_static_margin, absolute=1e-12
    )


# ---------------------------------------------------------------------------
# 5. Untrimmable designs and overweight designs carry real, graded cost.
# ---------------------------------------------------------------------------

def test_non_convergent_cruise_costs_the_maximum_penalty() -> None:
    from src.aero.aero_score import AeroScore
    from src.aero.main_aero import aero_main

    # The stock DesignVector does not trim; it must not be cheaper than a
    # marginally unstable design.
    design = DesignVector()
    with _quiet():
        mech = evaluate_mechanical_module(design, parameter_vector=PV)
        resolved = resolved_aerodynamic_design_vector(design, mech)
        thrust, fit = prop_main(
            resolved, PV, mission=2,
            prop_database=load_default_continuous_prop_database(), disp_res=False,
        )
        properties = mech.for_mission("M2")
        score = aero_main(
            design_vector=resolved,
            parameter_vector=PV,
            thrust_velocity=thrust,
            flight_time_fit=fit,
            mission=2,
            cg=properties.cg_m,
            inertia_matrix=properties.inertia_tensor_kg_m2,
            mass=properties.total_mass_kg,
        )
    assert not score.can_fly
    assert score.penalty == MAX_PENALTY
    assert AeroScore(can_fly=False).penalty == 0.0  # the unsafe default itself


def test_overweight_penalty_is_graded_above_the_limit() -> None:
    assert overweight_penalty(MAX_TAKEOFF_MASS_KG - 0.01) == 0.0
    at_limit = overweight_penalty(MAX_TAKEOFF_MASS_KG)
    assert at_limit == OVERWEIGHT_BASE_PENALTY
    heavier = [
        overweight_penalty(MAX_TAKEOFF_MASS_KG + over)
        for over in (1.0, 10.0, 100.0)
    ]
    assert heavier == sorted(heavier)
    assert heavier[0] > at_limit
    # Strictly increasing, so differential evolution has a direction to descend
    # instead of a plateau of ties.
    assert len(set(heavier)) == len(heavier)


# ---------------------------------------------------------------------------
# Local approx helper so this file does not require pytest to be importable.
# ---------------------------------------------------------------------------

class pytest_approx:  # noqa: N801 - mirrors pytest.approx usage
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
    print("All static-margin feedback tests passed.")
