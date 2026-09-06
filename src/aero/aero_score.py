"""
Aero scoring module.

Given cruise speed and stability analysis results, determines:

    lap_time  — estimated lap time on the DBF course [s]
    can_fly   — True if the design meets all minimum flyability requirements
    penalty   — 0.0 if can_fly is True; otherwise 0–10 on a log scale based
                on how far outside each stability requirement the design is

Flyability requirements (ALL must hold for can_fly = True):
    Cma < 0                           : longitudinally stable
    Cnb > 0                           : directionally stable
    static_margin > 0                 : CG ahead of neutral point
    spiral doubling time ≥ 2.5 s      : spiral divergence is slow enough to
                                        be corrected by the pilot

Spiral criterion: time to double bank angle must be at least 2.5 s, with 10 s
or better incurring no penalty at all.  The doubling time comes from the
4-state lateral solve in src/aero/stability_criteria.py, NOT from the
get_modes spiral approximation, which divides by Clb and is singular on this
zero-dihedral airframe.  Convergent spirals always pass.

When can_fly is False the penalty is a weighted, log-scale combination of
how far each hard constraint is violated.

Usage
-----
    from src.aero.aero_score import aero_score, AeroScore
    score: AeroScore = aero_score(cruise_condition, stability_result)
"""

import math

import numpy as np
from dataclasses import dataclass

from src.aero.custom_classes import CruiseCondition, StabilityResult
from src.vectors import ParameterVector

# ── DBF Course Geometry ────────────────────────────────────────────────────
# Per 26-27 DBF rules (Figure 3.1.1; confirmed from course diagram):
#   - 4 straight legs × 500 ft each = 2000 ft = 609.6 m total straight per lap
#   - 1 × 360° loop at the far (upwind/scoring) end
#   - 2 × 180° reversals per lap (one at each near-end waypoint)
#
# Total turning per lap = 1×360° + 2×180° = 4π rad.
#
# Turn speed follows the corner-velocity (load-factor) model (MAE 5070):
#   V_corner = sqrt(n_zs) * V_stall          [maximum sustained-turn speed]
#   V_turn   = min(V_cruise, V_corner)        [actual turn speed]
#   n_turn   = (V_turn / V_stall)^2          [actual load factor in turn]
#   Ω        = g * sqrt(n_turn^2 − 1) / V_turn   [rad/s]
#
# The structural limit load factor n_zs = 2.5 (civil limit) determines
# the minimum turn radius for a structurally limited aircraft.
STRAIGHT_LENGTH_M: float = 152.4         # 500 ft per straight leg [m]
STRAIGHTS_PER_LAP: int   = 4             # 4 legs × 500 ft = 2000 ft total straight
TURN_180_COUNT:    int   = 2             # number of 180° reversals per lap
TURN_180_RAD:      float = np.pi         # each 180° reversal [rad]
TURN_360_COUNT:    int   = 1             # number of 360° loops per lap
TURN_360_RAD:      float = 2.0 * np.pi   # each 360° loop [rad]
N_ZS:              float = 2.5           # structural limit load factor (civil)

# ── Mission Timing ─────────────────────────────────────────────────────────
# Each flight mission has a five-minute window.  Takeoff, the climb out to
# pattern altitude and the landing all consume part of it and none of that time
# is spent completing scored laps, so the window available for laps is shorter
# than the clock.  Twenty seconds is the team's reserve and matches the
# assumption behind the Mission-2 and Mission-3 best-team reference values in
# src/opt/score.py; those normalizers were produced by the same course model,
# so scoring and the propulsion energy budget must use the same window.
FLIGHT_WINDOW_S:   float = 300.0         # full mission clock [s]
GROUND_TIME_S:     float = 20.0          # takeoff + landing overhead [s]
USABLE_WINDOW_S:   float = FLIGHT_WINDOW_S - GROUND_TIME_S

# ── Flyability Thresholds ──────────────────────────────────────────────────
CMA_LIMIT:    float = 0.0   # Cma must be strictly below this
CNB_LIMIT:    float = 0.0   # Cnb must be strictly above this
SM_LIMIT:     float = 0.0   # static_margin must be strictly above this

# Spiral is judged on time to double bank angle, per the team's criterion:
# at or above 2.5 s is acceptable, 10 s or better is the target, and anything
# faster than 2.5 s is treated as unstable. The doubling time comes from the
# 4-state lateral solve in src/aero/stability_criteria.py, NOT from the
# get_modes spiral approximation, which divides by Clb and is singular on this
# zero-dihedral geometry.
SPIRAL_DOUBLING_TIME_MIN_S:   float = 2.5    # hard bound
SPIRAL_DOUBLING_TIME_IDEAL_S: float = 10.0   # no penalty at or above this
# Penalty charged exactly at the 2.5 s bound; the remainder of the 0-10 range
# is reserved for designs faster (worse) than the bound.
SPIRAL_BOUND_PENALTY: float = 2.0

# ── Penalty Scale Parameters ───────────────────────────────────────────────
# The "scale" for each constraint is the violation magnitude that drives that
# component's individual log penalty to exactly 10.  Smaller scale = penalty
# rises faster for small violations.
#
#   static margin:  scale=0.10  →  10 % MAC beyond boundary → full penalty
#   Cma:            scale=0.50  →  Cma = +0.50 /rad is severe instability
#   Cnb:            scale=0.10  →  Cnb = -0.10 /rad is severe instability
#   spiral:         graded on time-to-double; see _spiral_penalty
SM_PENALTY_SCALE:     float = 0.10          # [fraction of MAC]
CMA_PENALTY_SCALE:    float = 0.50          # [1/rad]
CNB_PENALTY_SCALE:    float = 0.10          # [1/rad]
# The spiral criterion is already normalized to [-1, 1], so a full violation
# maps to the cap.
ENDURANCE_PENALTY_SCALE: float = 1.0

# Upper bound on any single aero penalty, and the value assigned to a design
# that cannot be trimmed at all.
MAX_PENALTY: float = 10.0

# Relative weights applied to each stability penalty. The active weights are
# normalized when summed so the total stays in [0, 10].
#
# Rebalanced 2026-09-04 against what actually binds in a converged population
# of 500 real optimizer candidates:
#   p_sm      fired   0/500  (was the largest weight at 0.40)
#   p_cma     fired   0/500
#   p_cnb     fired 235/500  <- a real, now-accurate discriminator
#   p_spiral  fired 217/500  but saturated at the cap in 215 of them
# Battery endurance is no longer scored here. The propulsion module owns the
# segment-resolved takeoff, climb, straight, turn, and re-acceleration energy
# check; keeping the old quadratic flight-time fit here judged the same battery
# twice using inconsistent models. The remaining stability weights are
# normalized without changing their relative importance.
# Spiral is down-weighted because
# with zero wing dihedral it is violated by essentially every design and so
# cannot discriminate between them; it is kept as a smooth signal rather than
# a saturated step. See src/aero/stability_criteria.py.
W_SM:        float = 0.25
W_CMA:       float = 0.10
W_CNB:       float = 0.30   # directional stability — dominant real constraint
W_SPIRAL:    float = 0.10
W_ENDURANCE: float = 0.0
AERO_PENALTY_WEIGHT_SUM = W_SM + W_CMA + W_CNB + W_SPIRAL


# ──────────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AeroScore:
    """
    Output of aero_score().

    Attributes
    ----------
    lap_time : float
        Estimated seconds per lap on the DBF course.
    can_fly : bool
        True if ALL four flyability requirements are met:
        Cma < 0, Cnb > 0, static_margin > 0, spiral doubling time ≥ 4 s.
    penalty : float
        0.0 when can_fly is True.  Otherwise a value in (0, 10] on a log
        scale.  Use as a soft constraint in the optimizer.
    penalty_static_margin : float
        Component penalty from static-margin violation (before weighting).
    penalty_longitudinal : float
        Component penalty from Cma violation (before weighting).
    penalty_directional : float
        Component penalty from Cnb violation (before weighting).
    penalty_spiral : float
        Component penalty from spiral-mode violation (before weighting).
        0 if spiral doubling time ≥ 4 s (or spiral is stable).
    """
    can_fly:                 bool
    lap_time:                float = np.inf
    penalty:                 float = 0
    # Per-constraint breakdown (useful for debugging and grad-free optimizers)
    penalty_static_margin:   float | None = None
    penalty_longitudinal:    float | None = None
    penalty_directional:     float | None = None
    penalty_spiral:          float | None = None
    cruise_speed_mps:        float | None = None
    stall_speed_mps:         float | None = None


# ──────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────

def _log_penalty(violation: float, scale: float) -> float:
    """
    Smooth penalty in [0, 10] for a single constraint violation.

    Uses a log-base-2 scale so that:
        violation = 0      → penalty = 0   (no violation)
        violation = scale  → penalty = 10  (significant violation)
        violation > scale  → penalty still saturates toward 10 (capped)

    The log shape means:
        - Near-zero violations get small penalties (gentle near feasibility)
        - Violations at half the scale give ~5.85 (sensitive in middle range)
        - Beyond the scale the penalty is capped (no need to distinguish
          "very bad" from "extremely bad" for this purpose)

    Args
    ----
    violation : float
        How far outside the constraint boundary the value is. Must be >= 0;
        negative values (constraint satisfied) return 0.
    scale : float
        The violation magnitude that maps to a penalty of exactly 10.
    """
    if violation <= 0.0:
        return 0.0
    return min(10.0, 10.0 * np.log2(1.0 + violation / scale))


def _spiral_penalty(time_to_double_s: float) -> float:
    """Grade the spiral mode on time to double bank angle.

    0 at or above ``SPIRAL_DOUBLING_TIME_IDEAL_S`` (10 s), rising linearly in
    divergence rate to ``SPIRAL_BOUND_PENALTY`` at the ``2.5 s`` bound, then
    continuing on a log scale to the full cap for designs worse than the bound.
    A convergent spiral (infinite doubling time) scores 0.
    """

    if not math.isfinite(time_to_double_s):
        return 0.0  # convergent spiral, or no divergent root
    if time_to_double_s <= 0.0:
        return MAX_PENALTY
    rate = math.log(2.0) / time_to_double_s
    rate_ideal = math.log(2.0) / SPIRAL_DOUBLING_TIME_IDEAL_S
    rate_bound = math.log(2.0) / SPIRAL_DOUBLING_TIME_MIN_S
    if rate <= rate_ideal:
        return 0.0
    if rate <= rate_bound:
        span = rate_bound - rate_ideal
        return float(SPIRAL_BOUND_PENALTY * (rate - rate_ideal) / span)
    excess = (rate - rate_bound) / rate_bound
    return float(
        min(
            MAX_PENALTY,
            SPIRAL_BOUND_PENALTY
            + (MAX_PENALTY - SPIRAL_BOUND_PENALTY)
            * min(1.0, math.log2(1.0 + excess)),
        )
    )


def _compute_lap_time(
        cruise_speed: float,
        stall_speed: float,
        parameter_vector: ParameterVector,
) -> float:
    """
    Estimate lap time on the DBF course using the corner-velocity turn model.

    Course model
    ------------
    Four 500 ft straight legs (2000 ft = 609.6 m total), one 360° loop at the
    far (upwind) end, and two 180° reversals per lap.  Total turning = 4π rad.

    Turn performance uses the load-factor (corner-velocity) model from MAE 5070:

        V_corner = sqrt(n_zs) * V_stall      # max sustained-turn speed [m/s]
        V_turn   = min(V_cruise, V_corner)    # actual turn entry speed
        n_turn   = (V_turn / V_stall)²        # actual load factor in the turn
        Ω        = g * sqrt(n_turn² − 1) / V_turn   # sustained turn rate [rad/s]

    When V_cruise < V_corner the aircraft turns at cruise speed with a reduced
    load factor.  When V_cruise ≥ V_corner the aircraft decelerates to V_corner
    before entering the turn so the structural limit n_zs is not exceeded.

    Args
    ----
    cruise_speed : float
        True airspeed at trimmed cruise [m/s].
    stall_speed : float
        Stall speed at cruise weight, sea-level standard: sqrt(2W/(ρ·S·CL_max)) [m/s].
    parameter_vector : ParameterVector
        Shared physical constants. Uses parameter_vector.gravity [m/s²].

    Returns
    -------
    float
        Estimated lap time [s].  Returns 1e6 if the design cannot sustain a turn
        (n_turn ≤ 1, meaning cruise speed is at or below stall speed).
    """
    g = parameter_vector.gravity  # m/s²

    # Corner velocity: fastest speed at which full n_zs can be pulled
    V_corner = np.sqrt(N_ZS) * stall_speed

    # Actual turn speed: capped at corner velocity to respect structural limit
    V_turn = min(cruise_speed, V_corner)

    # Load factor and sustained turn rate
    n_turn = (V_turn / stall_speed) ** 2
    if n_turn <= 1.0 + 1e-9:
        # Aircraft at or below stall in the turn — cannot complete the course
        return 1e6
    omega = g * np.sqrt(n_turn ** 2 - 1.0) / V_turn  # rad/s

    # Straight-segment time
    t_straight = STRAIGHTS_PER_LAP * STRAIGHT_LENGTH_M / cruise_speed

    # Turn time: 1 × 360° loop + 2 × 180° reversals = 4π total
    total_turn_rad = (
        TURN_180_COUNT * TURN_180_RAD
        + TURN_360_COUNT * TURN_360_RAD
    )
    t_turns = total_turn_rad / omega

    return t_straight + t_turns


def _endurance_values(
    flight_time_fit: tuple[float, float, float],
    cruise_speed: float,
    lap_time: float,
    mission: int,
) -> tuple[float, float, float]:
    """Legacy flight-time-fit diagnostic; not used by ``aero_score``.

    Authoritative battery feasibility lives in ``mission_performance``.
    """
    if mission not in (1, 2, 3):
        raise ValueError(f"Mission must be 1, 2, or 3, got {mission}.")
    available = float(np.polyval(flight_time_fit, cruise_speed))
    if not np.isfinite(available):
        available = 0.0
    available = max(0.0, available)
    required = {1: 3.0 * lap_time, 2: 5.0 * lap_time, 3: 300.0}[mission]
    violation = max(0.0, required - available) / required
    return available, required, _log_penalty(violation, ENDURANCE_PENALTY_SCALE)


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def aero_score(
        cruise_condition: CruiseCondition,
        stability_result: StabilityResult,
        parameter_vector: ParameterVector,
        flight_time_fit: tuple[float, float, float],
        mission: int,
) -> AeroScore:
    """
    Score the aerodynamic performance and flyability of a design.

    Parameters
    ----------
    cruise_condition : CruiseCondition
        Output of cruise_analysis(). Provides trimmed cruise speed
        (cruise_condition.operating_point.velocity) and stall speed
        (cruise_condition.stall_speed) for the lap-time calculation.
    stability_result : StabilityResult
        Output of stability_analysis(). Must contain Cma, Cnb,
        static_margin, and spiral (ModeResult) at minimum.
    parameter_vector : ParameterVector
        Shared physical constants (gravity, rho, etc.).

    Returns
    -------
    AeroScore
        lap_time : estimated seconds per lap
        can_fly  : True if all stability requirements are met
        penalty  : 0 if can_fly, else 0–10 log-scale penalty

    Notes
    -----
    The lap-time model uses the corner-velocity turn model: cruise speed for
    straights, min(V_cruise, sqrt(n_zs)*V_stall) for turns with n_zs=2.5.
    Course: 4×500 ft straights + 1×360° loop + 2×180° reversals = 4π total turn.

    The penalty is 0 whenever can_fly is True.  All four gates must pass:
    Cma < 0, Cnb > 0, static_margin > 0, and spiral doubling time ≥ 4 s.
    """
    # ── Lap time ──────────────────────────────────────────────────────────
    cruise_speed = cruise_condition.operating_point.velocity
    stall_speed  = cruise_condition.stall_speed
    lap_time = _compute_lap_time(cruise_speed, stall_speed, parameter_vector)
    # Kept in the API while callers migrate. The propulsion module performs
    # the authoritative, segment-resolved battery-energy check.
    _ = flight_time_fit
    p_endurance = 0.0
    lap_time_ok = np.isfinite(lap_time) and lap_time < 1e6
    if not lap_time_ok:
        return AeroScore(
            lap_time=lap_time,
            can_fly=False,
            penalty=MAX_PENALTY,
            penalty_static_margin=0.0,
            penalty_longitudinal=0.0,
            penalty_directional=0.0,
            penalty_spiral=0.0,
            cruise_speed_mps=float(cruise_speed),
            stall_speed_mps=float(stall_speed),
        )

    # ── Flyability gates ──────────────────────────────────────────────────
    longitudinally_stable = stability_result.Cma < CMA_LIMIT          # Cma < 0
    directionally_stable  = stability_result.Cnb > CNB_LIMIT          # Cnb > 0
    cg_ahead_of_np        = stability_result.static_margin > SM_LIMIT # SM  > 0

    # Spiral: eigenvalue is real; positive means bank angle grows.
    # stability_result.spiral is a ModeResult; eigenvalue_real is its real part.
    spiral_time_to_double = (
        stability_result.spiral_time_to_double_s
        if stability_result.spiral_time_to_double_s is not None
        else math.inf
    )
    spiral_ok = spiral_time_to_double >= SPIRAL_DOUBLING_TIME_MIN_S

    can_fly = (
        longitudinally_stable and directionally_stable and cg_ahead_of_np
        and spiral_ok and lap_time_ok
    )

    # ── Penalty ──────────────────────────────────────────────────────────
    if can_fly:
        return AeroScore(
            lap_time=lap_time,
            can_fly=True,
            penalty=0.0,
            penalty_static_margin=0.0,
            penalty_longitudinal=0.0,
            penalty_directional=0.0,
            penalty_spiral=0.0,
            cruise_speed_mps=float(cruise_speed),
            stall_speed_mps=float(stall_speed),
        )

    # Compute how far each violated requirement is outside its boundary.
    sm_violation     = max(0.0, SM_LIMIT     - stability_result.static_margin)
    cma_violation    = max(0.0, stability_result.Cma - CMA_LIMIT)
    cnb_violation    = max(0.0, CNB_LIMIT    - stability_result.Cnb)
    p_spiral_direct = _spiral_penalty(spiral_time_to_double)

    p_sm     = _log_penalty(sm_violation,     SM_PENALTY_SCALE)
    p_cma    = _log_penalty(cma_violation,    CMA_PENALTY_SCALE)
    p_cnb    = _log_penalty(cnb_violation,    CNB_PENALTY_SCALE)
    p_spiral = p_spiral_direct

    # Normalize the active relative weights so the total stays in [0, 10].
    penalty = min(
        MAX_PENALTY,
        (
            W_SM * p_sm
            + W_CMA * p_cma
            + W_CNB * p_cnb
            + W_SPIRAL * p_spiral
            + W_ENDURANCE * p_endurance
        )
        / AERO_PENALTY_WEIGHT_SUM,
    )

    return AeroScore(
        lap_time=lap_time,
        can_fly=False,
        penalty=penalty,
        penalty_static_margin=p_sm,
        penalty_longitudinal=p_cma,
        penalty_directional=p_cnb,
        penalty_spiral=p_spiral,
        cruise_speed_mps=float(cruise_speed),
        stall_speed_mps=float(stall_speed),
    )
