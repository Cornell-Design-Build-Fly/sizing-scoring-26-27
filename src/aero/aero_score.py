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
    spiral doubling time > 4 s        : spiral divergence is slow enough to
                                        be corrected by the pilot / autopilot

Spiral criterion: a spiral growth rate λ_spiral ≤ ln(2)/4 ≈ 0.173 s⁻¹ means
the aircraft doubles its bank angle in ≥ 4 s when disturbed — accepted per
MIL-SPEC and typical DBF practice.  Stable spirals (λ ≤ 0) always pass.

When can_fly is False the penalty is a weighted, log-scale combination of
how far each hard constraint is violated.

Usage
-----
    from src.aero.aero_score import aero_score, AeroScore
    score: AeroScore = aero_score(cruise_condition, stability_result)
"""

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
LOOP_360_RAD:      float = 2.0 * np.pi   # 360° loop at far (upwind) end [rad]
TURN_180_COUNT:    int   = 2             # number of 180° reversals per lap
TURN_180_RAD:      float = np.pi         # each 180° reversal [rad]
N_ZS:              float = 2.5           # structural limit load factor (civil)

# ── Flyability Thresholds ──────────────────────────────────────────────────
CMA_LIMIT:    float = 0.0   # Cma must be strictly below this
CNB_LIMIT:    float = 0.0   # Cnb must be strictly above this
SM_LIMIT:     float = 0.0   # static_margin must be strictly above this

# Spiral: growth rate must not exceed ln(2)/T_double_min
# → eigenvalue.real ≤ SPIRAL_RATE_MAX means doubling time ≥ 4 s
SPIRAL_DOUBLING_TIME_MIN_S: float = 4.0
SPIRAL_RATE_MAX: float = np.log(2.0) / SPIRAL_DOUBLING_TIME_MIN_S  # ≈ 0.1733 s⁻¹

# ── Penalty Scale Parameters ───────────────────────────────────────────────
# The "scale" for each constraint is the violation magnitude that drives that
# component's individual log penalty to exactly 10.  Smaller scale = penalty
# rises faster for small violations.
#
#   static margin:  scale=0.10  →  10 % MAC beyond boundary → full penalty
#   Cma:            scale=0.50  →  Cma = +0.50 /rad is severe instability
#   Cnb:            scale=0.10  →  Cnb = -0.10 /rad is severe instability
#   spiral rate:    scale=SPIRAL_RATE_MAX  →  2× threshold (dt=2 s) → full penalty
SM_PENALTY_SCALE:     float = 0.10          # [fraction of MAC]
CMA_PENALTY_SCALE:    float = 0.50          # [1/rad]
CNB_PENALTY_SCALE:    float = 0.10          # [1/rad]
SPIRAL_PENALTY_SCALE: float = SPIRAL_RATE_MAX  # [s⁻¹]

# Weights applied to each component penalty before summing.
# Must sum to 1.0 so the weighted total stays in [0, 10].
W_SM:     float = 0.40   # static margin — most critical single metric
W_CMA:    float = 0.25   # longitudinal stability
W_CNB:    float = 0.15   # directional stability
W_SPIRAL: float = 0.20   # spiral mode


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
    lap_time:                float | None = None
    penalty:                 float | None = None

    # Per-constraint breakdown (useful for debugging and grad-free optimizers)
    penalty_static_margin:   float | None = None
    penalty_longitudinal:    float | None = None
    penalty_directional:     float | None = None
    penalty_spiral:          float | None = None


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
    total_turn_rad = LOOP_360_RAD + TURN_180_COUNT * TURN_180_RAD  # = 4π
    t_turns = total_turn_rad / omega

    return t_straight + t_turns


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def aero_score(
        cruise_condition: CruiseCondition,
        stability_result: StabilityResult,
        parameter_vector: ParameterVector,
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

    # ── Flyability gates ──────────────────────────────────────────────────
    longitudinally_stable = stability_result.Cma < CMA_LIMIT          # Cma < 0
    directionally_stable  = stability_result.Cnb > CNB_LIMIT          # Cnb > 0
    cg_ahead_of_np        = stability_result.static_margin > SM_LIMIT # SM  > 0

    # Spiral: eigenvalue is real; positive means bank angle grows.
    # Pass if growth rate ≤ SPIRAL_RATE_MAX (doubling time ≥ 4 s).
    # stability_result.spiral is a ModeResult; eigenvalue_real is its real part.
    spiral_rate = stability_result.spiral.eigenvalue_real
    spiral_ok   = spiral_rate <= SPIRAL_RATE_MAX

    can_fly = longitudinally_stable and directionally_stable and cg_ahead_of_np and spiral_ok

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
        )

    # Compute how far each violated requirement is outside its boundary.
    sm_violation     = max(0.0, SM_LIMIT     - stability_result.static_margin)
    cma_violation    = max(0.0, stability_result.Cma - CMA_LIMIT)
    cnb_violation    = max(0.0, CNB_LIMIT    - stability_result.Cnb)
    spiral_violation = max(0.0, spiral_rate  - SPIRAL_RATE_MAX)

    p_sm     = _log_penalty(sm_violation,     SM_PENALTY_SCALE)
    p_cma    = _log_penalty(cma_violation,    CMA_PENALTY_SCALE)
    p_cnb    = _log_penalty(cnb_violation,    CNB_PENALTY_SCALE)
    p_spiral = _log_penalty(spiral_violation, SPIRAL_PENALTY_SCALE)

    # Weighted sum; weights sum to 1.0 so total stays in [0, 10].
    penalty = min(10.0, W_SM * p_sm + W_CMA * p_cma + W_CNB * p_cnb + W_SPIRAL * p_spiral)

    return AeroScore(
        lap_time=lap_time,
        can_fly=False,
        penalty=penalty,
        penalty_static_margin=p_sm,
        penalty_longitudinal=p_cma,
        penalty_directional=p_cnb,
        penalty_spiral=p_spiral,
    )
