from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.vectors import DesignVector

SECONDS_PER_MISSION = 300.0
METERS_TO_FEET = 3.28
METERS_TO_INCHES = 39.37

# Ground mission references
DUCKS_TIME = 2.5
PUCKS_TIME = 1.5
BANNER_TIME = 7.0
BEST_GM_TIME_S = 25.0

# Mission 2 reference
BEST_M2_PROFIT = 2613

# Mission 3 references
BEST_M3_LAP_TIME_S = 46
BEST_BANNER_LENGTH_IN = 241
BEST_RAC = 0.90


@dataclass(frozen=True)
class ScoringReferences:
    """Reference values that normalize the mission scores."""

    seconds_per_mission: float = SECONDS_PER_MISSION
    meters_to_feet: float = METERS_TO_FEET
    meters_to_inches: float = METERS_TO_INCHES
    ducks_time_s: float = DUCKS_TIME
    pucks_time_s: float = PUCKS_TIME
    banner_time_s: float = BANNER_TIME
    best_gm_time_s: float = BEST_GM_TIME_S
    best_m2_profit: float = BEST_M2_PROFIT
    best_m3_lap_time_s: float = BEST_M3_LAP_TIME_S
    best_banner_length_in: float = BEST_BANNER_LENGTH_IN
    best_rac: float = BEST_RAC


DEFAULT_SCORING_REFERENCES = ScoringReferences()


def scoring_reference_values(refs: ScoringReferences | None = None) -> dict:
    """Return the score-normalization constants used for this run."""

    refs = DEFAULT_SCORING_REFERENCES if refs is None else refs
    best_m3_laps = int(refs.seconds_per_mission // refs.best_m3_lap_time_s)
    return {
        "seconds_per_mission": refs.seconds_per_mission,
        "meters_to_feet": refs.meters_to_feet,
        "meters_to_inches": refs.meters_to_inches,
        "ground": {
            "ducks_time_s": refs.ducks_time_s,
            "pucks_time_s": refs.pucks_time_s,
            "banner_time_s": refs.banner_time_s,
            "best_time_s": refs.best_gm_time_s,
            "normalization": "min(best_time_s, ground_time_s) / ground_time_s",
        },
        "mission_1": {
            "required_laps": 3.0,
            "time_limit_s": refs.seconds_per_mission,
        },
        "mission_2": {
            "best_profit": refs.best_m2_profit,
            "normalization": "max(best_profit, profit)",
        },
        "mission_3": {
            "best_lap_time_s": refs.best_m3_lap_time_s,
            "best_banner_length_in": refs.best_banner_length_in,
            "best_rac": refs.best_rac,
            "best_num_laps": best_m3_laps,
            "reference_performance": (
                best_m3_laps * refs.best_banner_length_in / refs.best_rac
            ),
            "normalization": "max(reference_performance, performance)",
        },
    }


def gm_score(
    dv: DesignVector,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> float:
    """Returns the ground mission score."""
    time_gm = 2.0 * (
        dv.ducks_num * refs.ducks_time_s + dv.pucks_num * refs.pucks_time_s
    ) + refs.banner_time_s
    normalization_time = min(refs.best_gm_time_s, time_gm)
    return normalization_time / time_gm


def m1_score(
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> float:
    """Returns the mission 1 score."""
    mission_time_s = lap_time_s * 3.0
    if mission_time_s < refs.seconds_per_mission:
        return 1.0
    return 0.0


def m2_score(
    dv: DesignVector,
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> float:
    """Returns the mission 2 score."""
    num_laps = int(refs.seconds_per_mission // lap_time_s)
    income_passengers = dv.ducks_num * (6.0 + 2.0 * num_laps)
    income_cargo = dv.pucks_num * (10.0 + 8.0 * num_laps)
    efficiency_factor = dv.batt_energy / 100.0
    cost = (
        num_laps
        * (10.0 + dv.ducks_num * 0.5 + dv.pucks_num * 2.0)
        * efficiency_factor
    )
    profit = (income_passengers + income_cargo) - cost
    normalization_profit = max(refs.best_m2_profit, profit)
    return 1.0 + (profit / normalization_profit)


<<<<<<< HEAD
def m2_score(dv: DesignVector, lap_time_s: float) -> float:
    """Returns the mission 2 score."""
    num_laps = int(SECONDS_PER_MISSION // lap_time_s)
    return _m2_score_for_laps(dv, num_laps)


def m2_optimization_score(dv: DesignVector, lap_time_s: float) -> float:
    """Continuously reward M2 speed between official lap thresholds."""
    return _m2_score_for_laps(dv, continuous_laps(lap_time_s))


def _m3_score_for_laps(dv: DesignVector, num_laps: float) -> float:
    """Evaluate M3 for either an official or relaxed lap count."""
    wing_span_ft = dv.wing_span * METERS_TO_FEET
    rac = 0.05 * wing_span_ft + 0.75
    best_num_laps = int(SECONDS_PER_MISSION // BEST_M3_LAP_TIME_S)
    performance = num_laps * dv.banner_length * METERS_TO_INCHES / rac
    reference_performance = best_num_laps * BEST_BANNER_LENGTH_IN / BEST_RAC
=======
def m3_score(
    dv: DesignVector,
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> float:
    """Returns the mission 3 score."""
    wing_span_ft = dv.wing_span * refs.meters_to_feet
    rac = 0.05 * wing_span_ft + 0.75
    num_laps = int(refs.seconds_per_mission // lap_time_s)
    best_num_laps = int(refs.seconds_per_mission // refs.best_m3_lap_time_s)
    performance = num_laps * dv.banner_length * refs.meters_to_inches / rac
    reference_performance = (
        best_num_laps * refs.best_banner_length_in / refs.best_rac
    )
>>>>>>> 56070c4e7dfac343ec5be23bac66a873d5be1183
    normalization_performance = max(reference_performance, performance)
    return 2.0 + (performance / normalization_performance)


<<<<<<< HEAD
def m3_score(dv: DesignVector, lap_time_s: float) -> float:
    """Returns the mission 3 score."""
    num_laps = int(SECONDS_PER_MISSION // lap_time_s)
    return _m3_score_for_laps(dv, num_laps)


def m3_optimization_score(dv: DesignVector, lap_time_s: float) -> float:
    """Continuously reward M3 speed between official lap thresholds."""
    return _m3_score_for_laps(dv, continuous_laps(lap_time_s))


def total_score(dv: DesignVector, lap_time_m1: float, lap_time_m2: float, lap_time_m3: float,
=======
def total_score(
    dv: DesignVector,
    lap_time_m1: float,
    lap_time_m2: float,
    lap_time_m3: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
>>>>>>> 56070c4e7dfac343ec5be23bac66a873d5be1183
) -> tuple[float, list[float]]:
    """Returns the total score and mission-by-mission breakdown."""
    gm = gm_score(dv, refs)
    m1 = m1_score(lap_time_m1, refs)
    m2 = 0.0
    m3 = 0.0

    if m1 > 0.0:
        m2 = m2_score(dv, lap_time_m2, refs)
        if m2 > 0.0:
            m3 = m3_score(dv, lap_time_m3, refs)

    breakdown = [gm, m1, m2, m3]
    # print(f"Score breakdown: GM={gm:.2f}, M1={m1:.2f}, M2={m2:.2f}, M3={m3:.2f}")
    return gm + m1 + m2 + m3, breakdown


def total_optimization_score(
    dv: DesignVector,
    lap_time_m1: float,
    lap_time_m2: float,
    lap_time_m3: float,
) -> tuple[float, list[float]]:
    """Return relaxed lap credit while preserving official mission unlocks."""
    gm = gm_score(dv)
    official_m1 = m1_score(lap_time_m1)
    m1 = m1_optimization_score(lap_time_m1)
    m2 = 0.0
    m3 = 0.0

    if official_m1 > 0.0:
        official_m2 = m2_score(dv, lap_time_m2)
        m2 = m2_optimization_score(dv, lap_time_m2)
        if official_m2 > 0.0:
            m3 = m3_optimization_score(dv, lap_time_m3)

    breakdown = [gm, m1, m2, m3]
    return sum(breakdown), breakdown
