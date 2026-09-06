"""2026-27 DBF flight- and ground-mission scoring."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import math

from src.vectors import DesignVector


SECONDS_PER_MISSION = 300.0
M1_REQUIRED_LAPS = 3
M2_REQUIRED_LAPS = 5
GROUND_DROP_HEIGHT_IN = 60.0
POUNDS_TO_KG = 0.45359237

# M2/M3 planning references retained from Ubilot commit bc86afc.
GM_REFERENCE_SENSOR_MASS_KG = 40.0 * POUNDS_TO_KG
DEFAULT_BEST_M2_WEIGHT_PER_TIME_KG_S = 0.182332 * POUNDS_TO_KG
DEFAULT_BEST_M3_LAP_WEIGHT_KG = 252.890 * POUNDS_TO_KG


@dataclass(frozen=True)
class ScoringReferences:
    """Contest-wide maxima (or planning estimates) used for normalization."""

    best_m2_weight_per_time_kg_s: float = DEFAULT_BEST_M2_WEIGHT_PER_TIME_KG_S
    best_m3_lap_weight_kg: float = DEFAULT_BEST_M3_LAP_WEIGHT_KG
    best_ground_weight_height_kg_in: float = (
        GM_REFERENCE_SENSOR_MASS_KG * GROUND_DROP_HEIGHT_IN
    )
    seconds_per_mission: float = SECONDS_PER_MISSION
    m3_deployment_time_s: float = 0.0
    m3_recovery_time_s: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.best_m2_weight_per_time_kg_s,
            self.best_m3_lap_weight_kg,
            self.best_ground_weight_height_kg_in,
            self.seconds_per_mission,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Scoring references must be finite and positive.")
        overheads = (self.m3_deployment_time_s, self.m3_recovery_time_s)
        if any(not math.isfinite(value) or value < 0.0 for value in overheads):
            raise ValueError("M3 deployment/recovery times must be finite and nonnegative.")
        if sum(overheads) >= self.seconds_per_mission:
            raise ValueError("M3 deployment/recovery must fit in the flight window.")


DEFAULT_SCORING_REFERENCES = ScoringReferences()


def round_half_up(value: float, decimal_places: int = 2) -> float:
    """Apply the conventional rounding required by DBF rule 6.2."""

    quantum = Decimal(1).scaleb(-decimal_places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def _official_mass_kg(mass_kg: float) -> float:
    """Round a scoring weight to 0.01 lb and return its SI equivalent."""

    return round_half_up(mass_kg / POUNDS_TO_KG) * POUNDS_TO_KG


def _valid_time(value_s: float, *, allow_zero: bool = False) -> bool:
    return math.isfinite(value_s) and (value_s >= 0.0 if allow_zero else value_s > 0.0)


def scoring_reference_values(
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> dict[str, object]:
    return {
        "flight_window_s": refs.seconds_per_mission,
        "mission_1": {"required_laps": M1_REQUIRED_LAPS},
        "mission_2": {
            "required_laps": M2_REQUIRED_LAPS,
            "best_weight_per_time_kg_s": refs.best_m2_weight_per_time_kg_s,
            "best_weight_per_time_lb_s": refs.best_m2_weight_per_time_kg_s / POUNDS_TO_KG,
        },
        "mission_3": {
            "best_lap_weight_kg": refs.best_m3_lap_weight_kg,
            "best_lap_weight_lb": refs.best_m3_lap_weight_kg / POUNDS_TO_KG,
            "deployment_time_s": refs.m3_deployment_time_s,
            "recovery_time_s": refs.m3_recovery_time_s,
        },
        "ground": {
            "drop_height_in": GROUND_DROP_HEIGHT_IN,
            "reference_sensor_weight_lb": GM_REFERENCE_SENSOR_MASS_KG / POUNDS_TO_KG,
            "best_weight_height_kg_in": refs.best_ground_weight_height_kg_in,
        },
        "maximum_total_mission_score": 7.5,
    }


def gm_score(
    dv: DesignVector,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
    *,
    successful: bool = True,
) -> float:
    """Score a successful fixed-60-inch Ground Mission."""

    if not successful:
        return 0.0
    performance = _official_mass_kg(dv.sensor_weight_kg) * GROUND_DROP_HEIGHT_IN
    return 0.5 + min(1.0, performance / refs.best_ground_weight_height_kg_in)


def m1_score(
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
    *,
    takeoff_time_s: float = 0.0,
    successful_landing: bool = True,
) -> float:
    """Return one point when takeoff and three laps fit in five minutes."""

    if (
        not successful_landing
        or not _valid_time(lap_time_s)
        or not _valid_time(takeoff_time_s, allow_zero=True)
    ):
        return 0.0
    elapsed = round_half_up(takeoff_time_s + M1_REQUIRED_LAPS * lap_time_s)
    return float(elapsed <= refs.seconds_per_mission)


def m1_optimization_score(
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
    *,
    takeoff_time_s: float = 0.0,
) -> float:
    if not _valid_time(lap_time_s) or not _valid_time(takeoff_time_s, allow_zero=True):
        return 0.0
    elapsed = takeoff_time_s + M1_REQUIRED_LAPS * lap_time_s
    return min(1.0, refs.seconds_per_mission / elapsed)


def m2_score(
    payload_mass_kg: float,
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
    *,
    takeoff_time_s: float = 0.0,
    successful_landing: bool = True,
) -> float:
    """Score payload mass divided by elapsed time for exactly five laps."""

    if (
        not successful_landing
        or not math.isfinite(payload_mass_kg)
        or payload_mass_kg <= 0.0
        or not _valid_time(lap_time_s)
        or not _valid_time(takeoff_time_s, allow_zero=True)
    ):
        return 0.0
    elapsed = round_half_up(takeoff_time_s + M2_REQUIRED_LAPS * lap_time_s)
    if elapsed > refs.seconds_per_mission:
        return 0.0
    performance = _official_mass_kg(payload_mass_kg) / elapsed
    return 1.0 + min(1.0, performance / refs.best_m2_weight_per_time_kg_s)


def m2_optimization_score(
    payload_mass_kg: float,
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
    *,
    takeoff_time_s: float = 0.0,
) -> float:
    official = m2_score(payload_mass_kg, lap_time_s, refs, takeoff_time_s=takeoff_time_s)
    if official > 0.0:
        return official
    if (
        not math.isfinite(payload_mass_kg)
        or payload_mass_kg <= 0.0
        or not _valid_time(lap_time_s)
        or not _valid_time(takeoff_time_s, allow_zero=True)
    ):
        return 0.0
    elapsed = takeoff_time_s + M2_REQUIRED_LAPS * lap_time_s
    completion = min(1.0, refs.seconds_per_mission / elapsed)
    normalized = min(
        1.0,
        (payload_mass_kg / elapsed) / refs.best_m2_weight_per_time_kg_s,
    )
    return completion * (1.0 + normalized)


def completed_m3_laps(
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
    *,
    takeoff_time_s: float = 0.0,
) -> int:
    """Count whole laps after takeoff, deployment, and recovery overhead."""

    if not _valid_time(lap_time_s) or not _valid_time(takeoff_time_s, allow_zero=True):
        return 0
    lap_window = (
        refs.seconds_per_mission
        - takeoff_time_s
        - refs.m3_deployment_time_s
        - refs.m3_recovery_time_s
    )
    return max(0, int(lap_window // lap_time_s))


def m3_score(
    dv: DesignVector,
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
    *,
    takeoff_time_s: float = 0.0,
    successful_landing: bool = True,
) -> float:
    """Score whole laps multiplied by derived steel-sensor weight."""

    if not successful_landing:
        return 0.0
    laps = completed_m3_laps(lap_time_s, refs, takeoff_time_s=takeoff_time_s)
    if laps < 1:
        return 0.0
    performance = laps * _official_mass_kg(dv.sensor_weight_kg)
    return 2.0 + min(1.0, performance / refs.best_m3_lap_weight_kg)


def m3_optimization_score(
    dv: DesignVector,
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
    *,
    takeoff_time_s: float = 0.0,
) -> float:
    if completed_m3_laps(lap_time_s, refs, takeoff_time_s=takeoff_time_s) < 1:
        return 0.0
    lap_window = (
        refs.seconds_per_mission
        - takeoff_time_s
        - refs.m3_deployment_time_s
        - refs.m3_recovery_time_s
    )
    performance = (lap_window / lap_time_s) * dv.sensor_weight_kg
    return 2.0 + min(1.0, performance / refs.best_m3_lap_weight_kg)


def total_score(
    dv: DesignVector,
    lap_time_m1: float,
    lap_time_m2: float,
    lap_time_m3: float,
    m2_payload_mass_kg: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
    *,
    takeoff_time_m1_s: float = 0.0,
    takeoff_time_m2_s: float = 0.0,
    takeoff_time_m3_s: float = 0.0,
    successful_landing_m1: bool = True,
    successful_landing_m2: bool = True,
    successful_landing_m3: bool = True,
    successful_ground_mission: bool = True,
) -> tuple[float, list[float]]:
    """Return scores using the existing M1-waiver and M2-to-M3 unlock flow."""

    gm = gm_score(dv, refs, successful=successful_ground_mission)
    standalone_m1 = m1_score(
        lap_time_m1,
        refs,
        takeoff_time_s=takeoff_time_m1_s,
        successful_landing=successful_landing_m1,
    )
    m2 = m2_score(
        m2_payload_mass_kg,
        lap_time_m2,
        refs,
        takeoff_time_s=takeoff_time_m2_s,
        successful_landing=successful_landing_m2,
    )
    m1 = 1.0 if m2 > 0.0 else standalone_m1
    m3 = (
        m3_score(
            dv,
            lap_time_m3,
            refs,
            takeoff_time_s=takeoff_time_m3_s,
            successful_landing=successful_landing_m3,
        )
        if m2 > 0.0
        else 0.0
    )
    breakdown = [gm, m1, m2, m3]
    return sum(breakdown), breakdown


def total_optimization_score(
    dv: DesignVector,
    lap_time_m1: float,
    lap_time_m2: float,
    lap_time_m3: float,
    m2_payload_mass_kg: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
    *,
    takeoff_time_m1_s: float = 0.0,
    takeoff_time_m2_s: float = 0.0,
    takeoff_time_m3_s: float = 0.0,
) -> tuple[float, list[float]]:
    """Return smooth optimizer progress while preserving official unlock gates."""

    gm = gm_score(dv, refs)
    official_m2 = m2_score(
        m2_payload_mass_kg,
        lap_time_m2,
        refs,
        takeoff_time_s=takeoff_time_m2_s,
    )
    m1 = (
        1.0
        if official_m2 > 0.0
        else m1_optimization_score(
            lap_time_m1,
            refs,
            takeoff_time_s=takeoff_time_m1_s,
        )
    )
    m2 = m2_optimization_score(
        m2_payload_mass_kg,
        lap_time_m2,
        refs,
        takeoff_time_s=takeoff_time_m2_s,
    )
    m3 = (
        m3_optimization_score(
            dv,
            lap_time_m3,
            refs,
            takeoff_time_s=takeoff_time_m3_s,
        )
        if official_m2 > 0.0
        else 0.0
    )
    breakdown = [gm, m1, m2, m3]
    return sum(breakdown), breakdown
