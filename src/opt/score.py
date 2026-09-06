"""2026-27 DBF flight- and ground-mission scoring."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import math

from src.aero.aero_score import FLIGHT_WINDOW_S, GROUND_TIME_S
from src.vectors import DesignVector

SECONDS_PER_MISSION = FLIGHT_WINDOW_S
M1_REQUIRED_LAPS = 3
M2_REQUIRED_LAPS = 5
GROUND_DROP_HEIGHT_IN = 60.0
POUNDS_TO_KG = 0.45359237
GM_REFERENCE_SENSOR_MASS_KG = 35.0 * POUNDS_TO_KG
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
    # Takeoff, the climb out and the landing all burn mission clock without
    # completing a scored lap. The best-team M2 and M3 reference values below
    # were produced by a course model that reserved this same 20 s, and the
    # propulsion energy budget sizes the pack against the same usable window,
    # so all three have to agree. See src/aero/aero_score.py.
    ground_time_s: float = GROUND_TIME_S
    m3_deployment_time_s: float = 0.0
    m3_recovery_time_s: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.best_m2_weight_per_time_kg_s,
            self.best_m3_lap_weight_kg,
            self.best_ground_weight_height_kg_in,
            self.seconds_per_mission,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Scoring references must be finite and positive.")
        overheads = (
            self.ground_time_s,
            self.m3_deployment_time_s,
            self.m3_recovery_time_s,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in overheads):
            raise ValueError(
                "Ground, deployment and recovery times must be finite and nonnegative."
            )
        if sum(overheads) >= self.seconds_per_mission:
            raise ValueError(
                "Ground time and Mission-3 deployment/recovery must fit in the "
                "flight window."
            )

    @property
    def usable_window_s(self) -> float:
        """Mission clock available for scored laps."""

        return self.seconds_per_mission - self.ground_time_s


DEFAULT_SCORING_REFERENCES = ScoringReferences()


def round_half_up(value: float, decimal_places: int = 2) -> float:
    """Apply the conventional rounding required by the DBF rules."""

    quantum = Decimal(1).scaleb(-decimal_places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def _official_mass_kg(mass_kg: float) -> float:
    """Round a scoring weight to 0.01 lb, then return its SI equivalent."""

    return round_half_up(mass_kg / POUNDS_TO_KG) * POUNDS_TO_KG


def scoring_reference_values(
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> dict[str, object]:
    return {
        "flight_window_s": refs.seconds_per_mission,
        "ground_time_s": refs.ground_time_s,
        "usable_window_s": refs.usable_window_s,
        "mission_2": {
            "best_weight_per_time_kg_s": refs.best_m2_weight_per_time_kg_s,
            "required_laps": M2_REQUIRED_LAPS,
        },
        "mission_3": {
            "best_lap_weight_kg": refs.best_m3_lap_weight_kg,
            "deployment_time_s": refs.m3_deployment_time_s,
            "recovery_time_s": refs.m3_recovery_time_s,
        },
        "ground": {
            "drop_height_in": GROUND_DROP_HEIGHT_IN,
            "reference_sensor_weight_lb": 35.0,
            "best_weight_height_kg_in": refs.best_ground_weight_height_kg_in,
        },
    }


def gm_score(
    dv: DesignVector,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> float:
    """Score a successful fixed-60-inch Ground Mission."""

    performance = _official_mass_kg(dv.sensor_weight_kg) * GROUND_DROP_HEIGHT_IN
    return 0.5 + min(1.0, performance / refs.best_ground_weight_height_kg_in)


def m1_score(
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> float:
    """Return one point when three laps fit in the five-minute window."""

    if not math.isfinite(lap_time_s) or lap_time_s <= 0.0:
        return 0.0
    mission_time = round_half_up(M1_REQUIRED_LAPS * lap_time_s)
    return float(mission_time <= refs.usable_window_s)


def m1_optimization_score(
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> float:
    if not math.isfinite(lap_time_s) or lap_time_s <= 0.0:
        return 0.0
    return min(1.0, refs.usable_window_s / (M1_REQUIRED_LAPS * lap_time_s))


def m2_score(
    payload_mass_kg: float,
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> float:
    """Score delivered payload mass divided by the time for exactly five laps."""

    if (
        not math.isfinite(payload_mass_kg)
        or payload_mass_kg <= 0.0
        or not math.isfinite(lap_time_s)
        or lap_time_s <= 0.0
    ):
        return 0.0
    elapsed = round_half_up(M2_REQUIRED_LAPS * lap_time_s)
    if elapsed > refs.usable_window_s:
        return 0.0
    performance = _official_mass_kg(payload_mass_kg) / elapsed
    return 1.0 + min(1.0, performance / refs.best_m2_weight_per_time_kg_s)


def m2_optimization_score(
    payload_mass_kg: float,
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> float:
    official = m2_score(payload_mass_kg, lap_time_s, refs)
    if official > 0.0:
        return official
    if (
        not math.isfinite(lap_time_s)
        or lap_time_s <= 0.0
        or payload_mass_kg <= 0.0
    ):
        return 0.0
    completion = min(
        1.0,
        refs.usable_window_s / (M2_REQUIRED_LAPS * lap_time_s),
    )
    performance = payload_mass_kg / (M2_REQUIRED_LAPS * lap_time_s)
    normalized = min(1.0, performance / refs.best_m2_weight_per_time_kg_s)
    return completion * (1.0 + normalized)


def completed_m3_laps(
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> int:
    if not math.isfinite(lap_time_s) or lap_time_s <= 0.0:
        return 0
    lap_window = (
        refs.usable_window_s
        - refs.m3_deployment_time_s
        - refs.m3_recovery_time_s
    )
    return int(lap_window // lap_time_s)


def m3_score(
    dv: DesignVector,
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> float:
    """Score completed laps times sensor mass."""

    laps = completed_m3_laps(lap_time_s, refs)
    if laps < 1:
        return 0.0
    performance = laps * _official_mass_kg(dv.mission3_sensor_weight_kg)
    return 2.0 + min(1.0, performance / refs.best_m3_lap_weight_kg)


def m3_optimization_score(
    dv: DesignVector,
    lap_time_s: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> float:
    if completed_m3_laps(lap_time_s, refs) < 1:
        return 0.0
    lap_window = (
        refs.usable_window_s
        - refs.m3_deployment_time_s
        - refs.m3_recovery_time_s
    )
    performance = (lap_window / lap_time_s) * dv.mission3_sensor_weight_kg
    return 2.0 + min(1.0, performance / refs.best_m3_lap_weight_kg)


def total_score(
    dv: DesignVector,
    lap_time_m1: float,
    lap_time_m2: float,
    lap_time_m3: float,
    m2_payload_mass_kg: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> tuple[float, list[float]]:
    """Return GM + M1 + M2 + M3, including the successful-M2 M1 waiver."""

    gm = gm_score(dv, refs)
    standalone_m1 = m1_score(lap_time_m1, refs)
    m2 = m2_score(m2_payload_mass_kg, lap_time_m2, refs)
    m1 = 1.0 if m2 > 0.0 else standalone_m1
    m3 = m3_score(dv, lap_time_m3, refs) if m2 > 0.0 else 0.0
    breakdown = [gm, m1, m2, m3]
    return sum(breakdown), breakdown


def total_optimization_score(
    dv: DesignVector,
    lap_time_m1: float,
    lap_time_m2: float,
    lap_time_m3: float,
    m2_payload_mass_kg: float,
    refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> tuple[float, list[float]]:
    """Return relaxed progress while preserving official unlock gates."""

    gm = gm_score(dv, refs)
    official_m2 = m2_score(m2_payload_mass_kg, lap_time_m2, refs)
    m1 = (
        1.0
        if official_m2 > 0.0
        else m1_optimization_score(lap_time_m1, refs)
    )
    m2 = m2_optimization_score(m2_payload_mass_kg, lap_time_m2, refs)
    m3 = (
        m3_optimization_score(dv, lap_time_m3, refs)
        if official_m2 > 0.0
        else 0.0
    )
    breakdown = [gm, m1, m2, m3]
    return sum(breakdown), breakdown
