from src.vectors import DesignVector

SECONDS_PER_MISSION = 300.0
METERS_TO_FEET = 3.28
METERS_TO_INCHES = 39.37

# Ground mission references
DUCKS_TIME = 2.5
PUCKS_TIME = 1.5
BANNER_TIME = 7.0
BEST_GM_TIME_S = 18.0

# Mission 2 reference
BEST_M2_PROFIT = 1800.0

# Mission 3 references
BEST_M3_LAP_TIME_S = 55.0
BEST_BANNER_LENGTH_IN = 300.0
BEST_RAC = 0.90


def gm_score(dv: DesignVector) -> float:
    """Returns the ground mission score."""
    time_gm = 2.0 * (dv.ducks_num * DUCKS_TIME + dv.pucks_num * PUCKS_TIME) + BANNER_TIME
    return BEST_GM_TIME_S / time_gm


def m1_score(lap_time_s: float) -> float:
    """Returns the mission 1 score."""
    mission_time_s = lap_time_s * 3.0
    if mission_time_s < SECONDS_PER_MISSION:
        return 1.0
    return 0.0


def m2_score(dv: DesignVector, lap_time_s: float) -> float:
    """Returns the mission 2 score."""
    num_laps = int(SECONDS_PER_MISSION // lap_time_s)
    income_passengers = dv.ducks_num * (6.0 + 2.0 * num_laps)
    income_cargo = dv.pucks_num * (10.0 + 8.0 * num_laps)
    efficiency_factor = dv.batt_energy / 100.0
    cost = num_laps * (10.0 + dv.ducks_num * 0.5 + dv.pucks_num * 2.0) * efficiency_factor
    profit = (income_passengers + income_cargo) - cost
    return 1.0 + (profit / BEST_M2_PROFIT)


def m3_score(dv: DesignVector, lap_time_s: float) -> float:
    """Returns the mission 3 score."""
    wing_span_ft = dv.wing_span * METERS_TO_FEET
    rac = 0.05 * wing_span_ft + 0.75
    num_laps = int(SECONDS_PER_MISSION // lap_time_s)
    best_num_laps = int(SECONDS_PER_MISSION // BEST_M3_LAP_TIME_S)
    return 2.0 + (num_laps * dv.banner_length * METERS_TO_INCHES / rac) / (best_num_laps * BEST_BANNER_LENGTH_IN / BEST_RAC)


def total_score(dv: DesignVector, lap_time_m1: float, lap_time_m2: float, lap_time_m3: float,
) -> tuple[float, list[float]]:
    """Returns the total score and mission-by-mission breakdown."""
    gm = gm_score(dv)
    m1 = m1_score(lap_time_m1)
    m2 = 0.0
    m3 = 0.0

    if m1 > 0.0:
        m2 = m2_score(dv, lap_time_m2)
        if m2 > 0.0:
            m3 = m3_score(dv, lap_time_m3)

    breakdown = [gm, m1, m2, m3]
    # print(f"Score breakdown: GM={gm:.2f}, M1={m1:.2f}, M2={m2:.2f}, M3={m3:.2f}")
    return gm + m1 + m2 + m3, breakdown
