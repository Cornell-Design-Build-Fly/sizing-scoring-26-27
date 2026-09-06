"""Estimate maximum M2 payload/time performance with a fixed 42 lb sensor."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution

from src.aero.main_aero import aero_main
from src.main import MAX_TAKEOFF_MASS_KG, resolved_aerodynamic_design_vector
from src.mech.main_mech import evaluate_mechanical_module
from src.opt.score import M2_REQUIRED_LAPS, POUNDS_TO_KG, m2_score, round_half_up
from src.prop.continuous_prop_database import load_default_continuous_prop_database
from src.prop.prop_classes import battery_nominal_voltage_v
from src.prop.main_prop import prop_main
from src.vectors import (
    DesignVector,
    MAX_SENSOR_LENGTH_M,
    MIN_MISSION3_SENSOR_WEIGHT_KG,
    ParameterVector,
    SENSOR_DIAMETER_M,
    SENSOR_STEEL_DENSITY_KG_M3,
)


FIXED_SENSOR_WEIGHT_LB = 42.0
FIXED_SENSOR_WEIGHT_KG = FIXED_SENSOR_WEIGHT_LB * POUNDS_TO_KG
EMPTY_CONTAINER_WEIGHT_LB = 0.5
BAD_OBJECTIVE = 1.0e6
OUTPUT_ROOT = Path("data_dump") / "m2_max_42lb"
PARAMETERS = ParameterVector()
PROP_DATABASE = None

# Two payload containers would weigh at least 85 lb, so the 55 lb airplane
# limit analytically fixes extra_shipping_containers at zero for this study.
VARIABLES = (
    ("wing_span", 0.914, 1.8288),
    ("wing_chord", 0.12, 0.40),
    ("tail_arm", 0.30, 0.90),
    ("nose_length", 0.08, 0.30),
    ("sensor_length_m", 0.0, MAX_SENSOR_LENGTH_M),
    ("batt_capacity", 1.0, 4.5),
    ("prop_diameter_in", 10.0, 25.0),
    ("prop_pitch_in", 5.0, 18.0),
    ("motor_kv", 200.0, 500.0),
    ("motor_max_power", 1000.0, 3000.0),
    ("cruise_throttle", 0.50, 1.0),
    ("battery_choice", 0.0, 1.0),
)
NAMES = tuple(item[0] for item in VARIABLES)


def _minimum_length_for_fixed_weight_m() -> float:
    area = math.pi * (0.5 * SENSOR_DIAMETER_M) ** 2
    return FIXED_SENSOR_WEIGHT_KG / (SENSOR_STEEL_DENSITY_KG_M3 * area)


BOUNDS = tuple(
    (
        max(lower, _minimum_length_for_fixed_weight_m())
        if name == "sensor_length_m"
        else lower,
        upper,
    )
    for name, lower, upper in VARIABLES
)
INTEGRALITY = tuple(name == "battery_choice" for name in NAMES)


@dataclass(frozen=True)
class M2Evaluation:
    feasible: bool
    performance_lb_s: float
    payload_weight_lb: float
    five_lap_time_s: float
    lap_time_s: float
    takeoff_mass_lb: float
    aero_penalty: float
    mechanical_penalty: float
    reason: str
    design: DesignVector


def _battery_cell_count(choice: float) -> int:
    return 6 if int(round(choice)) == 0 else 8


def _design_from_x(x: np.ndarray) -> DesignVector:
    values = dict(zip(NAMES, map(float, x)))
    return DesignVector(
        wing_span=values["wing_span"],
        wing_chord=values["wing_chord"],
        tail_arm=values["tail_arm"],
        nose_length=values["nose_length"],
        extra_shipping_containers=0,
        sensor_length_m=values["sensor_length_m"],
        sensor_weight_kg=FIXED_SENSOR_WEIGHT_KG,
        mission3_sensor_weight_kg=MIN_MISSION3_SENSOR_WEIGHT_KG,
        batt_capacity=values["batt_capacity"],
        battery_cell_count=_battery_cell_count(values["battery_choice"]),
        prop_diameter_in=values["prop_diameter_in"],
        prop_pitch_in=values["prop_pitch_in"],
        motor_kv=values["motor_kv"],
        motor_max_power=values["motor_max_power"],
        cruise_throttle=values["cruise_throttle"],
        mission3_cruise_throttle=0.5,
    )


def _prop_database():
    global PROP_DATABASE
    if PROP_DATABASE is None:
        with redirect_stdout(StringIO()):
            PROP_DATABASE = load_default_continuous_prop_database()
    return PROP_DATABASE


def evaluate_m2(x: np.ndarray) -> M2Evaluation:
    design = _design_from_x(x)
    with redirect_stdout(StringIO()):
        mechanical = evaluate_mechanical_module(design, parameter_vector=PARAMETERS)
        resolved = resolved_aerodynamic_design_vector(design, mechanical)
        thrust_curve, flight_time_fit = prop_main(
            resolved,
            PARAMETERS,
            mission=2,
            prop_database=_prop_database(),
        )
        properties = mechanical.for_mission("M2")
        aero = aero_main(
            design_vector=resolved,
            parameter_vector=PARAMETERS,
            thrust_velocity=thrust_curve,
            flight_time_fit=flight_time_fit,
            mission=2,
            cg=properties.cg_m,
            inertia_matrix=properties.inertia_tensor_kg_m2,
            mass=properties.total_mass_kg,
        )

    payload_mass_kg = sum(
        item.mass_kg
        for item in properties.items
        if item.category == "mission_2_payload"
    )
    payload_weight_lb = payload_mass_kg / POUNDS_TO_KG
    lap_time = float(aero.lap_time)
    elapsed = (
        round_half_up(M2_REQUIRED_LAPS * lap_time)
        if math.isfinite(lap_time) and lap_time > 0.0
        else math.inf
    )
    performance = payload_weight_lb / elapsed if math.isfinite(elapsed) else 0.0
    m2_static_penalty = float(
        mechanical.penalty_static_margin_by_mission.get("M2", 0.0)
    )
    mechanical_penalty = float(mechanical.penalty_placement + m2_static_penalty)
    takeoff_mass_lb = properties.total_mass_kg / POUNDS_TO_KG

    reasons = []
    if not aero.can_fly:
        reasons.append("M2 aero/endurance gate failed")
    if mechanical_penalty > 0.0:
        reasons.append("M2 mechanical gate failed")
    if properties.total_mass_kg >= MAX_TAKEOFF_MASS_KG:
        reasons.append("55 lb takeoff limit failed")
    if elapsed > 300.0:
        reasons.append("five laps exceed 300 s")
    feasible = not reasons
    return M2Evaluation(
        feasible=feasible,
        performance_lb_s=float(performance),
        payload_weight_lb=float(payload_weight_lb),
        five_lap_time_s=float(elapsed),
        lap_time_s=lap_time,
        takeoff_mass_lb=float(takeoff_mass_lb),
        aero_penalty=float(aero.penalty),
        mechanical_penalty=mechanical_penalty,
        reason="; ".join(reasons),
        design=design,
    )


def objective(x: np.ndarray) -> float:
    try:
        evaluation = evaluate_m2(x)
    except Exception:
        return BAD_OBJECTIVE
    if evaluation.feasible:
        return -evaluation.performance_lb_s

    # Feasible candidates always have a negative objective. This graded
    # positive region gives DE direction while it approaches hard feasibility.
    overweight = max(0.0, evaluation.takeoff_mass_lb - 55.0)
    overtime = (
        max(0.0, evaluation.five_lap_time_s - 300.0) / 30.0
        if math.isfinite(evaluation.five_lap_time_s)
        else 10.0
    )
    return float(
        1000.0
        + overweight
        + overtime
        + evaluation.aero_penalty
        + evaluation.mechanical_penalty
    )


def _pd_ratio(x: np.ndarray) -> float:
    return float(x[NAMES.index("prop_pitch_in")] / x[NAMES.index("prop_diameter_in")])


def _battery_energy_wh(x: np.ndarray) -> float:
    capacity = float(x[NAMES.index("batt_capacity")])
    cells = _battery_cell_count(x[NAMES.index("battery_choice")])
    return capacity * battery_nominal_voltage_v(cells)


def _floatify(value):
    if isinstance(value, dict):
        return {key: _floatify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_floatify(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def run(maxiter: int = 50, popsize: int = 12, workers: int = -1, seed: int = 20260904):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"M2 maximum-performance optimization: fixed sensor={FIXED_SENSOR_WEIGHT_LB:.1f} lb, "
        f"maxiter={maxiter}, popsize={popsize}, workers={workers}",
        flush=True,
    )
    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        constraints=(
            NonlinearConstraint(_pd_ratio, 0.4, 0.8),
            NonlinearConstraint(_battery_energy_wh, 0.0, 100.0),
        ),
        integrality=INTEGRALITY,
        maxiter=maxiter,
        popsize=popsize,
        init="sobol",
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        seed=seed,
        workers=workers,
        updating="deferred",
        disp=True,
    )
    evaluation = evaluate_m2(np.asarray(result.x, dtype=float))
    if not evaluation.feasible:
        raise RuntimeError(
            "The optimization did not find a feasible 42 lb M2 design: "
            f"{evaluation.reason}"
        )

    report = {
        "objective": "maximize M2 payload weight / official five-lap time",
        "fixed_sensor_weight_lb": FIXED_SENSOR_WEIGHT_LB,
        "extra_shipping_containers": 0,
        "performance_lb_s": evaluation.performance_lb_s,
        "payload_weight_lb": evaluation.payload_weight_lb,
        "five_lap_time_s": evaluation.five_lap_time_s,
        "lap_time_s": evaluation.lap_time_s,
        "takeoff_mass_lb": evaluation.takeoff_mass_lb,
        "official_m2_score_with_current_reference": m2_score(
            evaluation.payload_weight_lb * POUNDS_TO_KG,
            evaluation.lap_time_s,
        ),
        "design_vector": asdict(evaluation.design),
        "optimizer": {
            "maxiter": maxiter,
            "popsize": popsize,
            "workers": workers,
            "seed": seed,
            "nfev": int(result.nfev),
            "nit": int(result.nit),
            "success": bool(result.success),
            "message": str(result.message),
            "fun": float(result.fun),
        },
        "constraints": {
            "takeoff_mass_limit_lb": 55.0,
            "battery_energy_limit_wh": 100.0,
            "propeller_pitch_diameter_ratio": [0.4, 0.8],
            "minimum_sensor_length_for_steel_density_m": _minimum_length_for_fixed_weight_m(),
        },
    }
    (output_dir / "m2_max_report.json").write_text(
        json.dumps(_floatify(report), indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "optimizer_result.npz",
        x=np.asarray(result.x, dtype=float),
        population=np.asarray(result.population, dtype=float),
        population_energies=np.asarray(result.population_energies, dtype=float),
    )
    print(
        f"Best feasible M2 performance: {evaluation.performance_lb_s:.6f} lb/s "
        f"({evaluation.payload_weight_lb:.2f} lb in {evaluation.five_lap_time_s:.2f} s)",
        flush=True,
    )
    print(f"Artifacts: {output_dir.resolve()}", flush=True)
    return result, evaluation, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--popsize", type=int, default=12)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    run(args.maxiter, args.popsize, args.workers, args.seed)


if __name__ == "__main__":
    main()
