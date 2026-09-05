"""Deterministic and Monte-Carlo load envelopes for tow-system sizing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.tow.model import KG_TO_LBM, MPS_TO_FPS, M_TO_FT, TowConfig, simulate_tow


@dataclass(frozen=True)
class EnvelopeConfig:
    safety_factor: float = 1.5
    monte_carlo_runs: int = 24
    seed: int = 26027
    airspeed_fraction: float = 0.10
    bank_angle_delta_deg: float = 5.0
    roll_time_fraction: float = 0.20
    sensor_cd_fraction: float = 0.25
    rope_ea_fraction: float = 0.20
    damping_fraction: float = 0.50
    wind_limit_fps: float = 8.0
    initial_swing_limit_deg: float = 12.0
    initial_swing_rate_limit_deg_s: float = 20.0
    available_thrust_lbf: float | None = None
    allowable_load_factor: float | None = None
    rope_break_strength_lbf: float | None = None

    def __post_init__(self) -> None:
        if self.safety_factor < 1.0:
            raise ValueError("safety_factor must be at least 1.0.")
        if self.monte_carlo_runs < 0:
            raise ValueError("monte_carlo_runs cannot be negative.")


@dataclass(frozen=True)
class TowLoadEnvelope:
    typical_tension_mean_lbf: float
    typical_tension_rms_lbf: float
    nominal_peak_tension_lbf: float
    p95_mission_peak_tension_lbf: float
    p99_mission_peak_tension_lbf: float
    limit_tension_lbf: float
    ultimate_tension_lbf: float
    limit_tow_backward_lbf: float
    limit_tow_side_abs_lbf: float
    limit_tow_down_lbf: float
    limit_tow_resultant_lbf: float
    required_peak_forward_force_lbf: float
    required_peak_aircraft_load_factor: float
    minimum_sensor_below_offset_ft: float
    maximum_sensor_swing_deg: float
    sensor_reached_aircraft_altitude: bool
    rope_strength_margin: float | None
    thrust_margin_lbf: float | None
    load_factor_margin: float | None
    feasible: bool | None
    governing_case: str
    cases_run: int

    def to_dict(self) -> dict:
        return asdict(self)


def _case_metrics(name: str, config: TowConfig) -> dict[str, float | str]:
    result = simulate_tow(config)
    tow_norm = np.linalg.norm(result.tow_force_body_lbf, axis=1)
    req = result.required_aircraft_force_body_lbf
    perpendicular = np.linalg.norm(req[:, 1:3], axis=1)
    return {
        "case": name,
        "peak_tension_lbf": result.mission_peak_tension_lbf,
        "peak_tow_backward_lbf": float(np.max(result.tow_force_body_lbf[:, 0])),
        "peak_tow_side_abs_lbf": float(np.max(np.abs(result.tow_force_body_lbf[:, 1]))),
        "peak_tow_down_lbf": float(np.max(result.tow_force_body_lbf[:, 2])),
        "peak_tow_resultant_lbf": float(np.max(tow_norm)),
        "required_peak_forward_force_lbf": float(np.max(req[:, 0])),
        "required_peak_aircraft_load_factor": float(np.max(perpendicular) / config.aircraft_weight_lbf),
        "minimum_sensor_below_offset_ft": float(np.min(result.sensor_offset_body_ft[:, 2])),
        "maximum_sensor_swing_deg": float(np.max(result.swing_deg)),
        "tension_mean_lbf": float(np.mean(result.rope_tension_lbf)),
        "tension_rms_lbf": float(np.sqrt(np.mean(result.rope_tension_lbf**2))),
    }


def _deterministic_cases(base: TowConfig, env: EnvelopeConfig):
    yield "nominal", base
    yield "high_speed", replace(base, airspeed_fps=base.airspeed_fps * (1.0 + env.airspeed_fraction))
    yield "low_speed", replace(base, airspeed_fps=base.airspeed_fps * (1.0 - env.airspeed_fraction))
    yield "max_bank_fast_roll", replace(
        base,
        bank_angle_deg=min(75.0, base.bank_angle_deg + env.bank_angle_delta_deg),
        roll_time_s=base.roll_time_s * (1.0 - env.roll_time_fraction),
    )
    yield "low_rope_ea_low_damping", replace(
        base,
        rope_ea_lbf=base.rope_ea_lbf * (1.0 - env.rope_ea_fraction),
        rope_damping_ratio=base.rope_damping_ratio * (1.0 - env.damping_fraction),
    )
    yield "high_rope_ea_high_damping", replace(
        base,
        rope_ea_lbf=base.rope_ea_lbf * (1.0 + env.rope_ea_fraction),
        rope_damping_ratio=base.rope_damping_ratio * (1.0 + env.damping_fraction),
    )
    yield "gust_and_initial_swing", replace(
        base,
        wind_fps=(env.wind_limit_fps, env.wind_limit_fps, 0.0),
        initial_swing_deg=env.initial_swing_limit_deg,
        initial_swing_rate_deg_s=env.initial_swing_rate_limit_deg_s,
        sensor_cd_axial=base.sensor_cd_axial * (1.0 + env.sensor_cd_fraction),
        sensor_cd_broadside=base.sensor_cd_broadside * (1.0 + env.sensor_cd_fraction),
    )


def _random_case(base: TowConfig, env: EnvelopeConfig, rng: np.random.Generator, index: int):
    speed = base.airspeed_fps * (1.0 + rng.uniform(-env.airspeed_fraction, env.airspeed_fraction))
    bank = base.bank_angle_deg + rng.uniform(-env.bank_angle_delta_deg, env.bank_angle_delta_deg)
    roll = base.roll_time_s * (1.0 + rng.uniform(-env.roll_time_fraction, env.roll_time_fraction))
    cd_scale = 1.0 + rng.uniform(-env.sensor_cd_fraction, env.sensor_cd_fraction)
    return f"monte_carlo_{index + 1:03d}", replace(
        base,
        airspeed_fps=speed,
        bank_angle_deg=bank,
        roll_time_s=roll,
        rope_ea_lbf=base.rope_ea_lbf * (1.0 + rng.uniform(-env.rope_ea_fraction, env.rope_ea_fraction)),
        rope_damping_ratio=max(0.0, base.rope_damping_ratio * (1.0 + rng.uniform(-env.damping_fraction, env.damping_fraction))),
        sensor_cd_axial=base.sensor_cd_axial * cd_scale,
        sensor_cd_broadside=base.sensor_cd_broadside * cd_scale,
        wind_fps=(float(rng.uniform(-env.wind_limit_fps, env.wind_limit_fps)), float(rng.uniform(-env.wind_limit_fps, env.wind_limit_fps)), 0.0),
        initial_swing_deg=float(rng.uniform(-env.initial_swing_limit_deg, env.initial_swing_limit_deg)),
        initial_swing_rate_deg_s=float(rng.uniform(-env.initial_swing_rate_limit_deg_s, env.initial_swing_rate_limit_deg_s)),
    )


def run_load_envelope(base: TowConfig, envelope: EnvelopeConfig = EnvelopeConfig()):
    """Return a conservative sizing envelope and per-case audit records."""
    rng = np.random.default_rng(envelope.seed)
    cases = list(_deterministic_cases(base, envelope))
    cases.extend(_random_case(base, envelope, rng, i) for i in range(envelope.monte_carlo_runs))
    rows = [_case_metrics(name, config) for name, config in cases]
    peaks = np.asarray([row["peak_tension_lbf"] for row in rows], dtype=float)
    limit_index = int(np.argmax(peaks))
    nominal = rows[0]
    maxima = lambda key: max(float(row[key]) for row in rows)
    minima = lambda key: min(float(row[key]) for row in rows)
    limit = float(peaks[limit_index])
    strength_margin = None if envelope.rope_break_strength_lbf is None else envelope.rope_break_strength_lbf / (limit * envelope.safety_factor)
    thrust_margin = None if envelope.available_thrust_lbf is None else envelope.available_thrust_lbf - maxima("required_peak_forward_force_lbf")
    load_margin = None if envelope.allowable_load_factor is None else envelope.allowable_load_factor - maxima("required_peak_aircraft_load_factor")
    checks: list[bool] = []
    if strength_margin is not None:
        checks.append(strength_margin >= 1.0)
    if thrust_margin is not None:
        checks.append(thrust_margin >= 0.0)
    if load_margin is not None:
        checks.append(load_margin >= 0.0)
    feasible = None if not checks else all(checks)
    result = TowLoadEnvelope(
        typical_tension_mean_lbf=float(nominal["tension_mean_lbf"]),
        typical_tension_rms_lbf=float(nominal["tension_rms_lbf"]),
        nominal_peak_tension_lbf=float(nominal["peak_tension_lbf"]),
        p95_mission_peak_tension_lbf=float(np.quantile(peaks, 0.95)),
        p99_mission_peak_tension_lbf=float(np.quantile(peaks, 0.99)),
        limit_tension_lbf=limit,
        ultimate_tension_lbf=limit * envelope.safety_factor,
        limit_tow_backward_lbf=maxima("peak_tow_backward_lbf"),
        limit_tow_side_abs_lbf=maxima("peak_tow_side_abs_lbf"),
        limit_tow_down_lbf=maxima("peak_tow_down_lbf"),
        limit_tow_resultant_lbf=maxima("peak_tow_resultant_lbf"),
        required_peak_forward_force_lbf=maxima("required_peak_forward_force_lbf"),
        required_peak_aircraft_load_factor=maxima("required_peak_aircraft_load_factor"),
        minimum_sensor_below_offset_ft=minima("minimum_sensor_below_offset_ft"),
        maximum_sensor_swing_deg=maxima("maximum_sensor_swing_deg"),
        sensor_reached_aircraft_altitude=(
            minima("minimum_sensor_below_offset_ft") <= 0.0
        ),
        rope_strength_margin=strength_margin,
        thrust_margin_lbf=thrust_margin,
        load_factor_margin=load_margin,
        feasible=feasible,
        governing_case=str(rows[limit_index]["case"]),
        cases_run=len(rows),
    )
    return result, rows


def evaluate_design_tow_envelope(
    design_vector,
    *,
    aircraft_mass_kg: float,
    airspeed_mps: float,
    rope_length_m: float = 2.7432,
    rope_ea_lbf: float = 9000.0,
    envelope: EnvelopeConfig = EnvelopeConfig(),
):
    """Sizing-script adapter using the Mission-3 sensor in a DesignVector."""
    sensor_mass = float(design_vector.mission3_sensor_weight_kg)
    base = TowConfig(
        aircraft_weight_lbf=(aircraft_mass_kg - sensor_mass) * KG_TO_LBM,
        sensor_weight_lbf=sensor_mass * KG_TO_LBM,
        airspeed_fps=airspeed_mps * MPS_TO_FPS,
        rope_length_ft=rope_length_m * M_TO_FT,
        sensor_length_in=float(design_vector.mission3_sensor_length_m) / 0.0254,
    )
    if base.aircraft_weight_lbf <= 0.0:
        raise ValueError("aircraft_mass_kg must include and exceed the towed sensor mass.")
    return run_load_envelope(base, envelope)


def write_envelope_files(summary: TowLoadEnvelope, rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "tow_load_envelope.json").write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
    with (output / "tow_load_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data_dump/tow_envelope"))
    parser.add_argument("--monte-carlo", type=int, default=24)
    parser.add_argument("--safety-factor", type=float, default=1.5)
    parser.add_argument("--dt", type=float, default=0.004)
    args = parser.parse_args()
    summary, rows = run_load_envelope(
        TowConfig(dt_s=args.dt),
        EnvelopeConfig(monte_carlo_runs=args.monte_carlo, safety_factor=args.safety_factor),
    )
    write_envelope_files(summary, rows, args.output)
    print(json.dumps(summary.to_dict(), indent=2))
    print(f"Wrote {args.output / 'tow_load_envelope.json'}")
    print(f"Wrote {args.output / 'tow_load_cases.csv'}")


if __name__ == "__main__":
    main()
