"""Callable first-order dynamics model for a sensor towed on an elastic rope.

The aircraft path is prescribed.  Consequently, required aircraft forces are
feasibility demands, not a prediction that the aircraft can actually maintain
the path.  This is intentional for fast sizing-envelope studies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


G_FTPS2 = 32.174
KG_TO_LBM = 2.20462262185
M_TO_FT = 3.28083989501
MPS_TO_FPS = M_TO_FT


@dataclass(frozen=True)
class TowConfig:
    aircraft_weight_lbf: float = 20.0
    sensor_weight_lbf: float = 30.0
    airspeed_fps: float = 70.0
    bank_angle_deg: float = 35.0
    roll_time_s: float = 1.5
    straight_length_ft: float = 500.0
    rope_length_ft: float = 9.0
    rope_ea_lbf: float = 9000.0
    rope_damping_ratio: float = 0.03
    rope_diameter_in: float = 0.08
    rope_cd: float = 1.1
    sensor_diameter_in: float = 3.0
    sensor_length_in: float = 12.0
    sensor_cd_axial: float = 0.40
    sensor_cd_broadside: float = 1.05
    air_density_slug_ft3: float = 0.002377
    wind_fps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_heading_deg: float = 180.0
    initial_swing_deg: float = 0.0
    initial_swing_rate_deg_s: float = 0.0
    dt_s: float = 0.004

    def __post_init__(self) -> None:
        positive = (
            "aircraft_weight_lbf", "sensor_weight_lbf", "airspeed_fps",
            "roll_time_s", "straight_length_ft", "rope_length_ft",
            "rope_ea_lbf", "sensor_diameter_in", "sensor_length_in", "dt_s",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0.0 < self.bank_angle_deg < 89.0:
            raise ValueError("bank_angle_deg must lie between 0 and 89 degrees.")
        if self.rope_damping_ratio < 0.0:
            raise ValueError("rope_damping_ratio cannot be negative.")

    @property
    def rope_stiffness_lbf_ft(self) -> float:
        return self.rope_ea_lbf / self.rope_length_ft


@dataclass(frozen=True)
class TowSimulationResult:
    time_s: np.ndarray
    rope_tension_lbf: np.ndarray
    tow_force_body_lbf: np.ndarray  # columns: backward, left, down
    required_aircraft_force_body_lbf: np.ndarray  # forward, left, up
    sensor_offset_body_ft: np.ndarray  # aft, left, below
    swing_deg: np.ndarray
    rope_extension_ft: np.ndarray
    rope_taut: np.ndarray
    segment: tuple[str, ...]

    @property
    def mission_peak_tension_lbf(self) -> float:
        return float(np.max(self.rope_tension_lbf))

    @property
    def mission_peak_tow_lbf(self) -> float:
        return float(np.max(np.linalg.norm(self.tow_force_body_lbf, axis=1)))


def _schedule(config: TowConfig):
    bank_max = math.radians(config.bank_angle_deg)
    ramp_t = np.linspace(0.0, config.roll_time_s, 2001)
    ramp_phi = bank_max * 0.5 * (1.0 - np.cos(math.pi * ramp_t / config.roll_time_s))
    ramp_change = float(np.trapezoid(G_FTPS2 * np.tan(ramp_phi) / config.airspeed_fps, ramp_t))
    max_rate = G_FTPS2 * math.tan(bank_max) / config.airspeed_fps
    spec = (
        ("Straight 1", "straight", 0.0, 0),
        ("180 right", "turn", math.pi, -1),
        ("Straight 2", "straight", 0.0, 0),
        ("360 left", "turn", 2.0 * math.pi, 1),
        ("Straight 3", "straight", 0.0, 0),
        ("180 right", "turn", math.pi, -1),
        ("Straight 4", "straight", 0.0, 0),
    )
    result, cursor = [], 0.0
    for name, kind, angle, sign in spec:
        if kind == "straight":
            hold, duration = 0.0, config.straight_length_ft / config.airspeed_fps
        else:
            remaining = angle - 2.0 * ramp_change
            if remaining < 0.0:
                raise ValueError("roll_time_s is too long for the requested turns.")
            hold = remaining / max_rate
            duration = 2.0 * config.roll_time_s + hold
        result.append((name, kind, sign, hold, cursor, cursor + duration))
        cursor += duration
    return tuple(result), cursor, bank_max


def simulate_tow(config: TowConfig = TowConfig()) -> TowSimulationResult:
    """Run one complete nominal DBF-style course."""
    schedule, total_time, bank_max = _schedule(config)
    sensor_mass = config.sensor_weight_lbf / G_FTPS2
    aircraft_mass = config.aircraft_weight_lbf / G_FTPS2
    stiffness = config.rope_stiffness_lbf_ft
    damping = 2.0 * config.rope_damping_ratio * math.sqrt(stiffness * sensor_mass)
    wind = np.asarray(config.wind_fps, dtype=float)
    diameter_ft = config.sensor_diameter_in / 12.0
    length_ft = config.sensor_length_in / 12.0
    axial_area = math.pi * diameter_ft**2 / 4.0
    broadside_area = diameter_ft * length_ft
    rope_area = (config.rope_diameter_in / 12.0) * config.rope_length_ft

    def bank_command(t: float):
        seg = schedule[-1]
        for candidate in schedule:
            if t < candidate[5]:
                seg = candidate
                break
        name, kind, sign, hold, t0, _ = seg
        tau = max(0.0, t - t0)
        if kind == "straight":
            return 0.0, name
        if tau < config.roll_time_s:
            mag = bank_max * 0.5 * (1.0 - math.cos(math.pi * tau / config.roll_time_s))
        elif tau < config.roll_time_s + hold:
            mag = bank_max
        else:
            tau_out = min(config.roll_time_s, tau - config.roll_time_s - hold)
            mag = bank_max * 0.5 * (1.0 + math.cos(math.pi * tau_out / config.roll_time_s))
        return sign * mag, name

    def plane_motion(heading: float, bank: float):
        heading_rate = G_FTPS2 * math.tan(bank) / config.airspeed_fps
        forward = np.array((math.cos(heading), math.sin(heading), 0.0))
        left = np.array((-forward[1], forward[0], 0.0))
        return config.airspeed_fps * forward, config.airspeed_fps * heading_rate * left, heading_rate

    def rope_force(rp, vp, rs, vs):
        rel = rs - rp
        distance = float(np.linalg.norm(rel))
        if distance < 1e-12:
            return np.zeros(3), 0.0, distance, 0.0, False
        unit = rel / distance
        extension = distance - config.rope_length_ft
        extension_rate = float(np.dot(vs - vp, unit))
        if extension <= 0.0:
            return np.zeros(3), 0.0, distance, extension, False
        tension = max(0.0, stiffness * extension + damping * extension_rate)
        return -tension * unit, tension, distance, extension, tension > 0.0

    def drag(vrel, rope_axis):
        speed = float(np.linalg.norm(vrel))
        if speed < 1e-12:
            return np.zeros(3)
        flow = vrel / speed
        alignment = abs(float(np.dot(flow, rope_axis)))
        area_cd = (
            config.sensor_cd_axial * axial_area * alignment**2
            + config.sensor_cd_broadside * broadside_area * (1.0 - alignment**2)
        )
        # Half of the distributed rope drag is assigned to the sensor endpoint.
        area_cd += 0.5 * config.rope_cd * rope_area
        return -0.5 * config.air_density_slug_ft3 * area_cd * speed * vrel

    def rhs(t, y):
        heading = y[2]
        rs, vs = y[3:6], y[6:9]
        bank, _ = bank_command(t)
        vp, _, heading_rate = plane_motion(heading, bank)
        rp = np.array((y[0], y[1], 0.0))
        rel = rs - rp
        axis = rel / max(float(np.linalg.norm(rel)), 1e-12)
        force, _, _, _, _ = rope_force(rp, vp, rs, vs)
        force += drag(vs - wind, axis)
        force += np.array((0.0, 0.0, -config.sensor_weight_lbf))
        return np.concatenate((vp[:2], (heading_rate,), vs, force / sensor_mass))

    heading0 = math.radians(config.initial_heading_deg)
    vp0, _, _ = plane_motion(heading0, 0.0)
    downward = np.array((0.0, 0.0, -1.0))
    forward0 = vp0 / np.linalg.norm(vp0)
    # Iterate the steady straight-flight direction because cylinder and rope
    # drag depend on the angle between the relative wind and the tether.
    equilibrium_axis = downward.copy()
    for _ in range(8):
        external = np.array((0.0, 0.0, -config.sensor_weight_lbf))
        external += drag(vp0 - wind, equilibrium_axis)
        equilibrium_axis = external / np.linalg.norm(external)
    theta = math.radians(config.initial_swing_deg)
    # Apply the requested perturbation in the vertical/flight-direction plane
    # relative to the actual drag-deflected equilibrium.
    tangent_equilibrium = forward0 - np.dot(forward0, equilibrium_axis) * equilibrium_axis
    tangent_equilibrium /= max(np.linalg.norm(tangent_equilibrium), 1e-12)
    axis0 = math.cos(theta) * equilibrium_axis + math.sin(theta) * tangent_equilibrium
    # Equilibrium extension is a good initial condition even when a deliberate
    # initial-angle/rate perturbation is requested.
    equilibrium_force = np.array((0.0, 0.0, -config.sensor_weight_lbf))
    equilibrium_force += drag(vp0 - wind, equilibrium_axis)
    extension0 = float(np.linalg.norm(equilibrium_force)) / stiffness
    rs0 = (config.rope_length_ft + extension0) * axis0
    tangent0 = -math.sin(theta) * equilibrium_axis + math.cos(theta) * tangent_equilibrium
    vs0 = vp0 + math.radians(config.initial_swing_rate_deg_s) * config.rope_length_ft * tangent0
    y0 = np.concatenate(((0.0, 0.0, heading0), rs0, vs0))

    steps = int(math.ceil(total_time / config.dt_s)) + 1
    time = np.linspace(0.0, total_time, steps)
    states = np.empty((steps, 9), dtype=float)
    states[0] = y0
    for i in range(steps - 1):
        t, y = float(time[i]), states[i]
        h = float(time[i + 1] - time[i])
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = rhs(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = rhs(t + h, y + h * k3)
        states[i + 1] = y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    tension = np.zeros(steps)
    tow = np.zeros((steps, 3))
    required = np.zeros((steps, 3))
    offsets = np.zeros((steps, 3))
    swing = np.zeros(steps)
    extension = np.zeros(steps)
    taut = np.zeros(steps, dtype=bool)
    names: list[str] = []
    for i, t in enumerate(time):
        heading, rs, vs = states[i, 2], states[i, 3:6], states[i, 6:9]
        bank, name = bank_command(float(t))
        names.append(name)
        vp, ap, _ = plane_motion(heading, bank)
        rp = np.array((states[i, 0], states[i, 1], 0.0))
        forward = vp / np.linalg.norm(vp)
        left = np.array((-forward[1], forward[0], 0.0))
        frs, tension[i], distance, raw_extension, taut[i] = rope_force(rp, vp, rs, vs)
        frp = -frs
        tow[i] = (-np.dot(frp, forward), np.dot(frp, left), -frp[2])
        rel = rs - rp
        offsets[i] = (-np.dot(rel, forward), np.dot(rel, left), -rel[2])
        extension[i] = max(0.0, raw_extension)
        if distance > 1e-12:
            swing[i] = math.degrees(math.acos(np.clip(-(rel / distance)[2], -1.0, 1.0)))
        force_req = aircraft_mass * ap + np.array((0.0, 0.0, config.aircraft_weight_lbf)) - frp
        required[i] = (np.dot(force_req, forward), np.dot(force_req, left), force_req[2])

    return TowSimulationResult(time, tension, tow, required, offsets, swing, extension, taut, tuple(names))
