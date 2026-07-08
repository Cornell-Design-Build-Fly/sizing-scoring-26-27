from __future import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import math

@dataclass(frozen=True, slots=True)
class propeller:
    diameter: float  # in
    pitch: float  # in
    mass: float | None = None # kg

@dataclass(frozen=True, slots=True)
class motor:
    kv: float  # RPM/V
    kt = 60/(2*math.pi*kv): float  # Nm/A
    Rm: float  # Ohms
    max_power: float  # W
    I0: float  # A
    max_current: float  # A
    mass: float | None = None # kg

@dataclass(frozen=True, slots=True)
class battery:
    vnom: float  # V
    cells: int
    Rb: float  # Ohms
    Crat: float  # C
    capacity: float  # Ah
    mass: float | None = None # kg

@dataclass(frozen=True, slots=True)
class MotorCheckResult:
    passed: bool
    throttle: float
    flight_time_s: float
    power_w: float
    current_a: float
    voltage_sag_v: float
    voltage_required_v: float
@dataclass(frozen=True, slots=True)
class PropulsionCurveFit:
    throttled_thrust: np.ndarray
    max_thrust: np.ndarray
    throttled_time: np.ndarray
    max_time: np.ndarray

    sample_velocities_mps: np.ndarray
    throttled_thrust_samples: np.ndarray
    max_thrust_samples: np.ndarray
    throttled_time_samples: np.ndarray
    max_time_samples: np.ndarray


@dataclass(frozen=True, slots=True)
class PropInterpolants:
    thrust: Callable[[float, float, float, float], float]
    torque: Callable[[float, float, float, float], float]

rho = 1.225 # kg/m^3