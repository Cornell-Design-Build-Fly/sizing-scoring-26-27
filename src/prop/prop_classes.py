from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import math

MPS_TO_MPH = 2.2369

DEFAULT_VELOCITIES_MPS = np.linspace(0.001, 25.0, 8)

#DEFAULT_PROP_DATA_PATH = Path(__file__).resolve().parent / "data" / "prop_data.json"

DEFAULT_PROP_DIAMETER_IN = 14.0
DEFAULT_PROP_PITCH_IN = 10.0
DEFAULT_MOTOR_KV = 335.0
DEFAULT_MOTOR_MAX_POWER_W = 2200.0
DEFAULT_CRUISE_THROTTLE = 0.90
DEFAULT_MISSION3_CRUISE_THROTTLE = 0.85
DEFAULT_MAX_CURRENT_A = 100.0
DEFAULT_USABLE_BATTERY_FRACTION = 0.85

@dataclass(frozen=True, slots=True)
class Propeller:
    diameter: float  # in
    pitch: float  # in
    mass: float | None = None # kg

@dataclass(frozen=True, slots=True)
class Motor:
    kv: float  # RPM/V
    Rm: float  # Ohms
    max_power: float  # W
    I0: float  # A
    max_current: float  # A
    mass: float | None = None # kg
    def get_kt(self) -> float:
        return 60/(2*math.pi*self.kv)  # Nm/A

@dataclass(frozen=True, slots=True)
class Battery:
    vnom: float  # V
    cells: int
    Rb: float  # Ohms
    Crat: float  # C
    capacity: float  # Ah
    mass: float | None = None # kg
    useable_fraction: float = DEFAULT_USABLE_BATTERY_FRACTION

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