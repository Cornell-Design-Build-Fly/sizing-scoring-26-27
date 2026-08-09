from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import math

MPS_TO_MPH = 2.2369

# DEFAULT_VELOCITIES_MPS = np.linspace(0.01, 50, 4)


DEFAULT_VELOCITIES_MPS = np.array(
    [0.01, 9.5, 19.0, 28.35],
    dtype=float,
)

#DEFAULT_PROP_DATA_PATH = Path(__file__).resolve().parent / "data" / "prop_data.json"

DEFAULT_PROP_DIAMETER_IN = 14.0
DEFAULT_PROP_PITCH_IN = 10.0
DEFAULT_MOTOR_KV = 520.0
DEFAULT_MOTOR_MAX_POWER_W = 2000.0
DEFAULT_CRUISE_THROTTLE = 0.90
DEFAULT_MISSION3_CRUISE_THROTTLE = 0.85
DEFAULT_MAX_CURRENT_A = 100.0
DEFAULT_USABLE_BATTERY_FRACTION = 0.85

@dataclass(frozen=True, slots=True)
class Propeller:
    diameter: float  # in
    pitch: float  # in

@dataclass(frozen=True, slots=True)
class Motor:
    kv: float  # RPM/V
    max_power: float  # W
    max_current: float  # A
    def get_kt(self) -> float:
        return 60/(2*math.pi*self.kv)  # Nm/A
    def get_rm(self) -> float:
        c_R = np.array([0.3517732388, -0.0005385476, -0.0001855504, 0.0000002999, 0.0000000776, 0.0000000380,])
        Rm = c_R[0] + c_R[1]*self.kv + c_R[2]*self.max_power + c_R[3]*self.kv**2 + c_R[4]*self.kv*self.max_power + c_R[5]*self.max_power**2
        return Rm
    def get_I0(self) -> float:
        c_R = np.array([0.3517732388, -0.0005385476, -0.0001855504, 0.0000002999, 0.0000000776, 0.0000000380,])
        c_I = np.array([-0.5621009279, 0.0005335965, 0.0016292435, 0.0000005495, 0.0000006015, -0.0000004552])
        I0 = c_I[0] + c_I[1]*self.kv + c_I[2]*self.max_power + c_I[3]*self.kv**2 + c_I[4]*self.kv*self.max_power + c_I[5]*self.max_power**2
        return I0

@dataclass(frozen=True, slots=True)
class Battery:
    vnom: float  # V
    cells: int
    Crat: float  # C
    capacity: float  # Ah
    useable_fraction: float = DEFAULT_USABLE_BATTERY_FRACTION
    def get_Rb(self) -> float:
        return (0.013/self.capacity)*self.cells
    def get_useable_capacity(self) -> float:
        return self.capacity * self.useable_fraction


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