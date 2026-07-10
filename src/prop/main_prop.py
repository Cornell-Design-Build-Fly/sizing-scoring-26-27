# and here I am, catching you slacking, looking at the codebase for the first time...
from __future__ import annotations
from pathlib import Path
import re
import json
import math
from functools import lru_cache


import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from src.vectors import DesignVector, ParameterVector
from src.prop.prop_classes import (
    battery,
    motor,
    MotorCheckResult,
    PropInterpolants,
    PropulsionCurveFit,
)


''' MATLAB HELPER FOLDER'''
def battery_resistance(capacity_ah: float, num_cells: int) -> float:
    '''Calculates the internal resistance of a battery based on its capacity (ah) and number of cells.'''
    '''if capacity_ah <= 0:
        raise ValueError("Battery capacity must be positive.")
    if num_cells <= 0:
        raise ValueError("Number of cells must be positive.")'''
    return (0.013/capacity_ah)*num_cells

def motor_properties(kv: float, max_power_w: float) -> tuple[float, float]:
    '''Calcualtes the motor resistance and no-load current based on its KV and maximum power.'''
    if kv <= 0:
        raise ValueError("Motor KV must be positive.")
    if max_power_w <= 0:
        raise ValueError("Motor maximum power must be positive.")
    c_r = np.array([0.3517732388, -0.0005385476, -0.0001855504, 0.0000002999, 0.0000000776, 0.0000000380,])
    c_I = np.array([-0.5621009279, 0.0005335965, 0.0016292435, 0.0000005495, 0.0000006015, -0.0000004552])

    Rm = c_R[0] + c_R[1]*kv + c_R[2]*max_power_w + c_R[3]*kv**2 + c_R[4]*kv*max_power_w + c_R[5]*max_power_w**2;
    Inot = c_I[0] + c_I[1]*kv + c_I[2]*max_power_w + c_I[3]*kv**2 + c_I[4]*kv*max_power_w + c_I[5]*max_power_w**2;
    
    return Rm, Inot



'''MOTOR CHECK FUNCTION'''
def motor_check(torque_nm: float, rpm: float, motor: Motor, battery: Battery):
    if rpm <= 0:
        raise ValueError("RPM must be positive.")
    passed = True
    current = (torque / battery.get_kt()) + battery.I0 #(A) Current needed to sustain torque

    V_sag = battery.vnom - current*(battery.Rb) #(V) Voltage drop in battery under load
    V_req = rpm/motor.kv+current+motor.Rm #Voltage required due to EMF
    power = current*V_sag #(W) Power consumed by motor

    #Battery flight time calculation

    if current <= 1e-6:  #Avoid division by zero or very small power; use a small threshold
        t_flight = np.inf; #Effectively infinite time if no significant power drawn
    else:
        #E_battery is in Wh. P is in W. (Wh / W) = hours.
        #Convert hours to seconds by multiplying by 3600.
        t_flight = battery.capacity/current*3600.0

    #Throttle Required
    if V_sag <= 1e-6:  # Avoid division by zero or negative V_sag
        throttle = np.inf;  # Effectively infinite throttle required if V_sag is non-positive
    else:
        throttle = V_req / V_sag

    # --- Start of Failure conditions ---

    # Motor Power Limit Check
    # Motor Power will later set all values to zero if overshot
    # if P > P_max
    #     # fprintf('MOTOR CHECK FAIL: Power overload. RPM: %.0f, P_electrical: %.2f W > P_max_motor: %.2f W\n', RPM, P, P_max);
    #     pass = false;
    # end

    # Voltage Required vs Nominal Voltage
    if V_req > battery.vnom and passed:
        # fprintf('MOTOR CHECK FAIL: Insufficient nominal voltage. RPM: %.0f, V_req: %.2f V > V_nom: %.2f V\n', RPM, V_req, V_nom);
        passed = False

    # Throttle Limit Check (This is often the key RPM limiting factor)
    if throttle > 1.0 and passed:
        # fprintf('MOTOR CHECK FAIL: Throttle overload. RPM: %.0f, Throttle: %.3f > 1. (V_req: %.2f V, V_sag: %.2f V, I: %.2f A)\n', RPM, throttle, V_req, V_sag, I);
        passed = False
    
    # Check for V_sag becoming non-positive (battery completely depleted or calculation issue)
    if V_sag <= 0 and passed:
        # fprintf('MOTOR CHECK FAIL: Battery voltage sagged too low. RPM: %.0f, V_sag: %.2f V, I: %.2f A\n', RPM, V_sag, I);
        passed = False
    
    return passed



'''DATA MANAGEMENT'''

DEFAULT_PROP_DATA_PATH = Path(__file__).resolve().parent / "data" / "prop_data.json"


def parse_prop_key(key: str) -> tuple[float, float]:
    """
    Parses prop names like:
        x14x10E
        14x10
        14.5x10

    Returns:
        diameter_in, pitch_in
    """

    match = re.search(r"(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)", key)

    if match is None:
        raise ValueError(f"Could not parse prop diameter/pitch from key: {key}")

    diameter_in = float(match.group(1))
    pitch_in = float(match.group(2))

    return diameter_in, pitch_in


def parse_rpm_key(key: str) -> float | None:
    """
    Extracts RPM from keys like:
        RPM_10000
        10000
        rpm10000
    """

    match = re.search(r"(\d+(?:\.\d+)?)", key)

    if match is None:
        return None

    return float(match.group(1))


def as_1d_float_array(values) -> np.ndarray:
    """
    Converts JSON list data to a 1D NumPy float array.
    """

    return np.asarray(values, dtype=float).reshape(-1)


def load_prop_data_points(json_path: str | Path):
    """
    Temporary/simple prop data loader.

    This does NOT build interpolation yet.
    It only loads the raw prop data into arrays:
        diameter, pitch, velocity, rpm, thrust, torque
    """

    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Could not find prop data file: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    diameter_list = []
    pitch_list = []
    velocity_list = []
    rpm_list = []
    thrust_list = []
    torque_list = []

    for prop_key, prop_entry in raw_data.items():
        try:
            diameter_in, pitch_in = parse_prop_key(prop_key)
        except ValueError:
            continue

        if not isinstance(prop_entry, dict):
            continue

        for rpm_key, rpm_entry in prop_entry.items():
            rpm = parse_rpm_key(rpm_key)

            if rpm is None:
                continue

            if not isinstance(rpm_entry, dict):
                continue

            if "V" not in rpm_entry:
                continue

            if "Thrust_2" not in rpm_entry:
                continue

            if "Torque_2" not in rpm_entry:
                continue

            velocity = as_1d_float_array(rpm_entry["V"])
            thrust = as_1d_float_array(rpm_entry["Thrust_2"])
            torque = as_1d_float_array(rpm_entry["Torque_2"])

            n = min(len(velocity), len(thrust), len(torque))

            velocity = velocity[:n]
            thrust = thrust[:n]
            torque = torque[:n]

            valid = (
                np.isfinite(velocity)
                & np.isfinite(thrust)
                & np.isfinite(torque)
            )

            for v, t, q in zip(velocity[valid], thrust[valid], torque[valid]):
                diameter_list.append(diameter_in)
                pitch_list.append(pitch_in)
                velocity_list.append(v)
                rpm_list.append(rpm)
                thrust_list.append(t)
                torque_list.append(q)

    return {
        "diameter_in": np.array(diameter_list),
        "pitch_in": np.array(pitch_list),
        "velocity_mph": np.array(velocity_list),
        "rpm": np.array(rpm_list),
        "thrust_n": np.array(thrust_list),
        "torque_nm": np.array(torque_list),
    }