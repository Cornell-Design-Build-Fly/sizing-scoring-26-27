# and here I am, catching you slacking, looking at the codebase for the first time...
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from src.vectors import DesignVector, ParameterVector
from src.prop.prop_classes import (
    Battery,
    Motor,
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
    c_r = np.array([0.3517732388, -0.0005385476, -0.0001855504, 0.0000002999, 0.0000000776, 0.0000000380,])
    c_I = np.array([-0.5621009279, 0.0005335965, 0.0016292435, 0.0000005495, 0.0000006015, -0.0000004552])

    Rm = c_R[0] + c_R[1]*kv + c_R[2]*max_power_w + c_R[3]*kv**2 + c_R[4]*kv*max_power_w + c_R[5]*max_power_w**2;
    Inot = c_I[0] + c_I[1]*kv + c_I[2]*max_power_w + c_I[3]*kv**2 + c_I[4]*kv*max_power_w + c_I[5]*max_power_w**2;
    
    return Rm, Inot



'''MOTOR CHECK FUNCTION'''
def motor_check(torque_nm: float, rpm: float, motor: Motor, battery: Battery)
    pass = True
    current = (torque / Kt) + I0 #(A) Current needed to sustain torque

    V_sag = battery.vnom - current_A*(battery.Rb) #(V) Voltage drop in battery under load
    V_req = rpm/motor.kv+current+motor.Rm #Voltage required due to EMF
    power= current*V_sag #(W) Power consumed by motor

    #Battery flight time calculation

    if I <= 1e-6:  #Avoid division by zero or very small power; use a small threshold
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
    if V_req > battery.vnom and pass:
        # fprintf('MOTOR CHECK FAIL: Insufficient nominal voltage. RPM: %.0f, V_req: %.2f V > V_nom: %.2f V\n', RPM, V_req, V_nom);
        pass = False

    # Throttle Limit Check (This is often the key RPM limiting factor)
    if throttle > 1.0 and pass:
        # fprintf('MOTOR CHECK FAIL: Throttle overload. RPM: %.0f, Throttle: %.3f > 1. (V_req: %.2f V, V_sag: %.2f V, I: %.2f A)\n', RPM, throttle, V_req, V_sag, I);
        pass = False
    
    # Check for V_sag becoming non-positive (battery completely depleted or calculation issue)
    if V_sag <= 0 and pass:
        # fprintf('MOTOR CHECK FAIL: Battery voltage sagged too low. RPM: %.0f, V_sag: %.2f V, I: %.2f A\n', RPM, V_sag, I);
        pass = False

