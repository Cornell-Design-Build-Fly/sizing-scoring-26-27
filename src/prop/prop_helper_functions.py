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
    Battery,
    Motor,
    MotorCheckResult,
    PropInterpolants,
    PropulsionCurveFit,
    MPS_TO_MPH,
    DEFAULT_VELOCITIES_MPS,
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
    c_R = np.array([0.3517732388, -0.0005385476, -0.0001855504, 0.0000002999, 0.0000000776, 0.0000000380,])
    c_I = np.array([-0.5621009279, 0.0005335965, 0.0016292435, 0.0000005495, 0.0000006015, -0.0000004552])

    Rm = c_R[0] + c_R[1]*kv + c_R[2]*max_power_w + c_R[3]*kv**2 + c_R[4]*kv*max_power_w + c_R[5]*max_power_w**2;
    Inot = c_I[0] + c_I[1]*kv + c_I[2]*max_power_w + c_I[3]*kv**2 + c_I[4]*kv*max_power_w + c_I[5]*max_power_w**2;
    
    return Rm, Inot



'''MOTOR CHECK FUNCTION'''
def motor_check(torque: float, rpm: float, motor: Motor, battery: Battery):
    if rpm <= 0:
        raise ValueError("RPM must be positive.")
    passed = True
    current = (torque / motor.get_kt()) + motor.I0 #(A) Current needed to sustain torque

    V_sag = battery.vnom - current*(battery.Rb) #(V) Voltage drop in battery under load
    V_req = rpm/motor.kv+current*motor.Rm #Voltage required due to EMF
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
        throttle = np.inf  # Effectively infinite throttle required if V_sag is non-positive
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
    
    return MotorCheckResult(
        passed=passed,
        current_a=current,
        voltage_sag_v=V_sag,
        voltage_required_v=V_req,
        throttle=throttle,
        power_w=power,
        flight_time_s=t_flight
    )

def _get_value(obj, name: str, default):
    """
    Gets an attribute from DesignVector or ParameterVector.
    Uses default if the field does not exist.
    """
    return getattr(obj, name, default)


def make_motor_from_design(
    design_vector: DesignVector,
    parameter_vector: ParameterVector = ParameterVector,
) -> Motor:
    """
    Creates a Motor object from the DesignVector and ParameterVector.

        Motor(kv, Rm, max_power, I0, max_current, mass=None)
    """

    kv = float(_get_value(design_vector, "motor_kv", 335.0))
    max_power = float(_get_value(design_vector, "motor_max_power", 2200.0))
    max_current = float(_get_value(parameter_vector, "max_current", 100.0))

    Rm, I0 = motor_properties(
        kv=kv,
        max_power_w=max_power,
    )

    return Motor(
        kv=kv,
        Rm=Rm,
        max_power=max_power,
        I0=I0,
        max_current=max_current,
    )


def make_battery_from_design(
    design_vector: DesignVector,
    parameter_vector: ParameterVector = ParameterVector,
) -> Battery:
    """
    Creates a Battery object from the DesignVector and ParameterVector.

    Battery(vnom, cells, Rb, Crat, capacity, mass=None, useable_fraction=...)
    """

    capacity_ah = float(_get_value(design_vector, "batt_capacity", 4.5))
    vnom = float(_get_value(parameter_vector, "voltage", 22.2))

    cells_default = max(1, int(round(vnom / 3.7)))
    cells = int(_get_value(parameter_vector, "num_battery_cells", cells_default))

    useable_fraction = float(_get_value(parameter_vector, "usable_battery_fraction", 0.85))

    Rb = battery_resistance(capacity_ah=capacity_ah,num_cells=cells,)

    return Battery(
        vnom=vnom,
        cells=cells,
        Rb=Rb,
        Crat=0.0,  # Placeholder until C-rating is actually modeled
        capacity=capacity_ah * useable_fraction,
        useable_fraction=useable_fraction,
    )