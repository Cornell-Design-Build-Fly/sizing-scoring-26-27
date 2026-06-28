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

'''MATLAB TESTING FOLDER'''
def motor_check(torque_nm: float, rpm: float, motor: Motor, battery: Battery)
    