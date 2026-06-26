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
