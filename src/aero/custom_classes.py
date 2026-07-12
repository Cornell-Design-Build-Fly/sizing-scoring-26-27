import aerosandbox as asb
from aerosandbox import OperatingPoint
from dataclasses import dataclass

# The purpose of this file is to define custom classes for use in the aero module. There is essentially
# one class for each of the three aero analysis functions/files - cruise, stability, and aerodynamics. 

# (I think AirplaneAnalysisResult should be defined here instead of in vlm.py to make it simpler to find).

@dataclass(frozen=True)
class AirplaneAnalysisResult:
    """Compact output for a whole-airplane aerodynamic analysis run."""

    CL: float
    CD: float
    CY: float
    Cl: float
    Cm: float
    Cn: float
    L: float
    D: float
    Y: float
    l_b: float
    m_b: float
    n_b: float
    runtime_seconds: float
    converged: bool = True
    CDi: float | None = None
    CDp: float | None = None
    D_induced: float | None = None
    D_profile: float | None = None

@dataclass(frozen=True)
class CruiseCondition:
    """Compact output for a whole-airplane cruise analysis run. Contains the "operating point," which contains 
    the velocity, angle of attack, and other parameters at cruise (see spec in aerosandbox), as well as the 
    throttle setting."""

    operating_point: OperatingPoint
    throttle: float


@dataclass(frozen=True)
class StabilityResult:
    """Compact output for a whole-airplane stability analysis run."""

    phugoid: float
    short_period: float
    dutch_roll: float
    spiral: float
    roll_subsidence: float

    # static and dynamic stability metrics to be added once I figure that part out

@dataclass(frozen=True)
class AeroOutput:
    """Total output for aero module to be sent to scoring."""

    aero_result: asb.AirplaneAnalysisResult
    cruise_condition: CruiseCondition
    stability_result: StabilityResult