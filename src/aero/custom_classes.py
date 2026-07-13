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
    converged: bool | None = None # False to indicate if trim solved failed to converge
    throttle: float | None = None # TODO - Figure out throttle situation 

@dataclass(frozen=True)
class ModeResult:
    """Compact output for a single stability mode."""

    eigenvalue_real: float
    eigenvalue_imag: float
    damping_ratio: float
    eigenvalue_imag_approx: float | None = None # Only contained by phugoid mode
    damping_ratio_approx: float | None = None # Only contained by phugoid mode

@dataclass(frozen=True)
class StabilityResult:
    """Compact output for a whole-airplane stability analysis run."""

    phugoid: ModeResult
    short_period: ModeResult
    dutch_roll: ModeResult
    spiral: ModeResult
    roll_subsidence: ModeResult

@dataclass(frozen=True)
class AeroOutput:
    """Total output for aero module to be sent to scoring. If 
    converged==False, then the first three args are None."""

    aero_result: AirplaneAnalysisResult | None = None 
    cruise_condition: CruiseCondition | None = None
    stability_result: StabilityResult | None = None
    converged: bool


def dict_to_mode_result(mode_dict: dict) -> ModeResult:
    """Converts a dictionary of mode results to a ModeResult object."""
    return ModeResult(
        eigenvalue_real=mode_dict["eigenvalue_real"],
        eigenvalue_imag=mode_dict["eigenvalue_imag"],
        damping_ratio=mode_dict["damping_ratio"],
        eigenvalue_imag_approx=mode_dict.get("eigenvalue_imag_approx"),
        damping_ratio_approx=mode_dict.get("damping_ratio_approx"),
    )