import aerosandbox as asb
from aerosandbox import OperatingPoint
from dataclasses import dataclass

# The purpose of this file is to define custom classes for use in the aero module. There is essentially
# one class for each of the three aero analysis functions/files - cruise, stability, and aerodynamics. 

# (I think AirplaneAnalysisResult should be defined here instead of in vlm.py to make it simple to find).

# @dataclass(frozen=True)
# class AirplaneAnalysisResult:
#     """Compact output for a whole-airplane aerodynamic analysis run."""

#     CL: float
#     CD: float
#     CY: float
#     Cl: float
#     Cm: float
#     Cn: float
#     L: float
#     D: float
#     Y: float
#     l_b: float
#     m_b: float
#     n_b: float
#     runtime_seconds: float
#     converged: bool = True
#     CDi: float | None = None
#     CDp: float | None = None
#     D_induced: float | None = None
#     D_profile: float | None = None

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

    short_period_eigenvalues: np.ndarray
    short_period_eigenvectors: np.ndarray
    phugoid_eigenvalues: np.ndarray
    phugoid_eigenvectors: np.ndarray
    dutch_roll_eigenvalues: np.ndarray
    dutch_roll_eigenvectors: np.ndarray
    spiral_mode_eigenvalues: np.ndarray
    spiral_mode_eigenvectors: np.ndarray
    