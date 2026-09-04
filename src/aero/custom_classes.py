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
    stall_speed: float | None
    converged: bool | None = None # False to indicate if trim solved failed to converge
    throttle: float | None = None # TODO - Figure out throttle situation 
    elevator_deflection: float = 0.0
    tail_incidence: float = 0.0

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
    Cma: float
    Cnb: float
    # Geometric static margin, (x_np - x_cg) / c_ref. This is the single
    # definition used by the flyability gates, the mechanical placement target,
    # and the optimizer penalty.
    static_margin: float | None = None
    neutral_point_x_m: float | None = None
    # Diagnostic only: the legacy -Cma/CLa value. It is NOT a static margin,
    # because the calibrated Cma regression does not preserve the identity
    # dCma/dx_cg = CLa / c_ref. Retained so the two can be compared.
    static_margin_from_cma: float | None = None
    # Seconds for the spiral mode to double bank angle; inf when convergent.
    # Derived from the 4-state lateral solve in src/aero/stability_criteria.py,
    # not from get_modes' spiral approximation, which divides by Clb and is
    # singular on this zero-dihedral geometry.
    spiral_time_to_double_s: float | None = None

@dataclass(frozen=True)
class AeroOutput:
    """Total output for aero module to be sent to scoring. If 
    converged==False, then the first three args are None."""

    converged: bool
    aero_result: AirplaneAnalysisResult | None = None 
    cruise_condition: CruiseCondition | None = None
    stability_result: StabilityResult | None = None
