import aerosandbox as asb
from aerosandbox import OperatingPoint
from aerosandbox import optimization as opti 
from time import perf_counter

from src.aero.aero_analysis import aero_analysis
from src.aero.custom_classes import CruiseCondition
from src.aero.utils import require_scalar
from src.vectors import DesignVector, ASBDesignVector, ParameterVector
import numpy as np

def cruise_analysis_coarser(
        design_vector: DesignVector,
        parameter_vector: ParameterVector,
        thrust_velocity: tuple[float, float, float], # list containing a, b, c coefficients of parabola for curve. for now assume throttled thrust curve only
        cg: tuple[float, float, float],
        mass: float,
        mission: int,
) -> CruiseCondition:

    """
    Coarse version of initial cruise solver. Changes from full version: 
    - AeroBuildup not used for any aero calculations (to save from the time it takes to construct the object and run the full analysis
    for each new v/alpha/deflection).
    - Same optimization variables used (v, alpha, elevator deflection) 
    """

        # Write lift equation
                # Estimate Cl from alpha, design vector, elevator deflection
                # Sub in velocity
        
        # Write drag equation
                # Estimate Cd 
                # Sub in velocity, alpha, elevator deflection

        # Write moment equation
                # Need to look into how to do this

        # Set optimization conditions, create optimization variables, solve optimization problem.   



        # temporary
    return CruiseCondition(
                operating_point=OperatingPoint(
                    velocity=-1.0,
                    alpha=-999.0,
                    beta=0.0,
                    p=0.0,
                    q=0.0,
                    r=0.0,
                ),
                stall_speed=None,
                converged=False,
            )