import aerosandbox as asb
from aerosandbox import OperatingPoint
from aerosandbox import optimization as opti 
from src.aero import aero_analysis
from src.aero.custom_classes import CruiseCondition
from src.vectors import DesignVector
import numpy as np

def cruise_analysis(
        design_vector: DesignVector,
        thrust_velocity: np.array, # array containing a, b, c coefficients of parabola for curve. for now assume throttled thrust curve only
        cg: tuple[float, float, float],
        weight: float,
) -> CruiseCondition:
    """
    Perform cruise analysis for a given design vector. Includes ASB optimization methods and 
    calls to aero_analysis to perform force/moment balance. 

    Args:
        design_vector: The design vector representing the airplane configuration.
        thrust_velocity: Thrust vs velocity graph data determined in prop module.
        cg: The center of gravity of the airplane (x, y, z).
        weight: The weight of the airplane.
    """

    # Three optimization variables 
    velocity = opti.variable(init_guess=1.0, scale=0.05, lower_bound=0.0, upper_bound=50.0) # m/s
    alpha = opti.variable(init_guess=0.0, scale=0.05, lower_bound=-0.0, upper_bound=10.0) # deg
    throttle = opti.variable(init_guess=0.0, scale=0.05, lower_bound=0.0, upper_bound=1.0) # unitless

    # Define flight forces and moments as functions of velocity, alpha, and throttle
    lift = aero_analysis(design_vector, CruiseCondition(OperatingPoint(velocity=velocity, alpha=alpha), throttle)).L
    drag = aero_analysis(design_vector, CruiseCondition(OperatingPoint(velocity=velocity, alpha=alpha), throttle)).D
    thrust = thrust_velocity(velocity, throttle) # still not sure how this will work
    moment = [aero_analysis(design_vector, CruiseCondition(OperatingPoint(velocity=velocity, alpha=alpha), throttle)).l_b,
                aero_analysis(design_vector, CruiseCondition(OperatingPoint(velocity=velocity, alpha=alpha), throttle)).m_b,
                aero_analysis(design_vector, CruiseCondition(OperatingPoint(velocity=velocity, alpha=alpha), throttle)).n_b]
    
    # Constraints
    opti.subject_to(lift == weight)
    opti.subject_to(thrust == drag)
    opti.subject_to(moment[0] == 0.0)
    opti.subject_to(moment[1] == 0.0)
    opti.subject_to(moment[2] == 0.0)

    # Solve
    opti.solve()

    cruise_conditions = CruiseCondition(
        operating_point=OperatingPoint(
            velocity=opti.sol(velocity),
            alpha=opti.sol(alpha),
            beta=0.0,  
            p=0.0,     
            q=0.0,     
            r=0.0     
        ),
        throttle=opti.sol(throttle)
    )

    return cruise_conditions
