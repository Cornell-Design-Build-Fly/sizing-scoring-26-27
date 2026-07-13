import aerosandbox as asb
from aerosandbox import OperatingPoint
from aerosandbox import optimization as opti 
from src.aero.aero_analysis import aero_analysis
from src.aero.custom_classes import CruiseCondition
from src.vectors import DesignVector
import numpy as np

def cruise_analysis(
        design_vector: DesignVector,
        thrust_velocity: list[int, int, int], # list containing a, b, c coefficients of parabola for curve. for now assume throttled thrust curve only
        cg: tuple[float, float, float],
        mass: float,
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

    # Create an optimization problem
    opti = asb.Opti()  

    # Twp optimization variables 
    velocity = opti.variable(init_guess=18.0, scale=0.05, lower_bound=3.0, upper_bound=50.0) # m/s
    alpha = opti.variable(init_guess=4.0, scale=0.05, lower_bound=-4.0, upper_bound=15.0) # deg

    # Build the airplane
    airplane = design_vector.to_asb_airplane()

    # Operating point depends symbolically on velocity and alpha
    op_point = asb.OperatingPoint(
    velocity=velocity,
    alpha=alpha,
    beta=0.0,
    p=0.0,
    q=0.0,
    r=0.0,
    )

    # Symbolic aero analysis on plane
    aero = asb.AeroBuildup(
        airplane=airplane,
        op_point=op_point,
        xyz_ref=cg,
    ).run()

    # # Define flight forces and moments as functions of velocity, alpha, and throttle
    # # PROBLEM - aero_analysis specifically returns floats (not functions)
    # lift = aero_analysis(design_vector, CruiseCondition(OperatingPoint(velocity=velocity, alpha=alpha), cg,)).L
    # drag = aero_analysis(design_vector, CruiseCondition(OperatingPoint(velocity=velocity, alpha=alpha), cg,)).D
    # thrust = eval_thrust(velocity, thrust_velocity) # may or may not need to modify for throttle conditions
    # moment = [aero_analysis(design_vector, CruiseCondition(OperatingPoint(velocity=velocity, alpha=alpha), cg,)).l_b,
    #             aero_analysis(design_vector, CruiseCondition(OperatingPoint(velocity=velocity, alpha=alpha), cg,)).m_b,
    #             aero_analysis(design_vector, CruiseCondition(OperatingPoint(velocity=velocity, alpha=alpha), cg,)).n_b]
    
    # Define lift, drag, and pitching moment from AeroBuildup
    lift = aero["L"]
    drag = aero["D"]
    pitching_moment = aero["m_b"]
    
    # Define weight and thrust
    thrust = eval_thrust(velocity, thrust_velocity)
    weight = mass * 9.81  # N

    # Constraints
    opti.subject_to(lift == weight)
    opti.subject_to(drag == thrust)
    opti.subject_to(pitching_moment == 0)

    # Solve
    try:
        solution = opti.solve()

        solved_velocity = float(solution.value(velocity))
        solved_alpha = float(solution.value(alpha))

    except:
        # If failed to converge, return with converged=False
        return CruiseCondition(
        operating_point=OperatingPoint(
            velocity=-1,
            alpha=-999,
            beta=0.0,  
            p=0.0,     
            q=0.0,     
            r=0.0     
        ),
        converged=False,
        )

    # If successfully converged, return conditions
    return CruiseCondition(
        operating_point=OperatingPoint(
            velocity=solved_velocity,
            alpha=solved_alpha,
            beta=0.0,  
            p=0.0,     
            q=0.0,     
            r=0.0     
        ),
        converged=True,
    )



def eval_thrust(
        velocity: float,
        thrust_velocity: list[int, int, int], # list containing a, b, c coefficients of parabola for curve. for now assume throttled thrust curve only
) -> float:
    """
    Evaluate the thrust at a given velocity using the provided thrust-velocity curve.

    Args:
        velocity: The velocity at which to evaluate the thrust.
        thrust_velocity: A list containing the coefficients [a, b, c] of the quadratic equation representing the thrust-velocity curve.

    Returns:
        The evaluated thrust at the given velocity.
    """
    a, b, c = thrust_velocity
    return a * velocity**2 + b * velocity + c