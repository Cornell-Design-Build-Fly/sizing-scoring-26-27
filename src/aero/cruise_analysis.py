import aerosandbox as asb
from aerosandbox import OperatingPoint
from aerosandbox import optimization as opti 
from src.aero.aero_analysis import aero_analysis
from src.aero.custom_classes import CruiseCondition
from src.vectors import DesignVector, ASBDesignVector, ParameterVector
import numpy as np

def eval_thrust(
            velocity: float,
            thrust_velocity: tuple[float, float, float], # list containing a, b, c coefficients of parabola for curve. for now assume throttled thrust curve only
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

# TODO
def calc_stall_speed(
        design_vector: DesignVector,
        cruise_condition: CruiseCondition,
    ) -> float:
        """
        Estimate CL_max and stall speed using an angle-of-attack sweep.

        Returns:
        stall_speed_mps
        cl_max
        alpha_at_cl_max_deg
        """

def cruise_analysis(
        design_vector: DesignVector,
        parameter_vector: ParameterVector,
        thrust_velocity: tuple[float, float, float], # list containing a, b, c coefficients of parabola for curve. for now assume throttled thrust curve only
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
    airplane = ASBDesignVector.from_design_vector(design_vector).make_airplane()

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
    
    # Define lift, drag, and pitching moment from AeroBuildup
    lift = aero["L"]
    drag = aero["D"]
    pitching_moment = aero["m_b"]
    
    # Define weight and thrust
    thrust = eval_thrust(velocity, thrust_velocity)
    weight = mass * parameter_vector.gravity  # N

    # ------------------- Initial approach: 2 variables 3 equations ----------------

    # # Constraints
    # opti.subject_to(lift == weight)
    # opti.subject_to(drag == thrust)
    # opti.subject_to(pitching_moment == 0)

    # # Solve
    # try:
    #     solution = opti.solve()

    #     solved_velocity = float(solution.value(velocity))
    #     solved_alpha = float(solution.value(alpha))

    # except:
    #     # If failed to converge, return with converged=False
    #     return CruiseCondition(
    #     operating_point=OperatingPoint(
    #         velocity=-1,
    #         alpha=-999,
    #         beta=0.0,  
    #         p=0.0,     
    #         q=0.0,     
    #         r=0.0     
    #     ),
    #     converged=False,
    #     )
    
    # --------------------------------------------------------------------------------

    # ---------------------- New approach: residual solver ---------------------------
    lift_residual = (lift - weight) / weight
    drag_residual = (drag - thrust) / weight
    moment_residual = pitching_moment / (
        weight * airplane.c_ref
    )

    trim_error = (
        lift_residual**2
        + drag_residual**2
        + moment_residual**2
    )

    opti.minimize(trim_error)

   # Tolerances used to decide whether the resulting point is truly trimmed.
    LIFT_RESIDUAL_TOL = 1e-1
    DRAG_RESIDUAL_TOL = 1e-1
    MOMENT_RESIDUAL_TOL = 1e-1

    print("Trim residual: " + float(solution.value(trim_error)))

    try:
        solution = opti.solve()

        solved_velocity = float(solution.value(velocity))
        solved_alpha = float(solution.value(alpha))

        solved_lift_residual = abs(
            float(solution.value(lift_residual))
        )
        solved_drag_residual = abs(
            float(solution.value(drag_residual))
        )
        solved_moment_residual = abs(
            float(solution.value(moment_residual))
        )

        converged = (
            solved_lift_residual <= LIFT_RESIDUAL_TOL
            and solved_drag_residual <= DRAG_RESIDUAL_TOL
            and solved_moment_residual <= MOMENT_RESIDUAL_TOL
        )

    except RuntimeError:
        return CruiseCondition(
        operating_point=OperatingPoint(
            velocity=-1.0,
            alpha=-999.0,
            beta=0.0,  
            p=0.0,     
            q=0.0,     
            r=0.0     
        ),
        stall_speed=None,
        converged=False,
    )

    # Calculate and set stall speed
    RHO = parameter_vector.rho
    S_REF = design_vector.wing_area
    WEIGHT  = mass * parameter_vector.gravity
    CL = 0.5 # TODO - TEMPORARY, NEED TO FIX
    # TODO - fix stall speed- needs to use CL max instead of cruise CL
    stall_speed = (2 * WEIGHT / (RHO * S_REF * CL)) ** 0.5

    # Return solved values and whether converged within defined tolerances.
    return CruiseCondition(
        operating_point=OperatingPoint(
            velocity=solved_velocity,
            alpha=solved_alpha,
            beta=0.0,  
            p=0.0,     
            q=0.0,     
            r=0.0     
        ),
        stall_speed=stall_speed,
        converged=converged,
    )