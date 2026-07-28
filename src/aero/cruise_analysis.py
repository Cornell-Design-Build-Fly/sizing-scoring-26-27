import aerosandbox as asb
from aerosandbox import OperatingPoint
from aerosandbox import optimization as opti 
from time import perf_counter

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
    mass: float,
    parameter_vector: ParameterVector,
) -> float:
    """
    Estimate aircraft stall speed from the NeuralFoil airfoil cl_max.

    The airfoil section cl_max is obtained from ``cl_max()`` and converted
    to an approximate finite-wing CL_max using an aspect-ratio correction.

    """
    if mass <= 0:
        raise ValueError(
            f"Aircraft mass must be positive, but got {mass} kg."
        )

    if design_vector.wing_area <= 0:
        raise ValueError(
            f"Wing area must be positive, but got "
            f"{design_vector.wing_area} m^2."
        )

    if parameter_vector.rho <= 0:
        raise ValueError(
            f"Air density must be positive, but got "
            f"{parameter_vector.rho} kg/m^3."
        )

    # Maximum 2D section lift coefficient from NeuralFoil.
    airfoil_cl_max = cl_max(
        design_vector=design_vector,
        cruise_condition=cruise_condition,
    )

    # Wing aspect ratio:
    #
    #              b^2
    #     AR = -----------
    #               S
    #
    aspect_ratio = (
        design_vector.wing_span**2
        / design_vector.wing_area
    )

    # Approximate finite-wing correction.
    #
    # A real finite wing generally has a lower maximum lift coefficient
    # than its 2D airfoil section. This correction approaches 1.0 as
    # aspect ratio increases.
    finite_wing_factor = aspect_ratio / (aspect_ratio + 2.0)

    aircraft_cl_max = airfoil_cl_max * finite_wing_factor

    if not np.isfinite(aircraft_cl_max) or aircraft_cl_max <= 0:
        raise ValueError(
            f"Calculated aircraft CL_max is invalid: {aircraft_cl_max}."
        )

    weight = mass * parameter_vector.gravity

    stall_speed_mps = np.sqrt(
        (2.0 * weight)
        / (
            parameter_vector.rho
            * design_vector.wing_area
            * aircraft_cl_max
        )
    )

    return float(stall_speed_mps)

def cl_max(
    design_vector: DesignVector,
    cruise_condition: CruiseCondition,
) -> float:
    """
    Estimates the maximum 2D airfoil lift coefficient using NeuralFoil.

    The airfoil is evaluated across a range of angles of attack at the
    Reynolds number and Mach number corresponding to the aircraft's
    cruise condition.

    """
    if not cruise_condition.converged:
        raise ValueError(
            "Cannot calculate cl_max because cruise trim did not converge."
        )

    operating_point = cruise_condition.operating_point

    velocity = require_scalar(operating_point.velocity)

    if velocity <= 0:
        raise ValueError(
            f"Cruise velocity must be positive, but got {velocity} m/s."
        )

    if design_vector.wing_chord <= 0:
        raise ValueError(
            "Wing chord must be positive to calculate Reynolds number."
        )

    # AeroSandbox calculates these using the atmosphere stored inside
    # the cruise OperatingPoint.
    reynolds_number = require_scalar(
        operating_point.reynolds(
            reference_length=design_vector.wing_chord
        )
    )

    mach = require_scalar(operating_point.mach())

    # Evaluate the airfoil every 0.5 degrees from 10 to 20 degrees (most viable airfoils will have max Cl at alpha in that range).
    alpha_values = np.linspace(10.0, 20.0, 21)

    airfoil = asb.Airfoil(design_vector.wing_airfoil)

    polar = airfoil.get_aero_from_neuralfoil(
        alpha=alpha_values,
        Re=reynolds_number,
        mach=mach,
        model_size="small",
    )

    cl_values = np.asarray(polar["CL"], dtype=float).reshape(-1)

    finite_mask = np.isfinite(cl_values)

    if not np.any(finite_mask):
        raise ValueError(
            "NeuralFoil did not return any finite lift-coefficient values."
        )

    # Do not allow failed values to be selected as the maximum.
    valid_cl_values = np.where(finite_mask, cl_values, -np.inf)

    max_index = int(np.argmax(valid_cl_values))
    maximum_cl = float(valid_cl_values[max_index])
    alpha_at_cl_max = float(alpha_values[max_index])

    # If the largest CL is at the upper boundary, the sweep may not have
    # extended far enough to capture the actual peak.
    if max_index == len(alpha_values) - 1:
        raise ValueError(
            "The maximum NeuralFoil CL occurred at the upper alpha limit "
            f"of {alpha_at_cl_max:.1f} degrees. Increase the alpha range "
            "before treating this value as cl_max."
        )

    return maximum_cl


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
    print("[aero] Preparing cruise trim optimization...", flush=True)
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
        xyz_ref=np.array(cg), # match data type of cg
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
    lift_residual = (lift - weight) / weight # type: ignore
    drag_residual = (drag - thrust) / weight # type: ignore
    moment_residual = pitching_moment / (
        weight * airplane.c_ref
    ) # type: ignore

    trim_error = (
        lift_residual**2
        + drag_residual**2
        + moment_residual**2
    )

    opti.minimize(trim_error) # type: ignore

   # Tolerances used to decide whether the resulting point is truly trimmed.
    LIFT_RESIDUAL_TOL = 1e-2
    DRAG_RESIDUAL_TOL = 1e-2
    MOMENT_RESIDUAL_TOL = 1e-2

    optimization_start = perf_counter()
    print("[aero] Solving cruise trim optimization (this may take a while)...", flush=True)
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

    except RuntimeError as exc:
        print(
            f"[aero] Cruise trim optimization failed after "
            f"{perf_counter() - optimization_start:.2f} s: {exc}",
            flush=True,
        )

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

    print(
        f"[aero] Cruise trim optimization finished in "
        f"{perf_counter() - optimization_start:.2f} s "
        f"(converged={converged}, velocity={solved_velocity:.2f} m/s, "
        f"alpha={solved_alpha:.2f} deg, "
        f"trim residual={float(solution.value(trim_error)):.3e}).",
        flush=True,
    )

    # COnstruct cruise condition object, no stall speed for now
    cruise_condition = CruiseCondition(
        operating_point=OperatingPoint(
            velocity=solved_velocity,
            alpha=solved_alpha,
            beta=0.0,  
            p=0.0,     
            q=0.0,     
            r=0.0     
        ),
        stall_speed=None,
        converged=converged,
    )

    # Calculate and set stall speed
    stall_speed = stall_speed(
        design_vector,
        cruise_condition,
        mass,
        parameter_vector,
    )

    cruise_condition.stall_speed = stall_speed

    # Return solved values and whether converged within defined tolerances.
    return cruise_condition
