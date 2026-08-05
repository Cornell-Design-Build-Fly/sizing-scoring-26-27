import aerosandbox as asb
from aerosandbox import OperatingPoint
from aerosandbox import optimization as opti 
from time import perf_counter

from src.aero.aero_analysis import aero_analysis
from src.aero.custom_classes import CruiseCondition
from src.aero.utils import require_scalar
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


def elevator_trim_setup(opti, design_vector, velocity, thrust_velocity):
    control = opti.variable(
        init_guess=0.0, scale=10.0, lower_bound=-20.0, upper_bound=20.0
    )
    airplane = ASBDesignVector.from_design_vector(design_vector).make_airplane(
        elevator_deflection=control,
    )
    return "elevator", control, airplane, eval_thrust(velocity, thrust_velocity)


def tail_incidence_trim_setup(opti, design_vector, velocity, thrust_velocity):
    control = opti.variable(
        init_guess=0.0, scale=2.0, lower_bound=-4.0, upper_bound=4.0
    )
    airplane = ASBDesignVector.from_design_vector(design_vector).make_airplane(
        tail_incidence=control,
    )
    return "tail incidence", control, airplane, eval_thrust(velocity, thrust_velocity)


def throttle_trim_setup(opti, design_vector, velocity, thrust_velocity):
    control = opti.variable(
        init_guess=0.9, scale=0.5, lower_bound=0.0, upper_bound=1.0
    )
    airplane = ASBDesignVector.from_design_vector(design_vector).make_airplane()
    thrust = control * eval_thrust(velocity, thrust_velocity)
    return "throttle", control, airplane, thrust


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

def m3_drag(
        design_vector: DesignVector,
        velocity: float,
        mission: int,
        parameter_vector: ParameterVector,
) -> float:
    """
    Calculate additional drag from Mission 3-specific attachments.

    For the 2025-2026 competition, this consists only of the towed
    banner, modeled with an aspect ratio of 5 and a drag coefficient of 0.05.
    """

    if mission != 3:
        return 0.0

    return 0.005 * parameter_vector.rho * velocity**2 * design_vector.banner_length**2


def cruise_analysis(
        design_vector: DesignVector,
        parameter_vector: ParameterVector,
        thrust_velocity: tuple[float, float, float], # list containing a, b, c coefficients of parabola for curve. for now assume throttled thrust curve only
        cg: tuple[float, float, float],
        mass: float,
        mission: int,
        debug: bool = False,
) -> CruiseCondition:
    """
    Perform thorough and accurate cruise analysis for a given design vector. Includes ASB optimization methods and 
    calls to aero_analysis to perform force/moment balance. Also incorporates thorough stall speed calculation 
    using NeuralFoil data to find Clmax. 

    Args:
        design_vector: The design vector representing the airplane configuration.
        thrust_velocity: Thrust vs velocity graph data determined in prop module.
        cg: The center of gravity of the airplane (x, y, z).
        weight: The weight of the airplane.
    """

    # Create an optimization problem
    if debug:
        print("[aero] Preparing cruise trim optimization...", flush=True)
    opti = asb.Opti()  

    # Flight-condition variables
    velocity = opti.variable(init_guess=18.0, scale=20.0, lower_bound=3.0, upper_bound=50.0) # m/s
    alpha = opti.variable(init_guess=4.0, scale=5.0, lower_bound=-4.0, upper_bound=15.0) # deg

    control_name, trim_control, airplane, thrust = elevator_trim_setup(
        opti, design_vector, velocity, thrust_velocity
    )
    # control_name, trim_control, airplane, thrust = tail_incidence_trim_setup(
    #     opti, design_vector, velocity, thrust_velocity
    # )
    # control_name, trim_control, airplane, thrust = throttle_trim_setup(
    #     opti, design_vector, velocity, thrust_velocity
    # )

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
        include_wave_drag=False, # speeds up calculation to set this as false
    ).run()
    
    # Define lift, drag, and pitching moment from AeroBuildup
    lift = aero["L"]
    drag = aero["D"] + m3_drag(design_vector, velocity, mission, parameter_vector)
    pitching_moment = aero["m_b"]
    
    # Define weight and thrust
    weight = mass * parameter_vector.gravity  # N

    # Residual minimization approach
    lift_residual = (lift - weight) / weight
    drag_residual = (drag - thrust) / weight
    moment_residual = pitching_moment / (weight * airplane.c_ref)
    trim_error = lift_residual**2 + drag_residual**2 + moment_residual**2
    opti.subject_to(lift == weight)
    opti.subject_to(drag == thrust)
    opti.subject_to(pitching_moment == 0)

   # Tolerances used to decide whether the resulting point is truly trimmed.
    LIFT_RESIDUAL_TOL = 1e-2
    DRAG_RESIDUAL_TOL = 1e-2
    MOMENT_RESIDUAL_TOL = 1e-2

    optimization_start = perf_counter()
    if debug:
        print("[aero] Solving cruise trim optimization (this may take a while)...", flush=True)
    try:
        solution = opti.solve(verbose=False)

        solved_velocity = float(solution.value(velocity))
        solved_alpha = float(solution.value(alpha))
        solved_trim_control = float(solution.value(trim_control))

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
        if debug:
            print(
                f"[aero] Cruise trim optimization failed after {perf_counter() - optimization_start:.2f} s: {exc}",
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

    if debug:
        print(
            f"[aero] Cruise trim optimization finished in {perf_counter() - optimization_start:.2f} s "
            f"(converged={converged}, velocity={solved_velocity:.2f} m/s, alpha={solved_alpha:.2f} deg, "
            f"{control_name}={solved_trim_control:.2f}, trim residual={float(solution.value(trim_error)):.3e}, "
            f"lift residual={solved_lift_residual:.2%}, drag residual={solved_drag_residual:.2%}, "
            f"moment residual={solved_moment_residual:.2%}).",
            flush=True,
        )

    # Construct cruise condition object, no stall speed for now
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
        throttle=solved_trim_control if control_name == "throttle" else None,
        elevator_deflection=(
            solved_trim_control if control_name == "elevator" else 0.0
        ),
        tail_incidence=(
            solved_trim_control if control_name == "tail incidence" else 0.0
        ),
    )

    # Stall speed depends on a valid cruise condition. Let aero_main handle an
    # unconverged trim result without attempting the NeuralFoil calculation.
    if not cruise_condition.converged:
        return cruise_condition

    # Calculate and set stall speed.
    calculated_stall_speed = calc_stall_speed(
        design_vector,
        cruise_condition,
        mass,
        parameter_vector,
    )

    return CruiseCondition(
        operating_point=cruise_condition.operating_point,
        stall_speed=calculated_stall_speed,
        converged=True,
        throttle=cruise_condition.throttle,
        elevator_deflection=cruise_condition.elevator_deflection,
        tail_incidence=cruise_condition.tail_incidence,
    )
