import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox import OperatingPoint
from aerosandbox import optimization as opti 
from src.aero.custom_classes import CruiseCondition, AirplaneAnalysisResult
from src.aero.utils import require_scalar
from src.vectors import DesignVector
from src.aero.aerobuildup import run_aerobuildup_on_design_vector
from src.aero.vlm import run_vlm_on_design_vector
from src.aero.lifting_line import run_lifting_line_on_design_vector
from src.aero.nonlinear_lifting_line import (
    run_nonlinear_lifting_line_on_design_vector,
)

# Just change return statement to choose which analysis method to use.

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


def aero_analysis (
        design_vector: DesignVector,
        cruise_condition: CruiseCondition,
        cg: tuple[float, float, float],

) -> AirplaneAnalysisResult:
    """
    Perform aerodynamic analysis for a given design vector and cruise condition.

    Args:
        design_vector: The design vector representing the airplane configuration.
        cruise_condition: The cruise condition containing the operating point and throttle setting.
    """
    
    # Aero Buildup Method
    aero_buildup_result = run_aerobuildup_on_design_vector(
        design_vector=design_vector,
        cg=cg,
        velocity=cruise_condition.operating_point.velocity, # type: ignore
        alpha=cruise_condition.operating_point.alpha, # type: ignore
        beta=0.0,  
        p=0.0,  
        q=0.0,  
        r=0.0,
    )

    # # VLM Method
    # vlm_result = run_vlm_on_design_vector(
    #     design_vector=design_vector,
    #     cg=cg,
    #     velocity=cruise_condition.operating_point.velocity,
    #     alpha=cruise_condition.operating_point.alpha,
    #     beta=0.0,  
    #     p=0.0, 
    #     q=0.0,  
    #     r=0.0,  
    #     spanwise_resolution=6,  # Default resolution, can be adjusted (no idea what to put here)
    #     chordwise_resolution=6,  # Default resolution, can be adjusted (no idea what to put here)
        
    #     # a few other optional params can be changed but didn't seem necessary for now
    #     )
    
    # # Lifting Line Method
    # lifting_line_result = run_lifting_line_on_design_vector(
    #     design_vector=design_vector,
    #     cg=cg,
    #     velocity=cruise_condition.operating_point.velocity,
    #     alpha=cruise_condition.operating_point.alpha,
    #     beta=0.0,  
    #     p=0.0,  
    #     q=0.0,  
    #     r=0.0,  
    # )

    # # Nonlinear Lifting Line Method
    # nonlinear_lifting_line_result = run_nonlinear_lifting_line_on_design_vector(
    #     design_vector=design_vector,
    #     cg=cg,
    #     velocity=cruise_condition.operating_point.velocity,
    #     alpha=cruise_condition.operating_point.alpha,   
    #     beta=0.0,  
    #     p=0.0, 
    #     q=0.0,  
    #     r=0.0, 
    # )


    return aero_buildup_result 