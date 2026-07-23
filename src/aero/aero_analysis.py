import aerosandbox as asb
from aerosandbox import OperatingPoint
from aerosandbox import optimization as opti 
from src.aero.custom_classes import CruiseCondition
from src.aero.custom_classes import AirplaneAnalysisResult
from src.vectors import DesignVector
from src.aero.aerobuildup import run_aerobuildup_on_design_vector
from src.aero.vlm import run_vlm_on_design_vector
from src.aero.lifting_line import run_lifting_line_on_design_vector
from src.aero.nonlinear_lifting_line import (
    run_nonlinear_lifting_line_on_design_vector,
)

# Just change return statement to choose which analysis method to use.

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
        velocity=cruise_condition.operating_point.velocity,
        alpha=cruise_condition.operating_point.alpha,
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