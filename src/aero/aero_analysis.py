import aerosandbox as asb
from aerosandbox import OperatingPoint
from aerosandbox import optimization as opti 
from src.aero.custom_classes import CruiseCondition
from src.vectors import DesignVector
import vlm
import lifting_line
import nonlinear_lifting_line

# I know this is currently a lot of code for no reason because it is basically already present in the vlm, lifting_line,
# and nonlinear_lifting_line files, but I just made this file for now for flexibility later on if we want to
# add/modify aero analysis beyond the three methods. 

def aero_analysis (
        design_vector: DesignVector,
        cruise_condition: CruiseCondition,

) -> vlm.AirplaneAnalysisResult:
    """
    Perform aerodynamic analysis for a given design vector and cruise condition.

    Args:
        design_vector: The design vector representing the airplane configuration.
        cruise_condition: The cruise condition containing the operating point and throttle setting.
    """
    
    # Aero Buildup Method
    aero_buildup_result = vlm.run_aero_buildup_on_design_vector(
        design_vector=design_vector,
        velocity=cruise_condition.operating_point.velocity,
        alpha=cruise_condition.operating_point.alpha,
        beta=0.0,  
        p=0.0,  
        q=0.0,  
        r=0.0,
    )

    # VLM Method
    vlm_result = vlm.run_vlm_on_design_vector(
        design_vector=design_vector,
        velocity=cruise_condition.operating_point.velocity,
        alpha=cruise_condition.operating_point.alpha,
        beta=0.0,  
        p=0.0, 
        q=0.0,  
        r=0.0,  
        spanwise_resolution=6,  # Default resolution, can be adjusted (no idea what to put here)
        chordwise_resolution=6,  # Default resolution, can be adjusted (no idea what to put here)
        
        # a few other optional params can be changed but didn't seem necessary for now
        )
    
    # Lifting Line Method
    lifting_line_result = lifting_line.run_lifting_line_on_design_vector(
        design_vector=design_vector,
        velocity=cruise_condition.operating_point.velocity,
        alpha=cruise_condition.operating_point.alpha,
        beta=0.0,  
        p=0.0,  
        q=0.0,  
        r=0.0,  
    )

    # Nonlinear Lifting Line Method
    nonlinear_lifting_line_result = nonlinear_lifting_line.run_nonlinear_lifting_line_on_design_vector(
        design_vector=design_vector,
        velocity=cruise_condition.operating_point.velocity,
        alpha=cruise_condition.operating_point.alpha,   
        beta=0.0,  
        p=0.0, 
        q=0.0,  
        r=0.0, 
    )


    return vlm_result # could also return lifting_line_result and nonlinear_lifting_line_result if desired 