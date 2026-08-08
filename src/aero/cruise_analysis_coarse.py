from time import perf_counter

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox import OperatingPoint

from src.aero.custom_classes import CruiseCondition
from src.aero.drag_model import drag_coefficients, fuselage_drag_geometry
from src.vectors import DesignVector, ParameterVector


def cruise_analysis_coarse(
    design_vector: DesignVector,
    parameter_vector: ParameterVector,
    thrust_velocity: tuple[float, float, float],
    cg: tuple[float, float, float],
    mass: float,
    mission: int,
    debug: bool = False,
) -> CruiseCondition:
    """
    Trim cruise with a fast algebraic aerodynamic model.

    Coarse version of initial cruise solver. Changes from full version: 
    - AeroBuildup not used for any aero calculations (to save from the time it takes to construct the object and run the full analysis
    for each new v/alpha/deflection).
    - Same optimization variables used (v, alpha, elevator deflection) 
    - Same ASB optimization approach
    """
    if mass <= 0 or parameter_vector.rho <= 0:
        raise ValueError("Mass and air density must be positive.")

    opti = asb.Opti()
    velocity = opti.variable(init_guess=18.0, scale=20.0, lower_bound=3.0, upper_bound=50.0)
    alpha = opti.variable(init_guess=4.0, scale=5.0, lower_bound=-4.0, upper_bound=15.0)
    elevator = opti.variable(init_guess=0.0, scale=10.0, lower_bound=-20.0, upper_bound=20.0)

    # Lifting-line slopes; assumes attached, linear flow.
    wing_ar = design_vector.wing_span**2 / design_vector.wing_area
    tail_ar = design_vector.hstab_span**2 / design_vector.hstab_area
    wing_slope = 2 * np.pi / (1 + 2 / wing_ar)
    tail_slope = 2 * np.pi / (1 + 2 / tail_ar)
    alpha_rad, elevator_rad = np.radians(alpha), np.radians(elevator)
    wing_cl_est = wing_slope * (alpha_rad - np.radians(-2.0))
    wing_cl = 0.064247 + wing_cl_est * (0.910703 + 0.443763 / wing_ar)
    tail_cl = tail_slope * (0.929043 * alpha_rad + 0.815490 * elevator_rad)
    tail_ratio = design_vector.hstab_area / design_vector.wing_area
    total_cl = wing_cl + tail_ratio * tail_cl

    # Quarter-chord forces; assumes wing Cm_ac = -0.05.
    wing_ac = 0.25 * design_vector.wing_chord
    tail_ac = design_vector.tail_arm + 0.25 * design_vector.hstab_chord
    wing_cm_ac = -0.047069 - 0.021599 * alpha_rad + 0.067132 / wing_ar - 0.044650 * design_vector.wing_chord
    fuselage_length = design_vector.nose_length + design_vector.tail_arm + max(design_vector.hstab_chord, design_vector.vstab_chord)
    body_cm = design_vector.fuselage_height * fuselage_length**2 / (design_vector.wing_area * design_vector.wing_chord) * (
        0.002201 + 0.059479 * alpha_rad + 0.000757 * design_vector.nose_length / fuselage_length
        - 0.057975 * cg[0] / fuselage_length
    )
    cm = (
        wing_cm_ac
        + wing_cl * (cg[0] - wing_ac) / design_vector.wing_chord
        - tail_ratio * tail_cl * (tail_ac - cg[0]) / design_vector.wing_chord
        + body_cm
    )

    # Reynolds-aware profile, fuselage, and interacting induced drag.
    fuselage_geometry = fuselage_drag_geometry(design_vector)
    cd = sum(drag_coefficients(design_vector, parameter_vector, velocity, wing_cl, tail_cl, fuselage_geometry).values())

    dynamic_pressure = 0.5 * parameter_vector.rho * velocity**2
    lift = dynamic_pressure * design_vector.wing_area * total_cl
    drag = dynamic_pressure * design_vector.wing_area * cd
    if mission == 3:
        drag += 0.005 * parameter_vector.rho * velocity**2 * design_vector.banner_length**2
    a, b, c = thrust_velocity
    thrust = a * velocity**2 + b * velocity + c
    weight = mass * parameter_vector.gravity

    lift_residual = (lift - weight) / weight
    drag_residual = (drag - thrust) / weight
    moment_residual = dynamic_pressure * design_vector.wing_area * cm / weight
    trim_error = lift_residual**2 + drag_residual**2 + moment_residual**2
    opti.minimize(trim_error)

    start = perf_counter()
    try:
        solution = opti.solve(verbose=False)
        solved_velocity = float(solution.value(velocity))
        solved_alpha = float(solution.value(alpha))
        solved_elevator = float(solution.value(elevator))
        residuals = [abs(float(solution.value(r))) for r in (lift_residual, drag_residual, moment_residual)]
        converged = max(residuals) <= 1e-2
    except RuntimeError:
        return CruiseCondition(OperatingPoint(velocity=-1.0, alpha=-999.0), None, False)

    # Assumes section CL_max = 1.45 with a finite-wing correction.
    cl_max = 1.45 * wing_ar / (wing_ar + 2.0)
    stall_speed = np.sqrt(2 * weight / (parameter_vector.rho * design_vector.wing_area * cl_max))
    if debug:
        print(f"[aero] Coarse trim finished in {perf_counter() - start:.4f} s (converged={converged}).", flush=True)
    return CruiseCondition(
        operating_point=OperatingPoint(velocity=solved_velocity, alpha=solved_alpha),
        stall_speed=float(stall_speed) if converged else None,
        converged=converged,
        elevator_deflection=solved_elevator,
    )
