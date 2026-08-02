from time import perf_counter

import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox import OperatingPoint

from src.aero.custom_classes import CruiseCondition
from src.vectors import DesignVector, ParameterVector


def cruise_analysis_coarse(
    design_vector: DesignVector,
    parameter_vector: ParameterVector,
    thrust_velocity: tuple[float, float, float],
    cg: tuple[float, float, float],
    mass: float,
    mission: int,
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
    wing_cl = wing_slope * (alpha_rad - np.radians(-2.0))  # Estimated zero-lift angle.
    tail_cl = tail_slope * (0.65 * alpha_rad + 0.60 * elevator_rad)  # Downwash and elevator estimates.
    tail_ratio = design_vector.hstab_area / design_vector.wing_area
    total_cl = wing_cl + 0.90 * tail_ratio * tail_cl  # Assumed tail dynamic-pressure ratio.

    # Quarter-chord forces; assumes wing Cm_ac = -0.05.
    wing_ac = 0.25 * design_vector.wing_chord
    tail_ac = design_vector.tail_arm + 0.25 * design_vector.hstab_chord
    cm = (
        -0.05
        + wing_cl * (cg[0] - wing_ac) / design_vector.wing_chord
        - 0.90 * tail_ratio * tail_cl * (tail_ac - cg[0]) / design_vector.wing_chord
    )

    # Empirical parasite drag plus Oswald induced drag.
    cd0 = (
        0.018
        + 0.012 * (design_vector.hstab_area + design_vector.vstab_area) / design_vector.wing_area
        + 0.08 * design_vector.fuselage_width * design_vector.fuselage_height / design_vector.wing_area
    )
    cd = (
        cd0
        + wing_cl**2 / (np.pi * 0.85 * wing_ar)
        + 0.90 * tail_ratio * tail_cl**2 / (np.pi * 0.80 * tail_ar)
    )

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
    print(f"[aero] Coarse trim finished in {perf_counter() - start:.4f} s (converged={converged}).", flush=True)
    return CruiseCondition(
        operating_point=OperatingPoint(velocity=solved_velocity, alpha=solved_alpha),
        stall_speed=float(stall_speed) if converged else None,
        converged=converged,
        elevator_deflection=solved_elevator,
    )
