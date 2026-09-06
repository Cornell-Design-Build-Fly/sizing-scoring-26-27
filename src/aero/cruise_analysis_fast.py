from time import perf_counter

import numpy as np
from aerosandbox import OperatingPoint
from scipy.optimize import brentq, minimize_scalar

from src.aero.custom_classes import CruiseCondition
from src.aero.drag_model import drag_coefficients, fuselage_drag_geometry
from src.aero.tow_line_model import tow_line_force_components
from src.vectors import DesignVector, ParameterVector


def _trim_violation(alpha, elevator, drag, thrust, weight):
    """Search scales: 10% weight force mismatch or 5 degrees excess = 1.

    Both thrust excess and shortfall matter at the selected fixed throttle.
    This objective guides failed designs; it never grants mission completion.
    """
    force = np.abs(drag - thrust) / (0.10 * weight)
    angle = np.maximum(0., np.maximum(-4. - alpha, alpha - 15.)) / 5.
    control = np.maximum(0., np.abs(elevator) - 20.) / 5.
    value = force + angle + control
    return np.where(np.isfinite(value), value, np.inf)


def cruise_analysis_fast(
    design_vector: DesignVector,
    parameter_vector: ParameterVector,
    thrust_velocity: tuple[float, float, float],
    cg: tuple[float, float, float],
    mass: float,
    mission: int,
    debug: bool = False,
) -> CruiseCondition:
    """Trim cruise by eliminating alpha and elevator before solving velocity."""
    if mass <= 0 or parameter_vector.rho <= 0:
        raise ValueError("Mass and air density must be positive.")

    start = perf_counter()
    wing_ar = design_vector.wing_span**2 / design_vector.wing_area
    tail_ar = design_vector.hstab_span**2 / design_vector.hstab_area
    wing_slope = 2 * np.pi / (1 + 2 / wing_ar)
    tail_slope = 2 * np.pi / (1 + 2 / tail_ar)
    tail_ratio = design_vector.hstab_area / design_vector.wing_area
    wing_factor = 0.910703 + 0.443763 / wing_ar

    # Linear lift and moment coefficients in alpha and elevator (radians).
    wing_cl0 = 0.064247 + wing_slope * np.radians(2.0) * wing_factor
    wing_cla = wing_slope * wing_factor
    tail_cla, tail_cle = tail_slope * 0.929043, tail_slope * 0.815490
    cl0 = wing_cl0
    cla = wing_cla + tail_ratio * tail_cla
    cle = tail_ratio * tail_cle

    wing_ac = 0.25 * design_vector.wing_chord
    tail_ac = design_vector.tail_arm + 0.25 * design_vector.hstab_chord
    wing_lever = (cg[0] - wing_ac) / design_vector.wing_chord
    tail_lever = (tail_ac - cg[0]) / design_vector.wing_chord
    fuselage_length = design_vector.nose_length + design_vector.tail_arm + max(
        design_vector.hstab_chord, design_vector.vstab_chord
    )
    body_scale = (
        design_vector.fuselage_height
        * fuselage_length**2
        / (design_vector.wing_area * design_vector.wing_chord)
    )
    cm0 = (
        -0.047069
        + 0.067132 / wing_ar
        - 0.044650 * design_vector.wing_chord
        + wing_cl0 * wing_lever
        + body_scale
        * (
            0.002201
            + 0.000757 * design_vector.nose_length / fuselage_length
            - 0.057975 * cg[0] / fuselage_length
        )
    )
    cma = -0.021599 + wing_cla * wing_lever - tail_ratio * tail_cla * tail_lever + 0.059479 * body_scale
    cme = -tail_ratio * tail_cle * tail_lever
    trim_determinant = cla * cme - cle * cma
    if abs(trim_determinant) < 1e-10:
        return CruiseCondition(OperatingPoint(velocity=-1.0, alpha=-999.0), None, False,
                               trim_violation=float("inf"), trim_failure_reason="Singular lift/moment trim equations.")

    weight = mass * parameter_vector.gravity
    if mission == 3:
        # M3 mass is airplane-only; the sensor weight reaches it through the
        # tow line and therefore belongs in the required lift, not its inertia.
        weight += design_vector.sensor_weight_kg * parameter_vector.gravity
    thrust_a, thrust_b, thrust_c = thrust_velocity
    fuselage_geometry = fuselage_drag_geometry(design_vector)

    def state(velocity):
        q = 0.5 * parameter_vector.rho * velocity**2
        cl_required = weight / (q * design_vector.wing_area)
        lift_rhs = cl_required - cl0
        alpha_rad = (cme * lift_rhs + cle * cm0) / trim_determinant
        elevator_rad = (-cma * lift_rhs - cla * cm0) / trim_determinant
        wing_cl = wing_cl0 + wing_cla * alpha_rad
        tail_cl = tail_cla * alpha_rad + tail_cle * elevator_rad
        cd = sum(drag_coefficients(design_vector, parameter_vector, velocity, wing_cl, tail_cl, fuselage_geometry).values())
        drag = q * design_vector.wing_area * cd
        if mission == 3:
            tow_backward, _, _ = tow_line_force_components(
                design_vector, parameter_vector, velocity
            )
            drag += tow_backward
        thrust = thrust_a * velocity**2 + thrust_b * velocity + thrust_c
        return alpha_rad, elevator_rad, drag, thrust

    def drag_residual(velocity: float) -> float:
        _, _, drag, thrust = state(velocity)
        return drag - thrust

    # Bracket every crossing and prefer the valid solution nearest the former 18 m/s guess.
    velocity_grid = np.linspace(3.0, 50.0, 48)
    _, _, grid_drag, grid_thrust = state(velocity_grid)
    residuals = grid_drag - grid_thrust
    roots = []
    for left, right, f_left, f_right in zip(velocity_grid[:-1], velocity_grid[1:], residuals[:-1], residuals[1:]):
        if f_left == 0:
            roots.append(float(left))
        elif f_left * f_right < 0:
            roots.append(float(brentq(drag_residual, left, right)))

    if residuals[-1] == 0:
        roots.append(float(velocity_grid[-1]))

    candidates = []
    for velocity in roots:
        alpha_rad, elevator_rad, drag, thrust = state(velocity)
        alpha, elevator = np.degrees(alpha_rad), np.degrees(elevator_rad)
        if -4.0 <= alpha <= 15.0 and -20.0 <= elevator <= 20.0:
            candidates.append((abs(velocity - 18.0), velocity, alpha, elevator, abs(drag - thrust) / weight))
    if not candidates:
        def violation(velocity):
            a, e, drag, thrust = state(velocity)
            return _trim_violation(np.degrees(a), np.degrees(e), drag, thrust, weight)

        # Retain endpoints and invalid roots, then refine each sampled local
        # minimum. A coarse speed grid alone can hide a near-trim improvement.
        values = np.asarray(violation(velocity_grid), dtype=float)
        trials = [(float(value), float(v)) for value, v in zip(values, velocity_grid)]
        trials.extend((float(violation(v)), v) for v in roots)
        for i in range(1, len(velocity_grid) - 1):
            if values[i] <= values[i-1] and values[i] <= values[i+1] and np.isfinite(values[i]):
                result = minimize_scalar(violation, bounds=(velocity_grid[i-1], velocity_grid[i+1]),
                                         method="bounded", options={"xatol": 1e-5})
                if result.success:
                    trials.append((float(result.fun), float(result.x)))
        miss, velocity = min(trials)
        a, e, drag, thrust = state(velocity)
        alpha, elevator = float(np.degrees(a)), float(np.degrees(e))
        reason = (f"Cruise trim failed. Closest sampled/refined attempt: V={velocity:.2f} m/s, "
                  f"thrust-drag={float(thrust-drag):.2f} N, alpha={alpha:.2f} deg "
                  f"(allowed -4 to 15), elevator={elevator:.2f} deg (allowed -20 to 20).")
        return CruiseCondition(OperatingPoint(velocity=velocity, alpha=alpha), None, False,
                               elevator_deflection=elevator, trim_violation=miss,
                               trim_failure_reason=reason)

    _, velocity, alpha, elevator, residual = min(candidates)
    converged = residual <= 1e-2
    cl_max = 1.45 * wing_ar / (wing_ar + 2.0)
    stall_speed = np.sqrt(2 * weight / (parameter_vector.rho * design_vector.wing_area * cl_max))
    if debug:
        print(f"[aero] Fast trim finished in {perf_counter() - start:.4f} s (converged={converged}).", flush=True)
    return CruiseCondition(
        operating_point=OperatingPoint(velocity=velocity, alpha=alpha),
        stall_speed=float(stall_speed) if converged else None,
        converged=converged,
        elevator_deflection=float(elevator),
    )
