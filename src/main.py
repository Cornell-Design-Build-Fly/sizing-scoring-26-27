from dataclasses import replace
from time import perf_counter

from src.aero.main_aero import aero_main
from src.mech.main_mech import evaluate_mechanical_module
from src.prop.main_prop import prop_main
from src.vectors import DesignVector, ParameterVector
from src.opt.score import total_score


def main(
    dv: DesignVector,
    pv: ParameterVector,
    module_timings: dict[str, float] | None = None,
) -> tuple[float, list[float]]:
    """Evaluate mechanics, propulsion, and aerodynamics for all missions."""
    timings = module_timings if module_timings is not None else {}
    timings.update(
        mechanical=0.0,
        propulsion=0.0,
        aerodynamics=0.0,
        scoring=0.0,
    )

    start = perf_counter()
    mech_result = evaluate_mechanical_module(dv, parameter_vector=pv)
    timings["mechanical"] += perf_counter() - start

    # Use mech's resolved geometry downstream without changing the caller's
    # design vector or its starting-width input.
    resolved_dv = replace(
        dv,
        fuselage_width=mech_result.resolved_fuselage_width_m,
        fuselage_height=mech_result.resolved_fuselage_height_m,
    )

    # M1 run
    start = perf_counter()
    m1_thrust_curve, _ = prop_main(resolved_dv, pv, mission=1)
    timings["propulsion"] += perf_counter() - start
    m1_properties = mech_result.for_mission("M1")
    start = perf_counter()
    aero_m1 = aero_main(
        design_vector=resolved_dv,
        parameter_vector=pv,
        thrust_velocity=m1_thrust_curve,
        cg=m1_properties.cg_m,
        inertia_matrix=m1_properties.inertia_tensor_kg_m2,
        mass=m1_properties.total_mass_kg,
    )
    timings["aerodynamics"] += perf_counter() - start

    # M2 run
    start = perf_counter()
    m2_thrust_curve, _ = prop_main(resolved_dv, pv, mission=2)
    timings["propulsion"] += perf_counter() - start
    m2_properties = mech_result.for_mission("M2")
    start = perf_counter()
    aero_m2 = aero_main(
        design_vector=resolved_dv,
        parameter_vector=pv,
        thrust_velocity=m2_thrust_curve,
        cg=m2_properties.cg_m,
        inertia_matrix=m2_properties.inertia_tensor_kg_m2,
        mass=m2_properties.total_mass_kg,
    )
    timings["aerodynamics"] += perf_counter() - start

    # M3 run
    start = perf_counter()
    m3_thrust_curve, _ = prop_main(resolved_dv, pv, mission=3)
    timings["propulsion"] += perf_counter() - start
    m3_properties = mech_result.for_mission("M3")
    start = perf_counter()
    aero_m3 = aero_main(
        design_vector=resolved_dv,
        parameter_vector=pv,
        thrust_velocity=m3_thrust_curve,
        cg=m3_properties.cg_m,
        inertia_matrix=m3_properties.inertia_tensor_kg_m2,
        mass=m3_properties.total_mass_kg,
    )
    timings["aerodynamics"] += perf_counter() - start

    start = perf_counter()
    tot_score, breakdown = total_score(
        resolved_dv,
        aero_m1.lap_time,
        aero_m2.lap_time,
        aero_m3.lap_time,
    )
    tot_penalty = (
        mech_result.penalty
        + aero_m1.penalty
        + aero_m2.penalty
        + aero_m3.penalty
    )
    timings["scoring"] += perf_counter() - start
    return (tot_score - tot_penalty, breakdown)
