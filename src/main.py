from dataclasses import replace

from src.aero.main_aero import aero_main
from src.mech.main_mech import evaluate_mechanical_module
from src.prop.main_prop import prop_main
from src.prop.continuous_prop_database import ContinuousPropDatabase
from src.vectors import DesignVector, ParameterVector
from src.opt.score import (
    DEFAULT_SCORING_REFERENCES,
    ScoringReferences,
    total_optimization_score,
    total_score,
)


def main(
    dv: DesignVector,
    pv: ParameterVector,
    disp_res: bool = False,
    round_payload: bool = True,
    prop_database: ContinuousPropDatabase | None = None,
    return_details: bool = False,
    continuous_lap_scoring: bool = False,
    scoring_references: ScoringReferences = DEFAULT_SCORING_REFERENCES,
) -> tuple[float, list[float]] | tuple[float, list[float], dict]:
    """Evaluate mechanics, propulsion, and aerodynamics for all missions.

    The mechanical module now has one discrete path. ``round_payload`` is kept
    to preserve the old scoring behavior for callers that pass fractional
    optimizer payload counts.
    """
    scoring_dv = dv
    if round_payload:
        scoring_dv = replace(
            dv,
            extra_shipping_containers=round(dv.extra_shipping_containers),
        )

    mech_result = evaluate_mechanical_module(
        scoring_dv,
        parameter_vector=pv,
        disp_res=disp_res,
    )

    # Use mech's resolved geometry downstream without changing the caller's
    # design vector or its starting-width input.
    resolved_dv = replace(
        scoring_dv,
        fuselage_width=mech_result.resolved_fuselage_width_m,
        fuselage_height=mech_result.resolved_fuselage_height_m,
    )

    # M1 run
    m1_thrust_curve, m1_flight_time_fit = prop_main(
        resolved_dv,
        pv,
        mission=1,
        prop_database=prop_database,
        disp_res=disp_res,
    )
    m1_properties = mech_result.for_mission("M1")
    aero_m1 = aero_main(
        design_vector=resolved_dv,
        parameter_vector=pv,
        thrust_velocity=m1_thrust_curve,
        flight_time_fit=m1_flight_time_fit,
        mission=1,
        cg=m1_properties.cg_m,
        inertia_matrix=m1_properties.inertia_tensor_kg_m2,
        mass=m1_properties.total_mass_kg,
        debug=False,
    )

    # M2 run
    m2_thrust_curve, m2_flight_time_fit = prop_main(
        resolved_dv,
        pv,
        mission=2,
        prop_database=prop_database,
        disp_res=disp_res,
    )
    m2_properties = mech_result.for_mission("M2")
    aero_m2 = aero_main(
        design_vector=resolved_dv,
        parameter_vector=pv,
        thrust_velocity=m2_thrust_curve,
        flight_time_fit=m2_flight_time_fit,
        mission=2,
        cg=m2_properties.cg_m,
        inertia_matrix=m2_properties.inertia_tensor_kg_m2,
        mass=m2_properties.total_mass_kg,
        debug=False,
    )

    # M3 run
    m3_thrust_curve, m3_flight_time_fit = prop_main(
        resolved_dv,
        pv,
        mission=3,
        prop_database=prop_database,
        disp_res=disp_res,
    )
    m3_properties = mech_result.for_mission("M3")
    aero_m3 = aero_main(
        design_vector=resolved_dv,
        parameter_vector=pv,
        thrust_velocity=m3_thrust_curve,
        flight_time_fit=m3_flight_time_fit,
        mission=3,
        cg=m3_properties.cg_m,
        inertia_matrix=m3_properties.inertia_tensor_kg_m2,
        mass=m3_properties.total_mass_kg,
        disp_res=disp_res,
        debug=False,
    )

    score_function = total_optimization_score if continuous_lap_scoring else total_score
    m2_payload_mass_kg = sum(
        item.mass_kg
        for item in m2_properties.items
        if item.category == "mission_2_payload"
    )
    tot_score, breakdown = score_function(
        resolved_dv,
        aero_m1.lap_time,
        aero_m2.lap_time,
        aero_m3.lap_time,
        m2_payload_mass_kg,
        scoring_references,
    )
    tot_penalty = (
        mech_result.penalty
        + aero_m1.penalty
        + aero_m2.penalty
        + aero_m3.penalty
    )
    max_takeoff_mass_kg = max(
        properties.total_mass_kg
        for properties in (m1_properties, m2_properties, m3_properties)
    )
    if max_takeoff_mass_kg >= 55.0 * 0.45359237:
        tot_penalty += 10.0
    result = (tot_score - tot_penalty, breakdown)
    if return_details:
        return (*result, {"M1": aero_m1, "M2": aero_m2, "M3": aero_m3})
    return result
