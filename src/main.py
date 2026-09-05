from dataclasses import replace

from src.aero.main_aero import aero_main
from src.mech.main_mech import evaluate_mechanical_module
from src.mech.mass_properties import inertia_tensor_about_cg
from src.prop.main_prop import prop_main
from src.prop.mission_performance import (
    DEFAULT_PROPULSION_REQUIREMENTS,
    PROPULSION_INFEASIBLE_BASE_PENALTY,
    PropulsionRequirements,
    evaluate_mission_propulsion,
    propulsion_margin_bonus,
)
from src.prop.continuous_prop_database import (
    ContinuousPropDatabase,
    load_default_continuous_prop_database,
)
from src.vectors import DesignVector, ParameterVector
from src.opt.score import (
    DEFAULT_SCORING_REFERENCES,
    ScoringReferences,
    total_optimization_score,
    total_score,
)
from src.tow.surrogate import (
    DEFAULT_M3_DOWNWARD_LOAD_SURROGATE,
    DownwardLoadSurrogate,
)


POUNDS_TO_KG = 0.45359237
# AMA / rules 3.2.1: TOGW with payload must be under 55 lb.
MAX_TAKEOFF_MASS_KG = 55.0 * POUNDS_TO_KG
OVERWEIGHT_BASE_PENALTY = 10.0
# Extra penalty per kilogram over the limit. A flat step left ~80% of the
# sampled design space on a single plateau of exactly -10, so differential
# evolution saw mostly ties and had no direction to descend. The base step is
# kept because the limit is a hard legality cliff, not a soft preference.
OVERWEIGHT_PENALTY_PER_KG = 0.5


def overweight_penalty(max_takeoff_mass_kg: float) -> float:
    """Return the takeoff-weight penalty, graded above the 55 lb limit."""

    overshoot_kg = max_takeoff_mass_kg - MAX_TAKEOFF_MASS_KG
    if overshoot_kg < 0.0:
        return 0.0
    return OVERWEIGHT_BASE_PENALTY + OVERWEIGHT_PENALTY_PER_KG * overshoot_kg


def resolved_aerodynamic_design_vector(
    design_vector: DesignVector,
    mech_result,
) -> DesignVector:
    """Expand the aerodynamic fuselage to contain the mechanical assembly."""

    fuselage = next(
        item for item in mech_result.all_items if item.name == "Fuselage structure"
    )
    forward_edge_x_m = fuselage.position_m[0] - 0.5 * fuselage.dimensions_m[0]
    back_edge_x_m = fuselage.position_m[0] + 0.5 * fuselage.dimensions_m[0]
    return replace(
        design_vector,
        nose_length=max(float(design_vector.nose_length), -float(forward_edge_x_m)),
        fuselage_width=mech_result.resolved_fuselage_width_m,
        fuselage_height=mech_result.resolved_fuselage_height_m,
        fuselage_box_back_x_m=max(
            float(design_vector.wing_chord),
            float(back_edge_x_m),
        ),
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
    m3_tow_load_surrogate: DownwardLoadSurrogate | None = (
        DEFAULT_M3_DOWNWARD_LOAD_SURROGATE
    ),
    propulsion_requirements: PropulsionRequirements = (
        DEFAULT_PROPULSION_REQUIREMENTS
    ),
) -> tuple[float, list[float]] | tuple[float, list[float], dict]:
    """Evaluate mechanics, propulsion, and aerodynamics for all missions.

    The mechanical module now has one discrete path. ``round_payload`` is kept
    to preserve the old scoring behavior for callers that pass fractional
    optimizer payload counts.
    """
    scoring_dv = dv
    if prop_database is None:
        prop_database = load_default_continuous_prop_database()
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
    resolved_dv = resolved_aerodynamic_design_vector(scoring_dv, mech_result)

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
    m3_aero_mass_kg = m3_properties.total_mass_kg
    m3_supported_mass_kg = m3_properties.total_mass_kg
    m3_aero_cg = m3_properties.cg_m
    m3_aero_inertia = m3_properties.inertia_tensor_kg_m2
    representative_tow_down_lbf = None
    predicted_peak_tow_down_lbf = None
    if m3_tow_load_surrogate is not None:
        aircraft_only_items = tuple(
            item
            for item in m3_properties.items
            if item.category != "mission_3_payload"
        )
        aircraft_only_cg, aircraft_only_inertia = inertia_tensor_about_cg(
            aircraft_only_items
        )
        m3_aero_mass_kg = float(
            sum(item.mass_kg for item in aircraft_only_items)
        )
        sensor_mass_kg = float(resolved_dv.mission3_sensor_weight_kg)
        sensor_weight_lbf = sensor_mass_kg / POUNDS_TO_KG
        predicted_peak_tow_down_lbf = (
            m3_tow_load_surrogate.peak_downward_force_lbf(sensor_weight_lbf)
        )
        representative_tow_down_lbf = (
            m3_tow_load_surrogate.representative_downward_force_lbf(
                sensor_weight_lbf
            )
        )
        m3_supported_mass_kg = (
            m3_aero_mass_kg
            + m3_tow_load_surrogate.equivalent_supported_mass_kg(sensor_mass_kg)
        )
        m3_aero_cg = tuple(float(value) for value in aircraft_only_cg)
        m3_aero_inertia = aircraft_only_inertia
    aero_m3 = aero_main(
        design_vector=resolved_dv,
        parameter_vector=pv,
        thrust_velocity=m3_thrust_curve,
        flight_time_fit=m3_flight_time_fit,
        mission=3,
        cg=m3_aero_cg,
        inertia_matrix=m3_aero_inertia,
        mass=m3_aero_mass_kg,
        supported_mass=m3_supported_mass_kg,
        disp_res=disp_res,
        debug=False,
    )

    propulsion_result_list = []
    for mission, mass, supported_mass, aero in (
        (
            2,
            m2_properties.total_mass_kg,
            m2_properties.total_mass_kg,
            aero_m2,
        ),
        (3, m3_properties.total_mass_kg, m3_supported_mass_kg, aero_m3),
        (
            1,
            m1_properties.total_mass_kg,
            m1_properties.total_mass_kg,
            aero_m1,
        ),
    ):
        if aero.cruise_speed_mps is None or aero.stall_speed_mps is None:
            continue
        performance = evaluate_mission_propulsion(
            resolved_dv,
            pv,
            mission=mission,
            mass_kg=mass,
            supported_mass_kg=supported_mass,
            cruise_speed_mps=float(aero.cruise_speed_mps),
            stall_speed_mps=float(aero.stall_speed_mps),
            lap_time_s=float(aero.lap_time),
            prop_database=prop_database,
            requirements=propulsion_requirements,
        )
        propulsion_result_list.append(performance)
        # M2 is the prerequisite flight and M3 is the long energy case. There
        # is no value spending time evaluating lighter missions until each
        # prerequisite is propulsion-feasible.
        if not return_details and mission in (2, 3) and not performance.feasible:
            break
    propulsion_results = tuple(propulsion_result_list)
    propulsion_by_mission = {
        performance.mission: performance for performance in propulsion_results
    }
    # Course time is propulsion-aware: a turn can be lift-, structure-, or
    # thrust-limited.  Replace the preliminary aero-only corner-speed estimate
    # before competition scoring and reporting.
    if 1 in propulsion_by_mission:
        aero_m1 = replace(
            aero_m1,
            lap_time=propulsion_by_mission[1].modeled_lap_time_s,
        )
    if 2 in propulsion_by_mission:
        aero_m2 = replace(
            aero_m2,
            lap_time=propulsion_by_mission[2].modeled_lap_time_s,
        )
    if 3 in propulsion_by_mission:
        aero_m3 = replace(
            aero_m3,
            lap_time=propulsion_by_mission[3].modeled_lap_time_s,
        )
    propulsion_feasible = (
        {performance.mission for performance in propulsion_results} == {1, 2, 3}
        and all(performance.feasible for performance in propulsion_results)
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
        + aero_m2.penalty
        + aero_m3.penalty
    )
    propulsion_penalty = max(
        (performance.penalty for performance in propulsion_results),
        default=0.0,
    )
    if not propulsion_feasible:
        propulsion_penalty = max(
            propulsion_penalty,
            PROPULSION_INFEASIBLE_BASE_PENALTY,
        )
    tot_penalty += propulsion_penalty
    if continuous_lap_scoring:
        tot_score += propulsion_margin_bonus(
            propulsion_results,
            propulsion_requirements,
        )
    max_takeoff_mass_kg = max(
        properties.total_mass_kg
        for properties in (m1_properties, m2_properties, m3_properties)
    )
    tot_penalty += overweight_penalty(max_takeoff_mass_kg)
    result = (tot_score - tot_penalty, breakdown)
    if return_details:
        return (
            *result,
            {
                "M1": aero_m1,
                "M2": aero_m2,
                "M3": aero_m3,
                "penalty_total": tot_penalty,
                "penalty_mechanical": mech_result.penalty,
                "penalty_aero_m2": aero_m2.penalty,
                "penalty_aero_m3": aero_m3.penalty,
                "penalty_overweight": overweight_penalty(max_takeoff_mass_kg),
                "penalty_propulsion": propulsion_penalty,
                "propulsion_feasible": propulsion_feasible,
                "propulsion": {
                    f"M{performance.mission}": performance.to_dict()
                    for performance in propulsion_results
                },
                "max_takeoff_mass_kg": max_takeoff_mass_kg,
                "m3_tow_predicted_peak_down_lbf": predicted_peak_tow_down_lbf,
                "m3_tow_representative_down_lbf": representative_tow_down_lbf,
                "m3_aircraft_only_mass_kg": m3_aero_mass_kg,
                "m3_supported_mass_kg": m3_supported_mass_kg,
            },
        )
    return result
