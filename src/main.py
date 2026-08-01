from dataclasses import replace

from src.aero.main_aero import aero_main
from src.mech.main_mech import evaluate_mechanical_module
from src.prop.main_prop import prop_main
from src.vectors import DesignVector, ParameterVector
from src.opt.score import total_score


def main(
    dv: DesignVector,
    pv: ParameterVector,
    disp_res: bool = False,
) -> tuple[float, list[float]]:
    """Evaluate all missions and optionally save mechanical placement results."""
    mech_result = evaluate_mechanical_module(
        dv,
        parameter_vector=pv,
        disp_res=disp_res,
    )

    # Use mech's resolved geometry downstream without changing the caller's
    # design vector or its starting-width input.
    resolved_dv = replace(
        dv,
        fuselage_width=mech_result.resolved_fuselage_width_m,
        fuselage_height=mech_result.resolved_fuselage_height_m,
    )

    # M1 run
    m1_thrust_curve, _ = prop_main(resolved_dv, pv, mission=1)
    m1_properties = mech_result.for_mission("M1")
    aero_m1 = aero_main(
        design_vector=resolved_dv,
        parameter_vector=pv,
        thrust_velocity=m1_thrust_curve,
        mission=1,
        cg=m1_properties.cg_m,
        inertia_matrix=m1_properties.inertia_tensor_kg_m2,
        mass=m1_properties.total_mass_kg,
    )

    # M2 run
    m2_thrust_curve, _ = prop_main(resolved_dv, pv, mission=2)
    m2_properties = mech_result.for_mission("M2")
    aero_m2 = aero_main(
        design_vector=resolved_dv,
        parameter_vector=pv,
        thrust_velocity=m2_thrust_curve,
        mission=2,
        cg=m2_properties.cg_m,
        inertia_matrix=m2_properties.inertia_tensor_kg_m2,
        mass=m2_properties.total_mass_kg,
    )

    # M3 run
    m3_thrust_curve, _ = prop_main(resolved_dv, pv, mission=3)
    m3_properties = mech_result.for_mission("M3")
    aero_m3 = aero_main(
        design_vector=resolved_dv,
        parameter_vector=pv,
        thrust_velocity=m3_thrust_curve,
        mission=3,
        cg=m3_properties.cg_m,
        inertia_matrix=m3_properties.inertia_tensor_kg_m2,
        mass=m3_properties.total_mass_kg,
        disp_res=False,
    )

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
    return (tot_score - tot_penalty, breakdown)
