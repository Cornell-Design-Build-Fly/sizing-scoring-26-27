from dataclasses import replace

from src.vectors import DesignVector, ParameterVector

from src.aero.custom_classes import AeroOutput
from src.prop.main_prop import prop_main
from src.mech.main_mech import evaluate_mechanical_module
from src.aero.main_aero import aero_main


def main(
    dv: DesignVector,
    pv: ParameterVector,
) -> tuple[AeroOutput, AeroOutput, AeroOutput]:
    """Evaluate mechanics, propulsion, and aerodynamics for all missions."""

    mech_result = evaluate_mechanical_module(dv, parameter_vector=pv)

    # Mechanical packaging may widen the fuselage. Aero must analyze the final
    # cross-section rather than the starting dimensions in the design vector.
    aero_design = replace(
        dv,
        fuselage_width=mech_result.fuselage_width_m,
        fuselage_height=mech_result.fuselage_height_m,
    )

    # Prop section
    fit_m1 = prop_main(dv, pv, 1)
    fit_m2 = prop_main(dv, pv, 2)
    fit_m3 = prop_main(dv, pv, 3)

    outputs: list[AeroOutput] = []
    for mission, thrust_velocity in (
        ("M1", fit_m1),
        ("M2", fit_m2),
        ("M3", fit_m3),
    ):
        properties = mech_result.for_mission(mission)
        outputs.append(
            aero_main(
                design_vector=aero_design,
                thrust_velocity=thrust_velocity,
                cg=tuple(float(value) for value in properties.cg_m),
                inertia_matrix=properties.inertia_tensor_kg_m2,
                mass=properties.total_mass_kg,
            )
        )

    return outputs[0], outputs[1], outputs[2]


