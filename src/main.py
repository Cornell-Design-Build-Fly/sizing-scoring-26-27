from src.vectors import DesignVector, ParameterVector

from src.prop.main_prop import prop_main
from src.mech.main_mech import mech_main
from src.aero.main_aero import aero_main
from src.opt.score import total_score


def main(dv: DesignVector, pv: ParameterVector) -> tuple[float, list[float]]:
    """Runs a plane through all three missions and scores it"""
    
    # Mech section
    cg_m1, inertia_tensor_m1, weight_n_m1 = mech_main(dv, "M1")
    cg_m2, inertia_tensor_m2, weight_n_m2 = mech_main(dv, "M2")
    cg_m3, inertia_tensor_m3, weight_n_m3 = mech_main(dv, "M3")

    # Prop section
    fit_m1 = prop_main(dv, pv, 1)
    fit_m2 = prop_main(dv, pv, 2)
    fit_m3 = prop_main(dv, pv, 3)

    # Aero section
    output_m1 = aero_main(
        design_vector=dv,
        thrust_velocity=fit_m1,
        cg=cg_m1,
        inertia_matrix=inertia_tensor_m1,
        mass=weight_n_m1 / 9.81,
        sm=0.0,  # TODO - where is this used?
    )
    output_m2 = aero_main(
        design_vector=dv,
        thrust_velocity=fit_m2,
        cg=cg_m2,
        inertia_matrix=inertia_tensor_m2,
        mass=weight_n_m2 / 9.81,
        sm=0.0,  # TODO - where is this used?
    )
    output_m3 = aero_main(
        design_vector=dv,
        thrust_velocity=fit_m3,
        cg=cg_m3,
        inertia_matrix=inertia_tensor_m3,
        mass=weight_n_m3 / 9.81,
        sm=0.0,  # TODO - where is this used?
    )

    # return total_score(dv)


