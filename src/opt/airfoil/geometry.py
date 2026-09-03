from __future__ import annotations

from dataclasses import dataclass

import aerosandbox as asb
import numpy as np

from src.vectors import ASBDesignVector, DesignVector


@dataclass(frozen=True)
class AirfoilKulfanParameters:
    """Serializable Kulfan/CST airfoil parameters."""

    lower_weights: tuple[float, ...]
    upper_weights: tuple[float, ...]
    leading_edge_weight: float
    TE_thickness: float
    name: str = "optimized_airfoil"

    @classmethod
    def from_airfoil(
        cls,
        airfoil: asb.Airfoil | asb.KulfanAirfoil,
    ) -> "AirfoilKulfanParameters":
        kulfan = (
            airfoil
            if isinstance(airfoil, asb.KulfanAirfoil)
            else airfoil.to_kulfan_airfoil()
        )
        return cls(
            lower_weights=tuple(
                float(value) for value in np.asarray(kulfan.lower_weights).reshape(-1)
            ),
            upper_weights=tuple(
                float(value) for value in np.asarray(kulfan.upper_weights).reshape(-1)
            ),
            leading_edge_weight=float(kulfan.leading_edge_weight),
            TE_thickness=float(kulfan.TE_thickness),
            name=kulfan.name,
        )

    def to_airfoil(self) -> asb.KulfanAirfoil:
        return asb.KulfanAirfoil(
            name=self.name,
            lower_weights=np.asarray(self.lower_weights, dtype=float),
            upper_weights=np.asarray(self.upper_weights, dtype=float),
            leading_edge_weight=float(self.leading_edge_weight),
            TE_thickness=float(self.TE_thickness),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "lower_weights": list(self.lower_weights),
            "upper_weights": list(self.upper_weights),
            "leading_edge_weight": self.leading_edge_weight,
            "TE_thickness": self.TE_thickness,
        }


def make_airfoil_airplane(
    design_vector: DesignVector,
    airfoil: asb.Airfoil | asb.KulfanAirfoil | AirfoilKulfanParameters,
    *,
    name: str = "Airfoil Candidate",
) -> asb.Airplane:
    """Build the standard design-vector airplane with a custom main-wing airfoil."""

    if isinstance(airfoil, AirfoilKulfanParameters):
        airfoil_obj = airfoil.to_airfoil()
    else:
        airfoil_obj = airfoil

    base_airplane = ASBDesignVector.from_design_vector(design_vector).make_airplane(
        name=name
    )
    replacement_wings = []
    for wing in base_airplane.wings:
        if wing.name != "Main Wing":
            replacement_wings.append(wing)
            continue

        replacement_wings.append(
            asb.Wing(
                name="Main Wing",
                symmetric=wing.symmetric,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=xsec.xyz_le,
                        chord=xsec.chord,
                        twist=xsec.twist,
                        airfoil=airfoil_obj,
                        control_surfaces=xsec.control_surfaces,
                    )
                    for xsec in wing.xsecs
                ],
            )
        )

    return asb.Airplane(
        name=name,
        xyz_ref=base_airplane.xyz_ref,
        wings=replacement_wings,
        fuselages=base_airplane.fuselages,
        s_ref=base_airplane.s_ref,
        c_ref=base_airplane.c_ref,
        b_ref=base_airplane.b_ref,
    )
