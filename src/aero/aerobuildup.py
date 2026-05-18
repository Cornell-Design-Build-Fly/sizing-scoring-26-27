from __future__ import annotations

from time import perf_counter

import aerosandbox as asb

from src.aero.vlm import AirplaneAnalysisResult, require_scalar
from src.design_vector import ASBDesignVector, DesignVector


def run_aerobuildup_on_design_vector(
    design_vector: DesignVector,
    *,
    velocity: float = 18.0,
    alpha: float = 6.0,
    beta: float = 0.0,
    p: float = 0.0,
    q: float = 0.0,
    r: float = 0.0,
    model_size: str = "small",
    include_wave_drag: bool = True,
    airplane_name: str = "Design Vector Plane",
) -> AirplaneAnalysisResult:
    """Builds an AeroSandbox airplane from a design vector and runs AeroBuildup."""
    asb_design_vector = ASBDesignVector.from_design_vector(design_vector)
    airplane, _, _, _ = asb_design_vector.make_airplane(name=airplane_name)

    op_point = asb.OperatingPoint(
        velocity=velocity,
        alpha=alpha,
        beta=beta,
        p=p,
        q=q,
        r=r,
    )

    analysis = asb.AeroBuildup(
        airplane=airplane,
        op_point=op_point,
        model_size=model_size,
        include_wave_drag=include_wave_drag,
    )

    start_time = perf_counter()
    aero = analysis.run()
    runtime_seconds = perf_counter() - start_time

    return AirplaneAnalysisResult(
        CL=require_scalar(aero["CL"]),
        CD=require_scalar(aero["CD"]),
        CY=require_scalar(aero["CY"]),
        Cl=require_scalar(aero["Cl"]),
        Cm=require_scalar(aero["Cm"]),
        Cn=require_scalar(aero["Cn"]),
        L=require_scalar(aero["L"]),
        D=require_scalar(aero["D"]),
        Y=require_scalar(aero["Y"]),
        l_b=require_scalar(aero["l_b"]),
        m_b=require_scalar(aero["m_b"]),
        n_b=require_scalar(aero["n_b"]),
        runtime_seconds=runtime_seconds,
        converged=True,
        D_induced=require_scalar(aero["D_induced"]) if "D_induced" in aero else None,
        D_profile=require_scalar(aero["D_profile"]) if "D_profile" in aero else None,
    )
