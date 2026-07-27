from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import aerosandbox as asb

from src.vectors import ASBDesignVector, DesignVector
from src.aero.custom_classes import AirplaneAnalysisResult
from src.aero.utils import require_scalar

VLMAnalysisResult = AirplaneAnalysisResult


def run_vlm_on_design_vector(
    design_vector: DesignVector,
    *,
    velocity: float = 18.0,
    alpha: float = 6.0,
    beta: float = 0.0,
    p: float = 0.0,
    q: float = 0.0,
    r: float = 0.0,
    spanwise_resolution: int = 6,
    chordwise_resolution: int = 6,
    align_trailing_vortices_with_wind: bool = True,
    run_symmetric_if_possible: bool = False,
    verbose: bool = False,
    airplane_name: str = "Design Vector Plane",
) -> AirplaneAnalysisResult:
    """Builds an AeroSandbox airplane from a design vector and runs a VLM case."""
    asb_design_vector = ASBDesignVector.from_design_vector(design_vector)
    airplane = asb_design_vector.make_airplane(
        name=airplane_name
    )

    op_point = asb.OperatingPoint(
        velocity=velocity,
        alpha=alpha,
        beta=beta,
        p=p,
        q=q,
        r=r,
    )

    analysis = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=spanwise_resolution,
        chordwise_resolution=chordwise_resolution,
        align_trailing_vortices_with_wind=align_trailing_vortices_with_wind,
        run_symmetric_if_possible=run_symmetric_if_possible,
        verbose=verbose,
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
    )
