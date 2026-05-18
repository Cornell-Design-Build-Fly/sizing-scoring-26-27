from __future__ import annotations

from time import perf_counter

import aerosandbox as asb

from src.aero.vlm import AirplaneAnalysisResult
from src.design_vector import ASBDesignVector, DesignVector


def run_nonlinear_lifting_line_on_design_vector(
    design_vector: DesignVector,
    *,
    velocity: float = 18.0,
    alpha: float = 6.0,
    beta: float = 0.0,
    p: float = 0.0,
    q: float = 0.0,
    r: float = 0.0,
    spanwise_resolution: int = 8,
    align_trailing_vortices_with_wind: bool = True,
    verbose: bool = False,
    airplane_name: str = "Design Vector Plane",
    raise_on_failure: bool = False,
) -> AirplaneAnalysisResult:
    """Builds an AeroSandbox airplane from a design vector and runs NonlinearLiftingLine."""
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

    analysis = asb.NonlinearLiftingLine(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=spanwise_resolution,
        align_trailing_vortices_with_wind=align_trailing_vortices_with_wind,
        verbose=verbose,
    )

    start_time = perf_counter()
    try:
        aero = analysis.run()
        runtime_seconds = perf_counter() - start_time
    except Exception:
        runtime_seconds = perf_counter() - start_time
        if raise_on_failure:
            raise
        nan = float("nan")
        return AirplaneAnalysisResult(
            CL=nan,
            CD=nan,
            CY=nan,
            Cl=nan,
            Cm=nan,
            Cn=nan,
            L=nan,
            D=nan,
            Y=nan,
            l_b=nan,
            m_b=nan,
            n_b=nan,
            runtime_seconds=runtime_seconds,
            converged=False,
        )

    return AirplaneAnalysisResult(
        CL=float(aero["CL"]),
        CD=float(aero["CD"]),
        CY=float(aero["CY"]),
        Cl=float(aero["Cl"]),
        Cm=float(aero["Cm"]),
        Cn=float(aero["Cn"]),
        L=float(aero["L"]),
        D=float(aero["D"]),
        Y=float(aero["Y"]),
        l_b=float(aero["l_b"]),
        m_b=float(aero["m_b"]),
        n_b=float(aero["n_b"]),
        runtime_seconds=runtime_seconds,
        converged=True,
    )
