from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import aerosandbox as asb
import aerosandbox.numpy as asb_np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from src.aero.utils import require_scalar
from src.opt.winglets.geometry import (
    WINGLET_REGION_LIMIT_M,
    WingletGeometry,
    make_winglet_airplane,
)
from src.vectors import ASBDesignVector, DesignVector


SHOW_INTERACTIVE_AFTER_OPTIMIZATION = True
INTERACTIVE_VISUALIZATION_BACKEND = "plotly"


@dataclass(frozen=True)
class WingletOptimizationConfig:
    """Settings for matching baseline lift and minimizing drag."""

    design_vector: DesignVector = field(default_factory=DesignVector)
    velocity_mps: float = 18.0
    baseline_alpha_deg: float = 6.0
    alpha_bounds_deg: tuple[float, float] = (-4.0, 15.0)
    model_size: str = "small"
    include_wave_drag: bool = False
    winglet_airfoil: str = "naca0012"
    output_dir: Path = Path("data_dump") / "opt_winglets"
    save_visualization: bool = True
    show_interactive_after_optimization: bool = SHOW_INTERACTIVE_AFTER_OPTIMIZATION
    interactive_visualization_backend: str = INTERACTIVE_VISUALIZATION_BACKEND
    verbose: bool = False


@dataclass(frozen=True)
class WingletOptimizationResult:
    """Serializable summary of a completed winglet optimization."""

    winglet: WingletGeometry
    alpha_deg: float
    target_cl: float
    baseline_cd: float
    optimized_cd: float
    baseline_l_over_d: float
    optimized_l_over_d: float
    drag_reduction_percent: float
    runtime_seconds: float
    output_dir: Path


def _run_aerobuildup(
    airplane: asb.Airplane,
    *,
    velocity_mps: float,
    alpha_deg: float,
    model_size: str,
    include_wave_drag: bool,
) -> dict[str, Any]:
    op_point = asb.OperatingPoint(
        velocity=velocity_mps,
        alpha=alpha_deg,
        beta=0.0,
        p=0.0,
        q=0.0,
        r=0.0,
    )
    return asb.AeroBuildup(
        airplane=airplane,
        op_point=op_point,
        model_size=model_size,
        include_wave_drag=include_wave_drag,
    ).run()


def _baseline_airplane(config: WingletOptimizationConfig) -> asb.Airplane:
    return ASBDesignVector.from_design_vector(config.design_vector).make_airplane(
        name="Baseline Design Vector Plane"
    )


def _baseline_aero(config: WingletOptimizationConfig) -> dict[str, Any]:
    return _run_aerobuildup(
        _baseline_airplane(config),
        velocity_mps=config.velocity_mps,
        alpha_deg=config.baseline_alpha_deg,
        model_size=config.model_size,
        include_wave_drag=config.include_wave_drag,
    )


def _symbolic_scalar(value):
    shape = getattr(value, "shape", ())
    if shape in {(), (1,), (1, 1)}:
        try:
            return value[0]
        except (TypeError, IndexError):
            return value
    return value


def _make_symbolic_winglet(opti: asb.Opti) -> WingletGeometry:
    variables = {}
    for name, (lower, upper) in zip(WingletGeometry.opt_names(), WingletGeometry.bounds()):
        init_guess = 0.5 * (lower + upper)
        scale = max(abs(upper - lower), 1.0)
        variables[name] = opti.variable(
            init_guess=init_guess,
            lower_bound=lower,
            upper_bound=upper,
            scale=scale,
        )
    return WingletGeometry(**variables)


def _add_winglet_geometry_constraints(
    opti: asb.Opti,
    winglet: WingletGeometry,
    design_vector: DesignVector,
) -> None:
    span_semilength = design_vector.wing_span / 2.0
    cant_rad = winglet.cant_angle_deg * np.pi / 180.0
    tip_y = span_semilength - winglet.tip_inset_m
    tip_handle = 0.50 * winglet.blend_tension * asb_np.minimum(
        winglet.blend_length_m,
        winglet.height_m,
    )
    p2_y = tip_y - tip_handle * asb_np.cos(cant_rad)
    p2_z = winglet.height_m - tip_handle * asb_np.sin(cant_rad)

    opti.subject_to(winglet.blend_length_m <= WINGLET_REGION_LIMIT_M)
    opti.subject_to(winglet.height_m <= 1.50 * winglet.blend_length_m)
    opti.subject_to(winglet.tip_inset_m <= WINGLET_REGION_LIMIT_M)
    opti.subject_to(p2_y <= span_semilength)
    opti.subject_to(p2_y >= span_semilength - WINGLET_REGION_LIMIT_M)
    opti.subject_to(p2_z >= 0.0)


def _timestamped_output_dir(base_dir: Path) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"run_{stamp}"


def _floatify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _floatify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_floatify(item) for item in value]
    if isinstance(value, np.ndarray):
        return _floatify(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__float__"):
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return value


def _write_report(
    result: WingletOptimizationResult,
    config: WingletOptimizationConfig,
) -> Path:
    output_dir = result.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "result": {
            **asdict(result),
            "winglet": result.winglet.to_dict(),
        },
        "config": {
            **asdict(config),
            "design_vector": asdict(config.design_vector),
        },
        "assumptions": {
            "projected_wingspan_fixed_m": config.design_vector.wing_span,
            "winglet_region_limit_m": WINGLET_REGION_LIMIT_M,
            "objective": "Minimize AeroBuildup CD while matching the baseline CL at the configured baseline alpha.",
            "note": (
                "The winglet is a monotone-tapered blended curve. Its tip and blend "
                "control point stay inside the original wingtip station, so projected "
                "wingspan does not increase."
            ),
        },
    }
    path = output_dir / "winglet_optimization_report.json"
    path.write_text(json.dumps(_floatify(report), indent=2) + "\n", encoding="utf-8")
    return path


def _save_visualization(
    result: WingletOptimizationResult,
    config: WingletOptimizationConfig,
) -> Path:
    airplane = make_winglet_airplane(
        config.design_vector,
        result.winglet,
        name="Optimized Winglet Candidate",
        winglet_airfoil=config.winglet_airfoil,
    )
    path = result.output_dir / "optimized_winglet_geometry.png"
    airplane.draw_three_view(style="shaded", show=False)
    plt.gcf().savefig(path, dpi=200, bbox_inches="tight")
    plt.close(plt.gcf())
    return path


def _show_interactive_visualization(
    result: WingletOptimizationResult,
    config: WingletOptimizationConfig,
) -> None:
    airplane = make_winglet_airplane(
        config.design_vector,
        result.winglet,
        name="Optimized Winglet Candidate",
        winglet_airfoil=config.winglet_airfoil,
    )
    backend = config.interactive_visualization_backend
    if backend == "three_view":
        airplane.draw_three_view(style="shaded", show=False)
        plt.show()
    else:
        airplane.draw(backend=backend, show=True)


def run_winglet_optimization(
    config: WingletOptimizationConfig | None = None,
) -> WingletOptimizationResult:
    """Run an ASB Opti problem that keeps baseline CL and minimizes drag."""

    config = config or WingletOptimizationConfig()
    output_dir = _timestamped_output_dir(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    baseline = _baseline_aero(config)
    target_cl = require_scalar(baseline["CL"])
    baseline_cd = require_scalar(baseline["CD"])

    opti = asb.Opti()
    alpha = opti.variable(
        init_guess=config.baseline_alpha_deg,
        lower_bound=config.alpha_bounds_deg[0],
        upper_bound=config.alpha_bounds_deg[1],
        scale=5.0,
    )
    winglet = _make_symbolic_winglet(opti)
    _add_winglet_geometry_constraints(opti, winglet, config.design_vector)

    airplane = make_winglet_airplane(
        config.design_vector,
        winglet,
        name="Symbolic Winglet Candidate",
        winglet_airfoil=config.winglet_airfoil,
    )
    aero = _run_aerobuildup(
        airplane,
        velocity_mps=config.velocity_mps,
        alpha_deg=alpha,
        model_size=config.model_size,
        include_wave_drag=config.include_wave_drag,
    )

    cl = _symbolic_scalar(aero["CL"])
    cd = _symbolic_scalar(aero["CD"])
    opti.subject_to(cl == target_cl)
    opti.minimize(cd)
    solution = opti.solve(verbose=config.verbose)

    solved_winglet = WingletGeometry(
        **{
            name: float(solution.value(getattr(winglet, name)))
            for name in WingletGeometry.opt_names()
        }
    )
    solved_alpha = float(solution.value(alpha))
    solved_cd = float(solution.value(cd))
    runtime_seconds = perf_counter() - start

    result = WingletOptimizationResult(
        winglet=solved_winglet,
        alpha_deg=solved_alpha,
        target_cl=target_cl,
        baseline_cd=baseline_cd,
        optimized_cd=solved_cd,
        baseline_l_over_d=target_cl / baseline_cd,
        optimized_l_over_d=target_cl / solved_cd,
        drag_reduction_percent=100.0 * (baseline_cd - solved_cd) / baseline_cd,
        runtime_seconds=runtime_seconds,
        output_dir=output_dir,
    )

    report_path = _write_report(result, config)
    if config.save_visualization:
        _save_visualization(result, config)
    if config.show_interactive_after_optimization:
        _show_interactive_visualization(result, config)

    if config.verbose:
        print(f"Wrote report: {report_path}")
        print(
            "Optimized winglet: "
            f"CD {baseline_cd:.5f} -> {solved_cd:.5f}, "
            f"drag reduction {result.drag_reduction_percent:.2f}%"
        )

    return result


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive finite number.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize symmetric winglets on the repo design-vector aircraft."
    )
    parser.add_argument("--velocity", type=_positive_float, default=18.0, help=argparse.SUPPRESS)
    parser.add_argument("--baseline-alpha", type=float, default=6.0, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=Path("data_dump") / "opt_winglets", help=argparse.SUPPRESS)
    parser.add_argument("--model-size", default="small", help=argparse.SUPPRESS)
    parser.add_argument("--show-solver", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-visualization", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    result = run_winglet_optimization(
        WingletOptimizationConfig(
            velocity_mps=args.velocity,
            baseline_alpha_deg=args.baseline_alpha,
            model_size=args.model_size,
            output_dir=args.output_dir,
            save_visualization=not args.no_visualization,
            verbose=args.show_solver,
        )
    )
    print(f"Output: {result.output_dir}")
    print(f"Target CL: {result.target_cl:.4f}")
    print(f"Alpha: {result.alpha_deg:.3f} deg")
    print(f"CD: {result.baseline_cd:.5f} -> {result.optimized_cd:.5f}")
    print(f"L/D: {result.baseline_l_over_d:.2f} -> {result.optimized_l_over_d:.2f}")
    print(f"Drag reduction: {result.drag_reduction_percent:.2f}%")
    print(f"Winglet: {result.winglet.to_dict()}")


if __name__ == "__main__":
    main()
