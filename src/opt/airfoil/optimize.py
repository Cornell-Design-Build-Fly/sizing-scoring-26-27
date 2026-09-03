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
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from src.aero.utils import require_scalar
from src.opt.airfoil.geometry import AirfoilKulfanParameters, make_airfoil_airplane
from src.vectors import ASBDesignVector, DesignVector


OPTIMIZED_AIRFOIL_FILENAME = "optimized_airfoil.dat"
REPORT_FILENAME = "airfoil_optimization_report.json"
OVERLAY_PLOT_FILENAME = "airfoil_overlay.png"
POLAR_PLOT_FILENAME = "airfoil_polars.png"
AIRCRAFT_POLAR_PLOT_FILENAME = "aircraft_polars.png"


@dataclass(frozen=True)
class AirfoilOptimizationConfig:
    """Settings for matching baseline aircraft lift and minimizing drag."""

    design_vector: DesignVector = field(default_factory=DesignVector)
    velocity_mps: float = 18.0
    baseline_alpha_deg: float = 6.0
    alpha_bounds_deg: tuple[float, float] = (-4.0, 15.0)
    model_size: str = "small"
    include_wave_drag: bool = False
    output_dir: Path = Path("data_dump") / "opt_airfoil"
    save_plots: bool = True
    save_dat: bool = True
    coefficient_delta_bound: float = 0.25
    leading_edge_delta_bound: float = 0.25
    te_thickness_bounds: tuple[float, float] = (0.0, 0.025)
    min_local_thickness: float = 0.002
    max_local_thickness: float = 0.22
    thickness_sample_points: int = 31
    preserve_baseline_thickness_fraction: float = 0.90
    preserve_thickness_x_bounds: tuple[float, float] = (0.05, 0.95)
    minimum_max_thickness_fraction: float = 1.0
    polar_alpha_bounds_deg: tuple[float, float] = (-6.0, 16.0)
    polar_alpha_points: int = 89
    verbose: bool = False


@dataclass(frozen=True)
class AirfoilOptimizationResult:
    """Serializable summary of a completed airfoil optimization."""

    baseline_airfoil: str
    optimized_airfoil: AirfoilKulfanParameters
    alpha_deg: float
    target_cl: float
    target_cm: float
    baseline_cd: float
    optimized_cd: float
    baseline_cm: float
    optimized_cm: float
    baseline_l_over_d: float
    optimized_l_over_d: float
    drag_reduction_percent: float
    reynolds_number: float
    mach: float
    baseline_max_thickness: float
    optimized_max_thickness: float
    runtime_seconds: float
    output_dir: Path
    active_bounds: dict[str, list[str]]


def _run_aerobuildup(
    airplane: asb.Airplane,
    *,
    velocity_mps,
    alpha_deg,
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


def _baseline_airplane(config: AirfoilOptimizationConfig) -> asb.Airplane:
    return ASBDesignVector.from_design_vector(config.design_vector).make_airplane(
        name="Baseline Design Vector Plane"
    )


def _baseline_aero(config: AirfoilOptimizationConfig) -> dict[str, Any]:
    return _run_aerobuildup(
        _baseline_airplane(config),
        velocity_mps=config.velocity_mps,
        alpha_deg=config.baseline_alpha_deg,
        model_size=config.model_size,
        include_wave_drag=config.include_wave_drag,
    )


def _timestamped_output_dir(base_dir: Path) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"run_{stamp}"


def _symbolic_scalar(value):
    shape = getattr(value, "shape", ())
    if shape in {(), (1,), (1, 1)}:
        try:
            return value[0]
        except (TypeError, IndexError):
            return value
    return value


def _make_symbolic_airfoil(
    opti: asb.Opti,
    baseline: asb.KulfanAirfoil,
    config: AirfoilOptimizationConfig,
) -> asb.KulfanAirfoil:
    lower_init = np.asarray(baseline.lower_weights, dtype=float).reshape(-1)
    upper_init = np.asarray(baseline.upper_weights, dtype=float).reshape(-1)
    lower = opti.variable(
        init_guess=lower_init,
        lower_bound=lower_init - config.coefficient_delta_bound,
        upper_bound=lower_init + config.coefficient_delta_bound,
        scale=0.1,
        category="Airfoil Kulfan Lower Weights",
    )
    upper = opti.variable(
        init_guess=upper_init,
        lower_bound=upper_init - config.coefficient_delta_bound,
        upper_bound=upper_init + config.coefficient_delta_bound,
        scale=0.1,
        category="Airfoil Kulfan Upper Weights",
    )
    leading_edge_weight = opti.variable(
        init_guess=float(baseline.leading_edge_weight),
        lower_bound=(
            float(baseline.leading_edge_weight) - config.leading_edge_delta_bound
        ),
        upper_bound=(
            float(baseline.leading_edge_weight) + config.leading_edge_delta_bound
        ),
        scale=0.1,
        category="Airfoil Leading Edge Weight",
    )
    te_thickness = opti.variable(
        init_guess=float(np.clip(baseline.TE_thickness, *config.te_thickness_bounds)),
        lower_bound=config.te_thickness_bounds[0],
        upper_bound=config.te_thickness_bounds[1],
        scale=max(config.te_thickness_bounds[1] - config.te_thickness_bounds[0], 1e-3),
        category="Airfoil TE Thickness",
    )

    return asb.KulfanAirfoil(
        name="optimized_airfoil",
        lower_weights=lower,
        upper_weights=upper,
        leading_edge_weight=leading_edge_weight,
        TE_thickness=te_thickness,
    )


def _add_airfoil_geometry_constraints(
    opti: asb.Opti,
    airfoil: asb.KulfanAirfoil,
    baseline_airfoil: asb.KulfanAirfoil,
    config: AirfoilOptimizationConfig,
) -> None:
    x = np.linspace(0.02, 0.98, config.thickness_sample_points)
    thickness = airfoil.local_thickness(x_over_c=x)
    opti.subject_to(thickness >= config.min_local_thickness)
    opti.subject_to(thickness <= config.max_local_thickness)

    preserve_x = np.linspace(
        config.preserve_thickness_x_bounds[0],
        config.preserve_thickness_x_bounds[1],
        config.thickness_sample_points,
    )
    baseline_thickness = baseline_airfoil.local_thickness(x_over_c=preserve_x)
    opti.subject_to(
        airfoil.local_thickness(x_over_c=preserve_x)
        >= config.preserve_baseline_thickness_fraction * baseline_thickness
    )

    max_search_x = np.linspace(0.02, 0.98, 201)
    baseline_max_thickness_x = float(
        max_search_x[
            int(np.argmax(baseline_airfoil.local_thickness(x_over_c=max_search_x)))
        ]
    )
    baseline_max_thickness = float(
        baseline_airfoil.local_thickness(x_over_c=baseline_max_thickness_x)
    )
    opti.subject_to(
        airfoil.local_thickness(x_over_c=baseline_max_thickness_x)
        >= config.minimum_max_thickness_fraction * baseline_max_thickness
    )


def _max_thickness(airfoil: asb.KulfanAirfoil) -> float:
    x = np.linspace(0.02, 0.98, 201)
    return float(np.max(airfoil.local_thickness(x_over_c=x)))


def _airfoil_parameters_from_solution(
    solution,
    airfoil: asb.KulfanAirfoil,
    config: AirfoilOptimizationConfig,
) -> AirfoilKulfanParameters:
    lower = np.asarray(solution.value(airfoil.lower_weights), dtype=float).reshape(-1)
    upper = np.asarray(solution.value(airfoil.upper_weights), dtype=float).reshape(-1)
    te_thickness = float(solution.value(airfoil.TE_thickness))
    te_thickness = float(np.clip(te_thickness, *config.te_thickness_bounds))
    return AirfoilKulfanParameters(
        name="optimized_airfoil",
        lower_weights=tuple(float(value) for value in lower),
        upper_weights=tuple(float(value) for value in upper),
        leading_edge_weight=float(solution.value(airfoil.leading_edge_weight)),
        TE_thickness=te_thickness,
    )


def _airfoil_bounds_report(
    baseline: asb.KulfanAirfoil,
    optimized: AirfoilKulfanParameters,
    config: AirfoilOptimizationConfig,
    *,
    tolerance: float = 5e-4,
) -> dict[str, list[str]]:
    report: dict[str, list[str]] = {
        "lower_weights": [],
        "upper_weights": [],
        "leading_edge_weight": [],
        "TE_thickness": [],
    }
    lower_base = np.asarray(baseline.lower_weights, dtype=float).reshape(-1)
    upper_base = np.asarray(baseline.upper_weights, dtype=float).reshape(-1)
    lower_opt = np.asarray(optimized.lower_weights, dtype=float)
    upper_opt = np.asarray(optimized.upper_weights, dtype=float)

    for name, base, opt in [
        ("lower_weights", lower_base, lower_opt),
        ("upper_weights", upper_base, upper_opt),
    ]:
        lower_bound = base - config.coefficient_delta_bound
        upper_bound = base + config.coefficient_delta_bound
        for index, value in enumerate(opt):
            if value <= lower_bound[index] + tolerance:
                report[name].append(f"{index}: lower")
            if value >= upper_bound[index] - tolerance:
                report[name].append(f"{index}: upper")

    le_lower = float(baseline.leading_edge_weight) - config.leading_edge_delta_bound
    le_upper = float(baseline.leading_edge_weight) + config.leading_edge_delta_bound
    if optimized.leading_edge_weight <= le_lower + tolerance:
        report["leading_edge_weight"].append("lower")
    if optimized.leading_edge_weight >= le_upper - tolerance:
        report["leading_edge_weight"].append("upper")

    if optimized.TE_thickness <= config.te_thickness_bounds[0] + tolerance:
        report["TE_thickness"].append("lower")
    if optimized.TE_thickness >= config.te_thickness_bounds[1] - tolerance:
        report["TE_thickness"].append("upper")

    return {key: value for key, value in report.items() if value}


def _floatify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {str(key): _floatify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_floatify(item) for item in value]
    if isinstance(value, np.ndarray):
        return _floatify(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, int):
        return value
    if hasattr(value, "__float__"):
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return value


def _analysis_operating_point(config: AirfoilOptimizationConfig) -> asb.OperatingPoint:
    return asb.OperatingPoint(
        velocity=config.velocity_mps,
        alpha=config.baseline_alpha_deg,
        beta=0.0,
        p=0.0,
        q=0.0,
        r=0.0,
    )


def _write_report(
    result: AirfoilOptimizationResult,
    config: AirfoilOptimizationConfig,
) -> Path:
    output_dir = result.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "result": {
            **asdict(result),
            "optimized_airfoil": result.optimized_airfoil.to_dict(),
        },
        "config": {
            **asdict(config),
            "design_vector": asdict(config.design_vector),
        },
        "assumptions": {
            "objective": "Minimize full-aircraft AeroBuildup CD while matching the baseline aircraft CL and Cm at the configured baseline alpha.",
            "baseline_alpha_deg": config.baseline_alpha_deg,
            "velocity_mps": config.velocity_mps,
            "airfoil_parameterization": "AeroSandbox 8-weight Kulfan/CST seeded from DesignVector.wing_airfoil.",
            "geometry_note": "Aircraft planform, reference quantities, tail, and fuselage are unchanged; only the main-wing airfoil changes.",
            "thickness_note": (
                "The optimized airfoil must keep a configurable fraction of "
                "the baseline thickness distribution and at least the baseline "
                "sampled maximum thickness."
            ),
        },
    }
    path = output_dir / REPORT_FILENAME
    path.write_text(json.dumps(_floatify(report), indent=2) + "\n", encoding="utf-8")
    return path


def _save_dat(result: AirfoilOptimizationResult) -> Path:
    path = result.output_dir / OPTIMIZED_AIRFOIL_FILENAME
    result.optimized_airfoil.to_airfoil().to_airfoil().write_dat(path)
    return path


def _save_airfoil_overlay(
    baseline: asb.KulfanAirfoil,
    optimized: asb.KulfanAirfoil,
    result: AirfoilOptimizationResult,
) -> Path:
    path = result.output_dir / OVERLAY_PLOT_FILENAME
    baseline_coordinates = baseline.to_airfoil().coordinates
    optimized_coordinates = optimized.to_airfoil().coordinates

    fig, ax = plt.subplots(figsize=(8.0, 3.2), dpi=180)
    ax.plot(
        baseline_coordinates[:, 0],
        baseline_coordinates[:, 1],
        label=f"Baseline ({result.baseline_airfoil})",
        linewidth=2.0,
    )
    ax.plot(
        optimized_coordinates[:, 0],
        optimized_coordinates[:, 1],
        label="Optimized (CUFoil)",
        linewidth=2.0,
    )
    ax.axhline(0.0, color="0.75", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("Airfoil Overlay")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _airfoil_polar(
    airfoil: asb.KulfanAirfoil,
    *,
    alpha_values: np.ndarray,
    reynolds_number: float,
    mach: float,
    model_size: str,
) -> dict[str, np.ndarray]:
    polar = airfoil.get_aero_from_neuralfoil(
        alpha=alpha_values,
        Re=reynolds_number,
        mach=mach,
        model_size=model_size,
    )
    return {
        "CL": np.asarray(polar["CL"], dtype=float).reshape(-1),
        "CD": np.asarray(polar["CD"], dtype=float).reshape(-1),
        "CM": np.asarray(polar["CM"], dtype=float).reshape(-1),
    }


def _save_polar_plot(
    baseline: asb.KulfanAirfoil,
    optimized: asb.KulfanAirfoil,
    result: AirfoilOptimizationResult,
    config: AirfoilOptimizationConfig,
) -> Path:
    path = result.output_dir / POLAR_PLOT_FILENAME
    alpha_values = np.linspace(
        config.polar_alpha_bounds_deg[0],
        config.polar_alpha_bounds_deg[1],
        config.polar_alpha_points,
    )
    baseline_polar = _airfoil_polar(
        baseline,
        alpha_values=alpha_values,
        reynolds_number=result.reynolds_number,
        mach=result.mach,
        model_size=config.model_size,
    )
    optimized_polar = _airfoil_polar(
        optimized,
        alpha_values=alpha_values,
        reynolds_number=result.reynolds_number,
        mach=result.mach,
        model_size=config.model_size,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), dpi=180)
    plot_specs = [
        ("CL", "section cl"),
        ("CD", "section cd"),
        ("CM", "section cm"),
        ("L/D", "section cl/cd"),
    ]
    for ax, (key, ylabel) in zip(axes.reshape(-1), plot_specs):
        if key == "L/D":
            baseline_values = baseline_polar["CL"] / baseline_polar["CD"]
            optimized_values = optimized_polar["CL"] / optimized_polar["CD"]
        else:
            baseline_values = baseline_polar[key]
            optimized_values = optimized_polar[key]
        ax.plot(
            alpha_values,
            baseline_values,
            label=f"Baseline ({result.baseline_airfoil})",
            linewidth=2.0,
        )
        ax.plot(alpha_values, optimized_values, label="Optimized (CUFoil)", linewidth=2.0)
        ax.axvline(
            config.baseline_alpha_deg,
            color="0.65",
            linestyle="--",
            linewidth=1.0,
        )
        ax.axvline(result.alpha_deg, color="0.25", linestyle=":", linewidth=1.0)
        ax.set_xlabel("alpha [deg]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if key == "CD":
            ax.set_ylim(bottom=0.0)
    axes[0, 0].legend()
    fig.suptitle(
        f"2D Section NeuralFoil Polars at Re={result.reynolds_number:.2e}, Mach={result.mach:.3f}",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _aircraft_polar(
    airplane: asb.Airplane,
    *,
    alpha_values: np.ndarray,
    velocity_mps: float,
    model_size: str,
    include_wave_drag: bool,
) -> dict[str, np.ndarray]:
    aero = _run_aerobuildup(
        airplane,
        velocity_mps=velocity_mps,
        alpha_deg=alpha_values,
        model_size=model_size,
        include_wave_drag=include_wave_drag,
    )
    return {
        "CL": np.asarray(aero["CL"], dtype=float).reshape(-1),
        "CD": np.asarray(aero["CD"], dtype=float).reshape(-1),
        "Cm": np.asarray(aero["Cm"], dtype=float).reshape(-1),
    }


def _save_aircraft_polar_plot(
    optimized: asb.KulfanAirfoil,
    result: AirfoilOptimizationResult,
    config: AirfoilOptimizationConfig,
) -> Path:
    path = result.output_dir / AIRCRAFT_POLAR_PLOT_FILENAME
    alpha_values = np.linspace(
        config.polar_alpha_bounds_deg[0],
        config.polar_alpha_bounds_deg[1],
        config.polar_alpha_points,
    )
    baseline_polar = _aircraft_polar(
        _baseline_airplane(config),
        alpha_values=alpha_values,
        velocity_mps=config.velocity_mps,
        model_size=config.model_size,
        include_wave_drag=config.include_wave_drag,
    )
    optimized_polar = _aircraft_polar(
        make_airfoil_airplane(
            config.design_vector,
            optimized,
            name="Optimized Airfoil Aircraft",
        ),
        alpha_values=alpha_values,
        velocity_mps=config.velocity_mps,
        model_size=config.model_size,
        include_wave_drag=config.include_wave_drag,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), dpi=180)
    plot_specs = [
        ("CL", "aircraft CL"),
        ("CD", "aircraft CD"),
        ("Cm", "aircraft Cm"),
        ("L/D", "aircraft CL/CD"),
    ]
    for ax, (key, ylabel) in zip(axes.reshape(-1), plot_specs):
        if key == "L/D":
            baseline_values = baseline_polar["CL"] / baseline_polar["CD"]
            optimized_values = optimized_polar["CL"] / optimized_polar["CD"]
            baseline_marker = result.baseline_l_over_d
            optimized_marker = result.optimized_l_over_d
        else:
            baseline_values = baseline_polar[key]
            optimized_values = optimized_polar[key]
            baseline_marker = (
                result.target_cl
                if key == "CL"
                else result.baseline_cd
                if key == "CD"
                else result.baseline_cm
            )
            optimized_marker = (
                result.target_cl
                if key == "CL"
                else result.optimized_cd
                if key == "CD"
                else result.optimized_cm
            )

        ax.plot(
            alpha_values,
            baseline_values,
            label=f"Baseline ({result.baseline_airfoil})",
            linewidth=2.0,
        )
        ax.plot(alpha_values, optimized_values, label="Optimized (CUFoil)", linewidth=2.0)
        ax.scatter(
            [config.baseline_alpha_deg],
            [baseline_marker],
            color="C0",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.scatter(
            [result.alpha_deg],
            [optimized_marker],
            color="C1",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.axvline(
            config.baseline_alpha_deg,
            color="0.65",
            linestyle="--",
            linewidth=1.0,
        )
        ax.axvline(result.alpha_deg, color="0.25", linestyle=":", linewidth=1.0)
        if key in {"CL", "Cm"}:
            ax.axhline(baseline_marker, color="0.80", linestyle="-", linewidth=0.8)
        ax.set_xlabel("alpha [deg]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if key == "CD":
            ax.set_ylim(bottom=0.0)
    axes[0, 0].legend()
    fig.suptitle("Full-Aircraft AeroBuildup Polars", y=0.995)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def run_airfoil_optimization(
    config: AirfoilOptimizationConfig | None = None,
) -> AirfoilOptimizationResult:
    """Run an ASB Opti problem that keeps baseline aircraft CL and minimizes drag."""

    config = config or AirfoilOptimizationConfig()
    output_dir = _timestamped_output_dir(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    baseline = _baseline_aero(config)
    target_cl = require_scalar(baseline["CL"])
    baseline_cd = require_scalar(baseline["CD"])
    baseline_cm = require_scalar(baseline["Cm"])
    baseline_airfoil = asb.KulfanAirfoil(config.design_vector.wing_airfoil)

    op_point = _analysis_operating_point(config)
    reynolds_number = require_scalar(op_point.reynolds(config.design_vector.wing_chord))
    mach = require_scalar(op_point.mach())

    opti = asb.Opti()
    alpha = opti.variable(
        init_guess=config.baseline_alpha_deg,
        lower_bound=config.alpha_bounds_deg[0],
        upper_bound=config.alpha_bounds_deg[1],
        scale=5.0,
    )
    candidate_airfoil = _make_symbolic_airfoil(opti, baseline_airfoil, config)
    _add_airfoil_geometry_constraints(
        opti,
        candidate_airfoil,
        baseline_airfoil,
        config,
    )

    airplane = make_airfoil_airplane(
        config.design_vector,
        candidate_airfoil,
        name="Symbolic Airfoil Candidate",
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
    cm = _symbolic_scalar(aero["Cm"])
    opti.subject_to(cl == target_cl)
    opti.subject_to(cm == baseline_cm)
    opti.minimize(cd)
    solution = opti.solve(verbose=config.verbose)

    solved_airfoil = _airfoil_parameters_from_solution(
        solution,
        candidate_airfoil,
        config,
    )
    solved_alpha = float(solution.value(alpha))
    solved_cd = float(solution.value(cd))
    solved_cm = float(solution.value(cm))
    active_bounds = _airfoil_bounds_report(baseline_airfoil, solved_airfoil, config)
    optimized_airfoil = solved_airfoil.to_airfoil()
    runtime_seconds = perf_counter() - start

    result = AirfoilOptimizationResult(
        baseline_airfoil=config.design_vector.wing_airfoil,
        optimized_airfoil=solved_airfoil,
        alpha_deg=solved_alpha,
        target_cl=target_cl,
        target_cm=baseline_cm,
        baseline_cd=baseline_cd,
        optimized_cd=solved_cd,
        baseline_cm=baseline_cm,
        optimized_cm=solved_cm,
        baseline_l_over_d=target_cl / baseline_cd,
        optimized_l_over_d=target_cl / solved_cd,
        drag_reduction_percent=100.0 * (baseline_cd - solved_cd) / baseline_cd,
        reynolds_number=reynolds_number,
        mach=mach,
        baseline_max_thickness=_max_thickness(baseline_airfoil),
        optimized_max_thickness=_max_thickness(optimized_airfoil),
        runtime_seconds=runtime_seconds,
        output_dir=output_dir,
        active_bounds=active_bounds,
    )

    report_path = _write_report(result, config)
    if config.save_dat:
        _save_dat(result)
    if config.save_plots:
        _save_airfoil_overlay(baseline_airfoil, optimized_airfoil, result)
        _save_polar_plot(baseline_airfoil, optimized_airfoil, result, config)
        _save_aircraft_polar_plot(optimized_airfoil, result, config)

    if config.verbose:
        print(f"Wrote report: {report_path}")
        print(
            "Optimized airfoil: "
            f"CD {baseline_cd:.5f} -> {solved_cd:.5f}, "
            f"drag reduction {result.drag_reduction_percent:.2f}%"
        )
        if active_bounds:
            print(f"Active bounds: {active_bounds}")

    return result


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive finite number.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize the design-vector main-wing airfoil."
    )
    parser.add_argument("--velocity", type=_positive_float, default=18.0, help=argparse.SUPPRESS)
    parser.add_argument("--baseline-alpha", type=float, default=6.0, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=Path("data_dump") / "opt_airfoil", help=argparse.SUPPRESS)
    parser.add_argument("--model-size", default="small", help=argparse.SUPPRESS)
    parser.add_argument("--show-solver", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-plots", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    result = run_airfoil_optimization(
        AirfoilOptimizationConfig(
            velocity_mps=args.velocity,
            baseline_alpha_deg=args.baseline_alpha,
            model_size=args.model_size,
            output_dir=args.output_dir,
            save_plots=not args.no_plots,
            verbose=args.show_solver,
        )
    )
    print(f"Output: {result.output_dir}")
    print(f"Target aircraft CL: {result.target_cl:.4f}")
    print(f"Target aircraft Cm: {result.target_cm:.5f}")
    print(f"Alpha: {result.alpha_deg:.3f} deg")
    print(f"CD: {result.baseline_cd:.5f} -> {result.optimized_cd:.5f}")
    print(f"Cm: {result.baseline_cm:.5f} -> {result.optimized_cm:.5f}")
    print(f"L/D: {result.baseline_l_over_d:.2f} -> {result.optimized_l_over_d:.2f}")
    print(f"Drag reduction: {result.drag_reduction_percent:.2f}%")
    print(
        "Max thickness: "
        f"{result.baseline_max_thickness:.3f} -> "
        f"{result.optimized_max_thickness:.3f}"
    )
    if result.active_bounds:
        print(f"Active bounds: {result.active_bounds}")
    dat_path = result.output_dir / OPTIMIZED_AIRFOIL_FILENAME
    if dat_path.exists():
        print(f"DAT: {dat_path}")
    overlay_path = result.output_dir / OVERLAY_PLOT_FILENAME
    if overlay_path.exists():
        print(f"Overlay plot: {overlay_path}")
    polar_path = result.output_dir / POLAR_PLOT_FILENAME
    if polar_path.exists():
        print(f"Section polar plot: {polar_path}")
    aircraft_polar_path = result.output_dir / AIRCRAFT_POLAR_PLOT_FILENAME
    if aircraft_polar_path.exists():
        print(f"Aircraft polar plot: {aircraft_polar_path}")


if __name__ == "__main__":
    main()
