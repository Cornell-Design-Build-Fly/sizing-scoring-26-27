from __future__ import annotations

import csv
import inspect
import json
import math
import os
import platform
import time
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy
from scipy.optimize import NonlinearConstraint, differential_evolution
from tqdm.auto import tqdm

from src.main import main as score_aircraft
from src.mech.main_mech import evaluate_mechanical_module
from src.opt.view_results import (
    plot_final_population,
    plot_final_population_spread,
    plot_generation_convergence,
    plot_population_parallel_coordinates,
    plot_population_score_correlations,
    plot_population_score_vs_variables,
    plot_population_variable_histograms,
    print_best_result,
)
from src.prop.continuous_prop_database import (
    ContinuousPropDatabase,
    load_default_continuous_prop_database,
)
from src.vectors import ASBDesignVector, DesignVector, ParameterVector


BAD_OBJECTIVE = 1.0e6
PD_MIN = 0.4
PD_MAX = 0.8
MIN_DUCKS_PER_PUCK = 3.0
TARGET_EVALS_PER_SECOND = 80.0
TARGET_RUN_SECONDS = 3600.0

PROP_DATABASE: ContinuousPropDatabase | None = None
PARAMETER_VECTOR = ParameterVector()
CALLBACK_HISTORY: list[dict] = []
CALLBACK_GENERATION = 0
RUN_START_SECONDS: float | None = None
CALLBACK_POPULATION_SIZE = 0
CALLBACK_TARGET_CANDIDATES = 0
CALLBACK_SCORE_BEST = True
PROGRESS_BAR = None


@dataclass(frozen=True)
class ToplineConfig:
    """Settings for the long, SciPy-managed top-line DE run."""

    workers: int = -1
    popsize: int = 25
    maxiter: int | None = 300
    target_seconds: float = TARGET_RUN_SECONDS
    assumed_evals_per_second: float = TARGET_EVALS_PER_SECOND
    init: str = "sobol"
    mutation: tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    tol: float = 1.0e-5
    atol: float = 0.0
    polish: bool = False
    seed: int = 20260808
    output_dir: Path = Path("data_dump") / "opt_topline"
    round_payload: bool = True
    suppress_module_output: bool = True
    callback_score_best: bool = True
    save_best_visualization: bool = True


def _module_output_context(config: ToplineConfig):
    if not config.suppress_module_output:
        return nullcontext()
    return redirect_stdout(StringIO())


def _load_prop_database(config: ToplineConfig) -> ContinuousPropDatabase:
    with _module_output_context(config):
        return load_default_continuous_prop_database()


def _ensure_prop_database_loaded(config: ToplineConfig) -> float:
    global PROP_DATABASE

    if PROP_DATABASE is not None:
        return 0.0

    start = time.perf_counter()
    PROP_DATABASE = _load_prop_database(config)
    return time.perf_counter() - start


def _pd_ratio(x: np.ndarray) -> float:
    return float(x[9] / x[8])


PD_CONSTRAINT = NonlinearConstraint(_pd_ratio, PD_MIN, PD_MAX)


def _ducks_per_puck(x: np.ndarray) -> float:
    names = DesignVector.opt_names()
    ducks = x[names.index("ducks_num")]
    pucks = x[names.index("pucks_num")]
    return float(ducks / pucks)


DUCKS_PER_PUCK_CONSTRAINT = NonlinearConstraint(
    _ducks_per_puck,
    MIN_DUCKS_PER_PUCK,
    np.inf,
)


def _integrality_mask() -> np.ndarray:
    names = DesignVector.opt_names()
    return np.array(
        [name in {"ducks_num", "pucks_num"} for name in names],
        dtype=bool,
    )


def _expected_population_size(config: ToplineConfig) -> int:
    requested = config.popsize * len(DesignVector.bounds())
    if config.init == "sobol":
        return 1 << (requested - 1).bit_length()
    return requested


def _resolved_maxiter(config: ToplineConfig) -> int:
    if config.maxiter is not None:
        return config.maxiter

    target_evaluations = int(
        config.target_seconds
        * config.assumed_evals_per_second
    )
    population_size = _expected_population_size(config)
    return max(1, target_evaluations // population_size - 1)


def _target_candidate_count(config: ToplineConfig) -> int:
    return (_resolved_maxiter(config) + 1) * _expected_population_size(config)


def _objective(x: np.ndarray) -> float:
    """Top-level objective so SciPy can pickle it for worker processes."""

    if not np.all(np.isfinite(x)):
        return BAD_OBJECTIVE
    if not PD_MIN <= _pd_ratio(x) <= PD_MAX:
        return BAD_OBJECTIVE
    if _ducks_per_puck(x) < MIN_DUCKS_PER_PUCK:
        return BAD_OBJECTIVE

    config = ToplineConfig()
    try:
        _ensure_prop_database_loaded(config)
        design_vector = DesignVector.from_array(x)
        with _module_output_context(config):
            score, _ = cast(
                tuple[float, list[float]],
                score_aircraft(
                    design_vector,
                    PARAMETER_VECTOR,
                    disp_res=False,
                    round_payload=config.round_payload,
                    prop_database=PROP_DATABASE,
                ),
            )
    except Exception:
        return BAD_OBJECTIVE

    if not math.isfinite(score):
        return BAD_OBJECTIVE

    return -float(score)


def _callback(xk: np.ndarray, convergence: float) -> bool:
    global CALLBACK_GENERATION

    CALLBACK_GENERATION += 1
    elapsed = (
        time.perf_counter() - RUN_START_SECONDS
        if RUN_START_SECONDS is not None
        else 0.0
    )
    completed_candidates = min(
        CALLBACK_TARGET_CANDIDATES,
        CALLBACK_POPULATION_SIZE * (CALLBACK_GENERATION + 1),
    )
    rate = completed_candidates / elapsed if elapsed > 0.0 else 0.0
    eta = (
        (CALLBACK_TARGET_CANDIDATES - completed_candidates) / rate
        if rate > 0.0
        else math.nan
    )
    objective = _objective(xk) if CALLBACK_SCORE_BEST else math.nan
    score = -objective if math.isfinite(objective) else math.nan

    row = {
        "generation": CALLBACK_GENERATION,
        "estimated_completed_candidates": completed_candidates,
        "elapsed_seconds": elapsed,
        "estimated_candidates_per_second": rate,
        "eta_seconds": eta,
        "convergence": float(convergence),
        "best_objective": float(objective),
        "best_score": float(score),
    }
    for name, value in zip(DesignVector.opt_names(), xk):
        row[name] = float(value)
    CALLBACK_HISTORY.append(row)

    if PROGRESS_BAR is not None:
        delta = completed_candidates - PROGRESS_BAR.n
        if delta > 0:
            PROGRESS_BAR.update(delta)
        PROGRESS_BAR.set_postfix(
            {
                "gen": CALLBACK_GENERATION,
                "score": f"{score:.4g}",
                "conv": f"{convergence:.3g}",
            },
            refresh=False,
        )

    if CALLBACK_GENERATION == 1 or CALLBACK_GENERATION % 10 == 0:
        message = (
            "[topline] "
            f"gen={CALLBACK_GENERATION} "
            f"score={score:.6g} "
            f"conv={convergence:.3g} "
            f"elapsed={elapsed / 60.0:.1f} min "
            f"eta={eta / 60.0:.1f} min"
        )
        if PROGRESS_BAR is None:
            print(message, flush=True)
        else:
            tqdm.write(message)

    return False


def _differential_evolution_kwargs(config: ToplineConfig) -> dict:
    parameters = inspect.signature(differential_evolution).parameters
    kwargs = {
        "func": _objective,
        "bounds": DesignVector.bounds(),
        "constraints": (PD_CONSTRAINT, DUCKS_PER_PUCK_CONSTRAINT),
        "maxiter": _resolved_maxiter(config),
        "popsize": config.popsize,
        "polish": config.polish,
        "updating": "deferred",
        "callback": _callback,
        "tol": config.tol,
        "atol": config.atol,
        "disp": False,
        "workers": config.workers,
        "init": config.init,
        "mutation": config.mutation,
        "recombination": config.recombination,
    }

    if "integrality" in parameters:
        kwargs["integrality"] = _integrality_mask()
    if "rng" in parameters:
        kwargs["rng"] = np.random.default_rng(config.seed)
    elif "seed" in parameters:
        kwargs["seed"] = config.seed

    return kwargs


def _timestamped_output_dir(base_dir: Path) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"run_{stamp}"


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value)) # type: ignore
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_final_population(result, output_dir: Path) -> Path:
    population = np.asarray(result.population, dtype=float)
    energies = np.asarray(result.population_energies, dtype=float)
    order = np.argsort(energies)
    rows = []
    for rank, index in enumerate(order, start=1):
        objective = float(energies[index])
        row = {
            "rank": rank,
            "population_index": int(index),
            "objective": objective,
            "score": -objective,
            "finite": bool(np.isfinite(objective)),
        }
        for name, value in zip(DesignVector.opt_names(), population[index]):
            row[name] = float(value)
        rows.append(row)

    path = output_dir / "final_population.csv"
    _write_csv(path, rows)
    return path


def _best_design_report(result, config: ToplineConfig, output_dir: Path) -> dict:
    best_design = DesignVector.from_array(result.x)
    scoring_design = replace(
        best_design,
        ducks_num=round(best_design.ducks_num)
        if config.round_payload
        else best_design.ducks_num,
        pucks_num=round(best_design.pucks_num)
        if config.round_payload
        else best_design.pucks_num,
    )

    score, breakdown, details = cast(
        tuple[float, list[float], dict[str, Any]],
        score_aircraft(
            best_design,
            PARAMETER_VECTOR,
            disp_res=False,
            round_payload=config.round_payload,
            prop_database=PROP_DATABASE,
            return_details=True,
        ),
    )
    mech_result = evaluate_mechanical_module(
        scoring_design,
        parameter_vector=PARAMETER_VECTOR,
    )
    resolved_design = replace(
        scoring_design,
        fuselage_width=mech_result.resolved_fuselage_width_m,
        fuselage_height=mech_result.resolved_fuselage_height_m,
    )

    report = {
        "score": float(score),
        "objective": float(result.fun),
        "breakdown": {
            "ground": float(breakdown[0]),
            "m1": float(breakdown[1]),
            "m2": float(breakdown[2]),
            "m3": float(breakdown[3]),
        },
        "optimizer_vector": asdict(best_design),
        "scoring_vector": asdict(scoring_design),
        "resolved_vector": asdict(resolved_design),
        "mechanical": mech_result,
        "aero": details,
    }

    path = output_dir / "best_design_report.json"
    path.write_text(
        json.dumps(_json_safe(report), indent=2) + "\n",
        encoding="utf-8",
    )

    if config.save_best_visualization:
        import matplotlib.pyplot as plt

        geometry_path = output_dir / "best_design_geometry.png"
        ASBDesignVector.from_design_vector(resolved_design).make_airplane(
            name="Top-Line Optimized Design"
        ).draw_three_view(style="shaded")
        plt.gcf().savefig(geometry_path, dpi=200, bbox_inches="tight")
        plt.close(plt.gcf())
        report["geometry_path"] = str(geometry_path)

    return report


def _write_run_summary(
    result,
    config: ToplineConfig,
    output_dir: Path,
    elapsed_seconds: float,
    database_load_seconds: float,
    best_report: dict,
) -> Path:
    population_size = _expected_population_size(config)
    maxiter = _resolved_maxiter(config)
    target_candidates = _target_candidate_count(config)
    summary = {
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed_seconds,
        "database_load_seconds_parent": database_load_seconds,
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "success": bool(result.success),
        "message": str(result.message),
        "best_objective": float(result.fun),
        "best_score": -float(result.fun),
        "reevaluated_best_score": best_report["score"],
        "workers": config.workers,
        "variables": len(DesignVector.bounds()),
        "popsize": config.popsize,
        "expected_population_size": population_size,
        "maxiter": maxiter,
        "target_candidate_slots": target_candidates,
        "assumed_evals_per_second": config.assumed_evals_per_second,
        "target_seconds": config.target_seconds,
        "observed_nfev_per_second": int(result.nfev) / elapsed_seconds,
        "observed_candidate_slots_per_second": target_candidates / elapsed_seconds,
        "scipy_version": scipy.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "config": asdict(config),
        "bounds": {
            name: bounds
            for name, bounds in zip(DesignVector.opt_names(), DesignVector.bounds())
        },
        "integrality": {
            name: bool(flag)
            for name, flag in zip(DesignVector.opt_names(), _integrality_mask())
        },
    }

    path = output_dir / "run_summary.json"
    path.write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _save_plots(result, output_dir: Path) -> list[Path]:
    variable_names = DesignVector.opt_names()
    bounds = DesignVector.bounds()
    paths = {
        "generation_convergence": output_dir / "generation_convergence.png",
        "final_population_scores": output_dir / "final_population_scores.png",
        "final_population_spread": output_dir / "final_population_spread.png",
        "variable_histograms": output_dir / "final_variable_histograms.png",
        "score_vs_variables": output_dir / "final_score_vs_variables.png",
        "score_correlations": output_dir / "final_score_correlations.png",
        "parallel_coordinates": output_dir / "final_parallel_coordinates.png",
    }

    plot_generation_convergence(
        CALLBACK_HISTORY,
        save_path=str(paths["generation_convergence"]),
        show=False,
    )
    plot_final_population(
        result,
        save_path=str(paths["final_population_scores"]),
        show=False,
    )
    plot_final_population_spread(
        result,
        variable_names,
        bounds,
        save_path=str(paths["final_population_spread"]),
        show=False,
    )
    plot_population_variable_histograms(
        result.population,
        variable_names,
        bounds,
        save_path=str(paths["variable_histograms"]),
        show=False,
    )
    plot_population_score_vs_variables(
        result.population,
        result.population_energies,
        variable_names,
        save_path=str(paths["score_vs_variables"]),
        show=False,
    )
    plot_population_score_correlations(
        result.population,
        result.population_energies,
        variable_names,
        save_path=str(paths["score_correlations"]),
        show=False,
    )
    plot_population_parallel_coordinates(
        result.population,
        result.population_energies,
        variable_names,
        bounds,
        save_path=str(paths["parallel_coordinates"]),
        show=False,
    )
    return list(paths.values())


def run_topline_optimization(config: ToplineConfig | None = None):
    """Run the one-hour top-line SciPy differential-evolution optimization."""

    global CALLBACK_GENERATION
    global CALLBACK_HISTORY
    global CALLBACK_POPULATION_SIZE
    global CALLBACK_SCORE_BEST
    global CALLBACK_TARGET_CANDIDATES
    global PROGRESS_BAR
    global RUN_START_SECONDS

    config = config or ToplineConfig()
    output_dir = _timestamped_output_dir(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    CALLBACK_HISTORY = []
    CALLBACK_GENERATION = 0
    CALLBACK_POPULATION_SIZE = _expected_population_size(config)
    CALLBACK_TARGET_CANDIDATES = _target_candidate_count(config)
    CALLBACK_SCORE_BEST = config.callback_score_best

    print("Top-line differential evolution run")
    print(f"  output: {output_dir}")
    print(f"  workers: {config.workers}")
    print(f"  variables: {len(DesignVector.bounds())}")
    print(f"  population size: {CALLBACK_POPULATION_SIZE}")
    print(f"  maxiter: {_resolved_maxiter(config)}")
    print(f"  target candidate slots: {CALLBACK_TARGET_CANDIDATES}")
    print(
        "  assumed runtime: "
        f"{CALLBACK_TARGET_CANDIDATES / config.assumed_evals_per_second / 60.0:.1f} min"
    )

    database_load_seconds = _ensure_prop_database_loaded(config)
    print(f"  parent database load: {database_load_seconds:.3f} s")

    kwargs = _differential_evolution_kwargs(config)
    (output_dir / "de_kwargs_summary.json").write_text(
        json.dumps(
            _json_safe(
                {
                    key: value
                    for key, value in kwargs.items()
                    if key not in {"func", "callback", "constraints", "rng"}
                }
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    RUN_START_SECONDS = time.perf_counter()
    with tqdm(
        total=CALLBACK_TARGET_CANDIDATES,
        desc="Top-line evaluations",
        unit="eval",
    ) as progress_bar:
        PROGRESS_BAR = progress_bar
        try:
            result = differential_evolution(**kwargs)
        finally:
            if PROGRESS_BAR.n < CALLBACK_TARGET_CANDIDATES:
                PROGRESS_BAR.update(CALLBACK_TARGET_CANDIDATES - PROGRESS_BAR.n)
            PROGRESS_BAR = None
    elapsed_seconds = time.perf_counter() - RUN_START_SECONDS

    print_best_result(result, DesignVector.opt_names())
    best_report = _best_design_report(result, config, output_dir)
    summary_path = _write_run_summary(
        result,
        config,
        output_dir,
        elapsed_seconds,
        database_load_seconds,
        best_report,
    )
    generation_path = output_dir / "generation_history.csv"
    _write_csv(generation_path, CALLBACK_HISTORY)
    population_path = _write_final_population(result, output_dir)
    arrays_path = output_dir / "result_arrays.npz"
    np.savez(
        arrays_path,
        x=np.asarray(result.x, dtype=float),
        population=np.asarray(result.population, dtype=float),
        population_energies=np.asarray(result.population_energies, dtype=float),
    )
    plot_paths = _save_plots(result, output_dir)

    print("\nSaved top-line artifacts:")
    print(f"  summary: {summary_path}")
    print(f"  generation history: {generation_path}")
    print(f"  final population: {population_path}")
    print(f"  result arrays: {arrays_path}")
    print(f"  best design report: {output_dir / 'best_design_report.json'}")
    for path in plot_paths:
        print(f"  plot: {path}")

    return result


def main() -> None:
    run_topline_optimization()


if __name__ == "__main__":
    main()
