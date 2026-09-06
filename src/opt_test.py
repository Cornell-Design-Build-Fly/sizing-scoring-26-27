from __future__ import annotations

import math
import csv
import json
import os
import time
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint


from tqdm.auto import tqdm

from src.main import main
from src.mech.main_mech import evaluate_mechanical_module
from src.opt.score import scoring_reference_values
from src.opt.topline_opt import (
    ToplineConfig,
    _design_vector_from_optimizer,
    _integrality_mask,
    _optimizer_bounds,
    _optimizer_variable_names,
)
from src.opt.view_results import (
    plot_final_population,
    plot_final_population_spread,
    plot_generation_score_distribution,
    plot_penalty_history,
    plot_score_history,
    print_best_result,
)
from src.prop.continuous_prop_database import (
    ContinuousPropDatabase,
    load_default_continuous_prop_database,
)
from src.vectors import ASBDesignVector, DesignVector, ParameterVector


BAD_OBJECTIVE = 1.0e6
ROUND_PAYLOAD = True
MAXITER = 237
POPSIZE = 15
WORKERS = 10
DE_TOL = 1.0e-4
DE_ATOL = 0.0
OUTPUT_DIR = Path("data_dump") / "opt_preliminary"
SUPPRESS_MODULE_OUTPUT = True
REPORT_REJECTIONS = False
REJECTION_DETAIL_CHARS = 220
SAVE_BEST_DESIGN_VISUALIZATION = True
OPTIMIZER_CONFIG = ToplineConfig(
    battery_cell_count=8,
    optimize_battery_cell_count=False,
)
EVALUATION_HISTORY: list[dict] = []
PROGRESS_BAR = None
BEST_SCORE = -math.inf
PROP_DATABASE: ContinuousPropDatabase | None = None
PARAMETER_VECTOR = ParameterVector()
COMPLETED_GENERATIONS = 0
OPTIMIZATION_START: float | None = None

HISTORY_ARTIFACTS = (
    "history.csv",
    "score_history.png",
    "generation_score_distribution.png",
    "penalty_history.png",
)



def optimizer_variable_names() -> list[str]:
    return _optimizer_variable_names(OPTIMIZER_CONFIG)


def optimizer_bounds() -> list[tuple[float, float]]:
    return _optimizer_bounds(OPTIMIZER_CONFIG)


def optimizer_integrality() -> np.ndarray:
    return _integrality_mask(OPTIMIZER_CONFIG)


def optimizer_array_from_design(design: DesignVector) -> np.ndarray:
    return design.to_array()


def design_from_optimizer_array(x: np.ndarray) -> DesignVector:
    return _design_vector_from_optimizer(x, OPTIMIZER_CONFIG)


pd_constraint = NonlinearConstraint(
    lambda x: x[optimizer_variable_names().index("prop_pitch_in")]
    / x[optimizer_variable_names().index("prop_diameter_in")],
    0.4,   # minimum P/D
    0.8,   # maximum P/D
)

# Mission 3 flies its own propeller and needs the same geometric ratio bound.
mission3_pd_constraint = NonlinearConstraint(
    lambda x: x[optimizer_variable_names().index("mission3_prop_pitch_in")]
    / x[optimizer_variable_names().index("mission3_prop_diameter_in")],
    0.4,   # minimum P/D
    0.8,   # maximum P/D
)

def _prepare_output_dir() -> None:
    """Create the output directory and clear inapplicable old history."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not should_record_evaluations():
        for name in HISTORY_ARTIFACTS:
            path = OUTPUT_DIR / name
            if path.exists():
                path.unlink()


def _write_run_summary(result, elapsed_seconds: float) -> Path:
    """Save actual run size, timing, settings, and termination details."""

    path = OUTPUT_DIR / "run_summary.json"
    summary = {
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed_seconds,
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "success": bool(result.success),
        "message": str(result.message),
        "workers": resolved_worker_count(),
        "maxiter": MAXITER,
        "popsize": POPSIZE,
        "variables": len(optimizer_bounds()),
        "variable_names": optimizer_variable_names(),
        "bounds": {
            name: bounds
            for name, bounds in zip(optimizer_variable_names(), optimizer_bounds())
        },
        "integrality": {
            name: bool(flag)
            for name, flag in zip(
                optimizer_variable_names(), optimizer_integrality()
            )
        },
        "population_size": de_population_size(),
        "maximum_expected_evaluations": expected_de_evaluations(),
        "evaluations_per_second": int(result.nfev) / elapsed_seconds,
        "wall_seconds_per_evaluation": elapsed_seconds / int(result.nfev),
        "estimated_worker_seconds_per_evaluation": elapsed_seconds * resolved_worker_count() / int(result.nfev),
        "tol": DE_TOL,
        "atol": DE_ATOL,
        "history_recorded": should_record_evaluations(),
        "best_objective": float(result.fun),
        "best_score": -float(result.fun),
        "scoring_references": scoring_reference_values(),
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return path


def save_best_design_visualization(design: DesignVector) -> tuple[Path, Path]:
    """Save the rounded, mechanically resolved best design and three-view."""

    import matplotlib.pyplot as plt

    scoring_design = replace(
        design,
        extra_shipping_containers=(
            round(design.extra_shipping_containers)
            if ROUND_PAYLOAD
            else design.extra_shipping_containers
        ),
    )
    with _module_output_context():
        mech = evaluate_mechanical_module(scoring_design, parameter_vector=PARAMETER_VECTOR)
    resolved_design = replace(
        scoring_design,
        fuselage_width=mech.resolved_fuselage_width_m,
        fuselage_height=mech.resolved_fuselage_height_m,
    )
    vector_path = OUTPUT_DIR / "best_design_vector.json"
    geometry_path = OUTPUT_DIR / "best_design_geometry.png"
    vector_path.write_text(
        json.dumps(
            {
                "optimizer_vector": asdict(design),
                "visualized_resolved_vector": asdict(resolved_design),
                "scoring_references": scoring_reference_values(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ASBDesignVector.from_design_vector(resolved_design).make_airplane(
        name="Optimized Design"
    ).draw_three_view(style="shaded")
    plt.gcf().savefig(geometry_path, dpi=200, bbox_inches="tight")
    plt.close(plt.gcf())
    return vector_path, geometry_path


def _history_row(
    *,
    evaluation: int,
    x: np.ndarray,
    objective: float,
    score: float,
    breakdown: list[float] | None,
    status: str,
    message: str = "",
) -> dict:
    row = {
        "evaluation": evaluation,
        "generation": generation_for_evaluation(evaluation),
        "status": status,
        "objective": float(objective),
        "score": float(score),
        "raw_score": np.nan,
        "implied_penalty": np.nan,
        "message": message,
    }
    for index, name in enumerate(optimizer_variable_names()):
        row[name] = float(x[index])
    if breakdown is None:
        row.update({"ground": np.nan, "m1": np.nan, "m2": np.nan, "m3": np.nan})
    else:
        raw_score = float(sum(breakdown))
        row["raw_score"] = raw_score
        row["implied_penalty"] = raw_score - float(score)
        row.update(
            {
                "ground": float(breakdown[0]),
                "m1": float(breakdown[1]),
                "m2": float(breakdown[2]),
                "m3": float(breakdown[3]),
            }
        )
    return row


def _notify(message: str) -> None:
    """Write a progress-safe status line."""

    if PROGRESS_BAR is None:
        print(message, flush=True)
    else:
        tqdm.write(message)


def _next_evaluation_number() -> int:
    """Return a serial-mode evaluation number."""

    return len(EVALUATION_HISTORY) + 1


def _short_message(message: str, limit: int = REJECTION_DETAIL_CHARS) -> str:
    """Trim long exception text for optional live console reporting."""

    message = " ".join(message.split())
    if len(message) <= limit:
        return message
    return f"{message[: limit - 3]}..."


def _module_output_context():
    """Return a context manager that hides noisy module-level prints."""

    if not SUPPRESS_MODULE_OUTPUT:
        return nullcontext()
    return redirect_stdout(StringIO())


def _load_prop_database() -> ContinuousPropDatabase:
    """Build the shared prop database once, matching the main_test call path."""

    with _module_output_context():
        return load_default_continuous_prop_database()


def _ensure_prop_database_loaded() -> float:
    """Load the shared prop database if needed and return load seconds."""

    global PROP_DATABASE

    if PROP_DATABASE is not None:
        return 0.0

    start = time.perf_counter()
    PROP_DATABASE = _load_prop_database()
    return time.perf_counter() - start


def _random_pd_feasible_vector(rng: np.random.Generator) -> np.ndarray:
    """Sample a random optimizer vector that satisfies the P/D constraint."""

    bounds = optimizer_bounds()
    values = np.array(
        [
            rng.uniform(lower, upper)
            for lower, upper in bounds
        ],
        dtype=float,
    )

    names = optimizer_variable_names()
    for index, is_integer in enumerate(optimizer_integrality()):
        if is_integer:
            lower, upper = bounds[index]
            values[index] = rng.integers(int(lower), int(upper) + 1)
    for diameter_name, pitch_name in (
        ("prop_diameter_in", "prop_pitch_in"),
        ("mission3_prop_diameter_in", "mission3_prop_pitch_in"),
    ):
        diameter_index = names.index(diameter_name)
        pitch_index = names.index(pitch_name)
        diameter = values[diameter_index]
        pitch_lower, pitch_upper = bounds[pitch_index]

        constrained_lower = max(pitch_lower, 0.4 * diameter)
        constrained_upper = min(pitch_upper, 0.8 * diameter)
        if constrained_lower > constrained_upper:
            values[pitch_index] = np.clip(
                values[pitch_index],
                pitch_lower,
                pitch_upper,
            )
        else:
            values[pitch_index] = rng.uniform(
                constrained_lower,
                constrained_upper,
            )

    return values


def benchmark_fitness_runtime(
    sample_count: int = 100,
    *,
    seed: int = 1,
    warmup_count: int = 5,
    verbose: bool = True,
) -> dict:
    """Time warmed calls to the same objective function used by DE.

    This intentionally uses ``fitness(..., record=False)`` so timing includes
    the optimizer's exception handling and objective conversion, but excludes
    progress bars and serial history bookkeeping.
    """

    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if warmup_count < 0:
        raise ValueError("warmup_count cannot be negative.")

    database_load_seconds = _ensure_prop_database_loaded()

    baseline = optimizer_array_from_design(DesignVector())
    for _ in range(warmup_count):
        fitness(baseline, record=False)

    rng = np.random.default_rng(seed)
    rows = []
    for index in range(sample_count):
        x = _random_pd_feasible_vector(rng)
        start = time.perf_counter()
        objective = fitness(x, record=False)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "sample": index,
                "seconds": elapsed,
                "objective": float(objective),
                "status": (
                    "rejected"
                    if objective >= BAD_OBJECTIVE
                    else "ok"
                ),
            }
        )

    seconds = np.array(
        [row["seconds"] for row in rows],
        dtype=float,
    )
    ok_seconds = np.array(
        [
            row["seconds"]
            for row in rows
            if row["status"] == "ok"
        ],
        dtype=float,
    )
    rejected_count = sum(
        row["status"] == "rejected"
        for row in rows
    )

    summary = {
        "sample_count": sample_count,
        "seed": seed,
        "warmup_count": warmup_count,
        "database_load_seconds": database_load_seconds,
        "rejected_count": rejected_count,
        "ok_count": sample_count - rejected_count,
        "mean_seconds": float(np.mean(seconds)),
        "median_seconds": float(np.median(seconds)),
        "p95_seconds": float(np.percentile(seconds, 95)),
        "max_seconds": float(np.max(seconds)),
        "ok_mean_seconds": (
            float(np.mean(ok_seconds))
            if ok_seconds.size
            else math.nan
        ),
        "ok_median_seconds": (
            float(np.median(ok_seconds))
            if ok_seconds.size
            else math.nan
        ),
        "ok_p95_seconds": (
            float(np.percentile(ok_seconds, 95))
            if ok_seconds.size
            else math.nan
        ),
        "rows": rows,
    }

    if verbose:
        print("\nFitness runtime benchmark:")
        print(f"  samples: {sample_count}")
        print(f"  warmups: {warmup_count}")
        print(f"  database load: {database_load_seconds:.4f} s")
        print(
            f"  ok/rejected: {summary['ok_count']}/"
            f"{summary['rejected_count']}"
        )
        print(f"  all mean:   {summary['mean_seconds']:.4f} s")
        print(f"  all median: {summary['median_seconds']:.4f} s")
        print(f"  all p95:    {summary['p95_seconds']:.4f} s")
        print(f"  all max:    {summary['max_seconds']:.4f} s")
        if ok_seconds.size:
            print(f"  ok mean:    {summary['ok_mean_seconds']:.4f} s")
            print(f"  ok median:  {summary['ok_median_seconds']:.4f} s")
            print(f"  ok p95:     {summary['ok_p95_seconds']:.4f} s")

    return summary


def _record_evaluation(row: dict) -> None:
    """Store one objective evaluation and update live progress displays."""

    global BEST_SCORE

    EVALUATION_HISTORY.append(row)
    if PROGRESS_BAR is not None:
        PROGRESS_BAR.update(1)

    if row["status"] == "ok" and row["score"] > BEST_SCORE:
        BEST_SCORE = row["score"]
        _notify(
            f"[opt] new best score={row['score']:.6g} "
            f"at evaluation {row['evaluation']}"
        )


def expected_de_evaluations() -> int:
    """Expected objective calls for the current DE settings, excluding polish."""

    return (MAXITER + 1) * de_population_size()


def de_population_size() -> int:
    """Differential-evolution population size for the current settings."""

    return POPSIZE * len(optimizer_bounds())


def resolved_worker_count() -> int:
    """Return the number of local worker processes requested."""

    if WORKERS == -1:
        return os.cpu_count() or 1
    if WORKERS < -1 or WORKERS == 0:
        raise ValueError("WORKERS must be 1, a positive integer, or -1.")
    return WORKERS


def de_updating_mode() -> str:
    """SciPy requires deferred updating for parallel objective evaluation."""

    if resolved_worker_count() == 1:
        return "immediate"
    return "deferred"


def should_record_evaluations() -> bool:
    """Only serial runs can keep complete per-evaluation Python history."""

    return resolved_worker_count() == 1


def generation_for_evaluation(evaluation: int) -> int:
    """Return 0 for the initial DE population, then 1..MAXITER."""

    population_size = de_population_size()
    if evaluation <= population_size:
        return 0
    return int(math.ceil((evaluation - population_size) / population_size))


def fitness(x: np.ndarray, *, record: bool = True) -> float:
    """Objective for a preliminary integrated optimization run.

    SciPy minimizes, so this returns the negative of the project score. Any
    failed design gets a large finite objective so differential evolution can
    continue sampling.
    """

    record = record and should_record_evaluations()
    evaluation = _next_evaluation_number()
    try:
        design_vector = design_from_optimizer_array(x)
        if PROP_DATABASE is None:
            _ensure_prop_database_loaded()
        with _module_output_context():
            score, breakdown = main( # type: ignore
                design_vector,
                PARAMETER_VECTOR,
                disp_res=False,
                round_payload=ROUND_PAYLOAD,
                prop_database=PROP_DATABASE,
            )
    except Exception as exc:
        if record:
            if REPORT_REJECTIONS:
                _notify(
                    f"[opt] rejected evaluation {evaluation}: "
                    f"{_short_message(str(exc))}"
                )
            _record_evaluation(
                _history_row(
                    evaluation=evaluation,
                    x=np.asarray(x),
                    objective=BAD_OBJECTIVE,
                    score=-BAD_OBJECTIVE,
                    breakdown=None,
                    status="rejected",
                    message=str(exc),
                )
            )
        return BAD_OBJECTIVE

    if not math.isfinite(score):
        if record:
            _notify(
                f"[opt] nonfinite score for x={np.asarray(x)}; "
                f"breakdown={breakdown}"
            )
            _record_evaluation(
                _history_row(
                    evaluation=evaluation,
                    x=np.asarray(x),
                    objective=BAD_OBJECTIVE,
                    score=-BAD_OBJECTIVE,
                    breakdown=breakdown,
                    status="nonfinite",
                    message="Nonfinite score",
                )
            )
        return BAD_OBJECTIVE

    objective = -float(score)
    if record:
        _record_evaluation(
            _history_row(
                evaluation=evaluation,
                x=np.asarray(x),
                objective=objective,
                score=score,
                breakdown=breakdown,
                status="ok",
            )
        )
    return objective


def progress_callback(xk: np.ndarray, convergence: float) -> bool:
    """Print one concise progress line at the end of each DE generation."""

    global COMPLETED_GENERATIONS

    del xk
    COMPLETED_GENERATIONS += 1
    completed_evaluations = min(
        expected_de_evaluations(),
        de_population_size() * (COMPLETED_GENERATIONS + 1),
    )
    elapsed = time.perf_counter() - OPTIMIZATION_START if OPTIMIZATION_START is not None else 0.0
    rate = completed_evaluations / elapsed if elapsed > 0 else 0.0
    eta = (expected_de_evaluations() - completed_evaluations) / rate if rate > 0 else 0.0
    timing = (
        f", elapsed={elapsed:.1f}s, throughput={rate:.1f} eval/s, "
        f"worker_avg={elapsed * resolved_worker_count() / completed_evaluations:.4f}s/eval, "
        f"eta={eta:.1f}s"
    )

    if should_record_evaluations():
        sync_progress_bar_to_history()
        successful = [entry for entry in EVALUATION_HISTORY if entry["status"] == "ok"]
        rejected_count = sum(
            entry["status"] == "rejected" for entry in EVALUATION_HISTORY
        )
        if successful:
            best = max(successful, key=lambda entry: entry["score"])
            _notify(
                f"[opt] generation complete: evaluations={len(EVALUATION_HISTORY)}, "
                f"best_score={best['score']:.6g}, rejected={rejected_count}, "
                f"convergence={convergence:.3g}{timing}"
            )
        else:
            _notify(
                f"[opt] generation complete: evaluations={len(EVALUATION_HISTORY)}, "
                f"no successful designs yet, rejected={rejected_count}, "
                f"convergence={convergence:.3g}{timing}"
            )
    else:
        sync_progress_bar_to_count(completed_evaluations)
        _notify(
            f"[opt] generation complete: approx_evaluations={completed_evaluations}, "
            f"convergence={convergence:.3g}{timing}"
        )
    return False


def sync_progress_bar_to_count(count: int) -> None:
    """Update the parent progress bar to an approximate completed count."""

    if PROGRESS_BAR is None:
        return
    delta = min(count, expected_de_evaluations()) - PROGRESS_BAR.n
    if delta > 0:
        PROGRESS_BAR.update(delta)


def sync_progress_bar_to_history() -> None:
    """Update the parent progress bar from recorded objective rows."""

    if PROGRESS_BAR is None:
        return
    delta = len(EVALUATION_HISTORY) - PROGRESS_BAR.n
    if delta > 0:
        PROGRESS_BAR.update(delta)


def _differential_evolution_kwargs() -> dict:
    """Shared SciPy DE settings."""

    return {
        "func": fitness,
        "bounds": optimizer_bounds(),
        "constraints": (pd_constraint, mission3_pd_constraint),
        "maxiter": MAXITER,
        "popsize": POPSIZE,
        "polish": False,
        "updating": de_updating_mode(),
        "callback": progress_callback,
        "tol": DE_TOL,
        "atol": DE_ATOL,
        "disp": False,
        "integrality": optimizer_integrality(),
    }


def _run_serial_optimization():
    """Run DE in the parent process."""

    return differential_evolution(
        workers=1,
        **_differential_evolution_kwargs(),
    )


def _run_parallel_optimization():
    """Let SciPy own multiprocessing and skip per-evaluation history."""

    return differential_evolution(
        workers=WORKERS,
        **_differential_evolution_kwargs(),
    )


def run_preliminary_optimization():
    """Run a tiny DE pass through the full scoring stack."""

    global BEST_SCORE, COMPLETED_GENERATIONS, PROGRESS_BAR, OPTIMIZATION_START

    worker_count = resolved_worker_count()
    EVALUATION_HISTORY.clear()
    BEST_SCORE = -math.inf
    COMPLETED_GENERATIONS = 0
    OPTIMIZATION_START = time.perf_counter()

    with tqdm(
        total=expected_de_evaluations(),
        desc="Optimization evaluations",
        unit="eval",
    ) as progress_bar:
        PROGRESS_BAR = progress_bar
        try:
            if worker_count == 1:
                return _run_serial_optimization()
            return _run_parallel_optimization()
        finally:
            if should_record_evaluations():
                sync_progress_bar_to_history()
            else:
                sync_progress_bar_to_count(expected_de_evaluations())
            PROGRESS_BAR = None


def write_history_csv(path: Path) -> None:
    """Save objective-evaluation history for post-run inspection."""

    if not EVALUATION_HISTORY:
        return
    fieldnames = list(EVALUATION_HISTORY[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(EVALUATION_HISTORY)


def print_generation_summary() -> None:
    """Print best/average/worst net score and average penalty by generation."""

    successful = [entry for entry in EVALUATION_HISTORY if entry["status"] == "ok"]
    if not successful:
        print("\nNo successful evaluations to summarize.")
        return

    generations = sorted({entry["generation"] for entry in successful})
    print("\nGeneration summary:")
    print("  gen | best score | avg score | worst score | avg penalty")
    for generation in generations:
        rows = [entry for entry in successful if entry["generation"] == generation]
        scores = np.array([entry["score"] for entry in rows], dtype=float)
        penalties = np.array([entry["implied_penalty"] for entry in rows], dtype=float)
        print(
            f"  {generation:>3} | "
            f"{np.max(scores):>10.4f} | "
            f"{np.mean(scores):>9.4f} | "
            f"{np.min(scores):>11.4f} | "
            f"{np.nanmean(penalties):>11.4f}"
        )


def print_final_population_spread(result) -> None:
    """Print normalized final-population spread for each design variable."""

    if not hasattr(result, "population"):
        return
    population = np.asarray(result.population, dtype=float)
    bounds = np.asarray(optimizer_bounds(), dtype=float)
    spans = bounds[:, 1] - bounds[:, 0]
    normalized_std = np.std(population, axis=0) / spans

    print("\nFinal population spread:")
    print("  variable | std / range")
    for name, spread in sorted(
        zip(optimizer_variable_names(), normalized_std),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {name}: {spread:.4f}")


def main_opt_test() -> None:
    global PROP_DATABASE

    _prepare_output_dir()
    print("Loading prop database...")
    PROP_DATABASE = _load_prop_database()

    print("Optimization variables:")
    for name, bounds in zip(optimizer_variable_names(), optimizer_bounds()):
        print(f"  {name}: {bounds}")

    baseline = DesignVector()
    print("\nBaseline design:")
    for name, value in asdict(baseline).items():
        if name in optimizer_variable_names():
            print(f"  {name}: {value}")

    print("\nBaseline score check:")
    baseline_objective = fitness(
        optimizer_array_from_design(baseline),
        record=False,
    )
    print(f"  objective: {baseline_objective}")
    print(f"  score: {-baseline_objective}")

    print("\nStarting preliminary differential-evolution run...")
    print(f"Expected DE evaluations: {expected_de_evaluations()}")
    optimization_start = time.perf_counter()
    result = run_preliminary_optimization()
    elapsed_seconds = time.perf_counter() - optimization_start
    run_summary_path = _write_run_summary(result, elapsed_seconds)
    print(
        f"Optimization finished in {elapsed_seconds:.1f} s "
        f"with {result.nfev} evaluations across {result.nit} generations."
    )
    print(f"Termination: {result.message}")

    best_design = design_from_optimizer_array(result.x)
    print_best_result(result, optimizer_variable_names())
    if EVALUATION_HISTORY:
        print_generation_summary()
    else:
        print(
            "\nPer-evaluation history is disabled for parallel SciPy workers. "
            "Set WORKERS = 1 for history.csv, score history, generation scores, "
            "and penalty history."
        )
    print_final_population_spread(result)
    print("\nBest design vector:")
    print(best_design.disp_vars(optimization_names=optimizer_variable_names()))
    visualization_paths = None
    if SAVE_BEST_DESIGN_VISUALIZATION:
        try:
            visualization_paths = save_best_design_visualization(best_design)
        except Exception as exc:
            print(f"Could not save best-design visualization: {exc}")

    history_path = OUTPUT_DIR / "history.csv"
    score_history_path = OUTPUT_DIR / "score_history.png"
    generation_scores_path = OUTPUT_DIR / "generation_score_distribution.png"
    penalty_history_path = OUTPUT_DIR / "penalty_history.png"
    population_path = OUTPUT_DIR / "final_population_scores.png"
    population_spread_path = OUTPUT_DIR / "final_population_spread.png"
    if EVALUATION_HISTORY:
        write_history_csv(history_path)
        plot_score_history(
            EVALUATION_HISTORY,
            save_path=str(score_history_path),
            show=False,
        )
        plot_generation_score_distribution(
            EVALUATION_HISTORY,
            save_path=str(generation_scores_path),
            show=False,
        )
        plot_penalty_history(
            EVALUATION_HISTORY,
            save_path=str(penalty_history_path),
            show=False,
        )
    plot_final_population(
        result,
        save_path=str(population_path),
        show=False,
    )
    plot_final_population_spread(
        result,
        optimizer_variable_names(),
        optimizer_bounds(),
        save_path=str(population_spread_path),
        show=False,
    )
    print("\nSaved optimization artifacts:")
    print(f"  run summary: {run_summary_path}")
    if EVALUATION_HISTORY:
        print(f"  history: {history_path}")
        print(f"  score history: {score_history_path}")
        print(f"  generation scores: {generation_scores_path}")
        print(f"  penalty history: {penalty_history_path}")
    print(f"  final population: {population_path}")
    print(f"  final population spread: {population_spread_path}")
    if visualization_paths is not None:
        print(f"  best design vector: {visualization_paths[0]}")
        print(f"  best design geometry: {visualization_paths[1]}")


if __name__ == "__main__":
    main_opt_test()
