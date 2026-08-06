from __future__ import annotations

import math
import csv
import os
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from tqdm.auto import tqdm

from src.main import main
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
from src.vectors import DesignVector, ParameterVector


BAD_OBJECTIVE = 1.0e6
ROUND_PAYLOAD = False
MAXITER = 100
POPSIZE = 15
WORKERS = 10
OUTPUT_DIR = Path("data_dump") / "opt_preliminary"
SUPPRESS_MODULE_OUTPUT = True
REPORT_REJECTIONS = False
REJECTION_DETAIL_CHARS = 220
EVALUATION_HISTORY: list[dict] = []
PROGRESS_BAR = None
BEST_SCORE = -math.inf
PROP_DATABASE: ContinuousPropDatabase | None = None
PARAMETER_VECTOR = ParameterVector()
COMPLETED_GENERATIONS = 0


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
    for index, name in enumerate(DesignVector.opt_names()):
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

    return POPSIZE * len(DesignVector.bounds())


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
        design_vector = DesignVector.from_array(x)
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
                f"convergence={convergence:.3g}"
            )
        else:
            _notify(
                f"[opt] generation complete: evaluations={len(EVALUATION_HISTORY)}, "
                f"no successful designs yet, rejected={rejected_count}, "
                f"convergence={convergence:.3g}"
            )
    else:
        completed_evaluations = min(
            expected_de_evaluations(),
            de_population_size() * (COMPLETED_GENERATIONS + 1),
        )
        sync_progress_bar_to_count(completed_evaluations)
        _notify(
            f"[opt] generation complete: approx_evaluations={completed_evaluations}, "
            f"convergence={convergence:.3g}"
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
        "bounds": DesignVector.bounds(),
        "maxiter": MAXITER,
        "popsize": POPSIZE,
        "polish": False,
        "updating": de_updating_mode(),
        "callback": progress_callback,
        "disp": False,
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

    global BEST_SCORE, COMPLETED_GENERATIONS, PROGRESS_BAR

    worker_count = resolved_worker_count()
    EVALUATION_HISTORY.clear()
    BEST_SCORE = -math.inf
    COMPLETED_GENERATIONS = 0

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
    bounds = np.asarray(DesignVector.bounds(), dtype=float)
    spans = bounds[:, 1] - bounds[:, 0]
    normalized_std = np.std(population, axis=0) / spans

    print("\nFinal population spread:")
    print("  variable | std / range")
    for name, spread in sorted(
        zip(DesignVector.opt_names(), normalized_std),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {name}: {spread:.4f}")


def main_opt_test() -> None:
    global PROP_DATABASE

    print("Loading prop database...")
    PROP_DATABASE = _load_prop_database()

    print("Optimization variables:")
    for name, bounds in zip(DesignVector.opt_names(), DesignVector.bounds()):
        print(f"  {name}: {bounds}")

    baseline = DesignVector()
    print("\nBaseline design:")
    for name, value in asdict(baseline).items():
        if name in DesignVector.opt_names():
            print(f"  {name}: {value}")

    print("\nBaseline score check:")
    baseline_objective = fitness(baseline.to_array(), record=False)
    print(f"  objective: {baseline_objective}")
    print(f"  score: {-baseline_objective}")

    print("\nStarting preliminary differential-evolution run...")
    print(f"Expected DE evaluations: {expected_de_evaluations()}")
    result = run_preliminary_optimization()

    best_design = DesignVector.from_array(result.x)
    print_best_result(result, DesignVector.opt_names())
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
    print(best_design.disp_vars())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
        DesignVector.opt_names(),
        DesignVector.bounds(),
        save_path=str(population_spread_path),
        show=False,
    )
    print("\nSaved optimization artifacts:")
    if EVALUATION_HISTORY:
        print(f"  history: {history_path}")
        print(f"  score history: {score_history_path}")
        print(f"  generation scores: {generation_scores_path}")
        print(f"  penalty history: {penalty_history_path}")
    print(f"  final population: {population_path}")
    print(f"  final population spread: {population_spread_path}")


if __name__ == "__main__":
    main_opt_test()
