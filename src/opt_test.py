from __future__ import annotations

import math
import csv
from dataclasses import asdict
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
from src.vectors import DesignVector, ParameterVector


BAD_OBJECTIVE = 1.0e6
ROUND_PAYLOAD = False
MAXITER = 20
POPSIZE = 5
OUTPUT_DIR = Path("data_dump") / "opt_preliminary"
EVALUATION_HISTORY: list[dict] = []
PROGRESS_BAR = None
BEST_SCORE = -math.inf


def _history_row(
    *,
    x: np.ndarray,
    objective: float,
    score: float,
    breakdown: list[float] | None,
    status: str,
    message: str = "",
) -> dict:
    evaluation = len(EVALUATION_HISTORY) + 1
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

    try:
        design_vector = DesignVector.from_array(x)
        score, breakdown = main(
            design_vector,
            ParameterVector(),
            disp_res=False,
            round_payload=ROUND_PAYLOAD,
        )
    except Exception as exc:
        if record:
            _notify(f"[opt] rejected evaluation {len(EVALUATION_HISTORY) + 1}: {exc}")
            _record_evaluation(
                _history_row(
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

    del xk
    successful = [entry for entry in EVALUATION_HISTORY if entry["status"] == "ok"]
    if successful:
        best = max(successful, key=lambda entry: entry["score"])
        _notify(
            f"[opt] generation complete: evaluations={len(EVALUATION_HISTORY)}, "
            f"best_score={best['score']:.6g}, convergence={convergence:.3g}"
        )
    else:
        _notify(
            f"[opt] generation complete: evaluations={len(EVALUATION_HISTORY)}, "
            f"no successful designs yet, convergence={convergence:.3g}"
        )
    return False


def run_preliminary_optimization():
    """Run a tiny DE pass through the full scoring stack."""

    global BEST_SCORE, PROGRESS_BAR

    EVALUATION_HISTORY.clear()
    BEST_SCORE = -math.inf

    with tqdm(
        total=expected_de_evaluations(),
        desc="Optimization evaluations",
        unit="eval",
    ) as progress_bar:
        PROGRESS_BAR = progress_bar
        try:
            return differential_evolution(
                func=fitness,
                bounds=DesignVector.bounds(),
                maxiter=MAXITER,
                popsize=POPSIZE,
                polish=False,
                workers=1,
                updating="immediate",
                callback=progress_callback,
                disp=False,
            )
        finally:
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
    print_generation_summary()
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
    print(f"  history: {history_path}")
    print(f"  score history: {score_history_path}")
    print(f"  generation scores: {generation_scores_path}")
    print(f"  penalty history: {penalty_history_path}")
    print(f"  final population: {population_path}")
    print(f"  final population spread: {population_spread_path}")


if __name__ == "__main__":
    main_opt_test()
