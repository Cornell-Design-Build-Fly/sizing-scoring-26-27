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
from functools import partial
from io import StringIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy
from scipy.optimize import NonlinearConstraint, OptimizeResult, differential_evolution
from scipy.stats import qmc
from tqdm.auto import tqdm

from src.main import main as score_aircraft
from src.opt.score import (
    DEFAULT_SCORING_REFERENCES,
    ScoringReferences,
    scoring_reference_values,
)
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
TARGET_EVALS_PER_SECOND = 80.0
TARGET_RUN_SECONDS = 3600.0
TOP_CANDIDATE_LIMIT = 500

PROP_DATABASE: ContinuousPropDatabase | None = None
PARAMETER_VECTOR = ParameterVector()
CALLBACK_HISTORY: list[dict] = []
CALLBACK_GENERATION = 0
RUN_START_SECONDS: float | None = None
CALLBACK_POPULATION_SIZE = 0
CALLBACK_TARGET_CANDIDATES = 0
CALLBACK_SCORE_BEST = True
PAYLOAD_ARCHIVE: dict[int, dict[str, Any]] = {}
TOP_CANDIDATE_ARCHIVE: dict[tuple[float, ...], dict[str, Any]] = {}
NICHE_HISTORY: list[dict[str, Any]] = []
ACTIVE_ISLAND = 0
PROGRESS_BAR = None


@dataclass(frozen=True)
class ToplineConfig:
    """Settings for the long, SciPy-managed top-line DE run."""

    workers: int = 8
    popsize: int = 25
    maxiter: int | None = 300
    target_seconds: float = TARGET_RUN_SECONDS
    assumed_evals_per_second: float = TARGET_EVALS_PER_SECOND
    init: str = "sobol"
    island_count: int = 6
    epoch_generations: int = 10
    niche_radius: float = 0.18
    restart_duplicate_islands: bool = True
    mutation: tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    tol: float = 1.0e-5
    atol: float = 0.0
    polish: bool = False
    seed: int = 20260808
    output_dir: Path = Path("data_dump") / "opt_topline"
    round_payload: bool = True
    continuous_lap_scoring: bool = True
    suppress_module_output: bool = True
    callback_score_best: bool = True
    save_best_visualization: bool = True
    scoring_references: ScoringReferences = DEFAULT_SCORING_REFERENCES


ACTIVE_TOPLINE_CONFIG: ToplineConfig | None = None


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
    names = DesignVector.opt_names()
    return float(x[names.index("prop_pitch_in")] / x[names.index("prop_diameter_in")])


PD_CONSTRAINT = NonlinearConstraint(_pd_ratio, PD_MIN, PD_MAX)


def _integrality_mask() -> np.ndarray:
    names = DesignVector.opt_names()
    return np.array(
        [name == "extra_shipping_containers" for name in names],
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
    generations = _resolved_maxiter(config)
    island_count = max(1, config.island_count)
    per_island = [
        generations // island_count + (index < generations % island_count)
        for index in range(island_count)
    ]
    epoch_runs = sum(
        math.ceil(count / config.epoch_generations) if count else 0
        for count in per_island
    )
    return (generations + epoch_runs) * _expected_population_size(config)


def _initial_population(
    config: ToplineConfig,
    *,
    seed: int | None = None,
) -> str | np.ndarray:
    """Build a neutral, reproducible Sobol population over feasible designs."""
    if config.init != "sobol":
        return config.init

    bounds = np.asarray(DesignVector.bounds(), dtype=float)
    population_size = _expected_population_size(config)
    exponent = int(math.log2(population_size))
    unit_population = qmc.Sobol(
        d=len(bounds), scramble=True, seed=config.seed if seed is None else seed
    ).random_base2(exponent)
    population = qmc.scale(unit_population, bounds[:, 0], bounds[:, 1])

    names = DesignVector.opt_names()
    container_index = names.index("extra_shipping_containers")
    diameter_index = names.index("prop_diameter_in")
    pitch_index = names.index("prop_pitch_in")

    low, high = bounds[container_index].astype(int)
    population[:, container_index] = np.minimum(
        low + (unit_population[:, container_index] * (high - low + 1)).astype(int),
        high,
    )

    # Project propeller pitch into the P/D-feasible interval so population
    # slots are spent comparing aircraft rather than known constraint failures.
    diameter = population[:, diameter_index]
    pitch_lower, pitch_upper = bounds[pitch_index]
    feasible_lower = np.maximum(pitch_lower, PD_MIN * diameter)
    feasible_upper = np.minimum(pitch_upper, PD_MAX * diameter)
    pitch_unit = unit_population[:, pitch_index]
    population[:, pitch_index] = (
        feasible_lower + pitch_unit * (feasible_upper - feasible_lower)
    )
    return population


def _objective(x: np.ndarray, *, config: ToplineConfig | None = None) -> float:
    """Top-level objective so SciPy can pickle it for worker processes."""

    config = ToplineConfig() if config is None else config
    if not np.all(np.isfinite(x)):
        return BAD_OBJECTIVE
    if not PD_MIN <= _pd_ratio(x) <= PD_MAX:
        return BAD_OBJECTIVE
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
                    continuous_lap_scoring=config.continuous_lap_scoring,
                    prop_database=PROP_DATABASE,
                    scoring_references=config.scoring_references,
                ),
            )
    except Exception:
        return BAD_OBJECTIVE

    if not math.isfinite(score):
        return BAD_OBJECTIVE

    return -float(score)


def _update_payload_archive(
    population: np.ndarray,
    population_energies: np.ndarray,
    generation: int,
    archive: dict[int, dict[str, Any]] | None = None,
) -> int:
    """Retain the highest-scoring design seen for each payload combination."""
    archive = PAYLOAD_ARCHIVE if archive is None else archive
    population = np.asarray(population, dtype=float)
    energies = np.asarray(population_energies, dtype=float)
    variable_names = DesignVector.opt_names()
    if population.ndim != 2 or population.shape[1] != len(variable_names):
        raise ValueError("Population has an unexpected shape.")
    if energies.shape != (population.shape[0],):
        raise ValueError("Population energies have an unexpected shape.")

    container_index = variable_names.index("extra_shipping_containers")
    updates = 0
    for population_index, (vector, objective) in enumerate(zip(population, energies)):
        objective = float(objective)
        if not math.isfinite(objective) or objective >= BAD_OBJECTIVE:
            continue

        containers = int(round(vector[container_index]))
        score = -objective
        key = containers
        previous = archive.get(key)
        if previous is not None and score <= previous["score"]:
            continue

        row = {
            "extra_shipping_containers": containers,
            "score": score,
            "objective": objective,
            "generation": int(generation),
            "population_index": int(population_index),
        }
        for name, value in zip(variable_names, vector):
            row[name] = float(value)
        row["extra_shipping_containers"] = containers
        archive[key] = row
        updates += 1

    return updates


def _update_top_candidate_archive(
    population: np.ndarray,
    population_energies: np.ndarray,
    generation: int,
    archive: dict[tuple[float, ...], dict[str, Any]] | None = None,
    limit: int = TOP_CANDIDATE_LIMIT,
) -> int:
    """Retain the best unique evaluated population candidates seen so far."""
    archive = TOP_CANDIDATE_ARCHIVE if archive is None else archive
    population = np.asarray(population, dtype=float)
    energies = np.asarray(population_energies, dtype=float)
    variable_names = DesignVector.opt_names()
    if population.ndim != 2 or population.shape[1] != len(variable_names):
        raise ValueError("Population has an unexpected shape.")
    if energies.shape != (population.shape[0],):
        raise ValueError("Population energies have an unexpected shape.")
    if limit <= 0:
        raise ValueError("Top-candidate archive limit must be positive.")

    updates = 0
    for vector, objective_value in zip(population, energies):
        objective = float(objective_value)
        if not math.isfinite(objective) or objective >= BAD_OBJECTIVE:
            continue
        key = tuple(float(value) for value in vector)
        previous = archive.get(key)
        if previous is not None and objective >= previous["objective"]:
            continue
        row = {
            "score": -objective,
            "objective": objective,
            "generation": int(generation),
        }
        row.update(
            {name: float(value) for name, value in zip(variable_names, vector)}
        )
        archive[key] = row
        updates += 1

    if len(archive) > limit:
        keep = sorted(
            archive.items(), key=lambda item: item[1]["objective"]
        )[:limit]
        archive.clear()
        archive.update(keep)
    return updates


def _callback(intermediate_result) -> bool:
    global CALLBACK_GENERATION

    CALLBACK_GENERATION += 1
    xk = np.asarray(intermediate_result.x, dtype=float)
    convergence = float(intermediate_result.convergence)
    _update_payload_archive(
        intermediate_result.population,
        intermediate_result.population_energies,
        CALLBACK_GENERATION,
    )
    _update_top_candidate_archive(
        intermediate_result.population,
        intermediate_result.population_energies,
        CALLBACK_GENERATION,
    )
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
    config = ACTIVE_TOPLINE_CONFIG or ToplineConfig()
    objective = _objective(xk, config=config) if CALLBACK_SCORE_BEST else math.nan
    score = -objective if math.isfinite(objective) else math.nan

    row = {
        "generation": CALLBACK_GENERATION,
        "island": ACTIVE_ISLAND,
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


def _differential_evolution_kwargs(
    config: ToplineConfig,
    *,
    maxiter: int | None = None,
    init: str | np.ndarray | None = None,
    seed: int | None = None,
) -> dict:
    parameters = inspect.signature(differential_evolution).parameters
    kwargs = {
        "func": partial(_objective, config=config),
        "bounds": DesignVector.bounds(),
        "constraints": (PD_CONSTRAINT,),
        "maxiter": _resolved_maxiter(config) if maxiter is None else maxiter,
        "popsize": config.popsize,
        "polish": config.polish,
        "updating": "deferred",
        "callback": _callback,
        "tol": config.tol,
        "atol": config.atol,
        "disp": False,
        "workers": config.workers,
        "init": _initial_population(config, seed=seed) if init is None else init,
        "mutation": config.mutation,
        "recombination": config.recombination,
    }

    if "integrality" in parameters:
        kwargs["integrality"] = _integrality_mask()
    if "rng" in parameters:
        kwargs["rng"] = np.random.default_rng(config.seed if seed is None else seed)
    elif "seed" in parameters:
        kwargs["seed"] = config.seed if seed is None else seed

    return kwargs


def _normalized_design_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return Euclidean distance after scaling every design variable to [0, 1]."""
    bounds = np.asarray(DesignVector.bounds(), dtype=float)
    spans = bounds[:, 1] - bounds[:, 0]
    return float(np.linalg.norm((np.asarray(left) - np.asarray(right)) / spans))


def _duplicate_islands(states: list[dict[str, Any]], radius: float) -> set[int]:
    """Identify weaker islands whose champions occupy an existing niche."""
    if radius <= 0.0:
        raise ValueError("niche_radius must be positive.")
    ordered = sorted(
        range(len(states)), key=lambda index: float(states[index]["best_objective"])
    )
    representatives: list[int] = []
    duplicates: set[int] = set()
    for index in ordered:
        champion = states[index]["best_vector"]
        if any(
            _normalized_design_distance(champion, states[other]["best_vector"])
            <= radius
            for other in representatives
        ):
            duplicates.add(index)
        else:
            representatives.append(index)
    return duplicates


def _combined_island_result(
    states: list[dict[str, Any]],
    *,
    nfev: int,
    nit: int,
) -> OptimizeResult:
    population = np.vstack([state["population"] for state in states])
    energies = np.concatenate([state["energies"] for state in states])
    seen = {tuple(float(value) for value in row) for row in population}
    archived_vectors = []
    archived_energies = []
    for row in _top_candidate_rows():
        vector = np.asarray(
            [row[name] for name in DesignVector.opt_names()], dtype=float
        )
        key = tuple(float(value) for value in vector)
        if key in seen:
            continue
        seen.add(key)
        archived_vectors.append(vector)
        archived_energies.append(float(row["objective"]))
    if archived_vectors:
        population = np.vstack([population, np.asarray(archived_vectors)])
        energies = np.concatenate([energies, np.asarray(archived_energies)])
    best_index = int(np.nanargmin(energies))
    return OptimizeResult(
        x=np.asarray(population[best_index], dtype=float),
        fun=float(energies[best_index]),
        population=population,
        population_energies=energies,
        nfev=int(nfev),
        nit=int(nit),
        success=False,
        message="Completed configured multimodal island generation budget.",
    )


def _run_niching_islands(config: ToplineConfig) -> OptimizeResult:
    """Run full-range DE islands and restart weaker duplicate niches."""
    global ACTIVE_ISLAND

    if config.island_count <= 0:
        raise ValueError("island_count must be positive.")
    if config.epoch_generations <= 0:
        raise ValueError("epoch_generations must be positive.")

    total_generations = _resolved_maxiter(config)
    generation_budgets = [
        total_generations // config.island_count
        + (index < total_generations % config.island_count)
        for index in range(config.island_count)
    ]
    states: list[dict[str, Any]] = []
    for island in range(config.island_count):
        population = np.asarray(
            _initial_population(config, seed=config.seed + 1009 * island),
            dtype=float,
        )
        states.append(
            {
                "population": population,
                "energies": np.full(len(population), np.inf),
                "remaining": generation_budgets[island],
                "completed": 0,
                "restarts": 0,
                "best_vector": population[0].copy(),
                "best_objective": math.inf,
            }
        )

    total_nfev = 0
    total_nit = 0
    epoch = 0
    while any(state["remaining"] > 0 for state in states):
        epoch += 1
        for island, state in enumerate(states):
            if state["remaining"] <= 0:
                continue
            ACTIVE_ISLAND = island
            chunk = min(config.epoch_generations, state["remaining"])
            run_seed = config.seed + 100_003 * epoch + 1009 * island
            result = differential_evolution(
                **_differential_evolution_kwargs(
                    config,
                    maxiter=chunk,
                    init=state["population"],
                    seed=run_seed,
                )
            )
            state["population"] = np.asarray(result.population, dtype=float)
            state["energies"] = np.asarray(result.population_energies, dtype=float)
            best_index = int(np.nanargmin(state["energies"]))
            state["best_vector"] = state["population"][best_index].copy()
            state["best_objective"] = float(state["energies"][best_index])
            state["remaining"] -= chunk
            state["completed"] += int(result.nit)
            total_nfev += int(result.nfev)
            total_nit += int(result.nit)

        duplicates = _duplicate_islands(states, config.niche_radius)
        for island, state in enumerate(states):
            NICHE_HISTORY.append(
                {
                    "epoch": epoch,
                    "island": island,
                    "score": -float(state["best_objective"]),
                    "objective": float(state["best_objective"]),
                    "duplicate": island in duplicates,
                    "restarts": int(state["restarts"]),
                    **{
                        name: float(value)
                        for name, value in zip(
                            DesignVector.opt_names(), state["best_vector"]
                        )
                    },
                }
            )

        if config.restart_duplicate_islands:
            for island in duplicates:
                state = states[island]
                if state["remaining"] <= 0:
                    continue
                state["restarts"] += 1
                state["population"] = np.asarray(
                    _initial_population(
                        config,
                        seed=config.seed + 1_000_003 * state["restarts"] + island,
                    ),
                    dtype=float,
                )
                state["energies"] = np.full(len(state["population"]), np.inf)

    ACTIVE_ISLAND = 0
    return _combined_island_result(states, nfev=total_nfev, nit=total_nit)


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


def _payload_archive_rows() -> list[dict[str, Any]]:
    return [PAYLOAD_ARCHIVE[key] for key in sorted(PAYLOAD_ARCHIVE)]


def _write_payload_archive(output_dir: Path) -> tuple[Path, Path]:
    """Write the best design observed for each simulator count."""
    rows = _payload_archive_rows()
    csv_path = output_dir / "payload_score_archive.csv"
    _write_csv(csv_path, rows)

    variable_names = DesignVector.opt_names()
    counts = np.asarray(
        [row["extra_shipping_containers"] for row in rows], dtype=int
    )
    design_vectors = np.asarray(
        [[row[name] for name in variable_names] for row in rows],
        dtype=float,
    ).reshape(-1, len(variable_names))
    npz_path = output_dir / "payload_score_archive.npz"
    np.savez_compressed(
        npz_path,
        extra_shipping_container_counts=counts,
        scores=np.asarray([row["score"] for row in rows], dtype=float),
        objectives=np.asarray([row["objective"] for row in rows], dtype=float),
        generations=np.asarray([row["generation"] for row in rows], dtype=int),
        design_vectors=design_vectors,
        variable_names=np.asarray(variable_names),
    )
    return csv_path, npz_path


def _top_candidate_rows() -> list[dict[str, Any]]:
    """Return archived candidates ordered from highest to lowest score."""
    return sorted(TOP_CANDIDATE_ARCHIVE.values(), key=lambda row: row["objective"])


def _write_top_candidate_archive(output_dir: Path) -> tuple[Path, Path]:
    """Save the candidate data used by the top-500 parallel plot."""
    rows = _top_candidate_rows()
    csv_path = output_dir / "top_500_candidates.csv"
    _write_csv(csv_path, rows)
    variable_names = DesignVector.opt_names()
    population = np.asarray(
        [[row[name] for name in variable_names] for row in rows], dtype=float
    ).reshape(-1, len(variable_names))
    objectives = np.asarray([row["objective"] for row in rows], dtype=float)
    npz_path = output_dir / "top_500_candidates.npz"
    np.savez_compressed(
        npz_path,
        population=population,
        population_energies=objectives,
        scores=-objectives,
        generations=np.asarray([row["generation"] for row in rows], dtype=int),
        variable_names=np.asarray(variable_names),
    )
    return csv_path, npz_path


def _official_population_scores(result, config: ToplineConfig) -> np.ndarray:
    """Re-evaluate finalists using official integer-lap scoring."""
    scores = np.full(len(result.population), -BAD_OBJECTIVE, dtype=float)
    for index, x in enumerate(np.asarray(result.population, dtype=float)):
        try:
            with _module_output_context(config):
                score, _ = cast(
                    tuple[float, list[float]],
                    score_aircraft(
                        DesignVector.from_array(x),
                        PARAMETER_VECTOR,
                        disp_res=False,
                        round_payload=config.round_payload,
                        continuous_lap_scoring=False,
                        prop_database=PROP_DATABASE,
                    ),
                )
            if math.isfinite(score):
                scores[index] = float(score)
        except Exception:
            pass
    return scores


def _write_final_population(result, official_scores: np.ndarray, output_dir: Path) -> Path:
    population = np.asarray(result.population, dtype=float)
    energies = np.asarray(result.population_energies, dtype=float)
    optimization_scores = -energies
    order = np.lexsort((-optimization_scores, -official_scores))
    rows = []
    for rank, index in enumerate(order, start=1):
        objective = float(energies[index])
        row = {
            "rank": rank,
            "population_index": int(index),
            "optimization_objective": objective,
            "optimization_score": -objective,
            "official_score": float(official_scores[index]),
            "finite": bool(np.isfinite(objective)),
        }
        for name, value in zip(DesignVector.opt_names(), population[index]):
            row[name] = float(value)
        rows.append(row)

    path = output_dir / "final_population.csv"
    _write_csv(path, rows)
    return path


def _write_niche_champions(
    result,
    official_scores: np.ndarray,
    config: ToplineConfig,
    output_dir: Path,
) -> Path:
    """Save strong officially scored designs that occupy distinct niches."""
    population = np.asarray(result.population, dtype=float)
    optimization_scores = -np.asarray(result.population_energies, dtype=float)
    order = np.lexsort((-optimization_scores, -official_scores))
    selected: list[int] = []
    for index_value in order:
        index = int(index_value)
        if official_scores[index] <= -BAD_OBJECTIVE:
            continue
        if any(
            _normalized_design_distance(population[index], population[other])
            <= config.niche_radius
            for other in selected
        ):
            continue
        selected.append(index)
        if len(selected) >= max(config.island_count, 1):
            break

    rows = []
    for rank, index in enumerate(selected, start=1):
        row = {
            "niche_rank": rank,
            "population_index": index,
            "official_score": float(official_scores[index]),
            "optimization_score": float(optimization_scores[index]),
        }
        row.update(
            {
                name: float(value)
                for name, value in zip(DesignVector.opt_names(), population[index])
            }
        )
        rows.append(row)
    path = output_dir / "niche_champions.csv"
    _write_csv(path, rows)
    return path


def _best_design_report(
    result,
    official_best_index: int,
    config: ToplineConfig,
    output_dir: Path,
) -> dict:
    best_vector = np.asarray(result.population[official_best_index], dtype=float)
    best_design = DesignVector.from_array(best_vector)
    scoring_design = replace(
        best_design,
        extra_shipping_containers=round(best_design.extra_shipping_containers)
        if config.round_payload
        else best_design.extra_shipping_containers,
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
            continuous_lap_scoring=False,
            scoring_references=config.scoring_references,
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
        "official_score": float(score),
        "optimization_score": float(-result.population_energies[official_best_index]),
        "optimization_objective": float(result.population_energies[official_best_index]),
        "breakdown": {
            "ground": float(breakdown[0]),
            "m1": float(breakdown[1]),
            "m2": float(breakdown[2]),
            "m3": float(breakdown[3]),
        },
        "optimizer_vector": asdict(best_design),
        "scoring_vector": asdict(scoring_design),
        "resolved_vector": asdict(resolved_design),
        "scoring_references": scoring_reference_values(config.scoring_references),
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
        "best_optimization_objective": float(result.fun),
        "best_optimization_score": -float(result.fun),
        "best_official_score": best_report["official_score"],
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
        "scoring_references": scoring_reference_values(config.scoring_references),
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
        "top_500_parallel_coordinates": output_dir / "top_500_parallel_coordinates.png",
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
    top_rows = _top_candidate_rows()
    if top_rows:
        top_population = np.asarray(
            [[row[name] for name in variable_names] for row in top_rows],
            dtype=float,
        )
        top_energies = np.asarray(
            [row["objective"] for row in top_rows], dtype=float
        )
        plot_population_parallel_coordinates(
            top_population,
            top_energies,
            variable_names,
            bounds,
            top_fraction=1.0,
            title=f"Top {len(top_rows)} Evaluated Population Candidates",
            save_path=str(paths["top_500_parallel_coordinates"]),
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
    global PAYLOAD_ARCHIVE
    global TOP_CANDIDATE_ARCHIVE
    global NICHE_HISTORY
    global PROGRESS_BAR
    global RUN_START_SECONDS
    global ACTIVE_TOPLINE_CONFIG

    config = config or ToplineConfig()
    ACTIVE_TOPLINE_CONFIG = config
    output_dir = _timestamped_output_dir(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    CALLBACK_HISTORY = []
    CALLBACK_GENERATION = 0
    CALLBACK_POPULATION_SIZE = _expected_population_size(config)
    CALLBACK_TARGET_CANDIDATES = _target_candidate_count(config)
    CALLBACK_SCORE_BEST = config.callback_score_best
    PAYLOAD_ARCHIVE = {}
    TOP_CANDIDATE_ARCHIVE = {}
    NICHE_HISTORY = []

    print("Top-line differential evolution run")
    print(f"  output: {output_dir}")
    print(f"  workers: {config.workers}")
    print(f"  variables: {len(DesignVector.bounds())}")
    print(f"  population size: {CALLBACK_POPULATION_SIZE}")
    print(f"  maxiter: {_resolved_maxiter(config)}")
    print(f"  islands: {config.island_count}")
    print(f"  epoch generations: {config.epoch_generations}")
    print(f"  niche radius: {config.niche_radius}")
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
                    if key not in {"func", "callback", "constraints", "rng", "init"}
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
            result = _run_niching_islands(config)
        finally:
            if PROGRESS_BAR.n < CALLBACK_TARGET_CANDIDATES:
                PROGRESS_BAR.update(CALLBACK_TARGET_CANDIDATES - PROGRESS_BAR.n)
            PROGRESS_BAR = None
    elapsed_seconds = time.perf_counter() - RUN_START_SECONDS

    _update_payload_archive(
        result.population,
        result.population_energies,
        int(result.nit),
    )
    _update_top_candidate_archive(
        result.population,
        result.population_energies,
        int(result.nit),
    )

    print_best_result(result, DesignVector.opt_names())
    print("Re-ranking final population with official integer-lap scoring...", flush=True)
    official_scores = _official_population_scores(result, config)
    optimization_scores = -np.asarray(result.population_energies, dtype=float)
    official_order = np.lexsort((-optimization_scores, -official_scores))
    official_best_index = int(official_order[0])
    best_report = _best_design_report(
        result, official_best_index, config, output_dir
    )
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
    niche_history_path = output_dir / "niche_history.csv"
    _write_csv(niche_history_path, NICHE_HISTORY)
    population_path = _write_final_population(result, official_scores, output_dir)
    niche_champions_path = _write_niche_champions(
        result, official_scores, config, output_dir
    )
    payload_csv_path, payload_arrays_path = _write_payload_archive(output_dir)
    top_csv_path, top_arrays_path = _write_top_candidate_archive(output_dir)
    arrays_path = output_dir / "result_arrays.npz"
    np.savez(
        arrays_path,
        x=np.asarray(result.x, dtype=float),
        population=np.asarray(result.population, dtype=float),
        population_energies=np.asarray(result.population_energies, dtype=float),
        official_scores=official_scores,
        official_best_x=np.asarray(result.population[official_best_index], dtype=float),
    )
    plot_paths = _save_plots(result, output_dir)

    print("\nSaved top-line artifacts:")
    print(f"  summary: {summary_path}")
    print(f"  generation history: {generation_path}")
    print(f"  niche history: {niche_history_path}")
    print(f"  niche champions: {niche_champions_path}")
    print(f"  final population: {population_path}")
    print(f"  payload archive: {payload_csv_path}")
    print(f"  payload archive arrays: {payload_arrays_path}")
    print(f"  top-500 candidate archive: {top_csv_path}")
    print(f"  top-500 candidate arrays: {top_arrays_path}")
    print(f"  result arrays: {arrays_path}")
    print(f"  best design report: {output_dir / 'best_design_report.json'}")
    for path in plot_paths:
        print(f"  plot: {path}")

    return result


def main() -> None:
    run_topline_optimization()


if __name__ == "__main__":
    main()
