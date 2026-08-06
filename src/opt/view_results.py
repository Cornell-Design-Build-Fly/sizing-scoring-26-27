import matplotlib.pyplot as plt
import numpy as np

def score_distribution(breakdown: list[float]) -> None:
    """Plots the score distribution for each mission."""
    missions = ["Ground Mission", "Mission 1", "Mission 2", "Mission 3"]
    plt.bar(missions, breakdown)
    plt.ylabel("Score")
    plt.title("Score Distribution by Mission")
    plt.show()


def plot_score_history(
    history: list[dict],
    *,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot raw and best-so-far scores from objective evaluations."""

    successful = [entry for entry in history if entry["status"] == "ok"]
    if not successful:
        print("[opt] No successful evaluations to plot.")
        return

    evaluations = [entry["evaluation"] for entry in successful]
    scores = [entry["score"] for entry in successful]
    best_scores = np.maximum.accumulate(scores)
    generation_starts = []
    seen_generations = set()
    for entry in history:
        generation = entry.get("generation")
        if generation is None or generation in seen_generations:
            continue
        seen_generations.add(generation)
        generation_starts.append((entry["evaluation"], generation))

    plt.figure()
    plt.plot(evaluations, scores, ".", label="Trial evaluation score")
    plt.plot(evaluations, best_scores, "-", label="Best score so far")
    for evaluation, generation in generation_starts:
        if generation == 0:
            continue
        plt.axvline(evaluation, color="0.75", linewidth=0.8, linestyle="--")
    plt.xlabel("Objective evaluation")
    plt.ylabel("Score")
    plt.title("Optimization Score History")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_final_population(
    result,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot the final differential-evolution population score distribution."""

    if not hasattr(result, "population_energies"):
        print("[opt] Result does not include population energies.")
        return

    scores = -np.asarray(result.population_energies, dtype=float)
    finite_scores = scores[np.isfinite(scores)]
    if len(finite_scores) == 0:
        print("[opt] No finite final-population scores to plot.")
        return

    plt.figure()
    plt.hist(finite_scores, bins=min(12, max(3, len(finite_scores))))
    plt.xlabel("Score")
    plt.ylabel("Population count")
    plt.title("Final Population Score Distribution")
    plt.grid(True, axis="y", alpha=0.3)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_final_population_spread(
    result,
    variable_names: list[str],
    bounds: list[tuple[float, float]],
    *,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot normalized final-population spread for each design variable."""

    if not hasattr(result, "population"):
        print("[opt] Result does not include final population vectors.")
        return

    population = np.asarray(result.population, dtype=float)
    if population.ndim != 2 or population.shape[1] != len(variable_names):
        print("[opt] Final population has an unexpected shape.")
        return

    bounds_array = np.asarray(bounds, dtype=float)
    spans = bounds_array[:, 1] - bounds_array[:, 0]
    normalized_std = np.std(population, axis=0) / spans

    order = np.argsort(normalized_std)
    y = np.arange(len(variable_names))

    plt.figure(figsize=(8, max(4, 0.35 * len(variable_names))))
    plt.barh(y, normalized_std[order])
    plt.yticks(y, [variable_names[index] for index in order])
    plt.xlabel("Final population std / variable range")
    plt.title("Final Population Design Spread")
    plt.grid(True, axis="x", alpha=0.3)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_generation_score_distribution(
    history: list[dict],
    *,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot trial-score distributions by generation."""

    successful = [entry for entry in history if entry["status"] == "ok"]
    if not successful:
        print("[opt] No successful evaluations to plot by generation.")
        return

    generations = sorted({entry["generation"] for entry in successful})
    score_groups = [
        [entry["score"] for entry in successful if entry["generation"] == generation]
        for generation in generations
    ]

    plt.figure()
    plt.boxplot(score_groups, labels=[str(generation) for generation in generations])
    plt.xlabel("Generation")
    plt.ylabel("Trial evaluation score")
    plt.title("Trial Score Distribution by Generation")
    plt.grid(True, axis="y", alpha=0.3)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_penalty_history(
    history: list[dict],
    *,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot implied total penalty from objective evaluations."""

    successful = [
        entry
        for entry in history
        if entry["status"] == "ok" and np.isfinite(entry.get("implied_penalty", np.nan))
    ]
    if not successful:
        print("[opt] No successful evaluations with penalties to plot.")
        return

    evaluations = [entry["evaluation"] for entry in successful]
    penalties = [entry["implied_penalty"] for entry in successful]
    generation_starts = []
    seen_generations = set()
    for entry in history:
        generation = entry.get("generation")
        if generation is None or generation in seen_generations:
            continue
        seen_generations.add(generation)
        generation_starts.append((entry["evaluation"], generation))

    plt.figure()
    plt.plot(evaluations, penalties, ".", label="Implied total penalty")
    for evaluation, generation in generation_starts:
        if generation == 0:
            continue
        plt.axvline(evaluation, color="0.75", linewidth=0.8, linestyle="--")
    plt.xlabel("Objective evaluation")
    plt.ylabel("Penalty")
    plt.title("Optimization Penalty History")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def print_best_result(result, variable_names: list[str]) -> None:
    """Print a compact summary of a SciPy differential-evolution result."""

    print("\nOptimization result:")
    print(f"  success: {result.success}")
    print(f"  message: {result.message}")
    print(f"  best objective: {result.fun}")
    print(f"  best score: {-result.fun}")
    print("  best variables:")
    for name, value in zip(variable_names, result.x):
        print(f"    {name}: {value}")
