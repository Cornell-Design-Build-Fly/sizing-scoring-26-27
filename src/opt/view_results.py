import matplotlib

matplotlib.use("Agg")
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
    plt.boxplot(score_groups, tick_labels=[str(generation) for generation in generations])
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


def plot_generation_convergence(
    history: list[dict],
    *,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot callback-level best score and SciPy convergence metric."""

    if not history:
        print("[opt] No generation history to plot.")
        return

    generations = np.array([row["generation"] for row in history], dtype=float)
    scores = np.array([row["best_score"] for row in history], dtype=float)
    convergence = np.array([row["convergence"] for row in history], dtype=float)

    fig, score_axis = plt.subplots()
    score_axis.plot(generations, scores, "-", color="tab:blue", label="Best score")
    score_axis.set_xlabel("Generation")
    score_axis.set_ylabel("Best score", color="tab:blue")
    score_axis.tick_params(axis="y", labelcolor="tab:blue")
    score_axis.grid(True, alpha=0.3)

    convergence_axis = score_axis.twinx()
    convergence_axis.plot(
        generations,
        convergence,
        "-",
        color="tab:orange",
        alpha=0.75,
        label="SciPy convergence",
    )
    convergence_axis.set_ylabel("SciPy convergence", color="tab:orange")
    convergence_axis.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle("Top-Line DE Convergence")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_population_variable_histograms(
    population,
    variable_names: list[str],
    bounds: list[tuple[float, float]],
    *,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot final-population marginal distributions for every variable."""

    population = np.asarray(population, dtype=float)
    if population.ndim != 2 or population.shape[1] != len(variable_names):
        print("[opt] Final population has an unexpected shape.")
        return

    column_count = 3
    row_count = int(np.ceil(len(variable_names) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(12, max(3, 2.4 * row_count)),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for index, name in enumerate(variable_names):
        axis = axes[index]
        lower, upper = bounds[index]
        axis.hist(population[:, index], bins=24, color="0.35", edgecolor="white")
        axis.axvline(lower, color="0.75", linewidth=0.8)
        axis.axvline(upper, color="0.75", linewidth=0.8)
        axis.set_title(name)
        axis.grid(True, axis="y", alpha=0.25)

    for axis in axes[len(variable_names):]:
        axis.axis("off")

    fig.suptitle("Final Population Variable Distributions")
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_population_score_vs_variables(
    population,
    population_energies,
    variable_names: list[str],
    *,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot final-population score sensitivity scatter for each variable."""

    population = np.asarray(population, dtype=float)
    scores = -np.asarray(population_energies, dtype=float)
    finite_mask = np.isfinite(scores)
    if population.ndim != 2 or population.shape[1] != len(variable_names):
        print("[opt] Final population has an unexpected shape.")
        return
    if not np.any(finite_mask):
        print("[opt] No finite final-population scores to plot.")
        return

    population = population[finite_mask]
    scores = scores[finite_mask]

    column_count = 3
    row_count = int(np.ceil(len(variable_names) / column_count))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(12, max(3, 2.4 * row_count)),
    )
    axes = np.atleast_1d(axes).ravel()

    for index, name in enumerate(variable_names):
        axis = axes[index]
        scatter = axis.scatter(
            population[:, index],
            scores,
            c=scores,
            cmap="viridis",
            s=12,
            alpha=0.75,
        )
        axis.set_title(name)
        axis.set_ylabel("Score")
        axis.grid(True, alpha=0.25)

    for axis in axes[len(variable_names):]:
        axis.axis("off")

    fig.colorbar(scatter, ax=axes[:len(variable_names)], label="Score")
    fig.suptitle("Final Population Score vs Variables")
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_population_score_correlations(
    population,
    population_energies,
    variable_names: list[str],
    *,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot simple final-population Pearson correlations with score."""

    population = np.asarray(population, dtype=float)
    scores = -np.asarray(population_energies, dtype=float)
    finite_mask = np.isfinite(scores)
    if population.ndim != 2 or population.shape[1] != len(variable_names):
        print("[opt] Final population has an unexpected shape.")
        return
    if np.count_nonzero(finite_mask) < 3:
        print("[opt] Not enough finite final-population scores for correlations.")
        return

    population = population[finite_mask]
    scores = scores[finite_mask]
    correlations = []
    for index in range(population.shape[1]):
        variable = population[:, index]
        if np.std(variable) <= 0.0 or np.std(scores) <= 0.0:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(variable, scores)[0, 1]))

    correlations = np.asarray(correlations, dtype=float)
    order = np.argsort(np.abs(correlations))
    y = np.arange(len(variable_names))

    plt.figure(figsize=(8, max(4, 0.35 * len(variable_names))))
    plt.barh(y, correlations[order])
    plt.yticks(y, [variable_names[index] for index in order])
    plt.xlabel("Pearson correlation with final-population score")
    plt.title("Final Population Score Correlations")
    plt.axvline(0.0, color="0.2", linewidth=0.8)
    plt.grid(True, axis="x", alpha=0.3)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_population_parallel_coordinates(
    population,
    population_energies,
    variable_names: list[str],
    bounds: list[tuple[float, float]],
    *,
    top_fraction: float = 0.15,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot normalized final-population coordinates, emphasizing top designs."""

    population = np.asarray(population, dtype=float)
    energies = np.asarray(population_energies, dtype=float)
    scores = -energies
    finite_mask = np.isfinite(scores)
    if population.ndim != 2 or population.shape[1] != len(variable_names):
        print("[opt] Final population has an unexpected shape.")
        return
    if not np.any(finite_mask):
        print("[opt] No finite final-population scores for parallel plot.")
        return

    bounds_array = np.asarray(bounds, dtype=float)
    spans = bounds_array[:, 1] - bounds_array[:, 0]
    normalized = (population - bounds_array[:, 0]) / spans
    x = np.arange(len(variable_names))

    finite_indices = np.flatnonzero(finite_mask)
    finite_scores = scores[finite_indices]
    top_count = max(1, int(np.ceil(len(finite_indices) * top_fraction)))
    top_indices = finite_indices[np.argsort(finite_scores)[-top_count:]]

    plt.figure(figsize=(12, 6))
    for row in normalized[finite_mask]:
        plt.plot(x, row, color="0.75", alpha=0.08, linewidth=0.8)
    for index in top_indices:
        plt.plot(x, normalized[index], color="tab:blue", alpha=0.45, linewidth=1.2)

    best_index = int(np.nanargmin(energies))
    plt.plot(x, normalized[best_index], color="tab:red", linewidth=2.0, label="Best")
    plt.xticks(x, variable_names, rotation=45, ha="right")
    plt.ylabel("Normalized variable value")
    plt.title("Final Population Parallel Coordinates")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
