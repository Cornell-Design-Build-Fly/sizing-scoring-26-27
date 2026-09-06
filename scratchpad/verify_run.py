"""Does the reworked model actually find a better airplane?"""
from pathlib import Path
from src.opt import topline_opt as T

OUT = Path("data_dump/opt_flaps_energy")


def build_config():
    return T.ToplineConfig(
        popsize=12, maxiter=40, island_count=3, epoch_generations=10,
        workers=8, seed=20260906, output_dir=OUT,
        save_best_visualization=False, continuous_lap_scoring=False,
    )


if __name__ == "__main__":
    config = build_config()
    print("population size:", T._expected_population_size(config), flush=True)
    result = T.run_topline_optimization(config)
    print("DONE best objective:", getattr(result, "fun", None), flush=True)
