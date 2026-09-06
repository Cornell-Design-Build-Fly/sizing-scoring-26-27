"""End-to-end optimizer smoke: a tiny DE run through the real topline path."""
from pathlib import Path
from src.opt import topline_opt as T

config = T.ToplineConfig(
    popsize=2, maxiter=2, island_count=1, epoch_generations=1,
    workers=1, seed=7,
    output_dir=Path("/private/tmp/claude-501/-Users-ishanroy-Documents-GitHub-sizing-scoring-26-27/caf3f026-3662-48fe-b05c-2b92a588558b/scratchpad/topline_smoke"),
    save_best_visualization=False, continuous_lap_scoring=True,
)
names = T._optimizer_variable_names(config)
print("variables:", len(names))
print(names)
result = T.run_topline_optimization(config)
print("done; best objective:", getattr(result, "fun", None))
