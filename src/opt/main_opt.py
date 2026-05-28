from scipy.optimize import differential_evolution, OptimizeResult
from src.vectors import DesignVector
from src.opt.score import total_score

def fitness(x):
    dv = DesignVector.from_array(x)
    score, _ = total_score(dv, 100.0, 100.0, 100.0)
    return -score  # Negative because we want to maximize the score

def run_optimization() -> OptimizeResult:
    results = differential_evolution(
        func=fitness,
        bounds=DesignVector.bounds(),
        workers=-1,
        updating="deferred",
    )
    return results