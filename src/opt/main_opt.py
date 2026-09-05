from scipy.optimize import differential_evolution, OptimizeResult, NonlinearConstraint
from src.vectors import DesignVector
from src.opt.score import total_score
from src.main import main

pd_constraint = NonlinearConstraint(
    lambda x: x[9] / x[8],
    0.3,   # minimum P/D
    0.9,   # maximum P/D
)

def fitness(x):
    dv = DesignVector.from_array(x)
    score, _ = total_score(dv, 100.0, 100.0, 100.0)
    return -score  # Negative because we want to maximize the score

def run_optimization() -> OptimizeResult:
    results = differential_evolution(
        func=fitness,
        bounds=DesignVector.bounds(),
        constraints=(pd_constraint,),
        workers=-1,
        updating="deferred",
    )
    return results