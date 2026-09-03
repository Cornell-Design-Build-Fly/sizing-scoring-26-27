from scipy.optimize import differential_evolution, OptimizeResult, NonlinearConstraint
from src.vectors import DesignVector
from src.opt.score import total_score
from src.main import main

pd_constraint = NonlinearConstraint(
    lambda x: x[DesignVector.opt_names().index("prop_pitch_in")]
    / x[DesignVector.opt_names().index("prop_diameter_in")],
    0.3,   # minimum P/D
    0.9,   # maximum P/D
)

def fitness(x):
    dv = DesignVector.from_array(x)
    # Lightweight scoring-only estimate; the integrated optimizer uses src.main.
    container_mass_kg = dv.sensor_weight_kg + 0.5 * 0.45359237
    payload_mass_kg = (1 + round(dv.extra_shipping_containers)) * container_mass_kg
    score, _ = total_score(dv, 100.0, 60.0, 60.0, payload_mass_kg)
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
