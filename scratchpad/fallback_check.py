"""How many designs does the trim-throttle fallback recover?"""
import numpy as np
import src.main as M
from src.opt.topline_opt import ToplineConfig, _initial_population, _optimizer_variable_names
from src.vectors import DesignVector, ParameterVector
from src.prop.continuous_prop_database import load_default_continuous_prop_database

config = ToplineConfig(popsize=25, seed=11)
pop = _initial_population(config); names = _optimizer_variable_names(config)
db = load_default_continuous_prop_database(); pv = ParameterVector()
rng = np.random.default_rng(3)
rows = rng.choice(len(pop), size=200, replace=False)

def evaluate(throttles):
    M.CRUISE_TRIM_THROTTLES = throttles
    trimmed, scores = 0, []
    for i in rows:
        try:
            dv = DesignVector(**{n: float(pop[i][j]) for j, n in enumerate(names)})
            s, bd, det = M.main(dv, pv, prop_database=db, return_details=True, continuous_lap_scoring=True)
        except Exception:
            continue
        if det["penalty_aero_m2"] < 10.0 and det["penalty_aero_m3"] < 10.0:
            trimmed += 1
            scores.append(s)
    return trimmed, np.array(scores)

for label, thr in [("throttle 1.0 only", (1.0,)), ("with fallback", (1.0, 0.85, 0.70, 0.55))]:
    n, s = evaluate(thr)
    print(f"{label:20s}: {n:3d}/200 designs trim on both M2 and M3; "
          f"best score {s.max() if s.size else float('nan'):.3f}, median {np.median(s) if s.size else float('nan'):.2f}")
