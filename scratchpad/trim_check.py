"""How often does cruise trim fail at full throttle vs a reduced throttle?"""
from dataclasses import replace
import numpy as np
from src.opt.topline_opt import ToplineConfig, _initial_population, _optimizer_variable_names
from src.vectors import DesignVector, ParameterVector
from src.mech.main_mech import evaluate_mechanical_module
from src.main import resolved_aerodynamic_design_vector
from src.prop.main_prop import prop_main
from src.aero.main_aero import aero_main
from src.prop.continuous_prop_database import load_default_continuous_prop_database

config = ToplineConfig(popsize=25, seed=11)
pop = _initial_population(config)
names = _optimizer_variable_names(config)
db = load_default_continuous_prop_database(); pv = ParameterVector()
rng = np.random.default_rng(3)
rows = rng.choice(len(pop), size=min(200, len(pop)), replace=False)

counts = {t: 0 for t in (1.0, 0.85, 0.70)}
trimmed_sets = {t: set() for t in counts}
speeds = {t: [] for t in counts}
evaluated = 0
for i in rows:
    try:
        dv = DesignVector(**{n: float(pop[i][j]) for j, n in enumerate(names)})
        mech = evaluate_mechanical_module(dv, parameter_vector=pv)
        rdv = resolved_aerodynamic_design_vector(dv, mech)
        props = mech.for_mission("M2")
    except Exception:
        continue
    evaluated += 1
    for t in counts:
        d = replace(rdv, cruise_throttle=t)
        try:
            tc, ftf = prop_main(d, pv, mission=2, prop_database=db)
            aero = aero_main(design_vector=d, parameter_vector=pv, thrust_velocity=tc,
                             flight_time_fit=ftf, mission=2, cg=props.cg_m,
                             inertia_matrix=props.inertia_tensor_kg_m2, mass=props.total_mass_kg)
        except Exception:
            continue
        if aero.cruise_speed_mps is not None:
            counts[t] += 1
            trimmed_sets[t].add(int(i))
            speeds[t].append(float(aero.cruise_speed_mps))
print(f"{evaluated} designs built from the Sobol initial population")
for t in counts:
    v = np.array(speeds[t])
    print(f"  throttle {t:.2f}: trimmed {counts[t]:3d}/{evaluated}  "
          f"V median {np.median(v) if v.size else float('nan'):.1f}  max {v.max() if v.size else float('nan'):.1f}")
lost = (trimmed_sets[0.85] | trimmed_sets[0.70]) - trimmed_sets[1.0]
gained = trimmed_sets[1.0] - (trimmed_sets[0.85] | trimmed_sets[0.70])
print(f"trim only at reduced throttle (lost by pinning to 1.0): {len(lost)}")
print(f"trim only at full throttle: {len(gained)}")
