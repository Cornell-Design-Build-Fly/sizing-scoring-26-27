"""Best-shared vs best-split propeller, scanning both propellers."""
import json
import numpy as np
from src.vectors import DesignVector, ParameterVector
from src.main import main
from src.prop.continuous_prop_database import load_default_continuous_prop_database

rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]; names = DesignVector.opt_names()
base = {n: float(v[n]) if n in v else float(v[n.replace("mission3_prop_","prop_")]) for n in names}
db = load_default_continuous_prop_database(); pv = ParameterVector()

GRID = [(D, float(P)) for D in [10,12,14,16,18,20,22]
        for P in np.arange(0.4*D, 0.8*D+0.01, 0.1*D) if 4.0 <= P <= 18.0]

def scan(label, **fixed):
    kwbase = dict(base); kwbase.update(fixed)
    shared, split_m12, split_m3 = [], [], []
    for D, P in GRID:
        # shared: both props the same
        kw = dict(kwbase); kw.update(prop_diameter_in=float(D), prop_pitch_in=P,
                                     mission3_prop_diameter_in=float(D), mission3_prop_pitch_in=P)
        s, bd, det = main(DesignVector(**kw), pv, prop_database=db, return_details=True, continuous_lap_scoring=True)
        shared.append((s, D, P, bd))
        # M1/M2 prop varied, M3 prop held at the design's own
        kw = dict(kwbase); kw.update(prop_diameter_in=float(D), prop_pitch_in=P)
        s2, bd2, _ = main(DesignVector(**kw), pv, prop_database=db, return_details=True, continuous_lap_scoring=True)
        split_m12.append((bd2[1] + bd2[2], D, P))          # M1 + M2 points
        kw = dict(kwbase); kw.update(mission3_prop_diameter_in=float(D), mission3_prop_pitch_in=P)
        s3, bd3, _ = main(DesignVector(**kw), pv, prop_database=db, return_details=True, continuous_lap_scoring=True)
        split_m3.append((bd3[3], D, P))                    # M3 points
    best_shared = max(shared)
    best_m12 = max(split_m12); best_m3 = max(split_m3)
    kw = dict(kwbase); kw.update(prop_diameter_in=float(best_m12[1]), prop_pitch_in=best_m12[2],
                                 mission3_prop_diameter_in=float(best_m3[1]), mission3_prop_pitch_in=best_m3[2])
    s_split, bd_split, det = main(DesignVector(**kw), pv, prop_database=db, return_details=True, continuous_lap_scoring=True)
    m2m = det['propulsion'].get('M2',{}).get('inertial_mass_kg', float('nan'))
    m3m = det['propulsion'].get('M3',{}).get('inertial_mass_kg', float('nan'))
    print(f"{label}: M2 mass {m2m:.2f} kg, M3 mass {m3m:.2f} kg")
    print(f"  best shared  {best_shared[1]:.0f}x{best_shared[2]:.1f}  score {best_shared[0]:.4f}")
    print(f"  best split   M1/M2 {best_m12[1]:.0f}x{best_m12[2]:.1f}, M3 {best_m3[1]:.0f}x{best_m3[2]:.1f}  score {s_split:.4f}")
    print(f"  gain from splitting: {s_split - best_shared[0]:+.4f}\n")

scan("archived optimum")
scan("heavy M2 / light M3", sensor_weight_kg=6.0, mission3_sensor_weight_kg=0.4)
scan("heavy both", sensor_weight_kg=6.0, mission3_sensor_weight_kg=6.0)
