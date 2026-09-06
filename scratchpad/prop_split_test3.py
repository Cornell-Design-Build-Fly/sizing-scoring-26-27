"""Best shared propeller vs best independent M1/M2 + M3 propellers.

Full 2-D scan on total score (penalties included), so the comparison is honest.
"""
import itertools, json, sys
import numpy as np
from src.vectors import DesignVector, ParameterVector
from src.main import main
from src.prop.continuous_prop_database import load_default_continuous_prop_database

rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]; names = DesignVector.opt_names()
base = {n: float(v[n]) if n in v else float(v[n.replace("mission3_prop_","prop_")]) for n in names}
db = load_default_continuous_prop_database(); pv = ParameterVector()

PROPS = [(D, round(r*D, 1)) for D in [10, 13, 16, 20, 24] for r in (0.45, 0.6, 0.75)
         if 4.0 <= r*D <= 18.0]

def score(kw):
    s, bd, det = main(DesignVector(**kw), pv, prop_database=db, return_details=True, continuous_lap_scoring=True)
    return s, bd, det

def scan(label, **fixed):
    kwbase = dict(base); kwbase.update(fixed)
    table = {}
    for (D2, P2), (D3, P3) in itertools.product(PROPS, PROPS):
        kw = dict(kwbase)
        kw.update(prop_diameter_in=float(D2), prop_pitch_in=float(P2),
                  mission3_prop_diameter_in=float(D3), mission3_prop_pitch_in=float(P3))
        s, bd, det = score(kw)
        table[((D2,P2),(D3,P3))] = (s, bd, det)
    shared = {k: v for k, v in table.items() if k[0] == k[1]}
    bs_key, bs = max(shared.items(), key=lambda kv: kv[1][0])
    bp_key, bp = max(table.items(), key=lambda kv: kv[1][0])
    m2m = bp[2]['propulsion'].get('M2',{}).get('inertial_mass_kg', float('nan'))
    m3m = bp[2]['propulsion'].get('M3',{}).get('inertial_mass_kg', float('nan'))
    print(f"{label}: M2 {m2m:.2f} kg / M3 {m3m:.2f} kg   ({len(PROPS)}x{len(PROPS)} props scanned)")
    print(f"  best shared  {bs_key[0][0]:.0f}x{bs_key[0][1]:.1f}                    score {bs[0]:.4f}  breakdown {[round(x,3) for x in bs[1]]}")
    print(f"  best split   M1/M2 {bp_key[0][0]:.0f}x{bp_key[0][1]:.1f}, M3 {bp_key[1][0]:.0f}x{bp_key[1][1]:.1f}  score {bp[0]:.4f}  breakdown {[round(x,3) for x in bp[1]]}")
    print(f"  gain from splitting: {bp[0]-bs[0]:+.4f}\n", flush=True)

scan("archived optimum")
scan("heavy M2 / light M3", sensor_weight_kg=6.0, mission3_sensor_weight_kg=0.4)
scan("heavy both", sensor_weight_kg=6.0, mission3_sensor_weight_kg=6.0)
