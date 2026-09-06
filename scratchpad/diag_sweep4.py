import json
import numpy as np
from src.vectors import DesignVector, ParameterVector, maximum_sensor_weight_kg
from src.main import main
from src.prop.continuous_prop_database import load_default_continuous_prop_database

rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]; names = DesignVector.opt_names()
base = {n: float(v[n]) if n in v else float(v[n.replace("mission3_prop_","prop_")]) for n in names}
db = load_default_continuous_prop_database(); pv = ParameterVector()

def run(**over):
    kw = dict(base); kw.update(over)
    return main(DesignVector(**kw), pv, prop_database=db, return_details=True, continuous_lap_scoring=True)

print(f"{'w_kg':>5} {'score':>8} {'pen':>6} | {'M2 V':>6} {'lim':>6} {'lap':>6} {'E':>11} | {'M3 V':>6} {'lim':>6} {'lap':>6} {'laps':>4} {'E':>11} | {'toM2':>6}")
for w in [1.0, 2.0, 3.58, 5, 7, 9, 12]:
    if w > maximum_sensor_weight_kg(base['sensor_length_m'], base['sensor_diameter_m']):
        print(w, "density bound"); continue
    s, bd, det = run(sensor_weight_kg=w, mission3_sensor_weight_kg=w)
    p = det['propulsion']; m2 = p.get('M2', {}); m3 = p.get('M3', {})
    f = lambda d,k: d.get(k, float('nan'))
    print(f"{w:5.2f} {s:8.3f} {det['penalty_total']:6.2f} | "
          f"{f(m2,'cruise_speed_mps'):6.1f} {'ENER' if m2.get('energy_limited') else '-':>6} {f(m2,'modeled_lap_time_s'):6.1f} "
          f"{f(m2,'required_energy_wh'):5.1f}/{f(m2,'allowed_energy_wh'):5.1f} | "
          f"{f(m3,'cruise_speed_mps'):6.1f} {'ENER' if m3.get('energy_limited') else '-':>6} {f(m3,'modeled_lap_time_s'):6.1f} "
          f"{m3.get('completed_laps',0):4d} {f(m3,'required_energy_wh'):5.1f}/{f(m3,'allowed_energy_wh'):5.1f} | "
          f"{f(m2,'takeoff_distance_m'):6.1f}  {m2.get('limiting_constraint','-')}/{m3.get('limiting_constraint','-')}")
