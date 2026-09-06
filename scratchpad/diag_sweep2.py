import json, time
import numpy as np
from src.vectors import DesignVector, ParameterVector, maximum_sensor_weight_kg
from src.main import main
from src.prop.continuous_prop_database import load_default_continuous_prop_database

rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]
names = DesignVector.opt_names()
base = {n: float(v[n]) for n in names}
db = load_default_continuous_prop_database()
pv = ParameterVector()

def run(**over):
    kw = dict(base); kw.update(over)
    try:
        dv = DesignVector(**kw)
    except ValueError as e:
        return None, str(e)
    s, bd, det = main(dv, pv, prop_database=db, return_details=True, continuous_lap_scoring=True)
    return (s, bd, det), None

print(f"{'sensor_kg':>9} {'m3_kg':>7} {'score':>7} {'GM':>5} {'M1':>5} {'M2':>5} {'M3':>5} {'pen':>5} {'limit_M2':>22} {'limit_M3':>22} {'TOGW':>6} {'toM2':>6} {'E_M2':>11} {'E_M3':>11}")
for w in [1.0, 2.0, 3.58, 5, 7, 9, 12, 15, 18]:
    lmax = maximum_sensor_weight_kg(base['sensor_length_m'], base['sensor_diameter_m'])
    if w > lmax:
        print(f"{w:9.2f}  exceeds density bound {lmax:.2f}")
        continue
    r, err = run(sensor_weight_kg=w, mission3_sensor_weight_kg=w)
    if err: print(w, err); continue
    s, bd, det = r
    p = det['propulsion']
    m2 = p.get('M2', {}); m3 = p.get('M3', {})
    print(f"{w:9.2f} {w:7.2f} {s:7.3f} {bd[0]:5.2f} {bd[1]:5.2f} {bd[2]:5.2f} {bd[3]:5.2f} {det['penalty_total']:5.2f} "
          f"{m2.get('limiting_constraint','-'):>22} {m3.get('limiting_constraint','-'):>22} {det['max_takeoff_mass_kg']:6.2f} "
          f"{m2.get('takeoff_distance_m',float('nan')):6.1f} "
          f"{m2.get('required_energy_wh',float('nan')):5.1f}/{m2.get('allowed_energy_wh',float('nan')):5.1f} "
          f"{m3.get('required_energy_wh',float('nan')):5.1f}/{m3.get('allowed_energy_wh',float('nan')):5.1f}")
