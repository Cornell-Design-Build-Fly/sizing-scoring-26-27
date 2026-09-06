"""Is a bigger battery actually worse, or did the old model just leave it there?"""
import json
import numpy as np
from src.vectors import DesignVector, ParameterVector, MAX_BATT_CAPACITY_AH
from src.main import main
from src.mech.main_mech import evaluate_mechanical_module
from src.prop.continuous_prop_database import load_default_continuous_prop_database

rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]; names = DesignVector.opt_names()
base = {n: float(v[n]) if n in v else float(v[n.replace("mission3_prop_","prop_")]) for n in names}
db = load_default_continuous_prop_database(); pv = ParameterVector()

def mass_of(cap):
    kw = dict(base); kw["batt_capacity"] = cap
    mech = evaluate_mechanical_module(DesignVector(**kw), parameter_vector=pv)
    return next(i.mass_kg for i in mech.all_items if i.name == "Battery")

print(f"battery mass: {mass_of(1.0):.3f} kg at 1.0 Ah -> {mass_of(MAX_BATT_CAPACITY_AH):.3f} kg at "
      f"{MAX_BATT_CAPACITY_AH:.3f} Ah (full 100 Wh pack)")
print()
print(f"{'Ah':>6} {'Wh_nom':>7} {'allowed':>8} {'score':>8} {'M2':>6} {'M3':>6} "
      f"{'M2 V':>6} {'M2 E':>12} {'M3 V':>6} {'M3 E':>12} {'M3 laps':>7} {'TOGW':>6}")
for cap in np.linspace(1.5, MAX_BATT_CAPACITY_AH, 9):
    kw = dict(base); kw["batt_capacity"] = float(cap)
    dv = DesignVector(**kw)
    s, bd, det = main(dv, pv, prop_database=db, return_details=True, continuous_lap_scoring=True)
    m2 = det['propulsion'].get('M2', {}); m3 = det['propulsion'].get('M3', {})
    f = lambda d,k: d.get(k, float('nan'))
    print(f"{cap:6.3f} {dv.batt_energy:7.1f} {f(m2,'allowed_energy_wh'):8.1f} {s:8.4f} "
          f"{bd[2]:6.3f} {bd[3]:6.3f} "
          f"{f(m2,'cruise_speed_mps'):6.1f} {f(m2,'required_energy_wh'):5.1f}/{f(m2,'allowed_energy_wh'):5.1f} "
          f"{f(m3,'cruise_speed_mps'):6.1f} {f(m3,'required_energy_wh'):5.1f}/{f(m3,'allowed_energy_wh'):5.1f} "
          f"{m3.get('completed_laps',0):7d} {det['max_takeoff_mass_kg']:6.2f}")
