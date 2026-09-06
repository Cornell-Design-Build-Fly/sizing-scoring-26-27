"""What do flaps and the 12 m/s landing limit do to the takeoff wall?"""
import json
from dataclasses import replace
import numpy as np
from src.vectors import DesignVector, ParameterVector, maximum_sensor_weight_kg
from src.main import main
from src.prop.mission_performance import DEFAULT_PROPULSION_REQUIREMENTS as REQ
from src.aero.flaps import FlapConfig
from src.prop.continuous_prop_database import load_default_continuous_prop_database

rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]; names = DesignVector.opt_names()
base = {n: float(v[n]) if n in v else float(v[n.replace("mission3_prop_","prop_")]) for n in names}
db = load_default_continuous_prop_database(); pv = ParameterVector()

NO_FLAPS = replace(REQ, flaps=FlapConfig(takeoff_deflection_deg=0.0, landing_deflection_deg=0.0))

def run(req, **over):
    kw = dict(base); kw.update(over)
    return main(DesignVector(**kw), pv, prop_database=db, return_details=True,
                continuous_lap_scoring=True, propulsion_requirements=req)

print(f"{'w_kg':>5} | {'arm':<22} {'score':>8} {'pen':>6} {'toM2':>7} {'V_LO':>6} {'Vs_cln':>7} {'Vs_land':>8} {'lim':>18}")
for w in (3.58, 5.0, 7.0, 9.0):
    for label, req in (("no flaps", NO_FLAPS), ("flaps", REQ)):
        if w > maximum_sensor_weight_kg(base['sensor_length_m'], base['sensor_diameter_m']):
            continue
        s, bd, det = run(req, sensor_weight_kg=w, mission3_sensor_weight_kg=w)
        m2 = det['propulsion'].get('M2', {})
        f = lambda k: m2.get(k, float('nan'))
        print(f"{w:5.2f} | {label:<22} {s:8.3f} {det['penalty_total']:6.2f} "
              f"{f('takeoff_distance_m'):7.1f} {f('liftoff_speed_mps'):6.1f} "
              f"{f('clean_stall_speed_mps'):7.2f} {f('landing_stall_speed_mps'):8.2f} "
              f"{m2.get('limiting_constraint','-'):>18}")
    print()
