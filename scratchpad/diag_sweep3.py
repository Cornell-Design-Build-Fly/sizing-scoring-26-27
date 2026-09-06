import json
import numpy as np
from src.vectors import DesignVector, ParameterVector, maximum_sensor_weight_kg
from src.main import main
from src.prop.continuous_prop_database import load_default_continuous_prop_database

rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]; names = DesignVector.opt_names()
base = {n: float(v[n]) for n in names}
db = load_default_continuous_prop_database(); pv = ParameterVector()

def run(**over):
    kw = dict(base); kw.update(over)
    dv = DesignVector(**kw)
    return main(dv, pv, prop_database=db, return_details=True, continuous_lap_scoring=True)

hdr = f"{'D':>5} {'P':>5} {'kv':>5} {'span':>5} {'chord':>5} {'batt':>5} {'score':>8} {'pen':>6} {'limM2':>20} {'limM3':>20} {'toM2':>7} {'E2':>11} {'E3':>11} {'lapM3':>6}"
print(hdr)
# heavy sensor, sweep prop + wing + battery
for D,P,kv,span,chord,batt in [
    (10.2,6.4,499,1.451,0.360,2.37),
    (16,8,400,1.8288,0.40,3.378),
    (20,8,300,1.8288,0.40,3.378),
    (24,10,250,1.8288,0.40,3.378),
    (24,10,250,1.8288,0.40,3.378),
]:
    for w in [3.58, 6.0, 9.0, 12.0]:
        try:
            s, bd, det = run(sensor_weight_kg=w, mission3_sensor_weight_kg=w,
                             prop_diameter_in=D, prop_pitch_in=P, motor_kv=kv,
                             wing_span=span, wing_chord=chord, batt_capacity=batt,
                             motor_max_power=3000.0, cruise_throttle=1.0, mission3_cruise_throttle=1.0)
        except ValueError as e:
            print(f"{D:5} {P:5} {kv:5} {span:5.2f} {chord:5.2f} {batt:5.2f} w={w}: {e}"); continue
        p=det['propulsion']; m2=p.get('M2',{}); m3=p.get('M3',{})
        print(f"{D:5.1f} {P:5.1f} {kv:5.0f} {span:5.2f} {chord:5.2f} {batt:5.2f} w={w:5.1f} {s:8.3f} {det['penalty_total']:6.2f} "
              f"{m2.get('limiting_constraint','-'):>20} {m3.get('limiting_constraint','-'):>20} "
              f"{m2.get('takeoff_distance_m',float('nan')):7.1f} "
              f"{m2.get('required_energy_wh',float('nan')):5.1f}/{m2.get('allowed_energy_wh',float('nan')):5.1f} "
              f"{m3.get('required_energy_wh',float('nan')):5.1f}/{m3.get('allowed_energy_wh',float('nan')):5.1f} "
              f"{m3.get('modeled_lap_time_s',float('nan')):6.1f}")
    print()
