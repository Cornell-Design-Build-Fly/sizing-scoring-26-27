import json, time
from src.vectors import DesignVector, ParameterVector
from src.main import main
from src.prop.continuous_prop_database import load_default_continuous_prop_database

rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]
names = DesignVector.opt_names()
base = {n: float(v[n]) for n in names if n in v}
missing = [n for n in names if n not in v]
print("new opt vars (defaulted from the M1/M2 prop):", missing)
for n in missing:
    base[n] = float(v[n.replace("mission3_prop_", "prop_")])
db = load_default_continuous_prop_database(); pv = ParameterVector()
dv = DesignVector(**base)
t0=time.time()
s, bd, det = main(dv, pv, prop_database=db, return_details=True, continuous_lap_scoring=True)
print(f"score={s:.4f} breakdown={[round(b,4) for b in bd]} eval={time.time()-t0:.2f}s")
print("penalties:", {k: round(x,3) if isinstance(x,float) else x for k,x in det.items() if k.startswith('penalty')})
for m, p in det['propulsion'].items():
    print(f"--- {m}: prop {p['propeller_diameter_in']:.1f}x{p['propeller_pitch_in']:.1f}")
    for k in ['aerodynamic_cruise_speed_mps','cruise_speed_mps','energy_limited','cruise_power_cap_w',
              'cruise_power_w','turn_power_w','turn_speed_mps','turn_load_factor','modeled_lap_time_s',
              'completed_laps','mission_flight_time_s','takeoff_energy_wh','straight_energy_wh',
              'turn_energy_wh','reacceleration_energy_wh','required_energy_wh','allowed_energy_wh',
              'limiting_constraint','takeoff_distance_m']:
        val = p.get(k)
        print(f"    {k}: {round(val,4) if isinstance(val,float) else val}")
