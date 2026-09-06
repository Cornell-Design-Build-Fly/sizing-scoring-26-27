import json
from dataclasses import replace
from src.vectors import DesignVector, ParameterVector
from src.main import main
from src.prop.mission_performance import DEFAULT_PROPULSION_REQUIREMENTS as REQ
from src.aero.flaps import FlapConfig
from src.prop.continuous_prop_database import load_default_continuous_prop_database

rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]; names = DesignVector.opt_names()
base = {n: float(v[n]) if n in v else float(v[n.replace("mission3_prop_","prop_")]) for n in names}
db = load_default_continuous_prop_database(); pv = ParameterVector()
NO_FLAPS = replace(REQ, flaps=FlapConfig(takeoff_deflection_deg=0.0, landing_deflection_deg=0.0))

print(f"{'arm':<10} {'Vs_clean':>9} {'Vs_TO':>7} {'V_LO':>7} {'V_climb':>8} {'gap':>6} {'climb_rate':>11} {'margin':>7}")
for label, req in (("no flaps", NO_FLAPS), ("flaps", REQ)):
    s, bd, det = main(DesignVector(**base), pv, prop_database=db, return_details=True,
                      continuous_lap_scoring=True, propulsion_requirements=req)
    m = det['propulsion']['M2']
    gap = m['climb_speed_mps'] - m['liftoff_speed_mps']
    print(f"{label:<10} {m['clean_stall_speed_mps']:9.2f} {m['takeoff_stall_speed_mps']:7.2f} "
          f"{m['liftoff_speed_mps']:7.2f} {m['climb_speed_mps']:8.2f} {gap:6.2f} "
          f"{m['climb_rate_mps']:11.2f} {m['climb_rate_margin_mps']:7.2f}")
    print(f"           climb energy charged: {m['climb_energy_wh']} Wh   "
          f"climb_altitude_m={req.climb_altitude_m}")
