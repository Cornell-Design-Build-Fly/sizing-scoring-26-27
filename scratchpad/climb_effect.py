"""Effect of charging the climb-out, flying it flapped, and paying for altitude."""
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

for label, req in (("no flaps", NO_FLAPS), ("flaps", REQ)):
    s, bd, det = main(DesignVector(**base), pv, prop_database=db, return_details=True,
                      continuous_lap_scoring=True, propulsion_requirements=req)
    print(f"=== {label}: score {s:.4f}  penalty {det['penalty_total']:.2f}")
    for m in ("M2", "M3"):
        p = det['propulsion'].get(m)
        if not p: continue
        print(f"  {m}: roll {p['takeoff_distance_m']:.1f} m -> V_LO {p['liftoff_speed_mps']:.2f}"
              f" -> accel {p['acceleration_distance_m']:.1f} m / {p['acceleration_time_s']:.2f} s"
              f" -> V_climb {p['climb_speed_mps']:.2f} -> climb {p['climb_time_s']:.1f} s"
              f" to {p['cruise_altitude_m']:.1f} m -> V_cruise {p['cruise_speed_mps']:.2f}")
        print(f"      energy Wh: roll {p['takeoff_energy_wh']:.2f}  accel {p['acceleration_energy_wh']:.2f}"
              f"  climb {p['climb_energy_wh']:.2f}  retract {p['flap_retraction_energy_wh']:.2f}"
              f"  course {p['cruise_energy_wh']:.2f}  turn-exit {p['reacceleration_energy_wh']:.2f}"
              f"  = {p['required_energy_wh']:.2f} / {p['allowed_energy_wh']:.2f}")
        print(f"      climb rate {p['climb_rate_mps']:.2f} m/s (floor 2.0), laps {p['completed_laps']}, lim {p['limiting_constraint']}")
