"""Where the 12 m/s landing limit and the wing-area box collide, and the
best takeoff flap setting."""
import json
from dataclasses import replace
import numpy as np
from src.vectors import DesignVector, ParameterVector, OPT_VARS
from src.main import main
from src.prop.mission_performance import DEFAULT_PROPULSION_REQUIREMENTS as REQ
from src.aero.flaps import FlapConfig, clean_cl_max, DEFAULT_FLAPS
from src.prop.continuous_prop_database import load_default_continuous_prop_database

rho, g = 1.225, 9.81
bounds = dict(OPT_VARS)
span_max = bounds["wing_span"][1]; chord_max = bounds["wing_chord"][1]
S = span_max * chord_max
ar = span_max**2 / S
clmax_land = DEFAULT_FLAPS.cl_max_for(ar, "landing")
clmax_clean = clean_cl_max(ar)
for v_land in (12.0, 14.0, 16.0):
    w_max_n = 0.5 * rho * v_land**2 * S * clmax_land
    print(f"Vstall_land <= {v_land:4.1f} m/s at max box wing "
          f"(S={S:.3f} m^2, AR={ar:.2f}, CLmax_land={clmax_land:.2f}) "
          f"-> TOGW <= {w_max_n/g:5.2f} kg = {w_max_n/g/0.45359237:5.1f} lb")
print(f"   (clean CLmax {clmax_clean:.2f}; flaps add {clmax_land-clmax_clean:+.2f})")
print(f"   MATLAB's own optimum needs S = 1.52 m^2 at the same 6 ft span, i.e. chord {1.52/span_max:.2f} m")

print("\nBest takeoff flap deflection on the archived design (M2 ground roll):")
rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]; names = DesignVector.opt_names()
base = {n: float(v[n]) if n in v else float(v[n.replace("mission3_prop_","prop_")]) for n in names}
db = load_default_continuous_prop_database(); pv = ParameterVector()
for deg in (0, 10, 15, 20, 25, 30, 40):
    req = replace(REQ, flaps=FlapConfig(takeoff_deflection_deg=float(deg)))
    s, bd, det = main(DesignVector(**base), pv, prop_database=db, return_details=True,
                      continuous_lap_scoring=True, propulsion_requirements=req)
    m2 = det['propulsion']['M2']
    print(f"  {deg:3d} deg: roll {m2['takeoff_distance_m']:6.2f} m  V_LO {m2['liftoff_speed_mps']:5.2f}  "
          f"climb_rate {m2['climb_rate_mps']:5.2f}  climb_Wh {m2['climb_energy_wh']:5.2f}  "
          f"E {m2['required_energy_wh']:5.2f}  score {s:.4f}")
