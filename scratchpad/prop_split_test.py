"""Does giving Mission 3 its own propeller buy anything?"""
import json
import numpy as np
from src.vectors import DesignVector, ParameterVector
from src.main import main
from src.prop.continuous_prop_database import load_default_continuous_prop_database

rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]; names = DesignVector.opt_names()
base = {n: float(v[n]) if n in v else float(v[n.replace("mission3_prop_","prop_")]) for n in names}
db = load_default_continuous_prop_database(); pv = ParameterVector()

def run(**over):
    kw = dict(base); kw.update(over)
    return main(DesignVector(**kw), pv, prop_database=db, return_details=True, continuous_lap_scoring=True)

s0, bd0, det0 = run()
print(f"shared prop {base['prop_diameter_in']:.1f}x{base['prop_pitch_in']:.1f}: "
      f"score={s0:.4f} M3={bd0[3]:.4f} laps={det0['propulsion']['M3']['completed_laps']} "
      f"lap={det0['propulsion']['M3']['modeled_lap_time_s']:.1f}s")

best = (s0, None)
rows = []
for D in [10, 12, 14, 16, 18, 20, 22, 24]:
    for P in np.arange(0.4*D, 0.8*D + 0.01, 0.1*D):
        if not (4.0 <= P <= 18.0):
            continue
        s, bd, det = run(mission3_prop_diameter_in=float(D), mission3_prop_pitch_in=float(P))
        m3 = det['propulsion'].get('M3', {})
        rows.append((s, D, float(P), bd[3], m3.get('completed_laps', 0),
                     m3.get('modeled_lap_time_s', float('nan')),
                     m3.get('cruise_speed_mps', float('nan')),
                     m3.get('required_energy_wh', float('nan'))))
rows.sort(reverse=True)
print(f"\n{'score':>8} {'D':>5} {'P':>5} {'M3':>7} {'laps':>4} {'lap_s':>7} {'V':>6} {'E_wh':>6}")
for r in rows[:8]:
    print(f"{r[0]:8.4f} {r[1]:5.0f} {r[2]:5.1f} {r[3]:7.4f} {r[4]:4d} {r[5]:7.1f} {r[6]:6.1f} {r[7]:6.1f}")
print("...")
for r in rows[-3:]:
    print(f"{r[0]:8.4f} {r[1]:5.0f} {r[2]:5.1f} {r[3]:7.4f} {r[4]:4d} {r[5]:7.1f} {r[6]:6.1f} {r[7]:6.1f}")
print(f"\nbest separate-M3-prop score {rows[0][0]:.4f} vs shared {s0:.4f}  (+{rows[0][0]-s0:.4f})")
