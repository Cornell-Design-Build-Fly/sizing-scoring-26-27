import json, time
from src.vectors import DesignVector, ParameterVector
from src.main import main
from src.prop.continuous_prop_database import load_default_continuous_prop_database
rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]; names = DesignVector.opt_names()
base = {n: float(v[n]) if n in v else float(v[n.replace("mission3_prop_","prop_")]) for n in names}
db = load_default_continuous_prop_database(); pv = ParameterVector()
dv = DesignVector(**base)
main(dv,pv,prop_database=db,return_details=True,continuous_lap_scoring=True)
for w,label in [(base['sensor_weight_kg'],'baseline (M3 energy-limited)'), (5.0,'heavy (both energy-limited)')]:
    kw=dict(base); kw.update(sensor_weight_kg=w, mission3_sensor_weight_kg=w)
    d=DesignVector(**kw)
    t0=time.time()
    for _ in range(15): main(d,pv,prop_database=db,return_details=True,continuous_lap_scoring=True)
    print(f"{label}: {(time.time()-t0)/15*1000:.0f} ms/eval")
