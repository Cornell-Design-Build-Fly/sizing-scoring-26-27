import json, cProfile, pstats, io
from src.vectors import DesignVector, ParameterVector
from src.main import main
from src.prop.continuous_prop_database import load_default_continuous_prop_database
rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]; names = DesignVector.opt_names()
base = {n: float(v[n]) if n in v else float(v[n.replace("mission3_prop_","prop_")]) for n in names}
base.update(sensor_weight_kg=5.0, mission3_sensor_weight_kg=5.0)
db = load_default_continuous_prop_database(); pv = ParameterVector(); dv = DesignVector(**base)
main(dv,pv,prop_database=db,return_details=True,continuous_lap_scoring=True)
pr=cProfile.Profile(); pr.enable()
for _ in range(10): main(dv,pv,prop_database=db,return_details=True,continuous_lap_scoring=True)
pr.disable()
s=io.StringIO(); pstats.Stats(pr,stream=s).sort_stats('cumulative').print_stats(14); print(s.getvalue())
