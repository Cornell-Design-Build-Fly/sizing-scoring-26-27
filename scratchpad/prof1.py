import json, time
from src.vectors import DesignVector, ParameterVector
from src.main import main
from src.prop.main_prop import prop_main
from src.prop.continuous_prop_database import load_default_continuous_prop_database
rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]; names = DesignVector.opt_names()
dv = DesignVector(**{n: float(v[n]) for n in names})
db = load_default_continuous_prop_database(); pv = ParameterVector()
main(dv,pv,prop_database=db,return_details=True,continuous_lap_scoring=True)  # warm
import cProfile, pstats, io
pr=cProfile.Profile(); pr.enable()
for _ in range(10):
    main(dv,pv,prop_database=db,return_details=True,continuous_lap_scoring=True)
pr.disable()
s=io.StringIO(); pstats.Stats(pr,stream=s).sort_stats('cumulative').print_stats(22); print(s.getvalue())
t0=time.time()
for _ in range(20): prop_main(dv,pv,mission=2,prop_database=db)
print("prop_main:", (time.time()-t0)/20)
