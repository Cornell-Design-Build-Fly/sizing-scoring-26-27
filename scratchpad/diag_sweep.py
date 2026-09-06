"""Diagnose what stops the optimum from getting heavier."""
import json, time, sys
from dataclasses import replace
import numpy as np

from src.vectors import DesignVector, ParameterVector, maximum_sensor_weight_kg
from src.main import main
from src.prop.continuous_prop_database import load_default_continuous_prop_database

rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]
names = DesignVector.opt_names()
base_kwargs = {n: float(v[n]) for n in names}
db = load_default_continuous_prop_database()
pv = ParameterVector()

def run(**over):
    kw = dict(base_kwargs); kw.update(over)
    dv = DesignVector(**kw)
    t0 = time.time()
    score, bd, det = main(dv, pv, prop_database=db, return_details=True, continuous_lap_scoring=True)
    return score, bd, det, time.time()-t0

score, bd, det, dt = run()
print(f"baseline score={score:.4f} breakdown={[round(b,4) for b in bd]} eval={dt:.1f}s")
print("penalties:", {k:v for k,v in det.items() if k.startswith('penalty')})
