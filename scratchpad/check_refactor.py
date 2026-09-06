"""Confirm the grid refactor reproduces solve_cruise_samples exactly."""
import numpy as np, importlib.util, sys, types
from src.vectors import DesignVector, ParameterVector
from src.prop.prop_helper_functions import make_motor_from_design, make_battery_from_design
from src.prop.continuous_prop_database import load_default_continuous_prop_database
from src.prop.prop_cruise_values import solve_cruise_samples

# load the pre-refactor module under a different name
spec = importlib.util.spec_from_file_location("old_pcv", "scratchpad/old_pcv.py")
old = importlib.util.module_from_spec(spec)
sys.modules["old_pcv"] = old
spec.loader.exec_module(old)

db = load_default_continuous_prop_database(); pv = ParameterVector()
rng = np.random.default_rng(0)
worst = 0.0
for trial in range(6):
    dv = DesignVector(
        prop_diameter_in=float(rng.uniform(10, 24)),
        prop_pitch_in=float(rng.uniform(5, 14)),
        motor_kv=float(rng.uniform(200, 650)),
        motor_max_power=float(rng.uniform(1000, 3000)),
        batt_capacity=float(rng.uniform(1.5, 3.3)),
    )
    m = make_motor_from_design(dv, pv); b = make_battery_from_design(dv, pv)
    vels = np.linspace(0.01, 45, 37)
    cl = min(m.max_current, b.get_max_current())
    for thr in (0.6, 1.0):
        for mt in (None, np.full(vels.size, 12.0)):
            a = old.solve_cruise_samples(dv.prop_diameter_in, dv.prop_pitch_in, vels, m, b, cl, thr, db,
                                         min_rpm=3000, max_rpm=20000, rpm_step=100, minimum_thrust_n=mt)
            c = solve_cruise_samples(dv.prop_diameter_in, dv.prop_pitch_in, vels, m, b, cl, thr, db,
                                     min_rpm=3000, max_rpm=20000, rpm_step=100, minimum_thrust_n=mt)
            for f in ("thrust_samples_n","flight_time_samples_s","selected_rpm","selected_current_a",
                      "selected_throttle","selected_power_w","valid_rpm_count","failed_mask"):
                x = np.asarray(getattr(a,f), float); y = np.asarray(getattr(c,f), float)
                assert x.shape == y.shape, f
                bad = ~np.isclose(x, y, rtol=0, atol=0, equal_nan=True)
                if bad.any():
                    print("MISMATCH", f, x[bad][:5], y[bad][:5]); sys.exit(1)
print("exact match on all fields, all trials")
