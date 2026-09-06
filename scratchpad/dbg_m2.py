import json
import numpy as np
from src.vectors import DesignVector, ParameterVector
from src.prop import mission_performance as mp
from src.prop.continuous_prop_database import load_default_continuous_prop_database
from src.mech.main_mech import evaluate_mechanical_module
from src.main import resolved_aerodynamic_design_vector
from src.prop.main_prop import prop_main
from src.aero.main_aero import aero_main

rep = json.load(open("data_dump/opt_topline/run_20260905_193357/best_design_report.json"))
v = rep["optimizer_vector"]; names = DesignVector.opt_names()
base = {n: float(v[n]) if n in v else float(v[n.replace("mission3_prop_","prop_")]) for n in names}
base.update(sensor_weight_kg=5.0, mission3_sensor_weight_kg=5.0)
dv = DesignVector(**base); pv = ParameterVector()
db = load_default_continuous_prop_database()
mech = evaluate_mechanical_module(dv, parameter_vector=pv)
rdv = resolved_aerodynamic_design_vector(dv, mech)
props = mech.for_mission("M2")
tc, ftf = prop_main(rdv, pv, mission=2, prop_database=db)
aero = aero_main(design_vector=rdv, parameter_vector=pv, thrust_velocity=tc, flight_time_fit=ftf,
                 mission=2, cg=props.cg_m, inertia_matrix=props.inertia_tensor_kg_m2, mass=props.total_mass_kg)
print("mass", props.total_mass_kg, "V", aero.cruise_speed_mps, "Vstall", aero.stall_speed_mps)

# monkeypatch to trace the bisection
orig = mp._CourseState
import types
trace = []
real_select = mp.select_cruise_points
res = mp.evaluate_mission_propulsion(rdv, pv, mission=2, mass_kg=props.total_mass_kg,
        cruise_speed_mps=float(aero.cruise_speed_mps), stall_speed_mps=float(aero.stall_speed_mps),
        lap_time_s=float(aero.lap_time), prop_database=db)
print("energy_limited", res.energy_limited, "req", res.required_energy_wh, "allowed", res.allowed_energy_wh)
print("cruise_power", res.cruise_power_w, "turn_power", res.turn_power_w, "V", res.cruise_speed_mps)

# manual: replicate course_state at several caps by calling with a forced cap via requirements hack
# instead, directly probe turn feasibility at reduced caps
from src.prop.prop_helper_functions import make_motor_from_design, make_battery_from_design
motor = make_motor_from_design(rdv, pv); batt = make_battery_from_design(rdv, pv)
d, p = rdv.propeller_for_mission(2)
W = props.total_mass_kg * pv.gravity
env = mp._build_turn_envelope(rdv, pv, mission=2, supported_weight_n=W,
        maximum_cruise_speed_mps=float(aero.cruise_speed_mps), stall_speed_mps=float(aero.stall_speed_mps),
        motor=motor, battery=batt, prop_database=db, diameter_in=d, pitch_in=p)
cl = min(motor.max_current, batt.get_max_current())
for cap in [None, 800, 700, 600, 500, 400, 300, 200]:
    t = mp._solve_turn(env, pv, current_limit_a=cl, throttle_limit=1.0,
                       motor_max_power_w=float(motor.max_power), battery_vnom_v=float(batt.vnom),
                       maximum_battery_power_w=cap, maximum_speed_mps=float(aero.cruise_speed_mps))
    print(f"cap={cap}: feasible={t.feasible} V={t.speed_mps:.2f} n={t.load_factor:.3f} omega={t.angular_rate_rad_s:.4f} P={t.battery_power_w:.1f}")
