from src.prop.continuous_prop_database import load_default_continuous_prop_database
from src.prop.mission_performance import evaluate_mission_propulsion
from src.vectors import DesignVector, ParameterVector
db = load_default_continuous_prop_database(); pv = ParameterVector()
for cap in (1.2, 2.0, 3.0):
    d = DesignVector(batt_capacity=cap)
    print(f"--- batt {cap} Ah")
    for m in (4.0, 5.0, 6.0, 7.0, 8.0):
        r = evaluate_mission_propulsion(d, pv, mission=3, mass_kg=m, cruise_speed_mps=30.0,
                                        stall_speed_mps=12.0, lap_time_s=40.0, prop_database=db)
        print(f"  m={m:.1f} EL={r.energy_limited!s:5} feas={r.feasible!s:5} V={r.cruise_speed_mps:5.1f} "
              f"lap={r.modeled_lap_time_s:6.1f} laps={r.completed_laps} "
              f"E={r.required_energy_wh:6.1f}/{r.allowed_energy_wh:5.1f} lim={r.limiting_constraint}")
