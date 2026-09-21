"""Recalculate direct battery times at the requested reduced throttle limits."""

from pathlib import Path
import csv
import json
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.prop.prop_classes import Motor, Battery, DEFAULT_VELOCITIES_MPS
from src.prop.prop_cruise_values import solve_cruise_samples
from src.prop.continuous_prop_database import load_default_continuous_prop_database


def main():
    settings = json.loads((OUT.parent / "inputs_and_fits.json").read_text(encoding="utf-8"))
    inputs = dict(settings["inputs"], cruise_throttle=0.7453, mission3_cruise_throttle=0.7225)
    motor = Motor(inputs["motor_kv"], inputs["motor_max_power"], settings["motor_current_limit_a"])
    battery = Battery(settings["nominal_pack_voltage_v"], inputs["battery_cell_count"],
                      settings["battery_c_rating"], inputs["batt_capacity"], settings["usable_battery_fraction"])
    current_limit = min(motor.max_current, battery.get_max_current())
    velocities = np.unique(np.r_[np.arange(0.0, 40.01, 1.0), DEFAULT_VELOCITIES_MPS])
    database = load_default_continuous_prop_database()
    rows = []
    for mission, prefix, cap in (("M1/M2", "", 0.7453), ("M3", "mission3_", 0.7225)):
        diameter, pitch = inputs[prefix + "prop_diameter_in"], inputs[prefix + "prop_pitch_in"]
        catalog_key = database.catalog.get_by_geometry(diameter, pitch).key
        result = solve_cruise_samples(
            diameter, pitch, velocities, motor, battery, current_limit, cap, database,
            min_rpm=3000, max_rpm=20000, rpm_step=100,
            knockdown=False, maximum_battery_power_w=None,
        )
        valid = ~result.failed_mask
        assert np.all(result.selected_throttle[valid] <= cap)
        assert np.all(result.selected_current_a[valid] <= current_limit)
        assert np.all(result.selected_power_w[valid] <= motor.max_power)
        np.testing.assert_allclose(result.flight_time_samples_s[valid],
                                   battery.get_useable_capacity() * 3600 / result.selected_current_a[valid],
                                   rtol=1e-12)
        for i, speed in enumerate(velocities):
            failed = bool(result.failed_mask[i])
            thrust = float(result.thrust_samples_n[i])
            positive = not failed and thrust > 0.0
            rows.append({
                "mission": mission, "catalog_propeller": catalog_key, "throttle_cap_fraction": cap,
                "airspeed_mps": float(speed), "selected_throttle_fraction": float(result.selected_throttle[i]),
                "rpm": float(result.selected_rpm[i]), "current_a": float(result.selected_current_a[i]),
                "thrust_n": thrust, "electrical_terminal_power_w": float(result.selected_power_w[i]),
                "raw_solver_battery_time_s": float(result.flight_time_samples_s[i]),
                "positive_thrust_battery_time_s": float(result.flight_time_samples_s[i]) if positive else None,
                "positive_thrust_battery_time_min": float(result.flight_time_samples_s[i]) / 60.0 if positive else None,
                "valid_rpm_count": int(result.valid_rpm_count[i]), "failed": failed,
                "status": "positive thrust" if positive else ("no valid operating point" if failed else "nonpositive thrust"),
            })
    with (OUT / "flight_time_vs_airspeed.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "inputs": inputs, "battery_c_rating": battery.Crat, "nominal_pack_voltage_v": battery.vnom,
        "usable_battery_fraction": battery.useable_fraction, "usable_capacity_ah": battery.get_useable_capacity(),
        "motor_current_limit_a": motor.max_current, "effective_current_limit_a": current_limit,
        "rpm_search": {"minimum": 3000, "maximum": 20000, "step": 100},
        "knockdown": False, "maximum_battery_power_w": None,
        "method": "Direct solve_cruise_samples; maximum available thrust at or below each throttle cap.",
    }
    (OUT / "inputs.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Flight-time outputs at reduced throttle", "",
        "M1/M2: 21x13E with throttle cap 0.7453. M3: 18x12E with throttle cap 0.7225.", "",
        f"Same motor and battery as the original run: {motor.kv:.14g} Kv, {motor.max_power:.14g} W, "
        f"8S / {battery.capacity:.15g} Ah, {battery.vnom:g} V nominal, {battery.Crat:g}C, "
        f"{battery.useable_fraction:.0%} usable capacity ({battery.get_useable_capacity():.9f} Ah).", "",
        "Direct solver values use `time_s = usable_capacity_Ah * 3600 / current_A`. "
        "Throttle values are upper limits: the 100 RPM grid can select slightly lower throttle. "
        "No quadratic approximation or mission energy power cap is applied.", "",
        "| Airspeed (m/s) | M1/M2 time (min) | M3 time (min) | M1/M2 thrust (N) | M3 thrust (N) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for speed in range(0, 41, 5):
        a, b = [row for row in rows if row["airspeed_mps"] == speed]
        times = [f"{row['positive_thrust_battery_time_min']:.3f}" if row["positive_thrust_battery_time_min"] is not None
                 else "N/A" for row in (a, b)]
        lines.append(f"| {speed} | {times[0]} | {times[1]} | {a['thrust_n']:.3f} | {b['thrust_n']:.3f} |")
    lines += ["", "Zero airspeed is static battery runtime. Each other row assumes constant operation at "
              "that selected maximum-thrust point. Actual mission endurance also depends on drag, takeoff, climb, "
              "turns, and throttle scheduling. A positive thrust value alone does not prove sustainable flight at that speed.", "",
              "N/A means the solver could not find an operating point with positive thrust under that throttle cap. "
              "The CSV retains raw solver results and status; its zero failure sentinels are not zero-second endurance predictions.", "",
              "All valid points were checked against throttle, current, and motor-power limits. Battery times were "
              "verified against usable capacity and selected current.", "",
              "[Full CSV at 1 m/s increments](flight_time_vs_airspeed.csv) | [Exact inputs](inputs.json)", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    for mission in ("M1/M2", "M3"):
        invalid = [row["airspeed_mps"] for row in rows if row["mission"] == mission and row["status"] != "positive thrust"]
        print(mission, "samples without positive thrust:", invalid)


if __name__ == "__main__":
    main()
