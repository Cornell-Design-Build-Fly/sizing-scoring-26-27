r"""Export the requested design's direct prop-solver results and production fits.

Run from the repository root with:
  .\dbf-venv\Scripts\python.exe data_dump/prop/optimized_plane_8s_21x13_18x12/generate_report.py

Motor/Battery construction matches prop_helper_functions with ParameterVector
defaults. Importing those helpers/main_prop is currently blocked by the unrelated
FIXED_OPT_VALUES['sensor_diameter_m'] validation in vectors.py. Use the actual
solve_cruise_samples function and reproduce main_prop's four-point quadratic fit.
No production source files are modified.
"""

from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.prop.continuous_prop_database import load_default_continuous_prop_database
from src.prop.prop_classes import (
    Battery, Motor, DEFAULT_BATTERY_C_RATING, DEFAULT_MAX_CURRENT_A,
    DEFAULT_USABLE_BATTERY_FRACTION, DEFAULT_VELOCITIES_MPS, MPS_TO_MPH,
    battery_nominal_voltage_v,
)
from src.prop.prop_cruise_values import solve_cruise_samples

INPUTS = {
    "batt_capacity": 3.371432234281863,
    "battery_cell_count": 8,
    "prop_diameter_in": 21.0,
    "prop_pitch_in": 13.0,
    "mission3_prop_diameter_in": 18.0,
    "mission3_prop_pitch_in": 12.0,
    "motor_kv": 243.60636381140134,
    "motor_max_power": 2119.323245920669,
    "takeoff_flap_deflection_deg": 20.0,
    "cruise_throttle": 1.0,
    "mission3_cruise_throttle": 1.0,
}
NEWTONS_PER_LBF = 4.4482216152605
NEWTONS_PER_KGF = 9.80665


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_flight_time_outputs():
    """Add flight-time exports from the saved run without changing its results."""
    settings = json.loads((OUT / "inputs_and_fits.json").read_text(encoding="utf-8"))
    with (OUT / "thrust_vs_airspeed.csv").open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    usable_capacity = settings["inputs"]["batt_capacity"] * settings["usable_battery_fraction"]
    rows = []
    for source in source_rows:
        current = float(source["current_a"])
        seconds = float(source["constant_point_battery_time_s"])
        np.testing.assert_allclose(seconds, usable_capacity * 3600.0 / current, rtol=1e-12)
        rows.append({
            "mission": source["mission"], "catalog_propeller": source["catalog_propeller"],
            "airspeed_mps": float(source["airspeed_mps"]),
            "direct_flight_time_s": seconds, "direct_flight_time_min": seconds / 60.0,
            "quadratic_fit_flight_time_s": float(source["quadratic_fit_battery_time_s"]),
            "quadratic_fit_flight_time_min": float(source["quadratic_fit_battery_time_s"]) / 60.0,
            "current_a": current, "thrust_n": float(source["direct_thrust_n"]),
            "rpm": float(source["rpm"]), "throttle_fraction": float(source["throttle_fraction"]),
        })
    write_csv("flight_time_vs_airspeed.csv", rows)
    fig, ax = plt.subplots(figsize=(10.5, 6.5), layout="constrained")
    for case, color in zip(settings["results"], ("#17699b", "#cf6a27")):
        case_rows = [r for r in rows if r["mission"] == case["mission"]]
        speeds = np.array([r["airspeed_mps"] for r in case_rows])
        times_min = np.array([r["direct_flight_time_min"] for r in case_rows])
        np.testing.assert_allclose(
            [r["quadratic_fit_flight_time_s"] for r in case_rows],
            np.polyval(case["battery_time_fit_abc_s"], speeds), rtol=1e-12,
        )
        label = f"{case['mission']}: {case['catalog_propeller']}"
        ax.plot(speeds, times_min, color=color, lw=2.3, label=label + " | direct solver")
        smooth_v = np.linspace(settings["fit_velocities_mps"][0], 40.0, 300)
        ax.plot(smooth_v, np.polyval(case["battery_time_fit_abc_s"], smooth_v) / 60.0,
                color=color, lw=1.5, ls="--", label=label + " | quadratic fit")
        fit_indices = np.searchsorted(speeds, settings["fit_velocities_mps"])
        ax.scatter(speeds[fit_indices], times_min[fit_indices], color=color, s=35, zorder=4)
    ax.set(xlabel="Airspeed [m/s]", ylabel="Constant operating point battery time [min]",
           xlim=(0, 40), ylim=(0, None))
    ax.set_title("Battery time at maximum available thrust\n8S, 3.3714 Ah, 85% usable capacity, throttle cap 1.0", pad=15)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(OUT / "flight_time_vs_airspeed.png", dpi=200)
    fig.savefig(OUT / "flight_time_vs_airspeed.pdf")
    plt.close(fig)
    section = [
        "## Flight time versus airspeed", "",
        "These direct flight-time outputs correspond to the same maximum-available-thrust operating points "
        "as the thrust table, with throttle capped at 1.0. Each value assumes the listed current is held constant.", "",
        f"Usable capacity = {settings['inputs']['batt_capacity']:.15g} Ah x "
        f"{settings['usable_battery_fraction']:.0%} = {usable_capacity:.9f} Ah. "
        "The model computes `time_s = usable_capacity_Ah * 3600 / current_A`.", "",
        "| Airspeed (m/s) | 21x13 time (s) | 21x13 time (min) | 18x12 time (s) | 18x12 time (min) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for speed in range(0, 41, 5):
        a, b = [r for r in rows if r["airspeed_mps"] == speed]
        section.append(f"| {speed} | {a['direct_flight_time_s']:.2f} | {a['direct_flight_time_min']:.3f} | "
                       f"{b['direct_flight_time_s']:.2f} | {b['direct_flight_time_min']:.3f} |")
    section += ["", "The zero-airspeed row is static battery runtime. At higher airspeeds, reduced propeller load "
                "can lower current and increase battery time while available thrust decreases. "
                "These values alone do not establish that the aircraft can sustain the listed speed. "
                "Mission endurance also depends on drag, takeoff, climb, turns, and throttle scheduling.", "",
                "The quadratic flight-time fits in the next section approximate the four default samples; "
                "the table above uses direct solver values. The CSV includes both direct and fitted time in seconds and minutes.", "",
                "Files: [flight-time CSV](flight_time_vs_airspeed.csv), [graph](flight_time_vs_airspeed.png), "
                "[PDF graph](flight_time_vs_airspeed.pdf). All direct times were checked against usable capacity "
                "and the saved current, and all fitted times were checked against the saved coefficients.", ""]
    report_path = OUT / "report.md"
    report = report_path.read_text(encoding="utf-8")
    next_heading = "## Returned curve construction"
    before, after = report.split(next_heading, 1)
    before = before.split("## Flight time versus airspeed", 1)[0]
    report_path.write_text(before + "\n".join(section) + "\n" + next_heading + after, encoding="utf-8")
    print("\n".join(section))


def main():
    motor = Motor(INPUTS["motor_kv"], INPUTS["motor_max_power"], DEFAULT_MAX_CURRENT_A)
    battery = Battery(
        vnom=battery_nominal_voltage_v(INPUTS["battery_cell_count"]),
        cells=INPUTS["battery_cell_count"], Crat=DEFAULT_BATTERY_C_RATING,
        capacity=INPUTS["batt_capacity"], useable_fraction=DEFAULT_USABLE_BATTERY_FRACTION,
    )
    current_limit = min(motor.max_current, battery.get_max_current())
    database = load_default_continuous_prop_database()
    # Keep the exact original four fit points in the direct solver grid.
    velocities = np.unique(np.r_[np.arange(0.0, 40.01, 1.0), DEFAULT_VELOCITIES_MPS])
    fit_indices = np.searchsorted(velocities, DEFAULT_VELOCITIES_MPS)
    all_rows, fit_rows, static_rows, summaries = [], [], [], []
    fig, ax = plt.subplots(figsize=(10.5, 6.5), layout="constrained")

    cases = (
        ("M1/M2", "", "cruise_throttle", "#17699b"),
        ("M3", "mission3_", "mission3_cruise_throttle", "#cf6a27"),
    )
    for mission, prefix, throttle_key, color in cases:
        diameter = INPUTS[prefix + "prop_diameter_in"]
        pitch = INPUTS[prefix + "prop_pitch_in"]
        surface = database.catalog.get_by_geometry(diameter, pitch)
        result = solve_cruise_samples(
            diameter, pitch, velocities, motor, battery, current_limit,
            INPUTS[throttle_key], database, min_rpm=3000, max_rpm=20000,
            rpm_step=100, knockdown=False, maximum_battery_power_w=None,
        )
        assert not np.any(result.failed_mask), f"Failed samples for {surface.key}"
        thrust_fit = np.polyfit(DEFAULT_VELOCITIES_MPS, result.thrust_samples_n[fit_indices], 2)
        time_fit = np.polyfit(DEFAULT_VELOCITIES_MPS, result.flight_time_samples_s[fit_indices], 2)

        # Verify exported operating points independently against the exact
        # catalog surface and the model's motor/battery equations.
        thrust_check, torque_check = surface.evaluate(velocities * MPS_TO_MPH, result.selected_rpm)
        np.testing.assert_allclose(result.thrust_samples_n, thrust_check, rtol=1e-10)
        np.testing.assert_allclose(result.selected_torque_nm, torque_check, rtol=1e-10)
        current_check = torque_check / motor.get_kt() + motor.get_I0()
        voltage_sag = battery.vnom - current_check * battery.get_Rb()
        throttle_check = (result.selected_rpm / motor.kv + current_check * motor.get_rm()) / voltage_sag
        np.testing.assert_allclose(result.selected_current_a, current_check, rtol=1e-10)
        np.testing.assert_allclose(result.selected_throttle, throttle_check, rtol=1e-10)
        np.testing.assert_allclose(result.selected_power_w, current_check * voltage_sag, rtol=1e-10)
        assert np.all(result.selected_current_a <= current_limit)
        assert np.all(result.selected_power_w <= motor.max_power)
        assert np.all(result.selected_throttle <= INPUTS[throttle_key])

        rows = []
        for i, speed in enumerate(velocities):
            thrust = float(result.thrust_samples_n[i])
            row = {
                "mission": mission, "catalog_propeller": surface.key,
                "airspeed_mps": float(speed), "airspeed_mph": float(speed * MPS_TO_MPH),
                "direct_thrust_n": thrust, "direct_thrust_lbf": thrust / NEWTONS_PER_LBF,
                "direct_thrust_kgf": thrust / NEWTONS_PER_KGF,
                "quadratic_fit_thrust_n": float(np.polyval(thrust_fit, speed)),
                "rpm": float(result.selected_rpm[i]),
                "current_a": float(result.selected_current_a[i]),
                "throttle_fraction": float(result.selected_throttle[i]),
                "battery_terminal_voltage_v": float(voltage_sag[i]),
                "electrical_terminal_power_w": float(result.selected_power_w[i]),
                "battery_nominal_equivalent_power_w": float(result.selected_current_a[i] * battery.vnom),
                "shaft_power_w": float(result.selected_shaft_power_w[i]),
                "torque_nm": float(result.selected_torque_nm[i]),
                "constant_point_battery_time_s": float(result.flight_time_samples_s[i]),
                "quadratic_fit_battery_time_s": float(np.polyval(time_fit, speed)),
                "valid_rpm_count": int(result.valid_rpm_count[i]),
                "failed": bool(result.failed_mask[i]),
            }
            rows.append(row)
        all_rows.extend(rows)
        fit_rows.extend(rows[i] for i in fit_indices)
        static_rows.extend(rows[i] for i in (0, 1))  # 0 and 0.01 m/s
        summaries.append({
            "mission": mission, "catalog_propeller": surface.key,
            "thrust_fit_abc_n": thrust_fit.tolist(),
            "battery_time_fit_abc_s": time_fit.tolist(),
            "static_at_zero_mps": rows[0], "mission_static_at_0p01_mps": rows[1],
        })

        label = f"{mission}: {diameter:g} x {pitch:g} in"
        ax.plot(velocities, result.thrust_samples_n, color=color, lw=2.3, label=label + " | direct solver")
        smooth_v = np.linspace(DEFAULT_VELOCITIES_MPS[0], 40, 300)
        ax.plot(smooth_v, np.polyval(thrust_fit, smooth_v), color=color, lw=1.5,
                ls="--", label=label + " | quadratic fit")
        ax.scatter(DEFAULT_VELOCITIES_MPS, result.thrust_samples_n[fit_indices],
                   color=color, s=35, zorder=4)

    write_csv("thrust_vs_airspeed.csv", all_rows)
    write_csv("default_prop_samples.csv", fit_rows)
    write_csv("static_thrust.csv", static_rows)
    settings = {
        "inputs": INPUTS, "nominal_pack_voltage_v": battery.vnom,
        "battery_c_rating": battery.Crat, "usable_battery_fraction": battery.useable_fraction,
        "nominal_battery_energy_wh": battery.vnom * battery.capacity,
        "usable_battery_energy_wh": battery.vnom * battery.get_useable_capacity(),
        "motor_current_limit_a": motor.max_current, "effective_current_limit_a": current_limit,
        "motor_resistance_ohm": float(motor.get_rm()), "motor_no_load_current_a": float(motor.get_I0()),
        "motor_torque_constant_nm_per_a": motor.get_kt(), "battery_resistance_ohm": battery.get_Rb(),
        "rpm_search": {"minimum": 3000, "maximum": 20000, "step": 100},
        "knockdown": False, "maximum_battery_power_w": None,
        "fit_velocities_mps": DEFAULT_VELOCITIES_MPS.tolist(),
        "prop_database": "src/prop/data/prop_data.json",
        "method": "solve_cruise_samples; same four-point quadratic fitting as main_prop.prop_main",
        "wrapper_import_issue": "vectors.py rejects FIXED_OPT_VALUES['sensor_diameter_m']",
        "results": summaries,
    }
    (OUT / "inputs_and_fits.json").write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    ax.set(xlabel="Airspeed [m/s]", ylabel="Available thrust [N]", xlim=(0, 40), ylim=(0, 95))
    ax.set_title("Optimized plane | thrust versus airspeed\n8S, 3.3714 Ah, 243.606 Kv, throttle cap 1.0", pad=15)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax.secondary_yaxis("right", functions=(lambda n: n / NEWTONS_PER_LBF,
                                           lambda lbf: lbf * NEWTONS_PER_LBF)).set_ylabel("Available thrust [lbf]")
    fig.savefig(OUT / "thrust_vs_airspeed.png", dpi=200)
    fig.savefig(OUT / "thrust_vs_airspeed.pdf")
    plt.close(fig)

    lines = [
        "# Optimized plane propulsion outputs", "",
        "Calculated with the repository's propeller database and actual `solve_cruise_samples` solver.", "",
        "Inputs: 8S / 3.371432234281863 Ah; 243.60636381140134 Kv; motor electrical-power cap 2119.323245920669 W; "
        "21x13E for Missions 1/2 and 18x12E for Mission 3; both throttle upper limits 1.0.", "",
        f"Defaults: nominal voltage {battery.vnom:.1f} V, {battery.Crat:g}C battery, {battery.useable_fraction:.0%} usable capacity, "
        f"motor current limit {motor.max_current:g} A. The battery sets the effective current limit to {current_limit:.6f} A. "
        "RPM search: 3000 through 20000 in steps of 100. No thrust knockdown or mission energy power cap.", "",
        "These are maximum available thrust operating points at each airspeed. Actual mission cruise throttle/speed "
        "also depends on aircraft drag and the mission energy budget. The 20-degree flap input is not used by this prop curve solver. "
        "These are model predictions, not thrust-stand measurements.", "",
        "## Thrust versus airspeed", "",
        "Direct columns are actual solver outputs. Fit columns evaluate the same quadratic construction used in `prop_main`.", "",
        "| Airspeed (m/s) | 21x13 direct (N) | 21x13 fit (N) | 18x12 direct (N) | 18x12 fit (N) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for speed in range(0, 41, 5):
        a, b = [r for r in all_rows if r["airspeed_mps"] == speed]
        lines.append(f"| {speed} | {a['direct_thrust_n']:.3f} | {a['quadratic_fit_thrust_n']:.3f} | "
                     f"{b['direct_thrust_n']:.3f} | {b['quadratic_fit_thrust_n']:.3f} |")
    lines += ["", "## Static operating points at exactly 0 m/s", "",
              "| Quantity | M1/M2: 21x13E | M3: 18x12E |", "|---|---:|---:|"]
    a, b = [r for r in static_rows if r["airspeed_mps"] == 0]
    for label, key, places in (
        ("Thrust (N)", "direct_thrust_n", 4), ("Thrust (lbf)", "direct_thrust_lbf", 3),
        ("Thrust (kgf)", "direct_thrust_kgf", 3), ("RPM", "rpm", 0),
        ("Current (A)", "current_a", 3), ("Throttle fraction", "throttle_fraction", 6),
        ("Loaded battery voltage (V)", "battery_terminal_voltage_v", 3),
        ("Terminal electrical power (W)", "electrical_terminal_power_w", 2),
        ("Shaft power (W)", "shaft_power_w", 2), ("Torque (Nm)", "torque_nm", 4),
        ("Constant-point battery time (s)", "constant_point_battery_time_s", 2),
    ):
        lines.append(f"| {label} | {a[key]:.{places}f} | {b[key]:.{places}f} |")
    lines += ["", "The mission code uses 0.01 m/s as its static point: "
              f"{summaries[0]['mission_static_at_0p01_mps']['direct_thrust_n']:.6f} N for 21x13E and "
              f"{summaries[1]['mission_static_at_0p01_mps']['direct_thrust_n']:.6f} N for 18x12E. "
              "The quadratic intercept is a fitted value and differs from direct static thrust.", "",
              "Battery time above assumes constant current at the listed point and 85% usable capacity; it is not mission endurance.", "",
              "## Returned curve construction", "",
              "Each curve has the form `a*V**2 + b*V + c`, with V in m/s. "
              "The default fit speeds are 0.01, 13.34, 26.67, and 40.00 m/s. "
              "Use the fits within that interval; the 0 m/s fit entry is the intercept, just outside it.", ""]
    for item in summaries:
        lines += [f"{item['mission']} / {item['catalog_propeller']}:", "",
                  f"- Thrust [N], `[a,b,c]`: `{item['thrust_fit_abc_n']}`",
                  f"- Constant-point battery time [s], `[a,b,c]`: `{item['battery_time_fit_abc_s']}`", ""]
    lines += ["## Files and verification", "",
              "- `thrust_vs_airspeed.csv`: full-precision results at every 1 m/s from 0 to 40, plus default fit samples.",
              "- `default_prop_samples.csv`: the four default operating points for each prop.",
              "- `static_thrust.csv`: operating points at 0 and 0.01 m/s for each prop.",
              "- `inputs_and_fits.json`: exact inputs, assumptions, fits, and static results.",
              "- `thrust_vs_airspeed.png` / `.pdf`: graph of direct outputs and quadratic fits.", "",
              "Every reported point passed the solver's source-data and operating-limit checks. "
              "Thrust and torque were cross-checked against each exact catalog surface, and selected current, throttle, "
              "and electrical power were verified using the motor/battery equations.", "",
              "The existing `src/vectors.py` configuration rejects `sensor_diameter_m` in `FIXED_OPT_VALUES`, "
              "which prevents importing the `prop_main` wrapper. This report directly uses its underlying solver "
              "and reproduces its battery/motor defaults and four-point fitting. Production source files were not edited.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    export_flight_time_outputs()
    print("\n".join(lines[:30]))
    print(f"\nSaved report and data to {OUT}")


if __name__ == "__main__":
    main()
