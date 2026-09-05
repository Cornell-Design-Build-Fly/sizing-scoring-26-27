# DBF Sizing Script Competition Year 26-27

Python dependencies for this repo are tracked in `requirements.txt`. Install them into a virtual environment before running project code.

## Install the packages

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Update `requirements.txt`

If you install a new package, refresh `requirements.txt` so the repo stays in sync with your environment.

```powershell
python -m pip freeze > requirements.txt
```

If you only want to refresh the file after several installs, run just the last command once you are done adding packages.

The current `requirements.txt` was generated from the packages already installed in this project's virtual environment.

## Towed-sensor sizing envelope

Run a nominal case, deterministic corner cases, and a seeded Monte Carlo
sample with:

```powershell
python -m src.tow.envelope --monte-carlo 24 --safety-factor 1.5
```

The command writes `data_dump/tow_envelope/tow_load_envelope.json` for sizing
code and `tow_load_cases.csv` for audit/plotting. The JSON separates nominal
mean/RMS loads, sampled mission-peak percentiles, the worst sampled limit
load, and the safety-factored ultimate load.

For programmatic use, call `evaluate_design_tow_envelope()` from
`src.tow.envelope`. It accepts a `DesignVector`, total Mission-3 aircraft mass,
and Mission-3 airspeed. The model uses a prescribed flight path; required
thrust and load factor are feasibility demands, not proof that the aircraft
can fly that path.

### Fast Mission-3 optimizer model

`src.tow.surrogate.DEFAULT_M3_DOWNWARD_LOAD_SURROGATE` is a quadratic fit to
a 2--50 lbf sensor-weight sweep. A positive residual correction prevents the
fitted peak from falling below any sweep point. Mission 3 uses 80% of that
predicted peak as its representative downward tow load by default:

```text
M3 supported mass = aircraft-only mass + 0.8 * fitted peak downward force / g
```

The actual sensor mass is removed first, so it is not counted both as onboard
weight and tether load. Pass `m3_tow_load_surrogate=None` to `src.main.main()`
to reproduce the previous onboard-sensor treatment. The current surrogate is
valid only for its documented nominal rope, course, speed, sensor geometry,
and 2--50 lbf calibration range; regenerate it when those assumptions change.

Regenerate the fit, CSV audit data, and plot with:

```powershell
python -m src.tow.surrogate --minimum-weight 2 --maximum-weight 50 --points 25 --load-fraction 0.8
```

## Propulsion flyability checks

Each candidate is now checked for more than steady cruise. The propulsion
model requires:

- takeoff within 60 m at 1.2 times stall speed;
- 200 ft of climb before the 500 ft turn marker, with takeoff roll consuming
  part of that distance, plus a 2.0 m/s climb-rate floor;
- enough usable battery energy for takeoff, climb, straight segments,
  propulsion-limited turns, and kinetic-energy recovery after each turn;
- a five-percent usable-energy margin;
- propeller tip Mach no greater than 0.75;
- motor, ESC/current, 25C battery-discharge, voltage-sag, and power limits.

Normal sizing runs use a fixed 8S pack and optimize capacity up to the 100 Wh
rules limit. Course timing and energy use separate fast straightaways from
slower turns. The turn solver enforces the lift, 2.5-g structural, available
thrust, current, voltage, and motor-power limits, then selects the least-power
propeller operating point that sustains the required turn force.

The thresholds live in `src.prop.mission_performance.PropulsionRequirements`.
Every evaluated mission receives the detailed speed-dependent ground-roll
integration; the optimistic constant-static-thrust distance is retained only
as a diagnostic lower bound. Any propulsion failure receives a hard base
penalty larger than the maximum competition score, so the optimizer cannot
trade away flyability for payload points. A small optimizer-only margin bonus
breaks ties among fully feasible airplanes, while official scores remain
unchanged.
