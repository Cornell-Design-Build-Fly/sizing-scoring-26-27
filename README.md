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

Missions 1 and 2 share one propeller and Mission 3 flies its own
(`prop_diameter_in`/`prop_pitch_in` and `mission3_prop_diameter_in`/
`mission3_prop_pitch_in`). Nothing in the rules forces a single propeller across
flights, and the loaded-container and towed-sensor cases differ enough in weight
and cruise speed that one compromise propeller was costing both. Leaving the
Mission-3 pair unset falls back to the shared propeller.

Scored designs use real two-blade catalog propellers. Three- and four-blade
entries are excluded when the source database is loaded, and each continuous
diameter/pitch request from the optimizer is resolved to the nearest eligible
catalog geometry before mass or performance is evaluated. Reports include the
exact catalog key, blade count, maximum RPM and RPM margin, peak shaft power,
and disk power loading. Source-data interpolation in velocity/RPM is allowed,
but an operating point outside the source surface's data hull is not eligible.

Each candidate is now checked for more than steady cruise. The propulsion
model requires:

- takeoff within 75 m at 1.2 times the *takeoff-flap* stall speed;
- a conservative ground-effect credit during the ground roll: only the wing,
  tail, and interaction induced-drag terms are multiplied by 0.90; profile,
  flap, and fuselage drag receive no reduction;
- a 2.0 m/s climb-rate floor, evaluated in the flapped climb configuration.
  The rules set no minimum course altitude, so the climb-to-altitude *distance*
  check is off by default; set `climb_distance_m` to enable it;
- enough usable battery energy for takeoff, climb, straight segments,
  propulsion-limited turns, and kinetic-energy recovery after each turn;
- a five-percent usable-energy margin;
- maximum propeller RPM no greater than 90% of the APC Thin Electric limit
  `150,000 / diameter_in`;
- propeller tip Mach no greater than 0.75;
- motor, ESC/current, 25C battery-discharge, voltage-sag, and power limits.

Normal sizing runs use a fixed 8S pack and optimize capacity up to the 100 Wh
rules limit. Course timing and energy use separate fast straightaways from
slower turns. The turn solver enforces the lift, 2.5-g structural, available
thrust, current, voltage, motor-power, and turn-radius limits, then selects the
least-power propeller operating point that sustains the required turn force.

### Energy sets the cruise power

The pack holds a fixed number of watt-hours and the mission has a fixed clock,
so the aircraft is flown at the highest power those two together allow rather
than at a throttle the optimizer picks. The aerodynamically trimmed cruise speed
is an upper bound; `evaluate_mission_propulsion` searches the pack-power caps
the battery can sustain for the whole mission and flies the quickest lap among
the affordable ones. A heavier airplane needs more power for the same speed, so
the same watt-hours buy it a slower lap and fewer of them — weight limits
performance physically instead of tripping a pass/fail check afterwards.
`cruise_throttle` and `mission3_cruise_throttle` are consequently not optimizer
variables; they remain as an optional hard throttle ceiling for studies.

### Flaps

`src/aero/flaps.py` models a plain 25%-chord flap over the inboard 60% of the
span as coefficient increments only — a shifted CLmax and an added CD0, with no
geometry, so the AeroSandbox airplane and the stability derivatives are
untouched. The optimizer selects takeoff deflection from 0 to 40 degrees for
this existing hardware; 20 degrees remains the ordinary `DesignVector` default.
Landing stays fixed at 40 degrees. Flap chord, span, and type are deliberately
fixed until their hardware mass, aileron-space, and control-authority costs are
modeled.

The wing has three different maximum lift coefficients and the model keeps them
apart. **Clean** applies to cruise and to every turn on the course — crediting
the 2.5 g turn envelope with flapped lift the aircraft is not carrying would be
the easy mistake here. **Takeoff** applies to the ground roll, the
acceleration to climb speed, and the climb — the flaps come up on reaching
`cruise_altitude_m` (200 ft by default) and the course is flown clean.
**Landing** is reported as a diagnostic and nothing else: there is no
landing-speed or landing-distance constraint in the model.

Every phase is charged. The mission energy is the ground roll, the flapped
acceleration from liftoff to climb speed, the climb to cruise altitude, the
acceleration to cruise speed once the flaps retract, then the straights, the
turns, and the turn exits. Two of those did not exist before: the model used to
jump from liftoff to climb speed for free, and with `climb_altitude_m` defaulting
to zero it paid nothing at all to reach altitude.

Twenty seconds of the 300 s window is reserved for takeoff and landing
(`FLIGHT_WINDOW_S`, `GROUND_TIME_S` in `src/aero/aero_score.py`), so laps and
the energy budget both fit inside a 280 s usable window. The best-team Mission-2
and Mission-3 normalizers in `src/opt/score.py` were produced by a course model
that reserves the same 20 s.

The thresholds live in `src.prop.mission_performance.PropulsionRequirements`.
Every evaluated mission receives the detailed speed-dependent ground-roll
integration; the optimistic constant-static-thrust distance is retained only
as a diagnostic lower bound. Any propulsion failure receives a hard base
penalty larger than the maximum competition score, so the optimizer cannot
trade away flyability for payload points. A small optimizer-only margin bonus
breaks ties among fully feasible airplanes, while official scores remain
unchanged.
