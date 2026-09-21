# Optimized plane propulsion outputs

Calculated with the repository's propeller database and actual `solve_cruise_samples` solver.

Inputs: 8S / 3.371432234281863 Ah; 243.60636381140134 Kv; motor electrical-power cap 2119.323245920669 W; 21x13E for Missions 1/2 and 18x12E for Mission 3; both throttle upper limits 1.0.

Defaults: nominal voltage 29.6 V, 25C battery, 85% usable capacity, motor current limit 100 A. The battery sets the effective current limit to 84.285806 A. RPM search: 3000 through 20000 in steps of 100. No thrust knockdown or mission energy power cap.

These are maximum available thrust operating points at each airspeed. Actual mission cruise throttle/speed also depends on aircraft drag and the mission energy budget. The 20-degree flap input is not used by this prop curve solver. These are model predictions, not thrust-stand measurements.

## Thrust versus airspeed

Direct columns are actual solver outputs. Fit columns evaluate the same quadratic construction used in `prop_main`.

| Airspeed (m/s) | 21x13 direct (N) | 21x13 fit (N) | 18x12 direct (N) | 18x12 fit (N) |
|---:|---:|---:|---:|---:|
| 0 | 86.111 | 86.541 | 59.279 | 59.610 |
| 5 | 81.736 | 79.294 | 54.805 | 55.358 |
| 10 | 72.686 | 71.518 | 51.054 | 50.387 |
| 15 | 64.241 | 63.211 | 45.653 | 44.697 |
| 20 | 56.060 | 54.374 | 38.343 | 38.288 |
| 25 | 45.387 | 45.006 | 31.035 | 31.160 |
| 30 | 35.370 | 35.109 | 22.376 | 23.314 |
| 35 | 25.255 | 24.681 | 14.559 | 14.748 |
| 40 | 14.145 | 13.723 | 5.791 | 5.464 |

## Static operating points at exactly 0 m/s

| Quantity | M1/M2: 21x13E | M3: 18x12E |
|---|---:|---:|
| Thrust (N) | 86.1111 | 59.2788 |
| Thrust (lbf) | 19.359 | 13.326 |
| Thrust (kgf) | 8.781 | 6.045 |
| RPM | 5700 | 6300 |
| Current (A) | 64.646 | 40.669 |
| Throttle fraction | 0.978441 | 0.992535 |
| Loaded battery voltage (V) | 27.606 | 28.345 |
| Terminal electrical power (W) | 1784.62 | 1152.79 |
| Shaft power (W) | 1481.75 | 1017.64 |
| Torque (Nm) | 2.4824 | 1.5425 |
| Constant-point battery time (s) | 159.58 | 253.67 |

The mission code uses 0.01 m/s as its static point: 86.104398 N for 21x13E and 59.274651 N for 18x12E. The quadratic intercept is a fitted value and differs from direct static thrust.

Battery time above assumes constant current at the listed point and 85% usable capacity; it is not mission endurance.

## Flight time versus airspeed

These direct flight-time outputs correspond to the same maximum-available-thrust operating points as the thrust table, with throttle capped at 1.0. Each value assumes the listed current is held constant.

Usable capacity = 3.37143223428186 Ah x 85% = 2.865717399 Ah. The model computes `time_s = usable_capacity_Ah * 3600 / current_A`.

| Airspeed (m/s) | 21x13 time (s) | 21x13 time (min) | 18x12 time (s) | 18x12 time (min) |
|---:|---:|---:|---:|---:|
| 0 | 159.58 | 2.660 | 253.67 | 4.228 |
| 5 | 148.94 | 2.482 | 242.98 | 4.050 |
| 10 | 147.04 | 2.451 | 229.60 | 3.827 |
| 15 | 146.50 | 2.442 | 225.36 | 3.756 |
| 20 | 148.98 | 2.483 | 235.95 | 3.932 |
| 25 | 163.11 | 2.719 | 257.45 | 4.291 |
| 30 | 185.03 | 3.084 | 311.68 | 5.195 |
| 35 | 227.72 | 3.795 | 404.46 | 6.741 |
| 40 | 326.17 | 5.436 | 660.10 | 11.002 |

The zero-airspeed row is static battery runtime. At higher airspeeds, reduced propeller load can lower current and increase battery time while available thrust decreases. These values alone do not establish that the aircraft can sustain the listed speed. Mission endurance also depends on drag, takeoff, climb, turns, and throttle scheduling.

The quadratic flight-time fits in the next section approximate the four default samples; the table above uses direct solver values. The CSV includes both direct and fitted time in seconds and minutes.

Files: [flight-time CSV](flight_time_vs_airspeed.csv), [graph](flight_time_vs_airspeed.png), [PDF graph](flight_time_vs_airspeed.pdf). All direct times were checked against usable capacity and the saved current, and all fitted times were checked against the saved coefficients.

## Returned curve construction

Each curve has the form `a*V**2 + b*V + c`, with V in m/s. The default fit speeds are 0.01, 13.34, 26.67, and 40.00 m/s. Use the fits within that interval; the 0 m/s fit entry is the intercept, just outside it.

M1/M2 / 21x13E:

- Thrust [N], `[a,b,c]`: `[-0.010604425655518963, -1.396266043459354, 86.54065505836083]`
- Constant-point battery time [s], `[a,b,c]`: `[0.23323276775066168, -5.3667809176772705, 163.64420166584043]`

M3 / 18x12E:

- Thrust [N], `[a,b,c]`: `[-0.014378179405326858, -0.7785266295411074, 59.6098045775025]`
- Constant-point battery time [s], `[a,b,c]`: `[0.580651270558198, -13.70642635916974, 266.54513803468103]`

## Files and verification

- `thrust_vs_airspeed.csv`: full-precision results at every 1 m/s from 0 to 40, plus default fit samples.
- `default_prop_samples.csv`: the four default operating points for each prop.
- `static_thrust.csv`: operating points at 0 and 0.01 m/s for each prop.
- `inputs_and_fits.json`: exact inputs, assumptions, fits, and static results.
- `thrust_vs_airspeed.png` / `.pdf`: graph of direct outputs and quadratic fits.

Every reported point passed the solver's source-data and operating-limit checks. Thrust and torque were cross-checked against each exact catalog surface, and selected current, throttle, and electrical power were verified using the motor/battery equations.

The existing `src/vectors.py` configuration rejects `sensor_diameter_m` in `FIXED_OPT_VALUES`, which prevents importing the `prop_main` wrapper. This report directly uses its underlying solver and reproduces its battery/motor defaults and four-point fitting. Production source files were not edited.
