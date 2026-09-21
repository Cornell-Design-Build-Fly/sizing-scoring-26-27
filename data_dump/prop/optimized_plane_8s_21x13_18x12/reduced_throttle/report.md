# Flight-time outputs at reduced throttle

M1/M2: 21x13E with throttle cap 0.7453. M3: 18x12E with throttle cap 0.7225.

Same motor and battery as the original run: 243.6063638114 Kv, 2119.3232459207 W, 8S / 3.37143223428186 Ah, 29.6 V nominal, 25C, 85% usable capacity (2.865717399 Ah).

Direct solver values use `time_s = usable_capacity_Ah * 3600 / current_A`. Throttle values are upper limits: the 100 RPM grid can select slightly lower throttle. No quadratic approximation or mission energy power cap is applied.

| Airspeed (m/s) | M1/M2 time (min) | M3 time (min) | M1/M2 thrust (N) | M3 thrust (N) |
|---:|---:|---:|---:|---:|
| 0 | 4.196 | 7.352 | 53.469 | 32.828 |
| 5 | 3.876 | 6.715 | 49.809 | 30.748 |
| 10 | 3.899 | 6.654 | 41.929 | 26.029 |
| 15 | 3.881 | 6.655 | 36.232 | 22.108 |
| 20 | 4.289 | 8.045 | 28.155 | 15.149 |
| 25 | 5.477 | 11.310 | 18.491 | 8.382 |
| 30 | 8.028 | 21.212 | 9.258 | 1.995 |
| 35 | 19.350 | N/A | 0.005 | 0.000 |
| 40 | N/A | N/A | 0.000 | 0.000 |

Zero airspeed is static battery runtime. Each other row assumes constant operation at that selected maximum-thrust point. Actual mission endurance also depends on drag, takeoff, climb, turns, and throttle scheduling. A positive thrust value alone does not prove sustainable flight at that speed.

N/A means the solver could not find an operating point with positive thrust under that throttle cap. The CSV retains raw solver results and status; its zero failure sentinels are not zero-second endurance predictions.

All valid points were checked against throttle, current, and motor-power limits. Battery times were verified against usable capacity and selected current.

[Full CSV at 1 m/s increments](flight_time_vs_airspeed.csv) | [Exact inputs](inputs.json)
