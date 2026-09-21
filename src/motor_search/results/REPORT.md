# Motor and propeller search — 8S 3300 mAh

Completed 703 / 703 motor rows; 557 have a feasible combination; 0 have input or evaluation errors.

Ranked by feasibility, then the sizing model's official mission score minus penalties. Faster M3, M2 and M1 lap times break ties. Infeasible rows are diagnostics, not recommendations.

| Rank | Manufacturer | Motor | Kv | Mass (g) | M1/M2 prop | M3 prop | Score | Feasible |
|---:|---|---|---:|---:|---|---|---:|---|
| 1 | Turnigy | AX-4008Q-620 (620) | 620 | 88 | 9x9E | 7x7E | 4.805210 | True |
| 2 | T-Motor | V3115-400 (400) | 400 | 113 | 11x12E | 11x10E | 4.804574 | True |
| 3 | Turnigy | SK3-4240-620 (620) | 620 | 194 | 9x9E | 7x6E | 4.804530 | True |
| 4 | T-Motor | MN4110-400 (400) | 400 | 152 | 12x12E | 11x8E | 4.804444 | True |
| 5 | T-Motor | Antigravity 5008-340 (340) | 340 | 135 | 14x14E | 12x12E | 4.804401 | True |
| 6 | T-Motor | MN5008-340 (340) | 340 | 135 | 14x14E | 13x10E | 4.804401 | True |
| 7 | T-Motor | Antigravity 4014-330 (330) | 330 | 149 | 14x14E | 12x10E | 4.804315 | True |
| 8 | T-Motor | MN4014-11² (330) | 330 | 149 | 14x14E | 12x10E | 4.804315 | True |
| 9 | T-Motor | MN4014-9² (400) | 400 | 149 | 14x14E | 9x9E | 4.804315 | True |
| 10 | T-Motor | MN5212-420 (420) | 420 | 205 | 11x12E | 11x8E | 4.804200 | True |
| 11 | T-Motor | MN4116-450 (450) | 450 | 212 | 10x12WE | 9x7.5E | 4.804186 | True |
| 12 | Scorpion | MII-4010-360 (360) | 360 | 152 | 14x14E | 11x10E | 4.803843 | True |
| 13 | Turnigy | ACK-4012CP-480 (480) | 480 | 152 | 10x12WE | 8x8E | 4.803814 | True |
| 14 | T-Motor | V3115-640 (640) | 640 | 113 | 7x11E | 7x7E | 4.803757 | True |
| 15 | Cobra | CM-4510/28 (420) | 420 | 211 | 11x12E | 11x8E | 4.803743 | True |
| 16 | Planet-Hobby | Joker 5050-10 V2 (450) | 450 | 285 | 10x12WE | 9x7.5E | 4.803729 | True |
| 17 | Cobra | C-4120/22 (430) | 430 | 293 | 12x12E | 11x7E | 4.803715 | True |
| 18 | T-Motor | MN4012-9² (480) | 480 | 132 | 12x12E | 8x8E | 4.803700 | True |
| 19 | Planet-Hobby | Joker 5050-8 V2 (575) | 575 | 285 | 8x8E | 8x6E | 4.803700 | True |
| 20 | Turnigy | TR4260-06² (500) | 500 | 280 | 12x12E | 8x8E | 4.803686 | True |

See `ranked_motors.csv` for all motors and `ranked_single_prop.csv` for the exhaustive one-prop comparison. Per-motor JSON files contain full winning designs, mission diagnostics and search coverage.

Assumptions: CSV Watts is treated as the model's electrical input power limit; Rm is ohms, Io is amps, and Mass is grams. Battery: 29.6 V nominal, 97.68 Wh, 85% usable, 25C (82.5 A), 100 A ESC ceiling. CSV does not specify motor-specific current, maximum voltage, or thermal-duration limits, so hardware 8S compatibility is unverified. Tail dimensions and disabled fuselage/sensor drag follow airframe.json. The existing M3 downward tow-load surrogate remains active.

Search covers every two-blade, unique-geometry propeller retained by the repository catalog, without a ground-clearance diameter restriction. Existing source-hull, 100 RPM grid, takeoff, climb, mission energy and RPM/tip-Mach checks are retained. Optimum means best feasible catalog combination under this discrete sizing model, not a hardware validation. When no feasible pair exists, the retained least-penalized diagnostic is not guaranteed optimal among pruned infeasible pairs.
