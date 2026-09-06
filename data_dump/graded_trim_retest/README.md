# Graded failed-trim penalties

The fast cruise solver searches for the closest attempted trim over 3..50 m/s when no valid root is found. It evaluates the existing speed grid and thrust-balance roots, then refines sampled local minima. At each speed the existing equations enforce lift and pitching-moment balance; remaining thrust mismatch and angle-limit violations define the miss.

`miss = abs(thrust-drag)/(0.1*required_lift) + max(0,-4-alpha,alpha-15)/5 + max(0,abs(elevator)-20)/5`

Angles are degrees. Required lift includes sensor weight for M3. Both thrust excess and shortfall matter because throttle is fixed for each design evaluation.

`penalty = 10 + 10*miss/(1+miss)` per failed mission. Missing/nonfinite diagnostics receive 20. These are optimization scales, not calibrated aerodynamic constants. A miss of 0.1, 1, or 5 gives penalties 10.91, 15, or 18.33. The floor equals the existing maximum downstream mission penalty, so bypassing downstream analysis cannot provide a cheaper failure. Valid trim receives no trim penalty and proceeds to normal flight/stability checks.

Failed trim retains can_fly=False, infinite lap time and zero flight points. Reports expose penalty_trim and a reason with the closest attempted speed, force mismatch, alpha and elevator. The attempted state is not a valid cruise condition.

Validation: controlled force-shortfall tests show improving balance strictly lowers penalties; angle-limit tests cover elevator and alpha violations. Thirteen designs from the prior constant-score population now have thirteen distinct scores (-50.45 to -43.64). The previous winner scores -49.779. Lower scores reflect the new penalty scale, not worse physics.

This change addresses the failed-trim plateau only. Downstream takeoff/climb/landing failure penalties remain capped and may still cause plateaus. It does not establish that any design meets all mission requirements or that the sampled/refined trim miss is a global minimum.

Optimizer integration check: population 32, five generations, 192 candidate slots. Best score improved from -34.8267 at generation 1 to -28.71475. The final best design passed trim in all three missions, then failed M1 climb (264.8 m versus 152.4 m), M2 takeoff acceleration, and M3 runway length (362.3 m versus 60 m). This demonstrates progress through trim, not a fully feasible aircraft. 23 regression tests passed.
