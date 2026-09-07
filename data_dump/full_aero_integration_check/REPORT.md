# Full-aero integration checks

Read-only production-code audit on main using the supplied optimizer_vector. The pasted report is not valid JSON throughout, so the complete optimizer_vector object was extracted separately. No production files were modified. Probe and numerical results are in this directory.

Existing integrated baseline score: 5.763199516606509. The full-aero calculations below are separate baseline evaluations at the thrust curves supplied by the existing main pipeline, not an optimization or revalidation at energy-limited flown speeds.

| Mission | Trim speed m/s | Cma | Cnb | Static margin | Short-period damping | Spiral doubling s |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 44.54 | -3.7971 | 0.01418 | 0.703 | 0.145 | 55.4 |
| M2 | 44.17 | -1.1894 | 0.00409 | 0.197 | 0.103 | 114.6 |
| M3 | 38.02 | -4.6537 | 0.01338 | 0.716 | 0.138 | 31.3 |

All three full trim solutions passed. Independent AeroBuildup force/moment residuals were below 1.2e-11 when normalized by supported weight and supported weight times chord. Mass inertia tensors had positive eigenvalues. All reported oscillatory modes had negative real parts; spiral real parts were positive (slow divergence). Numeric residuals validate consistency with the current equations, not physical completeness.

## Interfaces that work

- Supplied sensor and mission-specific propeller definitions load into main. Propulsion selects dimensions through propeller_for_mission; M1/M2 and M3 thrust curves differ as expected.
- Mechanics updates mass, CG and inertia when chord or tail arm changes. A +5% chord test changed M2 mass 11.2172 to 11.2374 kg and CG x 0.04812 to 0.04984 m.
- M3 aircraft inertia uses 3.5620 kg, while steady lift/stall use supported mass 9.6366 kg including the existing downward-load surrogate. Passing supported mass into full trim and aircraft-only mass properties into stability keeps the two roles separate. No extra sensor-weight term should be added again.

## Changes required before the proposed optimizer

1. Independent tail areas need explicit design fields and propagation through mechanics and ASB promotion. A manually increased horizontal-tail area of 0.07210 m2 became 0.06008 m2 after current ASB conversion. Tail mass uses a span-based structural model; it is not an independently validated tail-area mass law.
2. Tail incidence exists as an airplane-builder argument, but the default full trim routine fixes incidence at zero while solving elevator. A fixed per-design incidence must reach both trim and stability.
3. Full wing geometry contains no flap control surfaces. Current takeoff/landing flap effects are empirical increments; DEFAULT_FLAPS specifies 25% chord over the inboard 60% span. Merely swapping cruise and stability calls would leave takeoff aero coarse.
4. mission_performance computes phase drag from coarse drag_coefficients and sets energy-limited flown speeds after preliminary trim. Full aerodynamic force evaluation must be integrated into that path, and stability must be checked at the actual evaluation condition, not silently at the earlier full-throttle speed.
5. Full trim balances aerodynamic pitch only. Sensor drag and supported load enter force balance, but tow-attachment moments, thrust-line moments, and coupled sensor dynamics are not included. The baseline M3 residuals therefore apply to that reduced model.
6. Full stability uses AeroBuildup derivatives, but get_modes is explicitly a textbook approximation; the code replaces only the spiral result with a separate lateral-state-matrix result. Full derivatives are not equivalent to a complete coupled six-DOF/tow eigenanalysis. Mode objectives must state their model scope.
7. Full trim returns only a failure sentinel on solver failure; the proposed optimization needs meaningful residual diagnostics rather than one identical failed-trim score.

The full-analysis foundation is usable. These are implementation requirements for the proposed fixed-variables study; no optimizer has been implemented or launched.
