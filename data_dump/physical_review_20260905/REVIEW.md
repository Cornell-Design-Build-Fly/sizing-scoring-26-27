# Physical reasonableness review — 2026-09-05

The current model is not sufficiently physically consistent to trust its optimized aircraft or mission feasibility. Several problems are reproducible implementation errors; others are conceptual models being used beyond their validation range. No production source was changed during this review.

Scope: inventoried and syntax-parsed all 90 Python source files (19,090 lines), traced the active mechanics → propulsion → fast cruise → coarse stability → flight profile → scoring path, and inspected the alternative aerodynamic and optimization implementations. Supporting tests, demonstrations, plots, and exporters were inspected for interface and geometry consistency; they were not all executed. Historical outputs, ZIP backups, external library internals, and CAD manufacturing validity were not exhaustively audited. This is a code/model review, not experimental validation or verification of the current competition rulebook. Existing uncommitted work was retained.

The numerical examples are reproducible with `venv\Scripts\python -m data_dump.physical_review_20260905.probes` from the repository root. [Probe results](probe_results.json), [probe source](probes.py), and [source inventory with hashes](source_inventory.json) record the reviewed state. Full AeroBuildup comparisons are consistency checks against another model, not ground truth.

## Findings requiring correction

### 1. High — the optimizer's pitch/diameter constraint uses motor Kv divided by pitch

Locations: `src/opt/topline_opt.py:116`, `src/opt_test.py:67`, `src/opt/main_opt.py:6`, and `src/vectors.py:34`.

All three constraints use `x[9] / x[8]`. The current vector puts diameter at index 7, pitch at 8, and motor Kv at 9. Consequently the constrained quantity has units of RPM/V/inch and ranges from at least 200/18 = 11.11 upward; the requested interval is 0.4–0.8. No design in the current bounds can satisfy it. The default aircraft's true P/D is 0.7143, while this constraint reports 33.5.

`topline_opt` also still accesses removed `ducks_num` and `pucks_num` fields in its payload constraint/archive and marks those removed fields as integral. The optional container count is therefore not marked integral. Calling `_ducks_per_puck()` reproduces a `ValueError`. `main_opt.fitness()` additionally calls `total_score()` without its now-required payload mass argument and uses fixed 100-second lap times.

Correction: resolve indices from variable names, replace the old payload constraints/archive schema, and make final feasibility validation part of every optimization entry point.

### 2. High — coarse stability breaks the dependence of stability on CG

Locations: `src/aero/stability_analysis_coarse.py:39–40,59–61,70`; active selection at `src/aero/main_aero.py`.

The independent fitted scalings of CLa and Cma violate the moment-reference relationship. The fitted Cnb contains `-14.3090 * cnb_est`, reversing a major directional-stability contribution. Moving the CG aft shortens the stabilizing fin lever arm, yet this model predicts increasing directional stability.

For the same resolved baseline geometry, at 25 m/s and 4 degrees alpha:

| CG x (m) | Coarse neutral point (m) | Full neutral point (m) | Coarse Cnb (/rad) | Full Cnb (/rad) |
|---:|---:|---:|---:|---:|
| 0.02 | 0.14661 | 0.17326 | -0.11346 | +0.00182 |
| 0.08 | 0.17896 | 0.17366 | -0.04976 | -0.00182 |
| 0.14 | 0.21130 | 0.17406 | +0.01394 | -0.00546 |
| 0.20 | 0.24364 | 0.17446 | +0.07763 | -0.00909 |

The coarse neutral point follows the CG by almost 0.10 m despite unchanged geometry/operating point. The full model shifts only about 0.0012 m, and the directional stability sign disagrees in three of the four probes. These derivatives directly determine flight penalties.

Correction: compute forces and moments about one fixed reference, translate moments consistently to each CG, and derive stability from the same aerodynamic model used to trim. Refit physical contributions rather than independently fitting final derivatives.

### 3. High — propulsion polynomials are extrapolated far beyond their fitted speeds

Locations: `src/prop/prop_classes.py:16`, `src/prop/main_prop.py:171–182`, `src/aero/cruise_analysis_fast.py:107`.

Only four speeds, 0.01, 9.5, 19.0, and 28.35 m/s, determine both quadratics. Cruise searches through 50 m/s without a validity mask. Even the default M1 and M2 cruise solutions are approximately 40.02 and 38.81 m/s, beyond the sampled interval.

In 35 seeded in-bound propulsion designs, seven met the probe's large-error condition at 50 m/s. One 18.393 × 11.227-inch design predicts 25.12 N from the polynomial versus 1.53 N from the direct velocity/RPM solver. Another predicts positive 12.69 N where the direct model gives -6.98 N. These are internal discrepancies; the direct model also contains extrapolation and is not experimental truth.

Correction: sample the actual operating envelope and interpolate supported thrust/current operating points, carrying invalid-region flags downstream. A quadratic fit must not create feasible operating points where the underlying solver has none.

### 4. High — fuselage mass geometry and aerodynamic geometry describe different aircraft

Locations: `src/mech/airframe_assembly.py:214`, `src/mech/mission2_sizing.py:79–98`, `src/main.py:45–49`, `src/vectors.py:make_fuselage`.

Mechanics builds a payload-dependent local fuselage, then translates it to achieve exactly the target M2 CG. The downstream design vector receives only its width/height. Aero independently builds a body from `-nose_length` through the wing trailing edge and out to the tail.

For the default design, the mechanical shell occupies x = -0.23104 to +0.20016 m. The aerodynamic nose is -0.254 m, the full-width body extends to +0.307 m, and a tapered body continues aft. The modeled shell length and placement thus do not match the drag-producing body. The mechanical installation checks the rear tail clearance but does not constrain its front to the aerodynamic nose.

Correction: pass one resolved fuselage description, including longitudinal stations, to mass, packaging, drag, stability, and visualization. If the intended aircraft has a short pod plus exposed boom, represent that configuration explicitly in both models.

### 5. High — sustained turns and speed transitions are kinematic assumptions

Locations: `src/aero/flight_profile.py:193–242`, especially the fixed acceleration/deceleration constants and `_lap_phases()` signature.

The turn calculation receives speed and stall speed but no thrust curve, mass, drag model, or tow load. It limits bank by stall/nominal structural load factor and assumes the resulting turn can be sustained. It also imposes 1.5 m/s² acceleration toward cruise without checking available excess thrust; at the solved fixed-throttle cruise equilibrium, excess thrust is zero.

At 15 m/s, stall speed 10 m/s, and mass 4 kg, `_lap_phases()` returns a 72.82-second lap. The profile's own drag model requires 4.93 N in level flight but 6.50 N at 35 degrees bank. A 4.93 N propulsion system could support the former, not the latter, yet that distinction cannot enter this function. Induced drag scales with lift squared, so banked flight cannot use a level-flight feasibility check unchanged. [NASA induced-drag relation](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/).

Correction: enforce L = W/cos(bank), T ≥ D during a sustained turn, and integrate m dV/dt = T − D during transitions, using a realizable phase throttle schedule. Carry tow acceleration loads into M3 turns.

### 6. High — tow forces are included without their attachment moment

Locations: `src/aero/cruise_analysis_fast.py:73–97`, `src/aero/cruise_analysis_coarse.py:77–94`, `src/aero/cruise_analysis.py:280–297`, `src/aero/tow_line_model.py:37`.

The steady model correctly adds sensor weight to required lift and sensor drag to required thrust. None of the trim implementations adds the moment `(attachment − CG) × tow_force`. Thus the implementation silently assumes the tow force acts at the CG, despite mechanics positioning a release mechanism elsewhere.

If the modeled release mechanism is also the tow attachment, the baseline 30 m/s load produces about 3.01 N·m of pitch moment that is omitted. That number is conditional on attachment location: an actual bridle arrangement could give a different effective point, which the model must expose.

M3 stability also uses an airplane-only rigid-body model with no tow dynamics or tow-force derivatives. Its reported trim CL is calculated from airplane-only weight, even though cruise lift supports both aircraft and sensor. The stowed M3 CG/inertia are calculated by mechanics but never checked aerodynamically during takeoff/recovery.

Correction: model the effective attachment explicitly, include its moment in trim, distinguish stowed/deployed states, and label the present steady rope approximation accordingly. Steady tension is not a peak deployment/catch-load calculation.

### 7. High — motor current is treated as battery current at partial throttle

Locations: `src/prop/prop_cruise_values.py:231–284`; duplicated in `src/prop/prop_helper_functions.py:motor_check`.

`Q/Kt + I0` estimates motor current. That same current is used for battery sag, battery capacity depletion, and input power `I * V_sag`, while motor voltage is explicitly lower by the throttle ratio. In a PWM power converter, motor-side and supply-side currents differ. Under an ideal averaged model, battery current is approximately duty × motor current; converter losses modify that relationship. This is a systematic part-throttle error, usually overstating battery drain and sag, with consequences for motor selection and endurance. [Maxon explanation of motor versus supply current](https://support.maxongroup.com/hc/de/articles/360012273994-Motorstrom-Messung-bei-PWM-Leistungsendstufen).

Correction: solve battery sag and ESC/motor power balance together; distinguish motor-current, battery-current, and electrical/shaft-power limits. Clarify which power rating the motor regression represents.

### 8. Medium — climb energy is charged twice and phase power is not force-balanced

Locations: `src/aero/flight_profile.py:166–174,353–369`.

Climb rate already comes from `(T − D)V/W` using the selected propulsion operating point. The energy ledger integrates electrical power from that same operating envelope over climb time, then adds `mgh/0.70` again. The thrust power producing the altitude gain was already supplied by that electrical power. Adding a second altitude charge double-counts climb work under the model's own assumptions. If the first power term were only level-flight drag power the extra term could be appropriate, but the propulsion module returns available operating-point power, not required level-flight power. [MIT climb power balance](https://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node100.html).

Turn and transition power are selected from speed alone without solving their actual lift/drag/acceleration requirements. The climb approximation also uses L = W even at climb angles up to 30 degrees, instead of resolving the normal force balance.

Correction: integrate one consistent electrical power history, with mechanical power satisfying drag work plus changes in kinetic/potential energy. Keep mgh as a diagnostic rather than an additional charge when integrating total propulsion power.

### 9. Medium — landing omits descent, flare, and the approach-to-touchdown speed change

Locations: `src/aero/flight_profile.py:245–262,398–402`.

Landing time includes only the change from cruise to 1.3 Vs and rollout starting at 1.15 Vs. There is no descent from the modeled 200-foot flight altitude, no flare, and no elapsed time for 1.3 Vs → 1.15 Vs. The output claims the potential energy was dissipated but contains no trajectory or aerodynamic work that dissipates it.

Doubling altitude from 100 to 200 feet in the probe leaves landing time exactly unchanged at 10.5163 seconds. Braking additionally assumes full weight on the wheels immediately, neglecting residual aerodynamic lift.

Correction: model a feasible descent/approach/flare, continuous speed transitions, and wheel normal load. Resolve whether descent overlaps a scored course segment rather than simply inserting or omitting time.

The initial leg has a related bookkeeping ambiguity: the model requires takeoff/climb to occur within the first 500-foot leg, then adds them as overhead while retaining every full straight in each lap. Reconcile the trajectory with the actual lap timing boundaries before treating the score as an elapsed-time prediction.

### 10. High — payload growth can put the fuselage below the landing gear

Locations: `src/mech/airframe_assembly.py:place_landing_gear_under_wing_leading_edge`, `src/mech/models.py:351–354`, and payload-driven height resolution.

Landing-gear position/dimensions stay fixed while fuselage height grows. With three optional containers, the body bottom is z = -0.254 m; the lowest point of the modeled gear envelope is only -0.1416 m. Thus the fuselage extends 112.4 mm below the gear model. No clearance gate prevents takeoff/landing calculations for this geometry.

There is also no explicit propeller axis/ground-clearance constraint even though diameter ranges to 25 inches, and no support-polygon or rotation-clearance check. The existing gear item is an equivalent mass envelope, so its real contact geometry must be supplied to resolve these issues rather than inferred from its CG.

Correction: model wheel contact points, propeller hub/axis, and rotation attitude; size or reject the landing gear as packaging and propeller size change.

### 11. Medium — the active aero model ignores airfoil choice and uses an unvalidated stall rule

Locations: `src/aero/cruise_analysis_fast.py:26–67,128–129`, `src/aero/drag_model.py:93–105`; alternate stall calculation in `src/aero/cruise_analysis.py:calc_stall_speed`.

Fast cruise uses fixed camber/lift/moment coefficients and fixed section CLmax = 1.45. `wing_airfoil` never affects those calculations. Switching NACA 0012 → 2412 → 4412 in the probe leaves both cruise speed (31.8941 m/s) and stall speed (13.6067 m/s) identical.

The AR/(AR+2) factor is a lift-curve-slope-style approximation; it is not a general validated prediction of finite-wing CLmax. Full cruise improves airfoil dependence but evaluates CLmax at cruise Reynolds number, not the unknown stall Reynolds number, and does not solve a trimmed wing/tail stall limit. Neither path checks tail stall at high elevator deflections.

Correction: either explicitly restrict the fast model to its calibrated airfoil/geometry/Reynolds envelope or supply airfoil-dependent polars. Solve stall with Reynolds number and trim load distribution consistently. Treat the separate airfoil optimization gains as disconnected from mission predictions until integrated.

### 12. Medium — declared flight feasibility does not gate mission scoring consistently

Locations: `src/main.py:133–135`, `src/aero/aero_score.py:367–379`, `src/opt/score.py:total_score`.

`main()` passes `flight_profile_feasible` to the scoring success flags instead of `can_fly`. A design failing static/directional stability can therefore receive mission credit and unlock M3; only its final scalar objective receives a soft penalty. Soft penalties can be useful during search, but the breakdown should not present such missions as physically feasible. A small negative Cnb incurs an arbitrarily small penalty while leaving the mission reward intact.

Furthermore, `can_fly` checks static derivatives and spiral growth but ignores the computed short-period, Dutch-roll, roll, and phugoid stability results. The supplied `get_modes()` is itself an approximate rigid-aircraft model, not a full coupled eigenanalysis.

Correction: retain separate search rewards if useful, but require explicit finite, valid trim/flight/stability gates for accepted results and physical mission completion. Choose dynamic-mode acceptance limits deliberately and check the applicable modes.

## Modeling limitations and additional inconsistencies

- **Structural feasibility is not modeled.** Wing/spar/boom masses are empirical area/length scalings independent of supported payload and bending load. The 2.5 load factor is imposed without a strength, deflection, buckling, or attachment calculation. Ground score assumes a successful drop by default; no impact absorption, stroke, or sensor-retention check establishes that success. These are missing analyses, not proof that every candidate fails.
- **Atmosphere and battery settings are not shared consistently.** Fast aero uses `ParameterVector.rho`, but returned AeroSandbox operating points use their default atmosphere. Stability and full aero then use that atmosphere. Prop data have no runtime density scaling. `DesignVector.batt_energy` uses the class-level voltage rather than the parameter instance used to build the battery. Configurable battery usable fractions and mission duration likewise have multiple independent sources. Default values mostly agree; environmental/configuration changes do not propagate reliably.
- **Tail-volume definition changed without adjusting sizing.** Geometry now interprets `tail_arm` as leading-edge to leading-edge, while area remains Vh S c / tail_arm and Vv S b / tail_arm. Actual quarter-chord lever arms differ. If Vh/Vv are intended as conventional aerodynamic tail-volume coefficients, the derived areas do not enforce them exactly.
- **Inertia mathematics is sound, mass distribution is approximate.** The parallel-axis tensor and cylinder intrinsic inertia are correct, and the off-diagonal convention matches the installed AeroSandbox implementation. But a shell uses solid-box intrinsic inertia by default; the primary M2 cylinder-plus-container also defaults to a uniform box. All electronics share one position/envelope even as component mass ratios change. These can bias dynamic modes without corrupting total mass.
- **Body/appendage drag is incomplete.** Fast profile drag has no airfoil lift/deflection dependence beyond its induced terms, and no explicit gear, exposed hardware, cooling, or propwash contribution. Its empirical interacting wing/tail induced drag should be validated over the new domain. Sensor Cd = 1.2 at fixed broadside orientation is a stated approximation; orientation, rope drag, and transient swing are not solved. The steady force balance itself is reasonable under those assumptions.
- **Alternative model checks have stale interfaces.** `lifting_line.py:34` and `nonlinear_lifting_line.py:34` unpack a single Airplane as four values. NLL additionally uses undefined `cg` instead of its `xyz_ref` parameter. `aero_analysis()` does not forward trimmed elevator/incidence to its AeroBuildup wrapper. Several comparison scripts import removed `load_default_prop_database` or construct removed duck/puck/banner fields. Therefore historical comparison plots do not establish accuracy of the current code.
- **Plots are not independent trim validation.** `plot_aero_result.py` uses full AeroBuildup even when fast aero produced the trim and omits the sensor tow forces in M3 plotted equilibrium quantities. Its fixed-alpha velocity sweep is not a re-trimmed level-flight drag curve.
- **Winglet/airfoil studies are local aerodynamic studies.** Airfoil optimization sensibly constrains thickness and matches baseline CL and Cm; winglet optimization matches CL but does not enforce pitch trim or include mass/structural cost. Neither validates full mission performance. Winglet toe and tip incidence enter geometry as their sum, so they are redundant controls in the present parameterization.
- **Validation of custom inputs is incomplete.** `DesignVector` does not reject non-finite primary dimensions in its initial comparisons; `MassItem` accepts symmetric but physically impossible supplied inertia tensors without checking eigenvalues/principal-moment triangle inequalities. Optimizer bounds reduce exposure but do not make public APIs physically valid.

## Checks performed and what they establish

| Check | Result | Interpretation |
|---|---|---|
| Syntax/AST inventory of all 90 source files | Passed | Syntax validity only |
| Mechanical module's executable assertion suite | Passed | Current mass/placement bookkeeping meets its own assertions |
| Focused pytest run: flight profile, tow, scoring, endurance, mech file | 16 passed, 1 failed | Endurance test still expects M2 to require 300 s; implementation uses five laps, 200 s in that test |
| Payload archive pytest file | 1 passed, 1 failed | Removed duck/puck fields break archive test |
| Propulsion vectorized-solver check | Passed | Finite outputs and valid discrete RPM selection; does not establish battery-current physics |
| Continuous prop database check | Passed | Interpolation/blend behavior and finite extrapolation, not physical extrapolation accuracy |
| 35 seeded in-bound propulsion cases, seven speeds each | No selected positive-thrust shaft-efficiency violation found; seven large 50 m/s fit discrepancies | Limited sampling, not proof of validity everywhere |
| CG-reference, airfoil-change, landing-altitude, geometry-clearance probes | Reproduced findings above | Model consistency checks |
| End-to-end default design | Net score -29.19925; breakdown [0.80075, 0, 0, 0] | Current model rejects all flight missions |

The default model predicts M1 needs 294.5 m to take off and reach first-turn altitude against its configured 152.4 m allowance; M2 and M3 takeoff rolls are 127.0 and 143.4 m against its 60 m runway. These are model outputs, not reliable measured performance estimates given the findings above.

Recommended correction order: repair optimizer interfaces; unify geometry and aerodynamic force/moment references; replace unsupported propulsion fits and correct current/power accounting; then solve feasible flight phases, towing, landing, and ground/structural constraints. Revalidate final designs with the corrected full pipeline before using optimizer rankings for hardware decisions.
