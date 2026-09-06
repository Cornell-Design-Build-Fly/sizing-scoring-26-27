# Implemented calibration retest

Implemented practical-study training-only coefficients in `src/aero/static_derivatives.py`, called by the active coarse stability analysis. Lift and pitch share component slopes and actual CG arms; yaw uses the vertical tail quarter-chord arm and a destabilizing fuselage-volume term. Near-zero fitted terms were removed. The original study is preserved.

Reran 300 matched cases with fresh full AeroBuildup evaluations. Full reference differences: 0. Production predictions match the saved fitted equations within 1e-12. No factors were refitted during this retest.

| Dataset (45 holdout cases each) | Derivative | Before MAE /rad | After MAE /rad | Reduction |
|---|---|---:|---:|---:|
| under_55lb | CLa | 0.613161 | 0.381647 | 37.8% |
| under_55lb | Cma | 0.560850 | 0.173946 | 69.0% |
| under_55lb | Cnb | 0.078600 | 0.007853 | 90.0% |
| broad | CLa | 0.348909 | 0.385435 | -10.5% |
| broad | Cma | 0.707200 | 0.328009 | 53.6% |
| broad | Cnb | 0.130833 | 0.008664 | 93.4% |

The practical holdout excludes all geometries used to fit the coefficients (105 training cases from 35 geometries; 45 holdout cases from 15 geometries). Broad results use these same practical factors and are a transfer check; baseline geometry is shared between the studies, so the full broad sample is not entirely independent.

Practical holdout static-margin MAE fell from 0.1111 to 0.0300 MAC. Pitch has 1 false-stable and 1 false-unstable case; yaw has 4 false-stable and 1 false-unstable case. CLa remains provisional: original grouped training cross-validation MAE was about 5% worse than the old fit despite improved holdout error.

Validation: 18 tests passed across static derivatives, flight profile, tow line, and scoring. Checks cover CG translation using CLa*cos(alpha), elevator response, and cache invalidation when fuselage geometry changes. Pytest reported an unwritable cache warning, but all tests ran and passed.

Default M1/M2/M3 end-to-end pipeline completed before and after the change. Total score remains -29.19925 because flight-profile infeasibility fixes the mission penalties: M1 requires 294.5 m before its first turn (152.4 m available); M2/M3 need 127.0/143.4 m of runway (60 m available). Directional and spiral stability diagnostics change, as expected; see pipeline.json.

Limitations: comparison is at matched prescribed operating points, not a new trim calibration or flight validation. The fit covers the current NACA2412/default tails, zero tail incidence, alpha -2..10 degrees, elevator -8..8 degrees, and speed 20..42 m/s with practical M2 masses <=55 lb. Cruise trim uses its existing separate force/moment fits; other dynamic derivatives retain their previous calibration. Extrapolation remains provisional. The pitch CG relation omits axial-force contributions; full-aircraft body sideforce can reverse the fin-only yaw CG trend.

Reproduce: `venv\Scripts\python -m src.testing.retest_static_calibration` and `venv\Scripts\python -m pytest src/testing/static_derivatives_test.py src/testing/flight_profile_test.py src/testing/tow_line_test.py src/testing/opt_score_test.py -q`.
