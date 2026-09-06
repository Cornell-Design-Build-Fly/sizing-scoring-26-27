# CLa, Cma, and Cnb calibration study — September 6, 2026

**Result:** replacing independent final-derivative regressions with shared force contributions and explicit moment arms substantially improves Cma and Cnb. The proposed joint CLa/Cma fit is promising, but CLa improvement is not consistent across validation partitions. These results support a candidate replacement, not an assertion that all three derivatives are now perfectly calibrated. Production stability and cruise code were not changed.

## Evaluation design

Two studies were completed, totaling **300 full/coarse comparisons**:

- Initial broad study: 50 packing-feasible geometries, each evaluated at three longitudinal CG locations (150 cases). It samples current design bounds, including payload combinations that lead to very high mass. Forty-two of the 50 designs exceed 55 lb M2 mass.
- Practical study: another 50 geometries with M2 mass below the **55 lb limit used in the saved September 4 optimization**, again at three CG locations (150 cases). This is a historical engineering filter, not a claim about the current competition rules. M2 mass is 9.00–24.88 kg, with 0–2 optional containers.

Each study fits on 35 complete geometry groups (105 cases) and evaluates on 15 held-out geometry groups (45 cases). No geometry contributes CG variants to both partitions. Five-fold grouped cross-validation within the training geometries selects the candidate equation family. The broad study was exploratory; the practical study evaluates the resulting six candidate families on a new geometry sample. All reported holdout predictions use training-only coefficients, not the optional all-data refits.

The three CG stations are the mechanically resolved M2 CG minus 0.20 chord, that CG itself, and plus 0.20 chord. Speed, alpha, elevator, geometry, lateral CG, and vertical CG stay fixed within each triple. Both analyses use exactly the same state and moment reference. LHS proposals vary wing span/chord, tail arm, nose length, sensor mass, container count, speed, alpha, and elevator. Mechanics resolves body width/height. Invalid packaging and geometries with less than 30 mm between the wing trailing edge and tail leading edge are rejected and logged.

The selected operating envelope is 20–42 m/s, alpha -2 to +10 degrees, and elevator -8 to +8 degrees. Airfoil is NACA2412, tail NACA0012, with the installed default AeroSandbox atmosphere, beta/p/q/r zero. The comparison calls full `AeroBuildup.run_with_stability_derivatives(p=False, q=False, r=False)` and the actual production `estimate_stability_derivatives()` for its baseline. All 300 evaluations completed with finite CLa/Cma/Cnb.

**These are matched static aerodynamic derivative evaluations, not 300 fully trimmed cruise solutions.** The coarse helper's `converged=True` input enables evaluation at the deliberately prescribed state; it does not represent a trim solve. No propulsion feasibility, tow force, deployment dynamics, landing, or structural acceptance is inferred. Applying the fit to mission optimization still requires consistent cruise equations and a trimmed operating-envelope check.

## Practical-study validation results

Mean absolute error on **45 held-out cases from 15 unseen geometries**, derivatives in /rad:

| Quantity | Current calibration | Refit existing equation forms | Proposed shared-force fit | Reduction vs current |
|---|---:|---:|---:|---:|
| CLa | 0.61316 | 0.58173 | **0.38165** | **37.8%** |
| Cma | 0.56085 | 0.33469 | **0.17395** | **69.0%** |
| Cnb | 0.07860 | 0.04730 | **0.00785** | **90.0%** |

Held-out RMSE falls from 0.76466 to 0.47389 for CLa, 0.73694 to 0.22037 for Cma, and 0.09837 to 0.00969 for Cnb. Static-margin MAE, calculated as `-Cma/CLa`, falls from **11.11 to 3.00 percentage points of chord**. Maximum static-margin error remains 6.32 percentage points.

| Held-out sign decision | Current | Proposed | Remaining mistakes |
|---|---:|---:|---|
| Cma < 0 | 86.7% | **95.6%** | 1 false stable, 1 false unstable |
| Cnb > 0 | 66.7% | **88.9%** | 4 false stable, 1 false unstable |

There are 36 pitch-stable and 7 yaw-stable cases among these 45 full-model references. Sign accuracy alone is therefore insufficient; predicting every case yaw-unstable would already obtain 84.4%. The large reduction in Cnb numerical error is stronger evidence of improvement than its accuracy percentage alone. CLa is positive in every reference case, so its 100% sign agreement is not a discriminating test.

Training-group cross-validation gives a necessary qualification:

| Quantity | Current grouped-CV MAE | Proposed grouped-CV MAE |
|---|---:|---:|
| CLa | **0.42024** | 0.44114 |
| Cma | 0.57078 | **0.21307** |
| Cnb | 0.07164 | **0.00753** |

Cma and Cnb improve strongly in both CV and held-out results. **CLa does not:** its grouped-CV error is about 5% worse, despite the favorable final holdout. This prevents a blanket recommendation that the new lift calibration is universally better.

The broad study shows the same warning: held-out CLa MAE changes from 0.34891 to 0.38085, while Cma improves 0.70720 → 0.30374 and Cnb improves 0.13083 → 0.00719. The broad-study training coefficients, applied without refitting to all 150 practical cases, give MAE 0.40434 / 0.20664 / 0.00772 for CLa / Cma / Cnb. See `under_55lb/broad_model_transfer.json`.

## Proposed equation forms and factors

The following factors come from the **105 practical training cases** and are the factors that produced the held-out results above. Tiny bounded-fit coefficients below 1e-12 are written as zero. They multiply newly defined physical contributions; they are **not replacements to paste directly onto the old regression terms**.

Notation: S, c, b are main-wing reference area/chord/span; SH/SV are tail areas; cH/cV are tail chords. xCG is positive aft. Lf and Vf are the aerodynamic fuselage's length and volume. xf is its approximate volume centroid. wf is fuselage width. `alpha_deg` and `elevator_deg` are in degrees; `alpha_rad` is in radians.

```python
aw = 2*pi / (1 + 2/AR_w)
ah = 0.90 * 0.85 * (2*pi / (1 + 2/AR_h)) * SH/S
av = 0.90 * (2*pi / (1 + 2/AR_v)) * SV/S

fw = max(0, 1 - (alpha_deg/15)**2)
fh = max(0, 1 - ((alpha_deg + 0.65*elevator_deg)/20)**2)
B = wf * Lf/S * abs(sin(alpha_rad))

# Calibrated force-slope contributions, shared by lift and pitch:
W = 0.977488479 * aw * fw
H = 1.294199057 * ah * fh
F = 5.004886075 * B
CLa = W + H + F

xw = 0.25*c
xh = tail_arm + 0.25*cH
xv = tail_arm + 0.25*cV

Cma = cos(alpha_rad) * (
    W*(xCG-xw)/c
    + H*(xCG-xh)/c
    + F*(xCG-xf)/c
) + 1.784356002 * Vf/(S*c)

Cnb = 0.888619502 * av*(xv-xCG)/b - 1.937582988 * Vf/(S*b)
```

The lift/pitch candidate originally allowed additional nonnegative wing/tail constant-gain and Reynolds terms. Their coefficients reached zero in this training fit; the simplified equations above are equivalent on the studied envelope. The optional third yaw feature, a body-crossflow force times its lever arm, also reached zero in this practical fit.

The alpha/elevator attenuation functions are **chosen empirical basis functions**, not validated stall laws. The candidate should be confined to the tested envelope. Its shared-gain structure enforces `dCma/d(xCG/c) = CLa*cos(alpha)` at fixed geometry/state. This uses a low-angle normal-force approximation; an exact body-axis force treatment also needs axial-force/drag derivatives and their angle transformations. It does not claim an exactly CG-invariant `xCG - Cma/CLa*c` at finite alpha.

The fuselage centroid used in the script is `trapz(x*A(x), x)/trapz(A(x), x)` over the current fuselage cross sections. Fuselage volume and length come from the generated AeroSandbox geometry. Changing that geometry definition requires revalidation.

For reproducibility, `under_55lb/summary.json` also stores an all-150-case refit. Those coefficients are **not** the ones used for the holdout table, and have no fresh independent validation. In particular, do not combine the all-data coefficients with the training-only performance claim.

## What I recommend changing

1. **Cnb: replace the old six-term regression with explicit fin/body contributions.** The leading candidate factors are **0.88862 on the fin term** and **1.93758 on the destabilizing body-volume term**, using the vertical tail's own quarter chord. This is substantially better than simply refitting the old `-14.3090*cnb_est` structure. Retain full-model checks near Cnb = 0 because four held-out cases are still falsely classified stable.
2. **Cma: replace the independent final multiplier/intercept with consistent force-to-moment coupling.** The proposed joint formulation gives strong error and sign improvements. The first priority is the structure of the relationship; coefficient replacement alone does not fix CG dependence.
3. **CLa: treat the new factors as provisional.** Do not claim a universal improvement based solely on the favorable 45-case holdout. Validate more independent geometries and operating states, particularly the alpha/elevator combinations where errors remain large, before promoting the shared lift/pitch candidate to production. The current final CLa regression can be retained as a comparison; keeping its value while correcting reference-moment translation also improved Cma in this study, but that separate reference fit had large compensating coefficients and is not my preferred deployable model.
4. **Apply the final accepted lift/moment model consistently to cruise trim.** The existing fast/coarse cruise code has another set of calibrated slopes. Editing only the stability report would leave the trim and stability models disagreeing. This study does not change either production path.

## Correction to the earlier CG explanation

The fin's restoring contribution weakens as its lever arm shortens, but **the total aircraft Cnb does not have to decrease for every geometry and operating point**. In these full-model sweeps, Cnb decreases with aft CG in 48/50 broad geometries and 45/50 practical geometries. In the remaining cases, the full model has a different total side-force/reference response. The candidate always decreases because its fitted practical yaw model contains a stabilizing fin plus a CG-independent body couple. That leaves a known limitation even though its overall errors are much smaller.

Thus the original negative calibration factor was suspicious in the baseline probe, but its sign alone is not proof that every whole-aircraft increasing-Cnb trend is unphysical. This larger study establishes the actual numerical error and identifies cases that the proposed simple replacement still cannot represent exactly.

## Artifacts and reproducibility

- [Practical held-out comparison plot](under_55lb/holdout_comparison.png)
- [Practical raw full/coarse evaluations](under_55lb/raw_results.csv)
- [Practical predictions for all six model families](under_55lb/predictions.csv)
- [Practical train/CV/holdout metrics](under_55lb/metrics.csv)
- [Practical coefficients and metadata](under_55lb/summary.json)
- [Broad-domain metrics and coefficients](summary.json)
- [Dataset and algebraic validation checks](validation_checks.json)

Run from the repository root:

```powershell
.\venv\Scripts\python -m src.testing.calibrate_static_derivatives
.\venv\Scripts\python -m src.testing.calibrate_static_derivatives --practical
# Refit/replot saved evaluations without rerunning AeroBuildup:
.\venv\Scripts\python -m src.testing.calibrate_static_derivatives --practical --reuse
```

Checks verified 150 finite reference cases per study, 50 distinct geometry groups per study, disjoint 35/15 training/holdout geometry partitions, unchanged full CLa across each fixed-state CG triple, and the candidate's algebraic pitch-reference relationship. Saved hashes/version information identify the evaluated code. Existing mechanics/fuselage discrepancies from the earlier review remain; this is calibration against AeroBuildup for the current generated geometry, not experimental confirmation of the real aircraft.
