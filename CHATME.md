# CHATME

Purpose: compact bot-to-bot handoff log for this repo.

How to use:
- Add new entries at the top under `## Session Log`.
- Keep each entry short and delta-focused.
- Prefer links to files over long explanations.
- Do not repeat unchanged context from older entries unless it is newly relevant.
- If an old entry becomes obsolete, add a one-line correction in the new entry instead of rewriting history.
- Mention outputs written to `data_dump` so later bots know what artifacts already exist.

Suggested entry format:
```md
### YYYY-MM-DD - Agent
Changed:
- ...

Learned:
- ...

Artifacts:
- ...

Open notes:
- ...
```

## Stable Context

Repo basics:
- Python project rooted at `src/`.
- Main design geometry source of truth is [src/vectors.py](src/vectors.py).
- Testing/demo scripts live in [src/testing](src/testing).
- Saved analysis artifacts should go into [data_dump](data_dump).
- Virtual environment is expected at `venv`; most scripts were run with `.\venv\Scripts\python`.

Geometry model:
- `DesignVector` / `ASBDesignVector` use meters everywhere.
- Primary user-set geometry variables are `wing_span`, `wing_chord`, `tail_arm`, `nose_length`.
- Tail sizing is derived from fixed constants `V_H`, `V_V`, `AR_H`, `AR_V`.
- Fuselage is generated from one main driver, `nose_length`, plus a fixed square-ish body section of `0.13 m x 0.13 m`.
- Fuselage intent:
  - nose tip starts at `wing_le_x - nose_length`
  - short transition into full body width
  - full body through wing trailing edge
  - taper aft to tail tip

Aero solver expectations:
- `VLM` is fast and visually rich, but essentially lifting-surface/inviscid; it does not give a real stall break and does not properly model fuselage drag.
- `NonlinearLiftingLine` includes nonlinear/viscous sectional behavior and can show stall-like softening, but is much slower and can fail to converge at higher alpha.
- `LiftingLine` is fast on this geometry and is a good middle ground.
- `AeroBuildup` is the fastest of the ASB methods used here and is the most practical current path for fuselage/body drag effects, but it is not a panel/wake method.

## Session Log

### 2026-09-03 - Claude
Changed (restores static-margin feedback lost in `5e9bb57`):
- `stability_analysis_coarse` now reports the **geometric** static margin,
  `(x_np - x_cg) / c`, using the same `estimate_aerodynamic_center_x` the
  mechanical module places against. The old `-Cma/CLa` value is kept as
  `StabilityResult.static_margin_from_cma` (diagnostic only).
- `buffered_static_margin_penalty` moved to `src/mech/mass_properties.py` (one
  implementation) and gained `StaticMarginConfig.optimizer_penalty_floor`.
- `mechanical_evaluation` applies that penalty across
  `StaticMarginConfig.penalized_missions` (default `("M2", "M3")`), taking the
  worst mission so the total stays on the 0-10 scale.
- Mission-2 placement is now **clamped** to keep the body ahead of the tail and
  penalized by the static-margin error, instead of raising
  `PayloadPlacementError`. It still raises when the block cannot fit at any
  placement.
- Non-convergent cruise returns `penalty=MAX_PENALTY` instead of 0.
- The 55 lb overweight penalty is graded (`10 + 0.5/kg` over the limit).
- `best_design_report.json` gained `penalties` and `static_margin` blocks;
  `breakdown.sum_before_penalties - penalties.total` now equals `score` exactly.

Learned:
- `-Cma/CLa` was never a static margin. `cma = -0.867205 + 0.448798 * cma_est`
  scales `dCma/dx_cg` by 0.4488, breaking the identity `dCma/dx_cg = CLa / c`,
  so the implied neutral point **moved with the CG** (0.186 -> 0.219 m over a
  +/-6 cm sweep). Measured aero slope was -1.1337 /m against the geometric
  -2.5368 /m (= -1/chord). The full `stability_analysis.py` was already
  geometric; only the coarse model was wrong.
- Static margin produced **zero** penalty across 901 real designs before this:
  mech penalty hardcoded 0, M2 SM pinned to exactly 0.1200 by construction
  (std 0.0), M3 SM a free float (0.117 -> 1.062), aero `SM > 0` never tripping.
- Placement is one rigid translation for {fuselage, electronics, release
  mechanism, containers}; SM is exactly linear in it at -1/chord, so 1 cm of
  block travel = 2.5% MAC. **One DOF cannot independently set three mission
  static margins** - the M1-M2 spread (~0.10 MAC) is structural.
- `MechanicalModuleConfig` already validates that
  `Mission2Config.target_static_margin` (0.12) lies inside
  `StaticMarginConfig` [minimum, maximum]; only the nominal `target` (0.20)
  differs, so the two are range-consistent, not contradictory.
- Regression over 300 fixed designs: best known design unchanged at 7.2295,
  134/150 of the converged population unchanged, 16 dropped by exactly 10.0
  (non-convergent cruise now costing what it should), 28 previously-rejected
  designs now scored and ranked instead of tying at `BAD_OBJECTIVE`.

Artifacts:
- No new `data_dump` outputs. `src/testing/test_static_margin_feedback.py`
  covers all five fixes; each was mutation-checked by reverting the fix and
  confirming the test fails.

Open notes:
- **Not changed, needs a decision.** The solid-steel sensor model
  (`src/vectors.py:40`) forces `MIN_SENSOR_WEIGHT_KG = 5.456 kg (12.03 lb)` and
  ties length to weight; last year both were independent with a 1 lb floor.
  This 12x jump in minimum payload is what drives the 55 lb overruns.
- **Not changed.** `cnb_est` is multiplied by -14.309
  (`stability_analysis_coarse.py:59`); the `Cnb > 0` gate fails ~90% of the time
  off the converged region. Same class of regression problem as the Cma slope.
- `Cma` itself still carries the -0.867/0.4488 calibration and still feeds
  `get_modes`, so the dynamic modes are unchanged by this session. Correcting
  Cma would shift short-period/phugoid/spiral and needs its own validation.
- Placement still has one DOF. Adding a second (sliding the battery/electronics
  relative to the containers) is the next real gain.
- Pre-existing, unrelated: `test_continuous_lap_scoring.py::test_successful_m2_
  waives_m1_and_unlocks_m3` fails under pytest (passes on original code too).
  `relaxed_total` sits 4.6e-5 below `official_total` because `_official_mass_kg`
  rounds sensor mass to 0.01 lb while the relaxed path does not.
- Pre-existing, unrelated: `src/testing/aero_score_test.py` errors on stale
  `CruiseCondition` / `StabilityResult` constructor signatures.
- Running `python -m src.testing.<name>` on a pytest-style file only imports it
  and runs nothing. Use `pytest` for those.
- `pytest` was installed into `venv` for this work; it is not in
  `requirements.txt`.

### 2026-09-03 - Codex
Changed:
- Centralized battery cell-count validation and nominal-voltage/energy
  derivation in `src/prop/prop_classes.py`; 6S remains the default.
- Added fixed and joint-integer battery modes to the top-line optimizer. The
  prop-side `compare_battery_cells.py` runner supports matched fixed runs and
  an inclusive integer cell-count range.
- Mechanical battery mass, propulsion, and M2 energy scoring now share the
  `DesignVector.battery_cell_count` source of truth.

Learned:
- At equal 4.5 Ah, 6S -> 8S increases nominal voltage, energy, modeled pack
  resistance, and modeled battery mass by 33.3%, as expected.
- Forty-generation fixed runs scored 5.41057 (6S) and 5.39060 (8S); the
  comparable joint 6-8S integer run selected 8S at 5.38105. These close,
  max-iteration-limited results are validation runs, not a final design choice.
- If only 6S and 8S are legal, outer enumeration is more search-efficient and
  exact than spending one DE population across both choices. Joint bounds
  `(6, 8)` also admit 7S.

Artifacts:
- Fixed comparison: `data_dump/prop/battery_cell_comparison_40gen/comparison_20260903_181155/`
- Joint comparison: `data_dump/prop/battery_cell_joint_40gen/comparison_20260903_181811/`

Open notes:
- Battery packaging volume, C-rating, ESC/full-charge voltage compatibility,
  and PWM-side battery-current accounting are not yet modeled.

### 2026-07-03 - OpenAI
Changed:
- Implemented the complete mechanical mass-properties module in [src/mech](src/mech):
  - component mass ledger and configuration dataclasses;
  - wing/tail neutral-point estimate for static margin;
  - M1 electronics placement and fixed landing-gear placement;
  - constrained M2 duck/puck 3-D packing with non-overlap and static-margin targeting;
  - configurable M3 three-mass banner-system placement;
  - mission CG, weight, and full 3x3 inertia tensors using intrinsic inertia plus the parallel-axis theorem.
- Added the aero-compatible `mech_main()` entry point and richer `evaluate_mechanical_module()` result.
- Added [src/mech/README.md](src/mech/README.md) and regression coverage in [src/testing/mech_test.py](src/testing/mech_test.py).
- Made the AeroSandbox import in [src/vectors.py](src/vectors.py) lazy so the mechanical module can use `DesignVector` without importing AeroSandbox.
- Updated the mechanical defaults with the supplied current-year data:
  - 53 mm cubic duck bounding boxes;
  - `49 g / 0.259 m` linear structural density for each stabilizer;
  - one 21 g servo on each stabilizer;
  - two 100 g M3 mechanisms;
  - banner areal density of `0.233 kg / 2.9 m^2`, with area computed from banner length and height.

Learned:
- `DesignVector.tail_arm` is quarter-chord to quarter-chord in the current geometry, not leading-edge to leading-edge.
- With the updated tail mass and servo data, the baseline M1 result is approximately 3.264 kg with 6.93% estimated static margin. The unconstrained combined-electronics location required for the 15% target is approximately `x=-0.2945 m`, which lies about 40 mm ahead of the modeled nose tip; the module clips it to the physical bound and correctly flags M1 as outside the 10-20% range.
- Exact horizontal- and vertical-tail servo installation coordinates have not been supplied, so both servos currently sit at their stabilizer geometric centers.
- Banner density is interpreted as `0.233 kg / 2.9 m^2` rather than `0.233 g / 2.9 m^2`.
- Fuselage structural mass uses a `0.300 kg / (0.5 m * 0.457 m perimeter)`
  shell-area density, so it scales with both fuselage length and cross-sectional
  perimeter as the selected fuselage width changes.
- Motor, propeller, and battery masses support piecewise-linear interpolation
  through any number of catalogue points. Motor power, propeller diameter, and
  battery capacity come directly from `DesignVector`; interpolated motor and
  propeller masses remain separate ledger items at the shared electronics
  equivalent-CM location.

Artifacts:
- No new `data_dump` artifacts. Run `python -m src.testing.mech_test` for the baseline report.

Open notes:
- The optimizer should catch `PayloadPlacementError` and penalize packing-infeasible designs.
- Duck/puck counts remain continuous optimizer variables and are rounded by the mechanical module; they should eventually be implemented as integer/discrete optimizer variables.


### 2026-05-27 - Codex
Changed:
- Renamed the design-vector module to [src/vectors.py](src/vectors.py) and updated repo imports to point at `src.vectors`.
- Added optimizer-facing helpers in [src/vectors.py](src/vectors.py):
  - `OPT_VARS`
  - `DesignVector.to_array()`
  - `DesignVector.from_array()`
  - `DesignVector.bounds()`
- Expanded [src/vectors.py](src/vectors.py) with mission/prop fields used by scoring and optimization:
  - `ducks_num`
  - `pucks_num`
  - `banner_length`
  - `batt_capacity`
  - derived `batt_energy`
- Implemented mission scoring in [src/opt/score.py](src/opt/score.py) with `gm_score`, `m1_score`, `m2_score`, `m3_score`, and `total_score()`.
- Updated [src/opt/main_opt.py](src/opt/main_opt.py) to build `DesignVector` instances from optimizer arrays and wrapped the DE run in `run_optimization()` so imports do not start optimization as a side effect.

Learned:
- Current optimization/scoring pipeline is `DesignVector.from_array(x) -> total_score(dv, ...) -> -score` for SciPy DE.
- The optimizer currently treats `ducks_num` and `pucks_num` as continuous variables and they are truncated to ints inside `DesignVector.__post_init__`.
- `total_score()` in [src/opt/score.py](src/opt/score.py) still expects externally supplied lap times; aero/performance coupling into the optimizer is not wired yet.

Artifacts:
- No new `data_dump` artifacts were written in this session.

Open notes:
- `DesignVector.from_array()` currently zips `x` with `OPT_VARS` without a length check, so mismatches could fail silently.
- Score breakdown printing in [src/opt/score.py](src/opt/score.py) will likely spam output during large optimization runs.
- Check whether `pucks_num` default / intended feasible range is correct: the current default and optimizer bounds do not appear to match.

### 2026-05-17 - Codex
Changed:
- Added ASB-ready geometry helpers in [src/vectors.py](src/vectors.py):
  - `ASBDesignVector`
  - `make_airplane()`
  - fuselage generation from design vector
- Added aero wrappers:
  - [src/aero/vlm.py](src/aero/vlm.py)
  - [src/aero/nonlinear_lifting_line.py](src/aero/nonlinear_lifting_line.py)
  - [src/aero/lifting_line.py](src/aero/lifting_line.py)
  - [src/aero/aerobuildup.py](src/aero/aerobuildup.py)
  - exports updated in [src/aero/__init__.py](src/aero/__init__.py)
- Added/updated testing and visualization scripts:
  - [src/testing/asb_design_vector_airplane.py](src/testing/asb_design_vector_airplane.py)
  - [src/testing/vector_test.py](src/testing/vector_test.py)
  - [src/testing/asb_three_way_compare.py](src/testing/asb_three_way_compare.py)
  - [src/testing/nll_design_vector_viewer.py](src/testing/nll_design_vector_viewer.py)
  - [src/testing/geometry_flow_showcase.py](src/testing/geometry_flow_showcase.py)

Learned:
- Current ASB solver behavior on this repo’s geometry:
  - `VLM` average runtime in sweeps: about `0.08 s/case`
  - `LiftingLine` average runtime in sweeps: about `0.07 s/case`
  - `AeroBuildup` average runtime in sweeps: about `0.025 s/case`
  - `NonlinearLiftingLine` average runtime in sweeps: about `3.7 s/case`, with failures near the top of the alpha sweep
- `VLM` outputs only totals like `CL`, `CD`, `Cm`, etc.; no useful drag breakdown.
- `NonlinearLiftingLine` raw ASB output includes `CDi` and `CDp`.
- `LiftingLine` raw ASB output includes `wing_aero` and `fuselage_aero_components`.
- `AeroBuildup` raw ASB output includes `D_induced` and `D_profile`; wrapper stores these as `D_induced` and `D_profile` in `AirplaneAnalysisResult`.
- Weird `L/D` blow-ups near small `CL` were caused mainly by VLM having extremely small drag near zero lift; adding a baseline `CD0` is a reasonable future patch if needed.
- `VLM` and current `NonlinearLiftingLine` are not the right tools if accurate fuselage drag is the goal; use `AeroBuildup` or compare against `LiftingLine`.

Artifacts:
- VLM vs NLL alpha sweep:
  - [data_dump/vector_test_alpha_sweep.csv](data_dump/vector_test_alpha_sweep.csv)
  - [data_dump/vector_test_alpha_sweep.png](data_dump/vector_test_alpha_sweep.png)
- NLL viewer:
  - [data_dump/nll_viewer_summary.json](data_dump/nll_viewer_summary.json)
  - [data_dump/nll_viewer_plotly.html](data_dump/nll_viewer_plotly.html)
  - [data_dump/nll_viewer_wireframe.png](data_dump/nll_viewer_wireframe.png)
- Three-way ASB comparison:
  - [data_dump/asb_three_way_compare.csv](data_dump/asb_three_way_compare.csv)
  - [data_dump/asb_three_way_compare.png](data_dump/asb_three_way_compare.png)
- Geometry flow showcase:
  - [data_dump/geometry_flow_showcase_summary.json](data_dump/geometry_flow_showcase_summary.json)
  - [data_dump/geometry_flow_showcase_wake.html](data_dump/geometry_flow_showcase_wake.html)
  - [data_dump/geometry_flow_showcase_wireframe.png](data_dump/geometry_flow_showcase_wireframe.png)
  - [data_dump/geometry_flow_showcase_downwash.png](data_dump/geometry_flow_showcase_downwash.png)
  - [data_dump/geometry_flow_showcase_loading.png](data_dump/geometry_flow_showcase_loading.png)

Open notes:
- `NonlinearLiftingLine` viewer script supports `ASB_HEADLESS=1` to skip opening windows while still saving artifacts.
- `geometry_flow_showcase.py` is the current “cool visual” script; it gives a strong VLM panel/wake presentation without the slow nonlinear solve.
- If future work wants more realistic total drag, likely next step is to combine:
  - lift/moments from `LiftingLine` or `VLM`
  - fuselage/body drag from `AeroBuildup`
- If future work wants corrected `L/D`, add a configurable baseline `CD0` before computing `CL/CD`.
