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
