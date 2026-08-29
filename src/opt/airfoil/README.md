# Airfoil Optimization

This package optimizes the main-wing airfoil on the existing
`DesignVector` / `ASBDesignVector` airplane.

Modeling assumptions:
- The airplane geometry, reference area, chord, span, tail, and fuselage remain
  fixed.
- The starting airfoil is `DesignVector.wing_airfoil`.
- The optimized airfoil uses AeroSandbox's 8-weight Kulfan/CST
  parameterization, seeded from the starting airfoil.
- The operating condition matches the winglet optimizer defaults:
  18 m/s, 6 deg baseline alpha, `AeroBuildup` model size `small`, and no wave
  drag.
- The objective is to match the baseline aircraft `CL` and `Cm` at the baseline
  alpha, then minimize full-aircraft `AeroBuildup` `CD`.
- Thickness constraints preserve a configurable fraction of the baseline
  thickness distribution and require at least the baseline sampled maximum
  thickness, so the optimizer cannot win mostly by making the wing airfoil thin.

Run from the repo root:

```powershell
.\venv\Scripts\python -m src.opt.airfoil.optimize
```

Artifacts are written to `data_dump/opt_airfoil/run_*/`:
- `optimized_airfoil.dat`
- `airfoil_optimization_report.json`
- `airfoil_overlay.png`
- `airfoil_polars.png`: 2D section NeuralFoil polar comparison
- `aircraft_polars.png`: full-aircraft AeroBuildup polar comparison
