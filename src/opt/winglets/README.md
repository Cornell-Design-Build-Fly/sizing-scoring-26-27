# Winglet Optimization

This package optimizes symmetric winglets on top of the existing
`DesignVector` / `ASBDesignVector` geometry.

Modeling assumptions:
- All dimensions are meters.
- The airplane reference area, chord, and span remain those of the original
  design vector.
- The projected wingtip y-station is fixed at the original semispan, so the
  wingspan does not grow.
- The main wing remains flat to the blend start, then the final tip region is
  remorphed into a nonplanar winglet.
- The optimizer can shape the final 250 mm of each semispan.
- Chord remains full at the blend root and tapers monotonically to the tip.
- The trailing edge is directly parameterized from the original wing trailing
  edge to the final tip trailing edge; section chord and twist are derived from
  the resulting leading-edge/trailing-edge pair.
- Airfoils are blended gradually from the main-wing airfoil to the winglet
  airfoil.
- Active variables are blend length, height, tip inset/curl, cant angle,
  leading-edge sweep, leading-edge sweep exponent, tip chord ratio, taper rate,
  toe angle, tip incidence, and blend tension.
- The default objective is to match the baseline `CL` at the configured
  baseline alpha, then minimize `AeroBuildup` `CD`.

Run from the repo root:

```powershell
.\venv\Scripts\python -m src.opt.winglets.optimize
```

Artifacts are written to `data_dump/opt_winglets/run_*/`.
Each optimizer run also writes two one-winglet export files:
- `optimized_single_winglet.step`: CAD solid for Fusion editing.
- `optimized_single_winglet_mm.stl`: millimeter-scale mesh for Bambu Studio.

Both contain one right-side winglet and its root transition segment. Set
`STEP_EXPORT_SINGLE_WINGLET = False` and `STEP_EXPORT_FULL_AIRPLANE = True` in
`optimize.py` only if you want a larger context STEP export instead.

To reopen the latest optimized geometry in an interactive 3D viewer without
rerunning the optimizer:

```powershell
.\venv\Scripts\python -m src.opt.winglets.viewer
```

The viewer settings live at the top of `viewer.py`. Set `VIEW_MODE =
"three_view"` if you specifically want Matplotlib followed by `plt.show()`;
leave it as `"plotly"` for the browser-based interactive 3D model. To show the
model immediately after a fresh optimization, set
`SHOW_INTERACTIVE_AFTER_OPTIMIZATION = True` near the top of `optimize.py`.
Set `EXPORT_STEP = True` or `EXPORT_STL = True` in `viewer.py` to write a
one-winglet export for an existing run without rerunning the optimizer.
