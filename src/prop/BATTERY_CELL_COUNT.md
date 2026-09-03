# Battery cell-count optimization

`DesignVector.battery_cell_count` is the battery-pack source of truth. The
propulsion model derives nominal voltage as `3.7 V * cell_count`; nominal energy,
pack resistance, and mechanical battery mass all use that same count. The
default remains 6S. `DesignVector` also enforces the incoming 100 Wh propulsion
battery limit, so capacity must decrease as cell count increases.

## Fixed 6S/8S comparison

Run both cases with identical differential-evolution settings and seed:

```powershell
.\dbf-venv\Scripts\python.exe -m src.prop.compare_battery_cells `
  --cells 6 8 --mode fixed --maxiter 300 --popsize 25 --workers -1
```

For a shorter validation run, use `--maxiter 40 --popsize 12 --workers 12`.
Each fixed case gets its own normal top-line output directory, and the common
parent receives `comparison_summary.csv` and `comparison_summary.json`.

## Joint integer optimization

```powershell
.\dbf-venv\Scripts\python.exe -m src.prop.compare_battery_cells `
  --cells 6 8 --mode joint --maxiter 300 --popsize 25 --workers -1
```

Joint mode adds `battery_cell_count` to the SciPy integrality mask. `--cells 6
8` defines inclusive bounds, so 6S, 7S, and 8S are all legal. If the hardware
choices are only 6S and 8S, use fixed mode and select its best result; that
outer enumeration is exact for the two allowed choices and gives each choice a
full continuous-optimization budget.

Use `--mode both` to run the fixed cases and the joint case in one invocation.

## Model limits

The present model does not check ESC/full-charge voltage compatibility,
battery C-rating, or cell-count-dependent packaging volume. It also uses motor
current directly for battery sag and endurance at partial throttle. Treat the
comparison as internally consistent sizing guidance until those assumptions
are validated against the selected hardware.
