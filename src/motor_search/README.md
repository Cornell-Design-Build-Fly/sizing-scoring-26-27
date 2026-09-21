# Fixed-airframe motor search

Run from the repository root:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
.\dbf-venv\Scripts\python.exe -m src.motor_search.run --workers 12
```

`airframe.json` holds the supplied geometry, tail overrides, payloads, flap and
drag settings. The battery is fixed at **8S, 3.3 Ah**. Kv, input-power limit,
no-load current, resistance and motor mass come from every CSV row, including
duplicate product rows. Mass is converted from grams to kilograms. Other
electrical and mission assumptions are the existing sizing-model defaults.

The default search allows separate M1/M2 and M3 props. Add `--single-prop` to
require one prop for all missions, with a separate `--output` directory.
`--motor-rows 0 94 324` selects zero-based CSV rows for a small validation run.

## Outputs

- `results/REPORT.md`: top 20 and assumptions.
- `results/ranked_motors.csv`: best combination per motor, with feasibility,
  scores, penalties, lap times, energy, takeoff, climb and RPM diagnostics.
- `results/ranked_single_prop.csv`: best shared prop per motor.
- `results/all_single_prop_combinations.csv`: every motor/shared-prop result.
- `results/best_m3_for_each_m12.csv`: best M3 prop conditional on each M1/M2
  prop. Rows with no viable M1/M2 flight retain the shared-prop diagnostic.
- `results/propeller_catalog.csv`: exact catalog products searched.
- `results/motor_status.csv`, `errors.csv`: input/evaluation failures, explicitly
  distinguished from valid evaluations that fail flight constraints.
- `results/motors/NNNN.json`: resumable per-row checkpoints, winning input and
  derived design values, full mission diagnostics and search counts.
- `results/manifest.json`: fixed inputs, propulsion requirements, input/source
  fingerprint, completion counts and runtime.

The same command resumes a run only if its source, motor CSV, propeller data
and inputs match. Use a new output directory after changing them. A finished
manifest must have `completed_motors == motor_count`; inspect `errored_motors`
before treating its ranking as complete.

## Search and ranking

Every two-blade, unique-geometry product retained by the existing catalog
loader is included, without an extra diameter/pitch bound. The database's
existing duplicate-geometry representative is used; excluded variants are
not separately searched. No landing-gear clearance constraint was supplied.

All shared-prop combinations are evaluated first. Any M1/M2 prop failing
either mission's propulsion checks cannot yield a feasible pair and its
remaining M3 combinations are pruned. All M3 props are considered for every
remaining M1/M2 prop. For fixed M1/M2 diameter, its pitch changes propulsion
but not mechanical placement or M3 performance. Those M3 evaluations are
therefore reused, preserving the M1/M2-diameter effect on placement and M3 CG.
Each winner is then verified against a direct, full `src.main.main` evaluation.
`considered_pairs + pruned_m12_pairs` equals the square of the catalog size.
This is exhaustive over feasible catalog pairs, not a stochastic/local search.
Infeasible diagnostics are not guaranteed best among pruned infeasible pairs.

Ranking puts combinations passing propulsion, zero-penalty aero/mechanical
checks, and scoring M2/M3 completion first; next comes propulsion feasibility,
then the official competition score minus penalties. Faster M3, M2 and M1
laps break ties. All payloads are fixed, so close scores are expected.

The interpolation adapter skips computing outside-hull values that the flight
solver rejects anyway. It preserves the original inside-hull interpolation
and the 100 RPM operating-point grid. It does not relax any flight constraints.

CSV `Watts` is interpreted as electrical input power, consistent with the
existing solver; the CSV does not label rating duration. It also lacks
motor-specific maximum voltage/current, ESC choice, and thermal limits.
Consequently **8S hardware compatibility is unverified**, and modeled winners
require those checks before hardware selection. Default battery assumptions
are 25C, 85% usable, 29.6 V nominal, 97.68 Wh and the existing sag model.
The M3 downward tow-load surrogate remains enabled even with sensor drag off.

## Supporting model corrections

The search adds optional measured motor properties and tail/drag overrides;
unspecified overrides retain the existing model behavior. It also fixes an
invalid fixed-optimizer entry for the already-derived sensor diameter, and
filters each mission's mass ledger so only its installed propeller is counted.
M2 fuselage placement uses the M2 ledger. This last correction affects previous
results, which accidentally charged both mission propellers on every flight.

Validation:

```powershell
.\dbf-venv\Scripts\python.exe -m pytest src/motor_search/test_search.py src/testing/test_propeller_safety.py src/testing/test_static_margin_feedback.py -q
```
