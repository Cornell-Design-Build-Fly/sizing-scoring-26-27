# Mechanical module

The mechanical module maintains a component ledger and reports total mass,
center of gravity, static margin, and inertia for Missions 1, 2, and 3.

## Coordinate system

- SI units are used throughout.
- `x` is positive aft, `y` is positive toward the right wing, and `z` is
  positive upward.
- The main-wing root leading edge is `(0, 0, 0)`.
- `DesignVector.tail_arm` is wing-leading-edge to tail-leading-edge distance.

Static margin is `(neutral_point_x - cg_x) / wing_chord`. The completed loaded
fuselage is placed to make Mission 2 exactly 12% MAC. Mission 1 is then accepted
whenever its static margin is at or below the configured maximum (23% by
default); falling slightly below 10% does not trigger a width increase.

The longitudinal neutral point is the aircraft aerodynamic center from the
MAE 4070 formula-sheet method. It area- and lift-slope-weights the wing and
horizontal-tail quarter-chord locations and applies the formula-sheet
horizontal-tail correction `(AR_w - 2) / (AR_w + 2)`.

## Primary calls

The discrete evaluator rounds payload counts to whole pieces:

```python
from src.mech import evaluate_mechanical_module
from src.vectors import DesignVector

result = evaluate_mechanical_module(DesignVector())
print(result.input_fuselage_width_m)
print(result.resolved_fuselage_width_m)
print(result.input_fuselage_height_m)
print(result.resolved_fuselage_height_m)
print(result.fuselage_width_increases)
print(result.for_mission("M2").static_margin)
print(result.penalty)
```

Set `disp_res=True` only for a design whose placements should be saved:

```python
result = evaluate_mechanical_module(DesignVector(), disp_res=True)
```

This writes `m2_mass_placements.csv`, `m2_mass_placements.png`,
`m3_mass_placements.csv`, and `m3_mass_placements.png` under
`data_dump/mech_results`. The PNGs contain top and side projections. The same
flag is available on the repository-level `main(dv, pv, disp_res=True)` call.

The result distinguishes the starting fuselage width supplied by the design
from the resolved width selected by placement retries. Downstream geometry
should use `resolved_fuselage_width_m`. The older `fuselage_width_m` attribute
remains as a compatibility alias for the resolved value.
Fuselage height is reported with the same input/resolved naming even though
the current placement process does not resize it.

`result.penalty` and `result.penalty_static_margin` are finite values in
`[0, 10]`. The integrated scoring path subtracts `result.penalty` alongside
the three aerodynamic penalties.

The aero-compatible adapter remains:

```python
from src.mech import mech_main

cg_m, inertia_kg_m2, weight_n = mech_main(DesignVector(), mission="M2")
```

`main_mech.py` is the small public caller/facade. The implementation is grouped
by responsibility:

- `airframe_assembly.py` builds and translates airframe, electronics, and
  fuselage mass items.
- `mission2_sizing.py` resolves payload counts and selects an accepted fuselage
  width and placement.
- `mission3_placement.py` places the fixed-spacing banner system.
- `mission_properties.py` calculates mission mass, CG, inertia, and margin.
- `mechanical_evaluation.py` coordinates those functions and assembles the
  result.

Public imports and call signatures remain unchanged. The mechanical module has
one discrete evaluation path; non-integer payload inputs are rounded to whole
pieces as described above.

## Workflow

The module performs these operations in order:

1. Build the fixed airframe: wing, wing controls and integration, wing spar,
   horizontal and vertical tails, tail controls and integration, boom spar,
   and landing gear. No fuselage or electronics are included yet.
2. Screen the permitted fuselage widths using closed-form payload row counts,
   aft extents, masses, and longitudinal moments. Geometrically impossible
   widths are skipped without constructing their mass-item ledgers.
3. For each geometrically plausible width, solve the loaded-fuselage
   installation location that makes Mission 2 static margin exactly 12%, then
   calculate Mission 1 static margin without calculating either inertia tensor.
4. When the fuselage reaches the tail or Mission 1 is above its configured
   maximum, increase
   fuselage width by one duck width and repeat from step 2.
5. Accept the first feasible width, then build its complete individual-item
   ledger and calculate M1/M2 inertia once. The initial width plus at most four width
   increases are tested. If completed placements were rejected only by Mission
   1 static margin, preserve the last such placement. It has no optimizer
   penalty while its SM is within 15 percentage points of the configured
   acceptable range; outside that buffer it receives a finite logarithmic
   penalty from 0 to 10. If no physically valid placement was completed,
   `PayloadPlacementError` is raised.
6. Build Mission 3 using the same fixed-distance process as before, after the
   M1/M2 fuselage has been accepted.

The landing-gear center is fixed directly under the main-wing leading edge and
four inches below the wing plane; its placement does not depend on CG.

`DesignVector.fuselage_width` is the starting width and defaults to `0.0762 m`,
which fits the `0.0762 m` puck exactly. The default duck width is `0.053 m`, so
the attempted widths are `0.0762`, `0.1292`, `0.1822`, `0.2352`, and
`0.2882 m`. The starting and selected values are returned as
`result.input_fuselage_width_m` and `result.resolved_fuselage_width_m`;
`result.fuselage_width_increases` reports the number of increments.

`Mission2Config.maximum_width_increases` changes the retry count. Every step is
exactly one configured duck width.
`Mission2Config.target_static_margin` sets the loaded placement target and
defaults to `0.12`.
`Mission2Config.tail_leading_edge_clearance_m` can reserve additional space
ahead of the tail; its default is zero.

## Local electronics and M2 packing

The electronics front face defines local `x=0`. The existing packaging
profiles remain available:

| Fuselage classification | Area length | CM from front |
|---|---:|---:|
| skinny: width and height `< 0.127 m` | `0.254 m` | `0.135 m` |
| fat: every other cross-section | `0.228 m` | `0.119 m` |

Electronics are three inches below the wing. After the completed fuselage is
translated onto the airplane, its absolute envelope is available through
`result.electronics_layout`.

M2 packing is deterministic and does not use the airplane CG or tail position:

1. The first item of each payload type touches the electronics back face.
2. Each row's occupied width is centered on the fuselage centerline, including
   partial final rows and full rows with unused side clearance.
3. A row fills from negative `y` toward positive `y`, using bounding-box width
   plus configured clearance as pitch.
4. When the row is full, the next row moves aft by bounding-box length plus
   clearance.
5. Every complete payload bounding box must remain inside the fuselage width.
   The back of the electronics is the only longitudinal packing wall; the
   fuselage grows aft to the final payload edge.

Ducks and pucks retain their configured vertical layers. The default places
ducks three inches below the wing and pucks immediately below them. Whole
payload pieces determine fuselage length; no item is silently dropped.

## Fuselage and mass ledger

The fuselage runs from the electronics front face to the aft-most whole M2
payload face. With no whole M2 payload, it ends at the electronics back face.
After installation, its back must remain strictly ahead of the nearer tail
leading edge. Its structural mass is based on a `0.300 kg` reference fuselage
with `0.5 m` length and `0.457 m` cross-sectional perimeter. Mass scales with
both length and the rectangular cross-sectional perimeter, `2 * (width + height)`.

The permanent ledger still includes the battery, motor/propeller, ESC, and
other electronics.

## Motor, battery, and propeller mass regressions

Motor mass is evaluated directly from `DesignVector.motor_kv` in RPM/V and
`DesignVector.motor_max_power` in W using the supplied quadratic regression.
Battery mass is evaluated from `DesignVector.batt_capacity` in Ah and
`ParameterVector.voltage` in V. Propeller mass is evaluated directly from
`DesignVector.prop_diameter_in` using the supplied cubic regression.

```python
from src.mech import evaluate_mechanical_module
from src.vectors import DesignVector, ParameterVector

design = DesignVector(
    motor_kv=335.0,
    motor_max_power=875.0,
    prop_diameter_in=17.5,
    batt_capacity=5.5,
)
parameters = ParameterVector()  # nominal battery voltage defaults to 22.2 V
result = evaluate_mechanical_module(design, parameter_vector=parameters)
```

`Motor` and `Propeller` are separate ledger items. Motor mass is not
interpolated; its quadratic model uses both Kv and maximum power. Battery mass
in grams is `(28.4 * capacity_ah + 0.63) * (V_nom / 3.7)`. Propeller mass in
grams is `0.0181235*d^3 - 0.192008*d^2 + 1.17229*d + 9.76484`, where `d` is
diameter in inches. All regression results are converted to kilograms. The
components remain separate for auditing but share the same equivalent
electronics position and feed the same electronics point-mass calculation.

## Mission 3

Mission 3 starts from the accepted M1 airplane, not from Mission 2. It retains
the prior banner and two-mechanism model with explicit fixed distances from the
banner center. Unless an absolute center is configured, the group translates
together toward the configured static-margin target while preserving those
distances. Banner height is modeled as one fifth of banner length, making its
area `banner_length^2 / 5` for the areal-density mass calculation. Its physical
electronics/tail bounds are unchanged.

## Validation

```powershell
python -m src.testing.mech_test
python -m src.testing.mech_test_design_sweep
```

The regression coverage includes wing-leading-edge landing-gear placement,
fixed-airframe separation, exact 12% M2
placement, the one-sided 20% M1 check, M2 wall-to-wall row ordering, width
retry and failure signaling, discrete payload rounding, fuselage envelope
sizing, M3 fixed distances, mass
interpolation hooks, and positive-semidefinite inertia tensors.
