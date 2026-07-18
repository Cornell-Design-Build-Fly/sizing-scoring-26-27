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
whenever its static margin is at or below 20%; falling slightly below 10% does
not trigger a width increase.

## Primary calls

The discrete evaluator rounds payload counts to whole pieces:

```python
from src.mech import evaluate_mechanical_module
from src.vectors import DesignVector

result = evaluate_mechanical_module(DesignVector())
print(result.fuselage_width_m)
print(result.fuselage_width_increases)
print(result.for_mission("M2").static_margin)
```

The aero-compatible adapter remains:

```python
from src.mech import mech_main

cg_m, inertia_kg_m2, weight_n = mech_main(DesignVector(), mission="M2")
```

For continuous optimizer payload values, import the alternate module directly:

```python
from src.mech.main_mech_continuous import evaluate_mechanical_module_continuous

result = evaluate_mechanical_module_continuous(
    DesignVector(ducks_num=3.75, pucks_num=1.25)
)
```

## Workflow

The module performs these operations in order:

1. Build the fixed airframe: wing, wing controls and integration, wing spar,
   horizontal and vertical tails, tail controls and integration, boom spar,
   and landing gear. No fuselage or electronics are included yet.
2. Build a separate fuselage in local coordinates. Put the electronics at its
   front and pack all whole M2 payload pieces behind the electronics.
3. Install the completed loaded fuselage at the location that makes Mission 2
   static margin exactly 12%.
4. Check that the back of the installed fuselage is strictly ahead of the front
   of both tails, then remove the M2 payload mathematically and calculate
   Mission 1 static margin.
5. When the fuselage reaches the tail or Mission 1 is above 20%, increase
   fuselage width by one duck width and repeat from step 2.
6. Accept the first feasible width. The initial width plus at most four width
   increases are tested. If none works, `PayloadPlacementError` is raised with
   every attempted width and failure reason.
7. Build Mission 3 using the same fixed-distance process as before, after the
   M1/M2 fuselage has been accepted.

The landing-gear center is fixed directly under the main-wing leading edge and
four inches below the wing plane; its placement does not depend on CG.

`DesignVector.fuselage_width` is the starting width and defaults to `0.0762 m`,
which fits the `0.0762 m` puck exactly. The default duck width is `0.053 m`, so
the attempted widths are `0.0762`, `0.1292`, `0.1822`, `0.2352`, and
`0.2882 m`. The selected values are returned as
`result.fuselage_width_m` and `result.fuselage_width_increases`.

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
2. Its negative-y face touches the negative-y fuselage sidewall.
3. A row fills laterally across the fuselage, using bounding-box width plus
   configured clearance as pitch.
4. When the row is full, the next row moves aft by bounding-box length plus
   clearance.
5. Every complete payload bounding box must remain inside the fuselage width.
   The back of the electronics is the only longitudinal packing wall; the
   fuselage grows aft to the final payload edge.

Ducks and pucks retain their configured vertical layers. The default places
ducks three inches below the wing and pucks immediately below them. Whole
payload pieces determine fuselage length; no item is silently dropped.

## Continuous payload values

`main_mech_continuous.py` uses the identical workflow for the whole portions of
the requested payload amounts. It strictly floors each amount, physically packs
those whole pieces, and uses them for fuselage length and width selection.

Each fractional remainder is then added as an auditable zero-size point mass at
the floor-count M2 CG. This preserves the prior continuous method: fractional
mass changes total mass and weight, but does not change CG, static margin,
inertia about the CG, fuselage envelope, or selected fuselage width.

## Fuselage and mass ledger

The fuselage runs from the electronics front face to the aft-most whole M2
payload face. With no whole M2 payload, it ends at the electronics back face.
After installation, its back must remain strictly ahead of the nearer tail
leading edge. Its default structural model remains `0.300 / 0.5 kg/m`.

The permanent ledger still includes the battery, motor/propeller, ESC, and
other electronics. The battery model and the lightweight `LinearMassModel`
hooks for motor and propeller interpolation are unchanged.

## Mission 3

Mission 3 starts from the accepted M1 airplane, not from Mission 2. It retains
the prior banner and two-mechanism model with explicit fixed distances from the
banner center. Unless an absolute center is configured, the group translates
together toward the configured static-margin target while preserving those
distances. Its physical electronics/tail bounds are unchanged.

## Validation

```powershell
python -m src.testing.mech_test
python -m src.testing.mech_test_design_sweep_continuous
```

The regression coverage includes wing-leading-edge landing-gear placement,
fixed-airframe separation, exact 12% M2
placement, the one-sided 20% M1 check, M2 wall-to-wall row ordering, width
retry and failure signaling, continuous
fractional masses at CG, fuselage envelope sizing, M3 fixed distances, mass
interpolation hooks, and positive-semidefinite inertia tensors.
