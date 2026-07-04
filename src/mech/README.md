# Mechanical module

The mechanical module converts one `DesignVector` into component-level mass
properties for Missions 1, 2, and 3.

## Coordinate system and geometry

- Units are SI: metres, kilograms, newtons, and `kg m^2`.
- `x` is positive aft, `y` is positive toward the right wing, and `z` is
  positive upward.
- The main-wing root leading edge is `(0, 0, 0)`.
- The existing `DesignVector.tail_arm` is quarter-chord to quarter-chord, which
  matches `ASBDesignVector.make_airplane()`. The mechanical module deliberately
  uses that same definition.

Static margin is based on an estimated **neutral point**, not a fixed center of
pressure. The neutral point weights the wing and horizontal-tail aerodynamic
centers by finite-wing lift-curve slope, area, tail efficiency, tail dynamic
pressure, and downwash. The fuselage correction is configurable and defaults to
zero until it can be calibrated with VLM/AVL/flight-test data.

## Primary call

```python
from src.mech import evaluate_mechanical_module
from src.vectors import DesignVector

result = evaluate_mechanical_module(DesignVector())

m1 = result.for_mission("M1")
print(m1.total_mass_kg)
print(m1.cg_m)
print(m1.static_margin)
print(m1.inertia_tensor_kg_m2)

# NumPy structured component ledger:
print(result.component_array("M2"))
```

For the current aero interface:

```python
from src.mech import mech_main

cg, inertia_tensor, weight_n = mech_main(DesignVector(), mission="M2")
```

## Mass model

The default configuration includes:

- wing skin/structure from area density;
- one 21 g servo at the center of each wing half;
- 100 g wing integration mass;
- span-scaled wing spar;
- horizontal- and vertical-tail structure masses using the supplied
  `49 g / 0.259 m` linear density;
- one 21 g servo at the geometric center of each stabilizer;
- tail spar from wing trailing edge to the aft-most tail trailing edge;
- 25 g tail integration mass;
- fuselage mass scaled by fuselage length;
- 220 g landing gear fixed at the Mission-1 CG station and four inches below
  the final Mission-1 CG;
- combined motor/prop, ESC, battery, and other electronics mass.

The combined electronics point is placed once for Mission 1 and then remains
fixed for Missions 2 and 3. The solver targets 15% static margin and reports a
warning if physical position bounds prevent it. A design remains acceptable by
default between 10% and 20%.

The battery model is linear in capacity and passes through 4.5 Ah / 0.690 kg.
Replace the slope with a fit to measured candidate batteries when that data is
available.

## Mission 2 packing

Every duck and puck is represented individually with a mass and rectangular
bounding box. The default duck box is `0.053 m x 0.053 m x 0.053 m`; the puck
remains `0.0762 m x 0.0762 m x 0.0254 m`. Placement rules are independent for
each type:

```python
from dataclasses import replace
from src.mech import (
    MechanicalModuleConfig,
    PlacementRules,
    RelativePayloadRules,
    evaluate_mechanical_module,
)

config = MechanicalModuleConfig()
config = replace(
    config,
    mission2=replace(
        config.mission2,
        maximum_width_m=0.15,
        maximum_height_m=0.15,
        duck=replace(
            config.mission2.duck,
            dimensions_m=(0.053, 0.053, 0.053),
            rules=PlacementRules(
                allow_forward=True,
                allow_aft=True,
                allow_above=True,
                allow_below=False,
                allow_stacking=True,
            ),
        ),
        puck=replace(
            config.mission2.puck,
            dimensions_m=(0.0762, 0.0762, 0.0254),
            rules=PlacementRules(
                allow_forward=True,
                allow_aft=False,
                allow_above=False,
                allow_below=True,
                allow_stacking=False,
            ),
        ),
        # Optional rules between the two payload types. These may be combined.
        relative_payload_rules=RelativePayloadRules(
            pucks_forward_of_ducks=True,
            pucks_below_ducks=True,
        ),
    ),
)

result = evaluate_mechanical_module(DesignVector(), config)
```

`PlacementRules` are relative to the configured compartment reference point.
`RelativePayloadRules` additionally enforce relationships between all pucks and
all ducks, such as pucks ahead of and below the ducks.

The default deterministic multi-start packer is intended for repeated optimizer
calls. `Mission2Config.solver` may instead be set to `"beam"` for a wider
combinatorial search, `"milp"` for an exact discretized solve, or `"auto"` to
try all three in increasing order of cost. Infeasible compartment dimensions or
rules raise `PayloadPlacementError`; the optimizer should catch that exception
and assign an infeasibility penalty rather than accepting a massless payload.

## Mission 3

Mission 3 starts from the same fixed Mission-1 airplane, not the Mission-2
loading. The current-year defaults are two 100 g mechanism masses and a banner
areal density of `0.233 kg / 2.9 m^2`. Banner area is calculated as
`banner_length * banner_height`:

```python
from src.mech import MechanicalModuleConfig, Mission3Config

config = MechanicalModuleConfig(
    mission3=Mission3Config(
        forward_mechanism_mass_kg=0.100,
        aft_mechanism_mass_kg=0.100,
        banner_areal_density_kg_m2=0.233 / 2.9,
        banner_height_m=0.10,
        banner_center_x_m=None,  # solve location for target static margin
    )
)
```

The forward and aft mechanisms are placed `banner_height_m / 2` ahead of and
behind the banner center. Set `banner_center_x_m` to a number to force a manual
location, or leave it as `None` to solve for the target static margin. A fixed
`banner_mass_kg` overrides the density model. The legacy
`banner_linear_density_kg_m` option remains available for a measured mass per
unit length.

## Inertia tensor

Each component contributes:

1. its intrinsic rectangular-prism inertia about its own center; and
2. its translated inertia through the full three-dimensional parallel-axis
   theorem.

The returned tensor is about that mission's aircraft CG and includes off-diagonal
products of inertia. Set a component's dimensions to zero to treat it as a point
mass, or pass a measured/custom `intrinsic_inertia_kg_m2` to `MassItem`.

## Regression test

```powershell
python -m src.testing.mech_test
```

The test checks all three missions, positive-semidefinite symmetric inertia
tensors, payload non-overlap, maximum current duck/puck counts, directional and
stacking rules, and battery-mass scaling.
