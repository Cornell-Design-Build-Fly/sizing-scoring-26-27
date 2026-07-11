# Mechanical module

The mechanical module keeps one component ledger and reports total mass, center
of gravity, static margin, and inertia for Missions 1, 2, and 3.

## Coordinate system

- SI units are used throughout.
- `x` is positive aft, `y` is positive toward the right wing, and `z` is
  positive upward.
- The main-wing root leading edge is `(0, 0, 0)`.
- `DesignVector.tail_arm` is the distance from the main-wing leading edge to
  the common horizontal/vertical-tail leading-edge station.

The existing tail-volume sizing formulas continue to use that design variable
as their simple sizing arm. Neutral-point calculations use the actual derived
quarter-chord stations of the resulting wing and horizontal tail.

Static margin is `(neutral_point_x - cg_x) / wing_chord`. The neutral point is
estimated from the wing and horizontal-tail finite-wing lift-curve slopes,
areas, aerodynamic centers, tail efficiency, downwash, and dynamic-pressure
ratio.

## Primary call

```python
from src.mech import evaluate_mechanical_module
from src.vectors import DesignVector

result = evaluate_mechanical_module(DesignVector())

for mission in ("M1", "M2", "M3"):
    properties = result.for_mission(mission)
    print(properties.total_mass_kg)
    print(properties.cg_m)
    print(properties.static_margin)

print(result.component_array("M2"))
```

The aero-compatible adapter remains:

```python
from src.mech import mech_main

cg_m, inertia_kg_m2, weight_n = mech_main(DesignVector(), mission="M2")
```

## Permanent airplane and Mission 1

The default ledger contains:

- wing structure at `0.356 / 0.36258 kg/m^2`;
- one 21 g servo at the center of each wing half;
- 100 g wing integration mass;
- a span-scaled wing spar at `0.202 / 1.18 kg/m`;
- horizontal- and vertical-tail structure at the existing
  `0.049 / 0.259 kg/m` assumption;
- one 21 g servo at each tail surface's geometric center;
- a boom spar from wing trailing edge to the aft-most tail trailing edge at
  `0.202 / 1.18 kg/m`;
- 25 g tail integration mass;
- fuselage structure at `0.300 / 0.5 kg/m`;
- 220 g landing gear at the M1 CG station and four inches below the final M1
  CG;
- battery, motor/propeller, ESC, and other electronics.

The electronics equivalent CM is solved analytically so Mission 1 has exactly
20% static margin. It is then fixed for Missions 2 and 3. Electronics are three
inches below the wing.

`electronics.py` converts that required CM into a physical longitudinal area:

| Fuselage classification | Area length | CM from front |
|---|---:|---:|
| skinny: width **and** height `< 0.127 m` | `0.254 m` | `0.135 m` |
| fat: every other cross-section | `0.228 m` | `0.119 m` |

The resolved front, CM, and back locations are available through
`result.electronics_layout`. Optional electronics CM bounds are treated as hard
feasibility checks; they never clip the CM and silently spoil the exact target.

### Linear battery, motor, and propeller masses

The default battery line passes through 4.5 Ah / 0.690 kg and the origin. Its
slope can be replaced after fitting measured battery data.

The supplied 0.390 kg motor/propeller mass remains a combined ledger item until
separate fits are configured. `LinearMassModel` provides a dependency-free
two-point framework for those fits:

```python
from dataclasses import replace
from src.mech import LinearMassModel, MechanicalModuleConfig

config = MechanicalModuleConfig()
motor_fit = LinearMassModel.from_points(
    500.0, 0.200,
    1000.0, 0.400,
    input_name="motor power [W]",
)
propeller_fit = LinearMassModel.from_points(
    10.0, 0.040,
    20.0, 0.080,
    input_name="propeller diameter [in]",
)
config = replace(
    config,
    airframe=replace(
        config.airframe,
        motor_mass_model=motor_fit,
        propeller_mass_model=propeller_fit,
        motor_sizing_value=750.0,
        propeller_sizing_value=15.0,
    ),
)
```

Both propulsion fits and both sizing inputs must be supplied together. This
prevents the combined 0.390 kg fallback from being double-counted.

## Mission 2 center-out placement

Mission 2 uses one deterministic process with no optimizer or fallback solver:

1. Use the actual M1 CG as the starting x-plane and the fuselage centerline as
   the starting y-location.
2. Put the first duck and first puck at that same x/y location.
3. Put duck centers three inches below the wing and put pucks directly below
   the duck layer.
4. For each payload type, make a lattice whose pitch is that type's bounding
   box plus the configured clearance.
5. Take valid lattice cells in increasing physical distance from the starting
   point. Symmetric forward/aft and left/right cells are considered in a fixed
   order.
6. Reject cells whose full bounding box crosses the electronics back edge, the
   forward-most tail leading edge, or either fuselage sidewall. Expansion
   continues in directions that remain open.

If the requested count does not fit this exact process,
`PayloadPlacementError` is raised. No payload is silently dropped and no more
expensive search is attempted. Mission 2 static margin is calculated after
placement; payload positions are not moved to optimize it.

The default zero vertical clearance lets the current 53 mm duck and 25.4 mm
puck layers fit beneath a duck CM at `z=-0.0762 m` in the 130 mm fuselage. Use
`Mission2Config.vertical_clearance_m` when geometry provides more space.

## Mission 3

Mission 3 starts from the same permanent M1 airplane, not from Mission 2. It
uses two 100 g mechanism masses and the existing banner density model. The
mechanism locations are explicit fixed distances from the banner center:

```python
from dataclasses import replace

config = replace(
    config,
    mission3=replace(
        config.mission3,
        forward_mechanism_distance_m=0.075,
        aft_mechanism_distance_m=0.125,
    ),
)
```

Unless an absolute banner center is supplied, the group translates together to
the configured static-margin target while those relative distances remain
fixed. Every M3 bounding box remains between the electronics back edge and the
forward-most tail leading edge; a manual center range is intersected with those
physical walls.

## Validation

```powershell
python -m src.testing.mech_test
```

The regression covers the 20% M1 target, LE-to-LE tail arm, skinny/fat
electronics layouts, component mass fits, payload ordering and boundaries,
maximum payload counts, deterministic placement, M3 fixed distances, and
positive-semidefinite inertia tensors.
