# Mechanical mass-properties module

The mechanical module builds a component mass ledger and calculates mass,
center of gravity, inertia, and static margin for Missions 1, 2, and 3.

```python
from src.mech import evaluate_mechanical_module
from src.vectors import DesignVector

design = DesignVector(
    sensor_weight_kg=1.0,
    extra_shipping_containers=4,
)
result = evaluate_mechanical_module(design)

print(result.for_mission("M2").static_margin)
print(result.for_mission("M3").inertia_tensor_kg_m2)
```

## Mission payload model

The sensor is a solid steel rod with a 3-inch diameter. Its length is derived
from `DesignVector.sensor_weight_kg` using the steel density in `SensorConfig`:

```text
sensor length = sensor mass / (steel density * circular cross-sectional area)
```

Mission 2 always carries one sensor shipping container and may carry zero to
ten additional container simulators. Each container is 5 inches wide, 5 inches
high, and 2 inches longer than the sensor. Its modeled loaded mass is the
sensor mass plus the 0.5 lb empty-container mass.

Containers are loaded in this order:

1. Fill the center bay three containers wide.
2. Stack a second layer on that row, for a maximum 3-by-2 bay.
3. Fill equal-distance aft and forward bays alternately, one cell at a time.
4. Repeat at the next fore/aft distance if more bays are required.

The fuselage cross-section expands to enclose the occupied rows and layers.
The electronics, fuselage shell, permanent release mechanism, and Mission 2
containers are first assembled in local coordinates. The completed loaded
assembly is then translated onto the fixed airplane to make Mission 2 static
margin exactly `Mission2Config.target_static_margin` (12% by default).

The release mechanism is a permanent point-mass component carried in all
missions. It weighs 1/20 of the sensor and is placed at the top of the center
container stack.

Mission 3 removes the containers and carries only the sensor in addition to
the permanent airplane. The sensor center is placed at the installed Mission 1
airplane CG, so adding it does not move the CG. Its intrinsic inertia uses the
solid-cylinder equations rather than the generic rectangular-prism model.

## Outputs

`evaluate_mechanical_module()` returns a `MechanicalResult` with:

- `for_mission("M1" | "M2" | "M3")` for mission mass properties;
- input and resolved fuselage width/height;
- the installed electronics layout;
- a complete component ledger through `component_array()`;
- placement/static-margin warnings and optimizer penalty values.

Set `disp_res=True` to write Mission 2 and Mission 3 placement CSVs and images
under `data_dump/mech_results`.
