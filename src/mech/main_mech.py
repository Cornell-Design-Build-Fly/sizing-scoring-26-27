"""Main orchestration for aircraft mass, CG, static margin, and inertia."""

from __future__ import annotations

import numpy as np

from src.mech.mass_properties import (
    center_of_gravity,
    estimate_neutral_point_x,
    geometry_stations,
    inertia_tensor_about_cg,
    static_margin,
)
from src.mech.models import (
    ALL_MISSIONS,
    MassItem,
    MechanicalModuleConfig,
    MechanicalResult,
    MissionMassProperties,
)
from src.mech.payload_placement import PayloadPlacementError, place_mission2_payload
from src.vectors import DesignVector, ParameterVector


def _integer_payload_count(value: float, name: str, warnings: list[str]) -> int:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    rounded = int(round(float(value)))
    if not np.isclose(value, rounded, atol=1e-9):
        warnings.append(
            f"{name}={value:.6g} is not an integer; the mechanical module rounded it "
            f"to {rounded}. The optimizer should eventually treat payload counts as integers."
        )
    return rounded


def _mission_properties(
    *,
    mission: str,
    items: tuple[MassItem, ...],
    design_vector: DesignVector,
    neutral_point_x_m: float,
    config: MechanicalModuleConfig,
    placement_feasible: bool = True,
    warnings: tuple[str, ...] = (),
) -> MissionMassProperties:
    cg, inertia = inertia_tensor_about_cg(items)
    mass = float(sum(item.mass_kg for item in items))
    margin = static_margin(neutral_point_x_m, cg[0], design_vector.wing_chord)
    sm_cfg = config.static_margin
    return MissionMassProperties(
        mission=mission,
        items=items,
        total_mass_kg=mass,
        weight_n=mass * ParameterVector.gravity,
        cg_m=cg,
        inertia_tensor_kg_m2=inertia,
        static_margin=margin,
        static_margin_feasible=sm_cfg.minimum <= margin <= sm_cfg.maximum,
        placement_feasible=placement_feasible,
        warnings=warnings,
    )


def _place_landing_gear_relative_to_base_cg(
    items_without_gear: tuple[MassItem, ...],
    config: MechanicalModuleConfig,
) -> MassItem:
    """Place fixed landing gear at the M1 CG and 4 inches below it.

    The gear's own mass makes "4 inches below the final CG" an implicit
    equation. It is solved analytically here rather than approximated.
    """

    airframe = config.airframe
    mass_without_gear = sum(item.mass_kg for item in items_without_gear)
    cg_without_gear = center_of_gravity(items_without_gear)
    gear_mass = airframe.landing_gear_mass_kg
    offset = airframe.landing_gear_vertical_offset_m

    final_cg_z = cg_without_gear[2] - (gear_mass / mass_without_gear) * offset
    gear_position = np.array(
        [cg_without_gear[0], cg_without_gear[1], final_cg_z - offset], dtype=float
    )
    return MassItem(
        name="Landing gear",
        mass_kg=gear_mass,
        position_m=gear_position,
        dimensions_m=airframe.landing_gear_dimensions_m,
        missions=ALL_MISSIONS,
        category="airframe",
        notes="Fixed at the Mission-1 CG station and 4 inches below the final M1 CG.",
    )


def _base_airframe_items(
    design_vector: DesignVector,
    config: MechanicalModuleConfig,
    neutral_point_x_m: float,
) -> tuple[tuple[MassItem, ...], np.ndarray, tuple[str, ...]]:
    airframe = config.airframe
    stations = geometry_stations(design_vector)
    warnings: list[str] = []

    items: list[MassItem] = []

    wing_mass = airframe.wing_areal_density_kg_m2 * design_vector.wing_area
    items.append(
        MassItem(
            name="Main wing structure",
            mass_kg=wing_mass,
            position_m=(stations.wing_center_x_m, 0.0, 0.0),
            dimensions_m=(
                design_vector.wing_chord,
                design_vector.wing_span,
                airframe.wing_surface_thickness_m,
            ),
            missions=ALL_MISSIONS,
            category="airframe",
        )
    )

    servo_x = airframe.wing_servo_chord_fraction * design_vector.wing_chord
    for side, y_position in (
        ("left", -0.25 * design_vector.wing_span),
        ("right", 0.25 * design_vector.wing_span),
    ):
        items.append(
            MassItem(
                name=f"Wing servo ({side})",
                mass_kg=airframe.wing_servo_mass_kg,
                position_m=(servo_x, y_position, 0.0),
                dimensions_m=airframe.wing_servo_dimensions_m,
                missions=ALL_MISSIONS,
                category="controls",
            )
        )

    items.append(
        MassItem(
            name="Wing integration",
            mass_kg=airframe.wing_integration_mass_kg,
            position_m=(stations.wing_center_x_m, 0.0, 0.0),
            dimensions_m=airframe.wing_integration_dimensions_m,
            missions=ALL_MISSIONS,
            category="integration",
        )
    )

    spar_width, spar_height = airframe.spar_cross_section_m
    items.append(
        MassItem(
            name="Wing spar",
            mass_kg=airframe.spar_linear_density_kg_m * design_vector.wing_span,
            position_m=(
                airframe.wing_spar_chord_fraction * design_vector.wing_chord,
                0.0,
                0.0,
            ),
            dimensions_m=(spar_width, design_vector.wing_span, spar_height),
            missions=ALL_MISSIONS,
            category="airframe",
        )
    )

    # Tail structural mass is scaled by full stabilizer span using the supplied
    # 49 g per 0.259 m datum. One 21 g servo is also included for each tail
    # surface. Servo locations default to the geometric centers because exact
    # installation coordinates have not yet been supplied.
    items.extend(
        [
            MassItem(
                name="Horizontal tail structure",
                mass_kg=airframe.tail_linear_density_kg_m * design_vector.hstab_span,
                position_m=(stations.horizontal_tail_center_x_m, 0.0, 0.0),
                dimensions_m=(
                    design_vector.hstab_chord,
                    design_vector.hstab_span,
                    airframe.tail_surface_thickness_m,
                ),
                missions=ALL_MISSIONS,
                category="airframe",
            ),
            MassItem(
                name="Vertical tail structure",
                mass_kg=airframe.tail_linear_density_kg_m * design_vector.vstab_span,
                position_m=(
                    stations.vertical_tail_center_x_m,
                    0.0,
                    0.5 * design_vector.vstab_span,
                ),
                dimensions_m=(
                    design_vector.vstab_chord,
                    airframe.tail_surface_thickness_m,
                    design_vector.vstab_span,
                ),
                missions=ALL_MISSIONS,
                category="airframe",
            ),
            MassItem(
                name="Horizontal tail servo",
                mass_kg=airframe.tail_servo_mass_kg,
                position_m=(stations.horizontal_tail_center_x_m, 0.0, 0.0),
                dimensions_m=airframe.tail_servo_dimensions_m,
                missions=ALL_MISSIONS,
                category="controls",
                notes=(
                    "Placed at the horizontal stabilizer geometric center until "
                    "a measured installation location is supplied."
                ),
            ),
            MassItem(
                name="Vertical tail servo",
                mass_kg=airframe.tail_servo_mass_kg,
                position_m=(
                    stations.vertical_tail_center_x_m,
                    0.0,
                    0.5 * design_vector.vstab_span,
                ),
                dimensions_m=airframe.tail_servo_dimensions_m,
                missions=ALL_MISSIONS,
                category="controls",
                notes=(
                    "Placed at the vertical stabilizer geometric center until "
                    "a measured installation location is supplied."
                ),
            ),
        ]
    )

    tail_spar_length = stations.tail_te_x_m - stations.wing_te_x_m
    if tail_spar_length <= 0:
        raise ValueError("Computed tail-spar length is not positive.")
    items.append(
        MassItem(
            name="Tail spar",
            mass_kg=airframe.spar_linear_density_kg_m * tail_spar_length,
            position_m=(
                0.5 * (stations.wing_te_x_m + stations.tail_te_x_m),
                0.0,
                0.0,
            ),
            dimensions_m=(tail_spar_length, spar_width, spar_height),
            missions=ALL_MISSIONS,
            category="airframe",
        )
    )

    items.append(
        MassItem(
            name="Tail integration",
            mass_kg=airframe.tail_integration_mass_kg,
            position_m=(stations.horizontal_tail_center_x_m, 0.0, 0.0),
            dimensions_m=airframe.tail_integration_dimensions_m,
            missions=ALL_MISSIONS,
            category="integration",
        )
    )

    fuselage_center_x = 0.5 * (stations.nose_tip_x_m + stations.tail_te_x_m)
    fuselage_center_z = -0.5 * design_vector.fuselage_height
    items.append(
        MassItem(
            name="Fuselage structure",
            mass_kg=airframe.fuselage_linear_density_kg_m * stations.fuselage_length_m,
            position_m=(fuselage_center_x, 0.0, fuselage_center_z),
            dimensions_m=(
                stations.fuselage_length_m,
                design_vector.fuselage_width,
                design_vector.fuselage_height,
            ),
            missions=ALL_MISSIONS,
            category="airframe",
            notes="Uniform linear-density approximation from nose tip to tail trailing edge.",
        )
    )

    fixed_items = tuple(items)
    fixed_mass = sum(item.mass_kg for item in fixed_items)
    fixed_x_moment = sum(item.mass_kg * item.position_m[0] for item in fixed_items)

    electronics_mass = airframe.electronics_mass_kg(design_vector.batt_capacity)
    target_cg_x = neutral_point_x_m - config.static_margin.target * design_vector.wing_chord
    required_electronics_x = (
        target_cg_x * (fixed_mass + electronics_mass) - fixed_x_moment
    ) / electronics_mass

    # By default, use the exact equivalent electronics-group CG required to
    # achieve the target Mission-1 static margin. This is intentionally not
    # restricted to the currently modeled fuselage or nose geometry: the
    # returned x location is a packaging/design requirement for the combined
    # motor/prop, battery, ESC, and other electronics group. Callers may still
    # provide explicit bounds when they deliberately want to test a fixed
    # packaging envelope.
    if airframe.electronics_x_bounds_m is None:
        electronics_bounds = None
        electronics_x = float(required_electronics_x)
    else:
        electronics_bounds = airframe.electronics_x_bounds_m
        electronics_x = float(np.clip(required_electronics_x, *electronics_bounds))
    electronics_z = (
        -0.5 * design_vector.fuselage_height
        if airframe.electronics_z_m is None
        else airframe.electronics_z_m
    )
    electronics_position = np.array(
        [electronics_x, airframe.electronics_y_m, electronics_z], dtype=float
    )
    if electronics_bounds is not None and not np.isclose(
        electronics_x, required_electronics_x
    ):
        warnings.append(
            "The exact electronics location required for the target static margin was "
            f"x={required_electronics_x:.4f} m, outside the explicitly configured range "
            f"[{electronics_bounds[0]:.4f}, {electronics_bounds[1]:.4f}] m. "
            f"It was clipped to x={electronics_x:.4f} m."
        )

    items.append(
        MassItem(
            name="Combined electronics",
            mass_kg=electronics_mass,
            position_m=electronics_position,
            dimensions_m=airframe.electronics_dimensions_m,
            missions=ALL_MISSIONS,
            category="propulsion_and_electronics",
            notes=(
                "Combined motor/prop, ESC, battery, and other electronics point mass. "
                "Its M1 position is fixed for M2 and M3."
            ),
        )
    )

    gear = _place_landing_gear_relative_to_base_cg(tuple(items), config)
    items.append(gear)
    return tuple(items), electronics_position, tuple(warnings)


def _mission2_items(
    design_vector: DesignVector,
    base_items: tuple[MassItem, ...],
    electronics_position: np.ndarray,
    neutral_point_x_m: float,
    config: MechanicalModuleConfig,
    warnings: list[str],
) -> tuple[MassItem, ...]:
    stations = geometry_stations(design_vector)
    duck_count = _integer_payload_count(design_vector.ducks_num, "ducks_num", warnings)
    puck_count = _integer_payload_count(design_vector.pucks_num, "pucks_num", warnings)

    m2 = config.mission2
    payload_min_x = float(electronics_position[0] + m2.electronics_aft_clearance_m)
    tail_leading_edge_x = min(
        stations.horizontal_tail_le_x_m, stations.vertical_tail_le_x_m
    )
    payload_max_x = float(tail_leading_edge_x - m2.tail_leading_edge_clearance_m)

    if m2.compartment_x_bounds_m is None:
        x_bounds = (payload_min_x, payload_max_x)
    else:
        x_bounds = (
            max(m2.compartment_x_bounds_m[0], payload_min_x),
            min(m2.compartment_x_bounds_m[1], payload_max_x),
        )
    if not x_bounds[0] < x_bounds[1]:
        raise PayloadPlacementError(
            "Mission-2 longitudinal payload region is empty after applying the "
            "electronics and tail keep-out constraints. "
            f"Electronics-aft limit: x >= {payload_min_x:.4f} m; "
            f"tail-forward limit: x <= {payload_max_x:.4f} m."
        )

    target_cg_x = neutral_point_x_m - config.static_margin.target * design_vector.wing_chord
    reference_x = (
        target_cg_x if m2.relative_reference_x_m is None else m2.relative_reference_x_m
    )

    return place_mission2_payload(
        duck_count=duck_count,
        puck_count=puck_count,
        base_items=base_items,
        target_cg_x_m=target_cg_x,
        config=m2,
        x_bounds_m=x_bounds,
        reference_x_m=reference_x,
    )


def _mission3_items(
    design_vector: DesignVector,
    base_items: tuple[MassItem, ...],
    neutral_point_x_m: float,
    config: MechanicalModuleConfig,
    warnings: list[str],
) -> tuple[MassItem, ...]:
    m3 = config.mission3
    stations = geometry_stations(design_vector)
    banner_mass = m3.resolved_banner_mass_kg(design_vector.banner_length)
    front_mass = m3.forward_mechanism_mass_kg
    aft_mass = m3.aft_mechanism_mass_kg
    total_payload_mass = banner_mass + front_mass + aft_mass

    if total_payload_mass <= 0:
        warnings.append(
            "Mission-3 masses are all zero. Set mechanism masses and either a banner "
            "mass, banner areal density, or banner linear density in Mission3Config "
            "before using M3 results."
        )
        return ()

    half_height = 0.5 * m3.banner_height_m
    offset_moment = -front_mass * half_height + aft_mass * half_height

    if m3.banner_center_x_m is None:
        base_mass = sum(item.mass_kg for item in base_items)
        base_moment = sum(item.mass_kg * item.position_m[0] for item in base_items)
        target_cg_x = (
            neutral_point_x_m - config.static_margin.target * design_vector.wing_chord
        )
        center_x = (
            target_cg_x * (base_mass + total_payload_mass)
            - base_moment
            - offset_moment
        ) / total_payload_mass
    else:
        center_x = m3.banner_center_x_m

    if m3.banner_center_x_bounds_m is None:
        center_bounds = (
            stations.nose_tip_x_m + half_height + 0.020,
            stations.wing_te_x_m + 0.25 * design_vector.tail_arm - half_height,
        )
    else:
        center_bounds = m3.banner_center_x_bounds_m

    clipped_center_x = float(np.clip(center_x, *center_bounds))
    if not np.isclose(clipped_center_x, center_x):
        warnings.append(
            f"Mission-3 banner center x={center_x:.4f} m was outside its allowed "
            f"range and was clipped to x={clipped_center_x:.4f} m."
        )
    center_x = clipped_center_x
    center_y = m3.banner_center_y_m
    center_z = m3.banner_center_z_m

    return (
        MassItem(
            name="M3 forward mechanism",
            mass_kg=front_mass,
            position_m=(center_x - half_height, center_y, center_z),
            dimensions_m=m3.forward_mechanism_dimensions_m,
            missions=frozenset({"M3"}),
            category="mission_3_payload",
        ),
        MassItem(
            name="M3 banner",
            mass_kg=banner_mass,
            position_m=(center_x, center_y, center_z),
            dimensions_m=m3.banner_packed_dimensions_m,
            missions=frozenset({"M3"}),
            category="mission_3_payload",
        ),
        MassItem(
            name="M3 aft mechanism",
            mass_kg=aft_mass,
            position_m=(center_x + half_height, center_y, center_z),
            dimensions_m=m3.aft_mechanism_dimensions_m,
            missions=frozenset({"M3"}),
            category="mission_3_payload",
        ),
    )


def evaluate_mechanical_module(
    design_vector: DesignVector,
    config: MechanicalModuleConfig | None = None,
) -> MechanicalResult:
    """Evaluate M1, M2, and M3 mass properties for one design vector."""

    config = config or MechanicalModuleConfig()
    warnings: list[str] = []
    stations = geometry_stations(design_vector)
    neutral_point_x = estimate_neutral_point_x(
        design_vector, config.neutral_point, stations
    )

    base_items, electronics_position, base_warnings = _base_airframe_items(
        design_vector, config, neutral_point_x
    )
    warnings.extend(base_warnings)

    m1 = _mission_properties(
        mission="M1",
        items=base_items,
        design_vector=design_vector,
        neutral_point_x_m=neutral_point_x,
        config=config,
        warnings=base_warnings,
    )
    if not m1.static_margin_feasible:
        warnings.append(
            f"M1 static margin is {100*m1.static_margin:.2f}%, outside the configured "
            f"{100*config.static_margin.minimum:.1f}% to "
            f"{100*config.static_margin.maximum:.1f}% range."
        )

    m2_payload = _mission2_items(
        design_vector, base_items, electronics_position, neutral_point_x, config, warnings
    )
    m2_items = base_items + m2_payload
    m2 = _mission_properties(
        mission="M2",
        items=m2_items,
        design_vector=design_vector,
        neutral_point_x_m=neutral_point_x,
        config=config,
    )
    if not m2.static_margin_feasible:
        warnings.append(
            f"M2 static margin is {100*m2.static_margin:.2f}%, outside the configured range."
        )

    m3_payload = _mission3_items(
        design_vector, base_items, neutral_point_x, config, warnings
    )
    m3_items = base_items + m3_payload
    m3 = _mission_properties(
        mission="M3",
        items=m3_items,
        design_vector=design_vector,
        neutral_point_x_m=neutral_point_x,
        config=config,
    )
    if not m3.static_margin_feasible:
        warnings.append(
            f"M3 static margin is {100*m3.static_margin:.2f}%, outside the configured range."
        )

    target_cg_x = neutral_point_x - config.static_margin.target * design_vector.wing_chord
    # Larger static margin means a farther-forward CG, so the numerical bounds
    # are [CG at maximum margin, CG at minimum margin].
    acceptable_cg_range = (
        neutral_point_x - config.static_margin.maximum * design_vector.wing_chord,
        neutral_point_x - config.static_margin.minimum * design_vector.wing_chord,
    )

    all_items = base_items + m2_payload + m3_payload
    return MechanicalResult(
        neutral_point_x_m=neutral_point_x,
        wing_aerodynamic_center_x_m=stations.wing_ac_x_m,
        horizontal_tail_aerodynamic_center_x_m=stations.horizontal_tail_ac_x_m,
        target_cg_x_m=target_cg_x,
        acceptable_cg_x_range_m=acceptable_cg_range,
        electronics_position_m=electronics_position,
        all_items=all_items,
        missions={"M1": m1, "M2": m2, "M3": m3},
        warnings=tuple(dict.fromkeys(warnings)),
    )


def mech_main(
    design_vector: DesignVector,
    mission: str = "M1",
    config: MechanicalModuleConfig | None = None,
) -> tuple[tuple[float, float, float], np.ndarray, float]:
    """Compatibility entry point for the aerodynamic module.

    Returns ``(cg, inertia_tensor, weight_n)`` for the selected mission.
    Use :func:`evaluate_mechanical_module` when the component ledger, static
    margins, warnings, or all three mission results are needed.
    """

    result = evaluate_mechanical_module(design_vector, config)
    mission_result = result.for_mission(mission)
    return (
        tuple(float(value) for value in mission_result.cg_m),
        mission_result.inertia_tensor_kg_m2.copy(),
        mission_result.weight_n,
    )
