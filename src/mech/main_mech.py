"""Main orchestration for aircraft mass, CG, static margin, and inertia."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.mech.electronics import ElectronicsLayout, resolve_electronics_layout
from src.mech.mass_properties import (
    GeometryStations,
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
    tolerance = 1e-12
    return MissionMassProperties(
        mission=mission,
        items=items,
        total_mass_kg=mass,
        weight_n=mass * ParameterVector.gravity,
        cg_m=cg,
        inertia_tensor_kg_m2=inertia,
        static_margin=margin,
        static_margin_feasible=(
            sm_cfg.minimum - tolerance <= margin <= sm_cfg.maximum + tolerance
        ),
        placement_feasible=placement_feasible,
        warnings=warnings,
    )


def _place_landing_gear_under_wing_leading_edge(
    stations: GeometryStations,
    config: MechanicalModuleConfig,
) -> MassItem:
    """Place landing gear directly below the main-wing leading edge."""

    airframe = config.airframe
    gear_position = np.array(
        [
            stations.wing_le_x_m,
            0.0,
            -airframe.landing_gear_vertical_offset_m,
        ],
        dtype=float,
    )
    return MassItem(
        name="Landing gear",
        mass_kg=airframe.landing_gear_mass_kg,
        position_m=gear_position,
        dimensions_m=airframe.landing_gear_dimensions_m,
        missions=ALL_MISSIONS,
        category="airframe",
        notes=(
            "Installed directly below the main-wing leading edge, with its "
            "center 4 inches below the wing plane."
        ),
    )


def _fixed_airframe_items(
    design_vector: DesignVector,
    config: MechanicalModuleConfig,
) -> tuple[MassItem, ...]:
    """Build wing, boom, tails, controls, integration, and landing gear."""

    airframe = config.airframe
    stations = geometry_stations(design_vector)

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
        raise ValueError("Computed boom-spar length is not positive.")
    items.append(
        MassItem(
            name="Boom spar",
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

    gear = _place_landing_gear_under_wing_leading_edge(stations, config)
    return tuple(items) + (gear,)


def _fuselage_item_from_m2_envelope(
    design_vector: DesignVector,
    electronics_layout: ElectronicsLayout,
    mission2_payload: tuple[MassItem, ...],
    fuselage_width_m: float,
    config: MechanicalModuleConfig,
) -> MassItem:
    """Build a local fuselage around electronics and the occupied M2 envelope."""

    front_edge_x = electronics_layout.front_edge_x_m
    payload_back_edges = tuple(
        item.position_m[0] + 0.5 * item.dimensions_m[0]
        for item in mission2_payload
    )
    back_edge_x = max(
        (electronics_layout.back_edge_x_m, *payload_back_edges)
    )
    length = float(back_edge_x - front_edge_x)
    if length <= 0:
        raise RuntimeError("The locally packed fuselage length must be positive.")

    return MassItem(
        name="Fuselage structure",
        mass_kg=config.airframe.fuselage_linear_density_kg_m * length,
        position_m=(
            0.5 * (front_edge_x + back_edge_x),
            0.0,
            -0.5 * design_vector.fuselage_height,
        ),
        dimensions_m=(
            length,
            fuselage_width_m,
            design_vector.fuselage_height,
        ),
        missions=ALL_MISSIONS,
        category="airframe",
        notes=(
            "Fuselage-local uniform linear-density approximation from the "
            "electronics front edge to the aft-most Mission-2 payload edge."
        ),
    )


def _local_fuselage_assembly(
    design_vector: DesignVector,
    *,
    fuselage_width_m: float,
    duck_count: int,
    puck_count: int,
    config: MechanicalModuleConfig,
) -> tuple[tuple[MassItem, ...], tuple[MassItem, ...], ElectronicsLayout]:
    """Pack electronics and M2 payload before the fuselage is on the airplane."""

    airframe = config.airframe
    packaging = airframe.electronics_packaging
    local_layout = resolve_electronics_layout(
        cg_x_m=packaging.skinny_cg_from_front_m,
        fuselage_width_m=fuselage_width_m,
        fuselage_height_m=design_vector.fuselage_height,
        config=packaging,
        cg_y_m=airframe.electronics_y_m,
    )
    # A fat-width attempt can select a different electronics profile. Re-anchor
    # its front face at local x=0 after profile resolution.
    if not np.isclose(local_layout.front_edge_x_m, 0.0, atol=1e-12):
        local_layout = resolve_electronics_layout(
            cg_x_m=local_layout.cg_from_front_m,
            fuselage_width_m=fuselage_width_m,
            fuselage_height_m=design_vector.fuselage_height,
            config=packaging,
            cg_y_m=airframe.electronics_y_m,
        )

    electronics_position = local_layout.position_m
    electronics_items = tuple(
        MassItem(
            name=component_name,
            mass_kg=component_mass,
            position_m=electronics_position,
            dimensions_m=(local_layout.length_m, 0.0, 0.0),
            missions=ALL_MISSIONS,
            category="propulsion_and_electronics",
            notes=(
                f"Equivalent {local_layout.profile} electronics-area CM, packed "
                "inside the fuselage before airplane installation."
            ),
        )
        for component_name, component_mass in airframe.electronics_component_masses_kg(
            design_vector.batt_capacity
        )
    )
    if sum(item.mass_kg for item in electronics_items) <= 0:
        raise ValueError("Permanent electronics mass must be positive.")

    half_width = 0.5 * fuselage_width_m
    mission2_payload = place_mission2_payload(
        duck_count=duck_count,
        puck_count=puck_count,
        config=config.mission2,
        electronics_back_x_m=(
            local_layout.back_edge_x_m
            + config.mission2.electronics_aft_clearance_m
        ),
        y_bounds_m=(-half_width, half_width),
        z_bounds_m=(-design_vector.fuselage_height, 0.0),
    )
    fuselage = _fuselage_item_from_m2_envelope(
        design_vector,
        local_layout,
        mission2_payload,
        fuselage_width_m,
        config,
    )
    return (fuselage,) + electronics_items, mission2_payload, local_layout


def _translate_items_x(
    items: tuple[MassItem, ...], translation_x_m: float
) -> tuple[MassItem, ...]:
    """Translate a completed local assembly onto the airplane."""

    return tuple(
        replace(
            item,
            position_m=item.position_m + np.array((translation_x_m, 0.0, 0.0)),
        )
        for item in items
    )


def _translate_electronics_layout_x(
    layout: ElectronicsLayout, translation_x_m: float
) -> ElectronicsLayout:
    return replace(
        layout,
        front_edge_x_m=layout.front_edge_x_m + translation_x_m,
        cg_x_m=layout.cg_x_m + translation_x_m,
        back_edge_x_m=layout.back_edge_x_m + translation_x_m,
    )


def _mission3_items(
    design_vector: DesignVector,
    base_items: tuple[MassItem, ...],
    electronics_layout: ElectronicsLayout,
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

    forward_distance = m3.forward_mechanism_distance_m
    aft_distance = m3.aft_mechanism_distance_m
    offset_moment = -front_mass * forward_distance + aft_mass * aft_distance

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

    payload_min_x = electronics_layout.back_edge_x_m
    payload_max_x = min(
        stations.horizontal_tail_le_x_m, stations.vertical_tail_le_x_m
    )
    forward_half_length = 0.5 * m3.forward_mechanism_dimensions_m[0]
    banner_half_length = 0.5 * m3.banner_packed_dimensions_m[0]
    aft_half_length = 0.5 * m3.aft_mechanism_dimensions_m[0]

    # Convert each fixed item's face constraints to bounds on the shared banner
    # center.  Translating the group can never alter its configured spacings.
    physical_center_bounds = (
        max(
            payload_min_x + forward_distance + forward_half_length,
            payload_min_x + banner_half_length,
            payload_min_x - aft_distance + aft_half_length,
        ),
        min(
            payload_max_x + forward_distance - forward_half_length,
            payload_max_x - banner_half_length,
            payload_max_x - aft_distance - aft_half_length,
        ),
    )
    if m3.banner_center_x_bounds_m is None:
        center_bounds = physical_center_bounds
    else:
        center_bounds = (
            max(physical_center_bounds[0], m3.banner_center_x_bounds_m[0]),
            min(physical_center_bounds[1], m3.banner_center_x_bounds_m[1]),
        )

    if center_bounds[0] > center_bounds[1]:
        raise ValueError(
            "Mission-3 fixed mechanism distances leave no valid banner-center "
            "range between the electronics and tail."
        )
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
            position_m=(center_x - forward_distance, center_y, center_z),
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
            position_m=(center_x + aft_distance, center_y, center_z),
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

    fixed_items = _fixed_airframe_items(design_vector, config)
    duck_count = _integer_payload_count(design_vector.ducks_num, "ducks_num", warnings)
    puck_count = _integer_payload_count(design_vector.pucks_num, "pucks_num", warnings)
    m2_target_cg_x = (
        neutral_point_x
        - config.mission2.target_static_margin * design_vector.wing_chord
    )
    fixed_mass = sum(item.mass_kg for item in fixed_items)
    fixed_x_moment = sum(item.mass_kg * item.position_m[0] for item in fixed_items)

    m2_config = config.mission2
    width_increment = m2_config.duck.dimensions_m[1]
    attempt_failures: list[str] = []
    accepted: tuple[
        tuple[MassItem, ...],
        tuple[MassItem, ...],
        ElectronicsLayout,
        MissionMassProperties,
        MissionMassProperties,
        float,
        int,
    ] | None = None

    for width_increases in range(m2_config.maximum_width_increases + 1):
        fuselage_width = float(
            design_vector.fuselage_width + width_increases * width_increment
        )
        local_m1_group, local_m2_payload, local_layout = _local_fuselage_assembly(
            design_vector,
            fuselage_width_m=fuselage_width,
            duck_count=duck_count,
            puck_count=puck_count,
            config=config,
        )

        local_m2_group = local_m1_group + local_m2_payload
        group_mass = sum(item.mass_kg for item in local_m2_group)
        group_x_moment = sum(
            item.mass_kg * item.position_m[0] for item in local_m2_group
        )
        translation_x = float(
            (
                m2_target_cg_x * (fixed_mass + group_mass)
                - fixed_x_moment
                - group_x_moment
            )
            / group_mass
        )
        base_items = fixed_items + _translate_items_x(
            local_m1_group, translation_x
        )
        m2_payload = _translate_items_x(local_m2_payload, translation_x)
        electronics_layout = _translate_electronics_layout_x(
            local_layout, translation_x
        )

        fuselage = next(
            item for item in base_items if item.name == "Fuselage structure"
        )
        fuselage_back_x = float(
            fuselage.position_m[0] + 0.5 * fuselage.dimensions_m[0]
        )
        tail_front_x = float(
            min(stations.horizontal_tail_le_x_m, stations.vertical_tail_le_x_m)
        )
        permitted_fuselage_back_x = (
            tail_front_x - m2_config.tail_leading_edge_clearance_m
        )
        if fuselage_back_x >= permitted_fuselage_back_x - 1e-12:
            attempt_failures.append(
                f"width {fuselage_width:.4f} m puts the fuselage back at "
                f"x={fuselage_back_x:.4f} m, at or behind the permitted "
                f"tail-front limit x={permitted_fuselage_back_x:.4f} m"
            )
            continue

        electronics_bounds = config.airframe.electronics_x_bounds_m
        if electronics_bounds is not None and not (
            electronics_bounds[0]
            <= electronics_layout.cg_x_m
            <= electronics_bounds[1]
        ):
            raise ValueError(
                "The exact M2 placement requires electronics CM "
                f"x={electronics_layout.cg_x_m:.4f} m outside the configured "
                f"bounds [{electronics_bounds[0]:.4f}, "
                f"{electronics_bounds[1]:.4f}] m."
            )

        m2 = _mission_properties(
            mission="M2",
            items=base_items + m2_payload,
            design_vector=design_vector,
            neutral_point_x_m=neutral_point_x,
            config=config,
        )
        if not np.isclose(
            m2.static_margin,
            config.mission2.target_static_margin,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Translating the completed fuselage did not achieve the exact "
                "Mission-2 static-margin target."
            )
        m2 = replace(m2, static_margin_feasible=True)

        m1 = _mission_properties(
            mission="M1",
            items=base_items,
            design_vector=design_vector,
            neutral_point_x_m=neutral_point_x,
            config=config,
        )
        m1_is_acceptable = (
            m1.static_margin <= config.static_margin.maximum + 1e-12
        )
        m1 = replace(m1, static_margin_feasible=m1_is_acceptable)
        if not m1_is_acceptable:
            attempt_failures.append(
                f"width {fuselage_width:.4f} m gives M1 static margin "
                f"{100*m1.static_margin:.2f}%"
            )
            continue

        accepted = (
            base_items,
            m2_payload,
            electronics_layout,
            m1,
            m2,
            fuselage_width,
            width_increases,
        )
        break

    if accepted is None:
        detail = "; ".join(attempt_failures)
        raise PayloadPlacementError(
            "No fuselage width kept the fuselage ahead of the tail, produced "
            "exact M2 static margin, and kept M1 at or below "
            f"{100*config.static_margin.maximum:.1f}% after "
            f"{m2_config.maximum_width_increases} permitted width increases. "
            f"Attempts: {detail}"
        )

    (
        base_items,
        m2_payload,
        electronics_layout,
        m1,
        m2,
        fuselage_width,
        width_increases,
    ) = accepted
    if width_increases:
        warnings.append(
            f"Mission 2 selected fuselage width {fuselage_width:.4f} m after "
            f"{width_increases} duck-width increase(s)."
        )
    m1 = replace(m1, warnings=tuple(warnings))
    m2 = replace(m2, warnings=tuple(warnings))

    m3_payload = _mission3_items(
        design_vector,
        base_items,
        electronics_layout,
        neutral_point_x,
        config,
        warnings,
    )
    m3_items = base_items + m3_payload
    m3 = _mission_properties(
        mission="M3",
        items=m3_items,
        design_vector=design_vector,
        neutral_point_x_m=neutral_point_x,
        config=config,
    )
    m3_warnings = list(warnings)
    if not m3.static_margin_feasible:
        warning = (
            f"M3 static margin is {100*m3.static_margin:.2f}%, outside the configured range."
        )
        warnings.append(warning)
        m3_warnings.append(warning)
    m3 = replace(m3, warnings=tuple(dict.fromkeys(m3_warnings)))

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
        target_cg_x_m=m2_target_cg_x,
        acceptable_cg_x_range_m=acceptable_cg_range,
        electronics_position_m=electronics_layout.position_m,
        electronics_layout=electronics_layout,
        fuselage_width_m=fuselage_width,
        fuselage_width_increases=width_increases,
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
