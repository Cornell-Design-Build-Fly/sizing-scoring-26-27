"""Construction and translation of airframe and fuselage mass items."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.mech.electronics import ElectronicsLayout, resolve_electronics_layout
from src.mech.mass_properties import GeometryStations, geometry_stations
from src.mech.models import ALL_MISSIONS, MassItem, MechanicalModuleConfig
from src.mech.payload_placement import place_mission2_payload
from src.vectors import DesignVector


def place_landing_gear_under_wing_leading_edge(
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


def build_fixed_airframe_items(
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

    gear = place_landing_gear_under_wing_leading_edge(stations, config)
    return tuple(items) + (gear,)


def build_fuselage_item(
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
    back_edge_x = max((electronics_layout.back_edge_x_m, *payload_back_edges))
    length = float(back_edge_x - front_edge_x)
    if length <= 0:
        raise RuntimeError("The locally packed fuselage length must be positive.")
    cross_sectional_perimeter = 2.0 * (
        fuselage_width_m + design_vector.fuselage_height
    )

    return MassItem(
        name="Fuselage structure",
        mass_kg=(
            config.airframe.fuselage_shell_areal_density_kg_m2
            * length
            * cross_sectional_perimeter
        ),
        position_m=(
            0.5 * (front_edge_x + back_edge_x),
            0.0,
            -0.5 * design_vector.fuselage_height,
        ),
        dimensions_m=(length, fuselage_width_m, design_vector.fuselage_height),
        missions=ALL_MISSIONS,
        category="airframe",
        notes=(
            "Fuselage-local uniform shell-area-density approximation from the "
            "electronics front edge to the aft-most Mission-2 payload edge; "
            "mass scales with length and cross-sectional perimeter."
        ),
    )


def build_local_fuselage_assembly(
    design_vector: DesignVector,
    *,
    battery_nominal_voltage_v: float,
    fuselage_width_m: float,
    duck_count: int,
    puck_count: int,
    config: MechanicalModuleConfig,
) -> tuple[tuple[MassItem, ...], tuple[MassItem, ...], ElectronicsLayout]:
    """Pack electronics and M2 payload before installation on the airplane."""

    airframe = config.airframe
    packaging = airframe.electronics_packaging
    local_layout = resolve_electronics_layout(
        cg_x_m=packaging.skinny_cg_from_front_m,
        fuselage_width_m=fuselage_width_m,
        fuselage_height_m=design_vector.fuselage_height,
        config=packaging,
        cg_y_m=airframe.electronics_y_m,
    )
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
            design_vector.batt_capacity,
            battery_nominal_voltage_v,
            design_vector.motor_kv,
            design_vector.motor_max_power,
            design_vector.prop_diameter_in,
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
    fuselage = build_fuselage_item(
        design_vector,
        local_layout,
        mission2_payload,
        fuselage_width_m,
        config,
    )
    return (fuselage,) + electronics_items, mission2_payload, local_layout


def translate_mass_items_x(
    items: tuple[MassItem, ...], translation_x_m: float
) -> tuple[MassItem, ...]:
    """Translate a completed local mass-item assembly onto the airplane."""

    return tuple(
        replace(
            item,
            position_m=item.position_m + np.array((translation_x_m, 0.0, 0.0)),
        )
        for item in items
    )


def translate_electronics_layout_x(
    layout: ElectronicsLayout,
    translation_x_m: float,
) -> ElectronicsLayout:
    """Translate an electronics-layout record along the body x axis."""

    return replace(
        layout,
        front_edge_x_m=layout.front_edge_x_m + translation_x_m,
        cg_x_m=layout.cg_x_m + translation_x_m,
        back_edge_x_m=layout.back_edge_x_m + translation_x_m,
    )


__all__ = [
    "build_fixed_airframe_items",
    "build_fuselage_item",
    "build_local_fuselage_assembly",
    "place_landing_gear_under_wing_leading_edge",
    "translate_electronics_layout_x",
    "translate_mass_items_x",
]
