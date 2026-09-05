"""Deterministic Mission-2 shipping-container placement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.mech.models import MassItem, Mission2Config


class PayloadPlacementError(RuntimeError):
    """Raised when the requested payload cannot follow the fixed M2 process."""


@dataclass(frozen=True)
class PayloadPackingSummary:
    """Mass, moment, and envelope of a local shipping-container packing."""

    total_mass_kg: float
    x_moment_kg_m: float
    min_edge_x_m: float
    max_edge_x_m: float
    required_width_m: float
    required_height_m: float


def resolve_extra_container_count(
    value: float,
    config: Mission2Config,
    warnings: list[str],
) -> int:
    """Validate and round the number of additional M2 containers."""

    if not np.isfinite(value) or value < 0:
        raise ValueError("extra_shipping_containers must be finite and nonnegative.")
    rounded = int(round(float(value)))
    if rounded > config.maximum_extra_containers:
        raise ValueError(
            "extra_shipping_containers cannot exceed "
            f"{config.maximum_extra_containers}."
        )
    if not np.isclose(value, rounded, atol=1e-9):
        warnings.append(
            f"extra_shipping_containers={value:.6g} is not an integer; the "
            f"mechanical module rounded it to {rounded}."
        )
    return rounded


def _cell_order(
    total_count: int,
    config: Mission2Config,
) -> tuple[tuple[int, int, int], ...]:
    """Return ``(longitudinal_bay, column, layer)`` in loading order."""

    if not isinstance(total_count, int) or total_count < 1:
        raise ValueError(
            "Mission 2 total_count must be an integer of at least one."
        )

    center_column = 0.5 * (config.containers_across - 1)
    column_order = sorted(
        range(config.containers_across),
        key=lambda column: (abs(column - center_column), column),
    )
    cells = tuple(
        (column, layer)
        for layer in range(config.maximum_stack)
        for column in column_order
    )
    order: list[tuple[int, int, int]] = [
        (0, column, layer) for column, layer in cells
    ]
    distance = 1
    while len(order) < total_count:
        for column, layer in cells:
            # Body x is positive aft. Fill paired bays aft, forward, aft, ...
            order.append((distance, column, layer))
            if len(order) >= total_count:
                break
            order.append((-distance, column, layer))
            if len(order) >= total_count:
                break
        distance += 1
    return tuple(order[:total_count])


def _local_positions(
    *,
    total_count: int,
    container_dimensions_m: tuple[float, float, float],
    electronics_back_x_m: float,
    config: Mission2Config,
) -> tuple[tuple[float, float, float], ...]:
    length_x, width_y, height_z = container_dimensions_m
    order = _cell_order(total_count, config)
    most_forward_bay = min(bay for bay, _, _ in order)
    pitch_x = length_x + config.clearance_m
    pitch_y = width_y + config.clearance_m
    pitch_z = height_z + config.clearance_m

    # Leave enough room ahead of the center bay for every occupied forward bay.
    center_bay_x = (
        electronics_back_x_m
        + config.electronics_aft_clearance_m
        + 0.5 * length_x
        - most_forward_bay * pitch_x
    )
    column_offsets = tuple(
        (column - 0.5 * (config.containers_across - 1)) * pitch_y
        for column in range(config.containers_across)
    )
    used_layers = max(layer for _, _, layer in order) + 1
    fuselage_height = (
        used_layers * height_z + (used_layers - 1) * config.clearance_m
    )
    bottom_layer_z = -fuselage_height + 0.5 * height_z

    return tuple(
        (
            float(center_bay_x + bay * pitch_x),
            float(column_offsets[column]),
            float(bottom_layer_z + layer * pitch_z),
        )
        for bay, column, layer in order
    )


def summarize_mission2_payload(
    *,
    total_count: int,
    sensor_mass_kg: float,
    sensor_length_m: float,
    sensor_diameter_m: float,
    config: Mission2Config,
    electronics_back_x_m: float,
) -> PayloadPackingSummary:
    """Summarize deterministic M2 packing without creating mass items."""

    if not np.isfinite(sensor_mass_kg) or sensor_mass_kg <= 0:
        raise ValueError("sensor_mass_kg must be finite and positive.")
    if not np.isfinite(electronics_back_x_m):
        raise ValueError("electronics_back_x_m must be finite.")
    dimensions = config.container_dimensions_m(sensor_length_m, sensor_diameter_m)
    positions = _local_positions(
        total_count=total_count,
        container_dimensions_m=dimensions,
        electronics_back_x_m=electronics_back_x_m,
        config=config,
    )
    container_mass = sensor_mass_kg + config.empty_container_mass_kg
    half_length = 0.5 * dimensions[0]
    used_layers = len({round(position[2], 12) for position in positions})
    return PayloadPackingSummary(
        total_mass_kg=float(total_count * container_mass),
        x_moment_kg_m=float(
            container_mass * sum(position[0] for position in positions)
        ),
        min_edge_x_m=float(
            min(position[0] - half_length for position in positions)
        ),
        max_edge_x_m=float(
            max(position[0] + half_length for position in positions)
        ),
        required_width_m=float(
            2.0
            * max(abs(position[1]) + 0.5 * dimensions[1] for position in positions)
        ),
        required_height_m=float(
            used_layers * dimensions[2]
            + (used_layers - 1) * config.clearance_m
        ),
    )


def place_mission2_payload(
    *,
    total_count: int,
    sensor_mass_kg: float,
    sensor_length_m: float,
    sensor_diameter_m: float,
    config: Mission2Config,
    electronics_back_x_m: float,
) -> tuple[MassItem, ...]:
    """Pack the required container plus simulators in local coordinates."""

    if not np.isfinite(sensor_mass_kg) or sensor_mass_kg <= 0:
        raise ValueError("sensor_mass_kg must be finite and positive.")
    if not np.isfinite(electronics_back_x_m):
        raise ValueError("electronics_back_x_m must be finite.")
    dimensions = config.container_dimensions_m(sensor_length_m, sensor_diameter_m)
    positions = _local_positions(
        total_count=total_count,
        container_dimensions_m=dimensions,
        electronics_back_x_m=electronics_back_x_m,
        config=config,
    )
    container_mass = sensor_mass_kg + config.empty_container_mass_kg
    return tuple(
        MassItem(
            name=(
                "M2 sensor shipping container"
                if index == 1
                else f"M2 shipping container simulator {index - 1}"
            ),
            mass_kg=container_mass,
            position_m=position,
            dimensions_m=dimensions,
            missions=frozenset({"M2"}),
            category="mission_2_payload",
            notes=(
                "Mass includes a sensor-equivalent payload plus the 0.5 lb "
                "empty container. The center bay fills 3-wide and 2-high "
                "before paired aft/forward bays are filled alternately."
            ),
        )
        for index, position in enumerate(positions, start=1)
    )


__all__ = [
    "PayloadPackingSummary",
    "PayloadPlacementError",
    "place_mission2_payload",
    "resolve_extra_container_count",
    "summarize_mission2_payload",
]
