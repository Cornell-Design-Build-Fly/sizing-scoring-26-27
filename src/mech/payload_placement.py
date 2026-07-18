"""Fast, deterministic Mission-2 payload placement inside a local fuselage.

The fuselage is packed before it is installed on the airplane.  Each payload
type starts immediately behind the electronics, with its first item against
the negative-y sidewall.  Rows fill across the available width and then move
aft.  Static margin is deliberately not part of this local packing step.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np

from src.mech.models import MassItem, Mission2Config, PayloadTypeConfig


class PayloadPlacementError(RuntimeError):
    """Raised when the requested payload cannot follow the fixed M2 process."""


@dataclass(frozen=True)
class _FuselageLocation:
    row: int
    column: int
    x_m: float
    y_m: float


def _front_to_back_locations(
    *,
    payload: PayloadTypeConfig,
    count: int,
    electronics_back_x_m: float,
    y_bounds_m: tuple[float, float],
    clearance_m: float,
) -> tuple[_FuselageLocation, ...]:
    """Fill wall-to-wall rows beginning directly behind the electronics."""

    if count == 0:
        return ()
    if not payload.rules.allow_aft:
        raise PayloadPlacementError(
            f"The {payload.label} rules forbid the required aftward placement."
        )

    length_x, width_y, _ = payload.dimensions_m
    available_width = y_bounds_m[1] - y_bounds_m[0]
    columns = floor(
        (available_width + clearance_m + 1e-12) / (width_y + clearance_m)
    )
    if columns < 1:
        raise ValueError(
            "Mission2Config violates the required starting-width invariant: "
            f"{payload.label} width is {width_y:.4f} m but the fuselage width "
            f"is {available_width:.4f} m."
        )

    first_x = electronics_back_x_m + clearance_m + 0.5 * length_x
    first_y = y_bounds_m[0] + 0.5 * width_y
    pitch_x = length_x + clearance_m
    pitch_y = width_y + clearance_m
    return tuple(
        _FuselageLocation(
            row=index // columns,
            column=index % columns,
            x_m=float(first_x + (index // columns) * pitch_x),
            y_m=float(first_y + (index % columns) * pitch_y),
        )
        for index in range(count)
    )


def _payload_items(
    *,
    payload: PayloadTypeConfig,
    count: int,
    z_m: float,
    electronics_back_x_m: float,
    y_bounds_m: tuple[float, float],
    clearance_m: float,
) -> tuple[MassItem, ...]:
    locations = _front_to_back_locations(
        payload=payload,
        count=count,
        electronics_back_x_m=electronics_back_x_m,
        y_bounds_m=y_bounds_m,
        clearance_m=clearance_m,
    )
    return tuple(
        MassItem(
            name=f"{payload.label} {index}",
            mass_kg=payload.mass_kg,
            position_m=(location.x_m, location.y_m, z_m),
            dimensions_m=payload.dimensions_m,
            missions=frozenset({"M2"}),
            category="mission_2_payload",
            notes=(
                "Fuselage-local front-to-back placement; "
                f"row {location.row}, column {location.column}; first column "
                "is against the negative-y fuselage sidewall."
            ),
        )
        for index, location in enumerate(locations, start=1)
    )


def _vertical_layer_centers(
    config: Mission2Config,
    *,
    duck_count: int,
    puck_count: int,
) -> tuple[float, float]:
    """Return ``(duck_z, puck_z)`` from the configured relative ordering."""

    rules = config.relative_payload_rules
    both_types_present = duck_count > 0 and puck_count > 0
    if both_types_present and (
        rules.pucks_forward_of_ducks or rules.pucks_aft_of_ducks
    ):
        raise PayloadPlacementError(
            "The local row process starts both payload types at the electronics, "
            "so a global forward/aft type separation cannot also be imposed."
        )
    duck_height = config.duck.dimensions_m[2]
    puck_height = config.puck.dimensions_m[2]
    separation = 0.5 * (duck_height + puck_height) + config.vertical_clearance_m
    duck_z = config.duck_center_z_m

    if rules.pucks_above_ducks:
        puck_z = duck_z + separation
    elif rules.pucks_below_ducks:
        puck_z = duck_z - separation
    elif both_types_present:
        raise PayloadPlacementError(
            "Mission 2 requires an explicit vertical payload order. Set either "
            "pucks_below_ducks or pucks_above_ducks."
        )
    else:
        puck_z = duck_z
    return float(duck_z), float(puck_z)


def _validate_vertical_fit(
    *,
    payload: PayloadTypeConfig,
    count: int,
    center_z_m: float,
    z_bounds_m: tuple[float, float] | None,
) -> None:
    if count == 0 or z_bounds_m is None:
        return
    half_height = 0.5 * payload.dimensions_m[2]
    if (
        center_z_m - half_height < z_bounds_m[0] - 1e-12
        or center_z_m + half_height > z_bounds_m[1] + 1e-12
    ):
        raise PayloadPlacementError(
            f"The fixed {payload.label} layer crosses the fuselage vertical "
            f"bounds [{z_bounds_m[0]:.4f}, {z_bounds_m[1]:.4f}] m."
        )


def place_mission2_payload(
    *,
    duck_count: int,
    puck_count: int,
    config: Mission2Config,
    electronics_back_x_m: float,
    y_bounds_m: tuple[float, float],
    z_bounds_m: tuple[float, float] | None = None,
    base_items: tuple[MassItem, ...] = (),
    target_cg_x_m: float | None = None,
    x_bounds_m: tuple[float, float] | None = None,
    reference_x_m: float | None = None,
) -> tuple[MassItem, ...]:
    """Pack M2 payload in fuselage-local coordinates.

    The last four keywords are retained only to produce clear errors for code
    written for the former airplane-CG-centered API.  They cannot override the
    electronics back face or the fuselage sidewalls in the new workflow.
    """

    del base_items, target_cg_x_m
    if duck_count < 0 or puck_count < 0:
        raise ValueError("Mission-2 payload counts cannot be negative.")
    if not np.isfinite(electronics_back_x_m):
        raise ValueError("electronics_back_x_m must be finite.")
    if not (
        np.all(np.isfinite(y_bounds_m)) and y_bounds_m[0] < y_bounds_m[1]
    ):
        raise PayloadPlacementError("Mission-2 y bounds must be finite and increasing.")
    if z_bounds_m is not None and not (
        np.all(np.isfinite(z_bounds_m)) and z_bounds_m[0] < z_bounds_m[1]
    ):
        raise PayloadPlacementError("Mission-2 z bounds must be finite and increasing.")
    if x_bounds_m is not None or reference_x_m is not None:
        raise ValueError(
            "Mission 2 no longer accepts airplane x bounds or a CG reference; "
            "packing starts at electronics_back_x_m and grows aft."
        )

    duck_z, puck_z = _vertical_layer_centers(
        config, duck_count=duck_count, puck_count=puck_count
    )
    _validate_vertical_fit(
        payload=config.duck,
        count=duck_count,
        center_z_m=duck_z,
        z_bounds_m=z_bounds_m,
    )
    _validate_vertical_fit(
        payload=config.puck,
        count=puck_count,
        center_z_m=puck_z,
        z_bounds_m=z_bounds_m,
    )
    ducks = _payload_items(
        payload=config.duck,
        count=duck_count,
        z_m=duck_z,
        electronics_back_x_m=electronics_back_x_m,
        y_bounds_m=y_bounds_m,
        clearance_m=config.clearance_m,
    )
    pucks = _payload_items(
        payload=config.puck,
        count=puck_count,
        z_m=puck_z,
        electronics_back_x_m=electronics_back_x_m,
        y_bounds_m=y_bounds_m,
        clearance_m=config.clearance_m,
    )
    return ducks + pucks
