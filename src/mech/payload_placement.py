"""Fast, deterministic Mission-2 payload placement.

The packer implements one process: anchor the first item at the starting-CG
plane and aircraft centerline, then take valid lattice locations in increasing
distance from that point.  It does not optimize CG or fall back to a different
search strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import numpy as np

from src.mech.models import MassItem, Mission2Config, PayloadTypeConfig


class PayloadPlacementError(RuntimeError):
    """Raised when the requested payload cannot follow the fixed M2 process."""


@dataclass(frozen=True)
class _LatticeLocation:
    i: int
    j: int
    x_m: float
    y_m: float


def _integer_limits(
    *, lower_m: float, upper_m: float, anchor_m: float, pitch_m: float
) -> tuple[int, int]:
    """Convert physical center bounds to inclusive lattice-index bounds."""

    tolerance = 1e-12
    return (
        ceil((lower_m - anchor_m) / pitch_m - tolerance),
        floor((upper_m - anchor_m) / pitch_m + tolerance),
    )


def _direction_rank(i: int, j: int) -> int:
    """Stable tie order that keeps opposite center-out directions adjacent."""

    if i == 0 and j == 0:
        return 0
    if j == 0:
        return 1 if i < 0 else 2  # forward, then aft
    if i == 0:
        return 3 if j > 0 else 4  # right, then left
    if i < 0 and j > 0:
        return 5
    if i > 0 and j < 0:
        return 6
    if i < 0 and j < 0:
        return 7
    return 8


def _center_out_locations(
    *,
    payload: PayloadTypeConfig,
    count: int,
    anchor_x_m: float,
    anchor_y_m: float,
    x_bounds_m: tuple[float, float],
    y_bounds_m: tuple[float, float],
    clearance_m: float,
) -> tuple[_LatticeLocation, ...]:
    """Return the first ``count`` valid positions on a center-anchored lattice."""

    if count == 0:
        return ()

    length_x, width_y, _ = payload.dimensions_m
    x_center_bounds = (
        x_bounds_m[0] + 0.5 * length_x,
        x_bounds_m[1] - 0.5 * length_x,
    )
    y_center_bounds = (
        y_bounds_m[0] + 0.5 * width_y,
        y_bounds_m[1] - 0.5 * width_y,
    )
    if x_center_bounds[0] > x_center_bounds[1] or y_center_bounds[0] > y_center_bounds[1]:
        raise PayloadPlacementError(
            f"The {payload.label} bounding box is larger than the Mission-2 bay."
        )

    pitch_x = length_x + clearance_m
    pitch_y = width_y + clearance_m
    i_min, i_max = _integer_limits(
        lower_m=x_center_bounds[0],
        upper_m=x_center_bounds[1],
        anchor_m=anchor_x_m,
        pitch_m=pitch_x,
    )
    j_min, j_max = _integer_limits(
        lower_m=y_center_bounds[0],
        upper_m=y_center_bounds[1],
        anchor_m=anchor_y_m,
        pitch_m=pitch_y,
    )

    if not (i_min <= 0 <= i_max and j_min <= 0 <= j_max):
        raise PayloadPlacementError(
            f"The first {payload.label} cannot be centered at the starting CG "
            "without crossing a payload-bay boundary."
        )

    candidates: list[_LatticeLocation] = []
    rules = payload.rules
    for i in range(i_min, i_max + 1):
        if i < 0 and not rules.allow_forward:
            continue
        if i > 0 and not rules.allow_aft:
            continue
        for j in range(j_min, j_max + 1):
            candidates.append(
                _LatticeLocation(
                    i=i,
                    j=j,
                    x_m=float(anchor_x_m + i * pitch_x),
                    y_m=float(anchor_y_m + j * pitch_y),
                )
            )

    candidates.sort(
        key=lambda candidate: (
            round(
                (candidate.i * pitch_x) ** 2 + (candidate.j * pitch_y) ** 2,
                15,
            ),
            abs(candidate.i) + abs(candidate.j),
            _direction_rank(candidate.i, candidate.j),
            abs(candidate.i),
            abs(candidate.j),
        )
    )
    if len(candidates) < count:
        raise PayloadPlacementError(
            f"Requested {count} {payload.label.lower()} items, but the fixed "
            f"center-out lattice has capacity {len(candidates)} within the "
            "electronics, tail, and fuselage-side bounds."
        )
    return tuple(candidates[:count])


def _payload_items(
    *,
    payload: PayloadTypeConfig,
    count: int,
    z_m: float,
    anchor_x_m: float,
    anchor_y_m: float,
    x_bounds_m: tuple[float, float],
    y_bounds_m: tuple[float, float],
    clearance_m: float,
) -> tuple[MassItem, ...]:
    locations = _center_out_locations(
        payload=payload,
        count=count,
        anchor_x_m=anchor_x_m,
        anchor_y_m=anchor_y_m,
        x_bounds_m=x_bounds_m,
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
                "Fixed-layer, center-out placement; lattice index "
                f"({location.i}, {location.j})."
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
            "The fixed center-out process anchors the first duck and puck on the "
            "same CG plane, so a global forward/aft type separation cannot also "
            "be imposed."
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
        # With only one type there is no relative ordering to enforce. A
        # puck-only load uses the configured payload plane directly.
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
    base_items: tuple[MassItem, ...],
    target_cg_x_m: float | None = None,
    config: Mission2Config,
    x_bounds_m: tuple[float, float],
    reference_x_m: float | None = None,
    y_bounds_m: tuple[float, float] | None = None,
    z_bounds_m: tuple[float, float] | None = None,
) -> tuple[MassItem, ...]:
    """Place M2 payloads with the required deterministic center-out process.

    ``target_cg_x_m`` is retained as a compatibility keyword but is not used as
    an optimization target.  If ``reference_x_m`` is omitted, the actual base
    airplane CG supplies the starting plane.
    """

    del target_cg_x_m
    if duck_count < 0 or puck_count < 0:
        raise ValueError("Mission-2 payload counts cannot be negative.")
    if not x_bounds_m[0] < x_bounds_m[1]:
        raise PayloadPlacementError("Mission-2 x bounds must be increasing.")

    base_mass = sum(item.mass_kg for item in base_items)
    if base_mass <= 0:
        raise ValueError("Mission-2 placement requires positive base-airplane mass.")
    base_cg = sum(
        (item.mass_kg * item.position_m for item in base_items), start=np.zeros(3)
    ) / base_mass
    anchor_x_m = float(base_cg[0])
    anchor_y_m = float(base_cg[1])
    if reference_x_m is not None and not np.isclose(
        reference_x_m, anchor_x_m, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "reference_x_m cannot override the required actual Mission-1 CG plane."
        )

    if y_bounds_m is None:
        if config.maximum_width_m is None:
            raise ValueError(
                "y_bounds_m is required when Mission2Config.maximum_width_m is None."
            )
        half_width = 0.5 * config.maximum_width_m
        y_bounds_m = (
            config.compartment_center_y_m - half_width,
            config.compartment_center_y_m + half_width,
        )
    if not y_bounds_m[0] < y_bounds_m[1]:
        raise PayloadPlacementError("Mission-2 y bounds must be increasing.")
    if z_bounds_m is not None and not z_bounds_m[0] < z_bounds_m[1]:
        raise PayloadPlacementError("Mission-2 z bounds must be increasing.")

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
        anchor_x_m=anchor_x_m,
        anchor_y_m=anchor_y_m,
        x_bounds_m=x_bounds_m,
        y_bounds_m=y_bounds_m,
        clearance_m=config.clearance_m,
    )
    pucks = _payload_items(
        payload=config.puck,
        count=puck_count,
        z_m=puck_z,
        anchor_x_m=anchor_x_m,
        anchor_y_m=anchor_y_m,
        x_bounds_m=x_bounds_m,
        y_bounds_m=y_bounds_m,
        clearance_m=config.clearance_m,
    )
    return ducks + pucks
