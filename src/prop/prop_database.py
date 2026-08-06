from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, QhullError

from src.prop.prop_data import (
    DEFAULT_PROP_DATA_PATH,
    PropellerData,
    load_prop_data,
)

from src.prop.prop_extrapolation import (
    LocalLinearExtrapolator,
)


FloatArray = NDArray[np.float64]
QueryResult: TypeAlias = float | FloatArray


def _evaluate_interpolator(
    interpolator: LinearNDInterpolator,
    velocity_mph: ArrayLike,
    rpm: ArrayLike,
) -> QueryResult:
    """
    Evaluate an interpolator with scalar or array inputs.

    Velocity and RPM inputs are broadcast to a common shape.
    """

    velocity_array, rpm_array = np.broadcast_arrays(
        np.asarray(velocity_mph, dtype=np.float64),
        np.asarray(rpm, dtype=np.float64),
    )

    query_points = np.column_stack(
        (
            velocity_array.reshape(-1),
            rpm_array.reshape(-1),
        )
    )

    result = np.asarray(
        interpolator(query_points),
        dtype=np.float64,
    ).reshape(velocity_array.shape)

    if result.ndim == 0:
        return float(result)

    return result


@dataclass(frozen=True, slots=True)
class CatalogPropSurface:
    """
    MATLAB-style velocity/RPM surfaces for one catalog propeller.

    The inputs are:
        velocity_mph
        rpm

    The outputs are:
        thrust_n
        torque_nm
    """

    key: str
    diameter_in: float
    pitch_in: float

    points: FloatArray
    thrust_values: FloatArray
    torque_values: FloatArray

    extrapolator: LocalLinearExtrapolator
    triangulation: Delaunay
    thrust_interpolator: LinearNDInterpolator
    torque_interpolator: LinearNDInterpolator

    def evaluate(
        self,
        velocity_mph: ArrayLike,
        rpm: ArrayLike,
    ) -> tuple[QueryResult, QueryResult]:
        """
        Return thrust and torque.

        LinearNDInterpolator is used inside the convex hull.
        Local linear extrapolation is used outside it.
        """

        velocity_array, rpm_array = np.broadcast_arrays(
            np.asarray(
                velocity_mph,
                dtype=np.float64,
            ),
            np.asarray(
                rpm,
                dtype=np.float64,
            ),
        )

        output_shape = velocity_array.shape

        query_points = np.column_stack(
            (
                velocity_array.reshape(-1),
                rpm_array.reshape(-1),
            )
        )

        thrust = np.asarray(
            self.thrust_interpolator(query_points),
            dtype=np.float64,
        ).reshape(-1)

        torque = np.asarray(
            self.torque_interpolator(query_points),
            dtype=np.float64,
        ).reshape(-1)

        outside_mask = np.isnan(thrust) | np.isnan(torque)

        if np.any(outside_mask):
            extrapolated_thrust, extrapolated_torque = (
                self.extrapolator.evaluate(
                    query_points[outside_mask]
                )
            )

            thrust[outside_mask] = extrapolated_thrust
            torque[outside_mask] = extrapolated_torque

        thrust = thrust.reshape(output_shape)
        torque = torque.reshape(output_shape)

        if thrust.ndim == 0:
            return float(thrust), float(torque)

        return thrust, torque


    def thrust(
        self,
        velocity_mph: ArrayLike,
        rpm: ArrayLike,
    ) -> QueryResult:
        """Return thrust in newtons."""

        thrust, _ = self.evaluate(
            velocity_mph,
            rpm,
        )

        return thrust


    def torque(
        self,
        velocity_mph: ArrayLike,
        rpm: ArrayLike,
    ) -> QueryResult:
        """Return torque in newton-metres."""

        _, torque = self.evaluate(
            velocity_mph,
            rpm,
        )

        return torque

    def contains(
        self,
        velocity_mph: ArrayLike,
        rpm: ArrayLike,
    ) -> bool | NDArray[np.bool_]:
        """
        Return whether query points are inside the data convex hull.

        No extrapolation is performed here.
        """

        velocity_array, rpm_array = np.broadcast_arrays(
            np.asarray(velocity_mph, dtype=np.float64),
            np.asarray(rpm, dtype=np.float64),
        )

        query_points = np.column_stack(
            (
                velocity_array.reshape(-1),
                rpm_array.reshape(-1),
            )
        )

        inside = (
            self.triangulation.find_simplex(query_points) >= 0
        ).reshape(velocity_array.shape)

        if inside.ndim == 0:
            return bool(inside)

        return inside


def build_catalog_surface(
    propeller: PropellerData,
) -> CatalogPropSurface:
    """
    Build thrust and torque surfaces for one catalog propeller.

    No diameter/pitch interpolation occurs in this function.
    """

    velocity_parts: list[FloatArray] = []
    rpm_parts: list[FloatArray] = []
    thrust_parts: list[FloatArray] = []
    torque_parts: list[FloatArray] = []

    for rpm_table in propeller.rpm_data:
        point_count = rpm_table.sample_count

        velocity_parts.append(
            np.asarray(
                rpm_table.velocity_mph,
                dtype=np.float64,
            )
        )

        rpm_parts.append(
            np.full(
                point_count,
                rpm_table.rpm,
                dtype=np.float64,
            )
        )

        thrust_parts.append(
            np.asarray(
                rpm_table.thrust_n,
                dtype=np.float64,
            )
        )

        torque_parts.append(
            np.asarray(
                rpm_table.torque_nm,
                dtype=np.float64,
            )
        )

    velocity_mph = np.concatenate(velocity_parts)
    rpm = np.concatenate(rpm_parts)
    thrust_n = np.concatenate(thrust_parts)
    torque_nm = np.concatenate(torque_parts)

    points = np.column_stack((velocity_mph, rpm))

    unique_point_count = len(
        np.unique(points, axis=0)
    )

    if unique_point_count != len(points):
        duplicate_count = len(points) - unique_point_count

        raise ValueError(
            f'Propeller "{propeller.key}" contains '
            f"{duplicate_count} duplicate velocity/RPM points."
        )

    try:
        triangulation = Delaunay(points)
    except QhullError as error:
        raise ValueError(
            f'Could not triangulate propeller "{propeller.key}".'
        ) from error

    thrust_interpolator = LinearNDInterpolator(
        triangulation,
        thrust_n,
        fill_value=np.nan,
    )

    torque_interpolator = LinearNDInterpolator(
        triangulation,
        torque_nm,
        fill_value=np.nan,
    )

    extrapolator = LocalLinearExtrapolator.build(
        points=points,
        thrust_values=thrust_n,
        torque_values=torque_nm,
    )

    points.setflags(write=False)
    thrust_n.setflags(write=False)
    torque_nm.setflags(write=False)

    return CatalogPropSurface(
        key=propeller.key,
        diameter_in=propeller.diameter_in,
        pitch_in=propeller.pitch_in,
        points=points,
        thrust_values=thrust_n,
        torque_values=torque_nm,
        extrapolator=extrapolator,
        triangulation=triangulation,
        thrust_interpolator=thrust_interpolator,
        torque_interpolator=torque_interpolator,
    )


class CatalogPropDatabase:
    """
    Collection of MATLAB-style surfaces for catalog propellers.

    This is the foundation of the later continuous 4D database.
    """

    def __init__(
        self,
        surfaces: tuple[CatalogPropSurface, ...],
    ) -> None:
        if not surfaces:
            raise ValueError(
                "Catalog prop database cannot be empty."
            )

        self.surfaces = surfaces

        self._by_key = {
            surface.key: surface
            for surface in surfaces
        }

        self._by_geometry = {
            (
                surface.diameter_in,
                surface.pitch_in,
            ): surface
            for surface in surfaces
        }

        if len(self._by_key) != len(surfaces):
            raise ValueError(
                "Catalog database contains duplicate propeller keys."
            )

        if len(self._by_geometry) != len(surfaces):
            raise ValueError(
                "Catalog database contains duplicate geometries."
            )

    def get_by_key(
        self,
        propeller_key: str,
    ) -> CatalogPropSurface:
        """Return one surface using its JSON key."""

        try:
            return self._by_key[propeller_key]
        except KeyError as error:
            raise KeyError(
                f'Unknown propeller key: "{propeller_key}"'
            ) from error

    def get_by_geometry(
        self,
        diameter_in: float,
        pitch_in: float,
    ) -> CatalogPropSurface:
        """
        Return an exact catalog geometry.

        This function does not snap nearby geometries to catalog props.
        """

        geometry = (
            float(diameter_in),
            float(pitch_in),
        )

        try:
            return self._by_geometry[geometry]
        except KeyError as error:
            raise KeyError(
                f"No exact catalog propeller exists at "
                f"{diameter_in:g}x{pitch_in:g}."
            ) from error

    @property
    def propeller_count(self) -> int:
        return len(self.surfaces)


def build_catalog_database(
    propellers: tuple[PropellerData, ...],
) -> CatalogPropDatabase:
    """Build catalog surfaces for all loaded propellers."""

    surfaces = tuple(
        build_catalog_surface(propeller)
        for propeller in propellers
    )

    return CatalogPropDatabase(surfaces)


def load_catalog_prop_database(
    json_path: str | Path = DEFAULT_PROP_DATA_PATH,
) -> CatalogPropDatabase:
    """Load the JSON and build all catalog propeller surfaces."""

    propellers = load_prop_data(json_path)
    return build_catalog_database(propellers)


@lru_cache(maxsize=1)
def load_default_catalog_prop_database(
) -> CatalogPropDatabase:
    """
    Load and cache the default catalog database.

    The cache lasts only for the current Python process.
    """

    return load_catalog_prop_database(
        DEFAULT_PROP_DATA_PATH
    )

