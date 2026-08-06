from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import Delaunay, cKDTree

from src.prop.prop_database import (
    CatalogPropDatabase,
    CatalogPropSurface,
    QueryResult,
    load_default_catalog_prop_database,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GeometryBlend:
    """
    Catalog propellers and weights used for one diameter/pitch query.
    """

    surfaces: tuple[CatalogPropSurface, ...]
    weights: FloatArray
    method: str


class ContinuousPropDatabase:
    """
    Continuous propeller database.

    Inputs:
        diameter_in
        pitch_in
        velocity_mph
        rpm

    Outputs:
        thrust_n
        torque_nm

    Velocity/RPM behavior comes from the catalog surfaces.
    Diameter/pitch behavior is handled by this class.
    """

    def __init__(
        self,
        catalog_database: CatalogPropDatabase,
        extrapolation_neighbor_count: int = 8,
    ) -> None:
        if extrapolation_neighbor_count < 3:
            raise ValueError(
                "extrapolation_neighbor_count must be at least 3."
            )

        self.catalog = catalog_database
        self.extrapolation_neighbor_count = (
            extrapolation_neighbor_count
        )

        self.geometries = np.array(
            [
                [
                    surface.diameter_in,
                    surface.pitch_in,
                ]
                for surface in self.catalog.surfaces
            ],
            dtype=np.float64,
        )

        if len(np.unique(self.geometries, axis=0)) != len(
            self.geometries
        ):
            raise ValueError(
                "Continuous database received duplicate geometries."
            )

        self.diameter_min = float(
            np.min(self.geometries[:, 0])
        )
        self.diameter_max = float(
            np.max(self.geometries[:, 0])
        )

        self.pitch_min = float(
            np.min(self.geometries[:, 1])
        )
        self.pitch_max = float(
            np.max(self.geometries[:, 1])
        )

        self.geometry_minimum = np.array(
            [self.diameter_min, self.pitch_min],
            dtype=np.float64,
        )

        self.geometry_scale = np.array(
            [
                self.diameter_max - self.diameter_min,
                self.pitch_max - self.pitch_min,
            ],
            dtype=np.float64,
        )

        if np.any(self.geometry_scale <= 0.0):
            raise ValueError(
                "Geometry data must span diameter and pitch."
            )

        self.normalized_geometries = (
            self.geometries - self.geometry_minimum
        ) / self.geometry_scale

        self.geometry_triangulation = Delaunay(
            self.geometries
        )

        self.geometry_tree = cKDTree(
            self.normalized_geometries
        )

        self._blend_cache: dict[
            tuple[float, float],
            GeometryBlend,
        ] = {}

        self.geometries.setflags(write=False)
        self.normalized_geometries.setflags(write=False)

    def _validate_geometry(
        self,
        diameter_in: float,
        pitch_in: float,
    ) -> FloatArray:
        """Validate and return one diameter/pitch query."""

        point = np.array(
            [diameter_in, pitch_in],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(point)):
            raise ValueError(
                "Diameter and pitch must be finite."
            )

        if not (
            self.diameter_min
            <= diameter_in
            <= self.diameter_max
        ):
            raise ValueError(
                f"Diameter {diameter_in:g} is outside "
                f"{self.diameter_min:g} to "
                f"{self.diameter_max:g} inches."
            )

        if not (
            self.pitch_min
            <= pitch_in
            <= self.pitch_max
        ):
            raise ValueError(
                f"Pitch {pitch_in:g} is outside "
                f"{self.pitch_min:g} to "
                f"{self.pitch_max:g} inches."
            )

        return point

    def _find_exact_geometry(
        self,
        point: FloatArray,
    ) -> int | None:
        """Return the catalog index if the geometry matches exactly."""

        differences = np.abs(
            self.geometries - point
        )

        matching_indices = np.flatnonzero(
            np.all(
                differences <= 1.0e-12,
                axis=1,
            )
        )

        if len(matching_indices) == 0:
            return None

        if len(matching_indices) > 1:
            raise ValueError(
                "More than one exact catalog geometry was found."
            )

        return int(matching_indices[0])

    def _build_inside_hull_blend(
        self,
        point: FloatArray,
        simplex_index: int,
    ) -> GeometryBlend:
        """Build barycentric weights inside the geometry hull."""

        transform = self.geometry_triangulation.transform[
            simplex_index
        ]

        first_weights = transform[:2] @ (
            point - transform[2]
        )

        weights = np.append(
            first_weights,
            1.0 - np.sum(first_weights),
        ).astype(np.float64)

        vertex_indices = (
            self.geometry_triangulation.simplices[
                simplex_index
            ]
        )

        surfaces = tuple(
            self.catalog.surfaces[int(index)]
            for index in vertex_indices
        )

        weights.setflags(write=False)

        return GeometryBlend(
            surfaces=surfaces,
            weights=weights,
            method="triangulated",
        )

    def _build_outside_hull_blend(
        self,
        point: FloatArray,
    ) -> GeometryBlend:
        """
        Build local linear diameter/pitch extrapolation weights.

        This is only used inside the diameter/pitch bounding box but
        outside the convex hull of the catalog geometries.
        """

        normalized_query = (
            point - self.geometry_minimum
        ) / self.geometry_scale

        neighbor_count = min(
            self.extrapolation_neighbor_count,
            len(self.geometries),
        )

        while True:
            distances, indices = self.geometry_tree.query(
                normalized_query,
                k=neighbor_count,
            )

            distances = np.atleast_1d(
                np.asarray(
                    distances,
                    dtype=np.float64,
                )
            )

            indices = np.atleast_1d(
                np.asarray(
                    indices,
                    dtype=np.int64,
                )
            )

            offsets = (
                self.normalized_geometries[indices]
                - normalized_query
            )

            design_matrix = np.column_stack(
                (
                    np.ones(len(indices)),
                    offsets,
                )
            )

            distance_weights = 1.0 / np.maximum(
                distances,
                1.0e-12,
            )

            sqrt_weights = np.sqrt(
                distance_weights
            )

            weighted_design = (
                design_matrix
                * sqrt_weights[:, None]
            )

            rank = np.linalg.matrix_rank(
                weighted_design
            )

            if rank == 3:
                pseudoinverse = np.linalg.pinv(
                    weighted_design
                )

                # The fitted plane is centered on the query point.
                # Its intercept is therefore the output at the query.
                blend_weights = (
                    pseudoinverse[0]
                    * sqrt_weights
                )

                surfaces = tuple(
                    self.catalog.surfaces[int(index)]
                    for index in indices
                )

                blend_weights = np.asarray(
                    blend_weights,
                    dtype=np.float64,
                )

                blend_weights.setflags(write=False)

                return GeometryBlend(
                    surfaces=surfaces,
                    weights=blend_weights,
                    method="local-linear-extrapolation",
                )

            if neighbor_count == len(self.geometries):
                raise ValueError(
                    "Could not form a full-rank geometry plane."
                )

            neighbor_count = min(
                neighbor_count * 2,
                len(self.geometries),
            )

    def geometry_blend(
        self,
        diameter_in: float,
        pitch_in: float,
    ) -> GeometryBlend:
        """
        Determine which catalog props contribute to one geometry.

        This calculation occurs once per diameter/pitch query, not once
        per velocity/RPM point.
        """

        diameter_in = float(diameter_in)
        pitch_in = float(pitch_in)

        cache_key = (
            round(diameter_in, 12),
            round(pitch_in, 12),
        )

        cached_blend = self._blend_cache.get(
            cache_key
        )

        if cached_blend is not None:
            return cached_blend

        point = self._validate_geometry(
            diameter_in,
            pitch_in,
        )

        exact_index = self._find_exact_geometry(
            point
        )

        if exact_index is not None:
            weights = np.array(
                [1.0],
                dtype=np.float64,
            )

            weights.setflags(write=False)

            blend = GeometryBlend(
                surfaces=(
                    self.catalog.surfaces[exact_index],
                ),
                weights=weights,
                method="exact",
            )

            self._blend_cache[cache_key] = blend
            return blend

        simplex_index = int(
            self.geometry_triangulation.find_simplex(
                point
            )
        )

        if simplex_index >= 0:
            blend = self._build_inside_hull_blend(
                point,
                simplex_index,
            )
        else:
            blend = self._build_outside_hull_blend(
                point
            )

        self._blend_cache[cache_key] = blend

        return blend

    def evaluate(
        self,
        diameter_in: float,
        pitch_in: float,
        velocity_mph: ArrayLike,
        rpm: ArrayLike,
    ) -> tuple[QueryResult, QueryResult]:
        """Return thrust and torque for scalar or array inputs."""

        blend = self.geometry_blend(
            diameter_in,
            pitch_in,
        )

        total_thrust: FloatArray | None = None
        total_torque: FloatArray | None = None

        for weight, surface in zip(
            blend.weights,
            blend.surfaces,
        ):
            surface_thrust, surface_torque = (
                surface.evaluate(
                    velocity_mph,
                    rpm,
                )
            )

            surface_thrust_array = np.asarray(
                surface_thrust,
                dtype=np.float64,
            )

            surface_torque_array = np.asarray(
                surface_torque,
                dtype=np.float64,
            )

            if total_thrust is None:
                total_thrust = np.zeros_like(
                    surface_thrust_array,
                    dtype=np.float64,
                )

                total_torque = np.zeros_like(
                    surface_torque_array,
                    dtype=np.float64,
                )

            total_thrust += (
                float(weight)
                * surface_thrust_array
            )

            total_torque += (
                float(weight)
                * surface_torque_array
            )

        if total_thrust is None or total_torque is None:
            raise RuntimeError(
                "Geometry blend contained no surfaces."
            )

        if total_thrust.ndim == 0:
            return (
                float(total_thrust),
                float(total_torque),
            )

        return total_thrust, total_torque

    def thrust(
        self,
        diameter_in: float,
        pitch_in: float,
        velocity_mph: ArrayLike,
        rpm: ArrayLike,
    ) -> QueryResult:
        """Return thrust in newtons."""

        thrust, _ = self.evaluate(
            diameter_in,
            pitch_in,
            velocity_mph,
            rpm,
        )

        return thrust

    def torque(
        self,
        diameter_in: float,
        pitch_in: float,
        velocity_mph: ArrayLike,
        rpm: ArrayLike,
    ) -> QueryResult:
        """Return torque in newton-metres."""

        _, torque = self.evaluate(
            diameter_in,
            pitch_in,
            velocity_mph,
            rpm,
        )

        return torque


def build_continuous_prop_database(
    catalog_database: CatalogPropDatabase,
) -> ContinuousPropDatabase:
    """Build the continuous database from catalog surfaces."""

    return ContinuousPropDatabase(
        catalog_database
    )


@lru_cache(maxsize=1)
def load_default_continuous_prop_database(
) -> ContinuousPropDatabase:
    """Build and cache the default continuous database."""

    catalog_database = (
        load_default_catalog_prop_database()
    )

    return build_continuous_prop_database(
        catalog_database
    )