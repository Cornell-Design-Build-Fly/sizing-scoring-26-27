from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class LocalLinearExtrapolator:
    """
    Extrapolate thrust and torque using a local weighted plane.

    Velocity and RPM are normalized before finding nearby points so
    RPM's larger numerical magnitude does not dominate distance.
    """

    normalized_points: FloatArray
    output_values: FloatArray

    velocity_scale: float
    rpm_scale: float

    point_tree: cKDTree
    neighbor_count: int = 12

    @classmethod
    def build(
        cls,
        points: FloatArray,
        thrust_values: FloatArray,
        torque_values: FloatArray,
        neighbor_count: int = 12,
    ) -> LocalLinearExtrapolator:
        """Build an extrapolator for one catalog propeller."""

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                "points must have shape (sample_count, 2)."
            )

        if (
            len(points) != len(thrust_values)
            or len(points) != len(torque_values)
        ):
            raise ValueError(
                "Points, thrust, and torque must have equal lengths."
            )

        if neighbor_count < 3:
            raise ValueError(
                "neighbor_count must be at least 3."
            )

        velocity_scale = float(np.ptp(points[:, 0]))
        rpm_scale = float(np.ptp(points[:, 1]))

        if velocity_scale <= 0.0 or rpm_scale <= 0.0:
            raise ValueError(
                "Data must span both velocity and RPM."
            )

        scale = np.array(
            [velocity_scale, rpm_scale],
            dtype=np.float64,
        )

        normalized_points = points / scale

        # Column 0 is thrust, column 1 is torque.
        output_values = np.column_stack(
            (thrust_values, torque_values)
        )

        normalized_points.setflags(write=False)
        output_values.setflags(write=False)

        return cls(
            normalized_points=normalized_points,
            output_values=output_values,
            velocity_scale=velocity_scale,
            rpm_scale=rpm_scale,
            point_tree=cKDTree(normalized_points),
            neighbor_count=neighbor_count,
        )

    def evaluate(
        self,
        query_points: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """
        Extrapolate thrust and torque at N velocity/RPM points.

        query_points[:, 0] must contain velocity in mph.
        query_points[:, 1] must contain RPM.
        """

        if query_points.ndim != 2 or query_points.shape[1] != 2:
            raise ValueError(
                "query_points must have shape (query_count, 2)."
            )

        scale = np.array(
            [self.velocity_scale, self.rpm_scale],
            dtype=np.float64,
        )

        normalized_queries = query_points / scale

        predictions = np.empty(
            (len(query_points), 2),
            dtype=np.float64,
        )

        raw_point_count = len(self.normalized_points)

        for query_index, query in enumerate(normalized_queries):
            neighbor_count = min(
                self.neighbor_count,
                raw_point_count,
            )

            while True:
                distances, indices = self.point_tree.query(
                    query,
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

                # Center the local plane on the query point.
                offsets = (
                    self.normalized_points[indices] - query
                )

                design_matrix = np.column_stack(
                    (
                        np.ones(len(indices)),
                        offsets,
                    )
                )

                # Nearby raw samples receive more influence.
                weights = 1.0 / np.maximum(
                    distances,
                    1.0e-12,
                )

                sqrt_weights = np.sqrt(weights)

                weighted_design = (
                    design_matrix
                    * sqrt_weights[:, None]
                )

                weighted_outputs = (
                    self.output_values[indices]
                    * sqrt_weights[:, None]
                )

                coefficients, _, rank, _ = np.linalg.lstsq(
                    weighted_design,
                    weighted_outputs,
                    rcond=None,
                )

                if rank == 3:
                    # Because the plane is centered on the query,
                    # coefficient 0 is its value at the query.
                    predictions[query_index] = coefficients[0]
                    break

                # Nearby points can occasionally all lie on one line.
                # Expand the neighborhood until a 2D plane is possible.
                if neighbor_count == raw_point_count:
                    raise ValueError(
                        "Could not form a full-rank "
                        "extrapolation plane."
                    )

                neighbor_count = min(
                    neighbor_count * 2,
                    raw_point_count,
                )

        thrust = predictions[:, 0]
        torque = predictions[:, 1]

        return thrust, torque