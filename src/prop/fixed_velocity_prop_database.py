from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from src.prop.prop_classes import MPS_TO_MPH
from src.prop.prop_database import ContinuousPropDatabase, load_default_prop_database


FIXED_FIT_VELOCITIES_MPS = np.array([0.01, 9.5, 19.0, 28.35], dtype=float)
FIXED_FIT_VELOCITIES_MPH = FIXED_FIT_VELOCITIES_MPS * MPS_TO_MPH

DEFAULT_FIXED_PROP_PICKLE_PATH = (
    Path(__file__).resolve().parent / "data" / "prop_data_fixed_velocity_3d.pkl"
)

FIXED_PROP_CACHE_VERSION = 1


class FixedVelocity3DPropDatabase:
    """
    Faster prop database for fixed airspeed samples.

    Instead of interpolating over:
        diameter, pitch, velocity, rpm

    this stores one 3D interpolator per fixed velocity:
        diameter, pitch, rpm
    """

    def __init__(
        self,
        fixed_velocities_mps: np.ndarray,
        points_3d: np.ndarray,
        thrust_values: np.ndarray,
        torque_values: np.ndarray,
    ):
        self.fixed_velocities_mps = np.asarray(fixed_velocities_mps, dtype=float)
        self.fixed_velocities_mph = self.fixed_velocities_mps * MPS_TO_MPH
        self.points_3d = np.asarray(points_3d, dtype=float)

        thrust_values = np.asarray(thrust_values, dtype=float)
        torque_values = np.asarray(torque_values, dtype=float)

        self.diameter_bounds_in = (
            float(self.points_3d[:, 0].min()),
            float(self.points_3d[:, 0].max()),
        )
        self.pitch_bounds_in = (
            float(self.points_3d[:, 1].min()),
            float(self.points_3d[:, 1].max()),
        )
        self.rpm_bounds = (
            float(self.points_3d[:, 2].min()),
            float(self.points_3d[:, 2].max()),
        )
        self.velocity_bounds_mph = (
            float(self.fixed_velocities_mph.min()),
            float(self.fixed_velocities_mph.max()),
        )

        self.thrust_linear = []
        self.thrust_nearest = []
        self.torque_linear = []
        self.torque_nearest = []

        for i, velocity_mps in enumerate(self.fixed_velocities_mps):
            print(f"Building fixed-speed 3D interpolators at {velocity_mps:.2f} m/s")

            self.thrust_linear.append(
                LinearNDInterpolator(
                    self.points_3d,
                    thrust_values[i],
                    fill_value=np.nan,
                    rescale=True,
                )
            )
            self.thrust_nearest.append(
                NearestNDInterpolator(
                    self.points_3d,
                    thrust_values[i],
                    rescale=True,
                )
            )

            self.torque_linear.append(
                LinearNDInterpolator(
                    self.points_3d,
                    torque_values[i],
                    fill_value=np.nan,
                    rescale=True,
                )
            )
            self.torque_nearest.append(
                NearestNDInterpolator(
                    self.points_3d,
                    torque_values[i],
                    rescale=True,
                )
            )

    def _evaluate_3d_slice(
        self,
        linear_interp,
        nearest_interp,
        diameter_in,
        pitch_in,
        rpm,
    ) -> np.ndarray:
        d, p, r = np.broadcast_arrays(diameter_in, pitch_in, rpm)
        original_shape = d.shape

        query = np.column_stack(
            [
                d.ravel(),
                p.ravel(),
                r.ravel(),
            ]
        )

        values = linear_interp(query)

        bad = np.isnan(values)
        if np.any(bad):
            values[bad] = nearest_interp(query[bad])

        return values.reshape(original_shape)

    def _evaluate_many(
        self,
        linear_interps,
        nearest_interps,
        diameter_in,
        pitch_in,
        velocity_mph,
        rpm,
    ) -> np.ndarray:
        d, p, v, r = np.broadcast_arrays(
            diameter_in,
            pitch_in,
            velocity_mph,
            rpm,
        )

        output = np.empty(d.shape, dtype=float)
        matched = np.zeros(d.shape, dtype=bool)

        for i, fixed_velocity_mph in enumerate(self.fixed_velocities_mph):
            speed_mask = np.isclose(
                v,
                fixed_velocity_mph,
                rtol=0.0,
                atol=1e-5,
            )

            if not np.any(speed_mask):
                continue

            output[speed_mask] = self._evaluate_3d_slice(
                linear_interps[i],
                nearest_interps[i],
                d[speed_mask],
                p[speed_mask],
                r[speed_mask],
            )

            matched[speed_mask] = True

        if not np.all(matched):
            requested = np.unique(v[~matched])
            supported = self.fixed_velocities_mph
            raise ValueError(
                "FixedVelocity3DPropDatabase only supports the fixed velocity "
                f"samples {supported} mph. Got unsupported velocities {requested} mph."
            )

        return output

    def thrust_many(self, diameter_in, pitch_in, velocity_mph, rpm) -> np.ndarray:
        return self._evaluate_many(
            self.thrust_linear,
            self.thrust_nearest,
            diameter_in,
            pitch_in,
            velocity_mph,
            rpm,
        )

    def torque_many(self, diameter_in, pitch_in, velocity_mph, rpm) -> np.ndarray:
        return self._evaluate_many(
            self.torque_linear,
            self.torque_nearest,
            diameter_in,
            pitch_in,
            velocity_mph,
            rpm,
        )

    def thrust(self, diameter_in: float, pitch_in: float, velocity_mph: float, rpm: float) -> float:
        return float(self.thrust_many(diameter_in, pitch_in, velocity_mph, rpm))

    def torque(self, diameter_in: float, pitch_in: float, velocity_mph: float, rpm: float) -> float:
        return float(self.torque_many(diameter_in, pitch_in, velocity_mph, rpm))


def build_fixed_velocity_3d_prop_database(
    source_database: ContinuousPropDatabase | None = None,
    fixed_velocities_mps: np.ndarray = FIXED_FIT_VELOCITIES_MPS,
) -> FixedVelocity3DPropDatabase:
    """
    Builds the fixed-speed 3D database from the existing 4D database.

    Build-time:
        diameter, pitch, velocity, rpm -> thrust/torque

    Runtime after this:
        diameter, pitch, rpm -> thrust/torque
        for each fixed velocity slice
    """
    if source_database is None:
        source_database = load_default_prop_database()

    # Unique 3D operating coordinates: diameter, pitch, rpm.
    points_3d = source_database.points[:, [0, 1, 3]]
    points_3d = np.unique(points_3d, axis=0)

    diameter = points_3d[:, 0]
    pitch = points_3d[:, 1]
    rpm = points_3d[:, 2]

    fixed_velocities_mps = np.asarray(fixed_velocities_mps, dtype=float)
    fixed_velocities_mph = fixed_velocities_mps * MPS_TO_MPH

    thrust_values = []
    torque_values = []

    for velocity_mps, velocity_mph in zip(fixed_velocities_mps, fixed_velocities_mph):
        print(f"Sampling 4D prop database at {velocity_mps:.2f} m/s")

        velocity = np.full_like(rpm, velocity_mph, dtype=float)

        thrust_values.append(
            source_database.thrust_many(
                diameter,
                pitch,
                velocity,
                rpm,
            )
        )

        torque_values.append(
            source_database.torque_many(
                diameter,
                pitch,
                velocity,
                rpm,
            )
        )

    return FixedVelocity3DPropDatabase(
        fixed_velocities_mps=fixed_velocities_mps,
        points_3d=points_3d,
        thrust_values=np.vstack(thrust_values),
        torque_values=np.vstack(torque_values),
    )


def save_fixed_velocity_3d_prop_database(
    cache_path: Path = DEFAULT_FIXED_PROP_PICKLE_PATH,
) -> FixedVelocity3DPropDatabase:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    prop_database = build_fixed_velocity_3d_prop_database()

    payload = {
        "version": FIXED_PROP_CACHE_VERSION,
        "fixed_velocities_mps": FIXED_FIT_VELOCITIES_MPS,
        "prop_database": prop_database,
    }

    with cache_path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved fixed-velocity prop database to: {cache_path}")
    return prop_database


@lru_cache(maxsize=1)
def load_fixed_velocity_3d_prop_database() -> FixedVelocity3DPropDatabase:
    cache_path = DEFAULT_FIXED_PROP_PICKLE_PATH

    if cache_path.exists():
        with cache_path.open("rb") as file:
            payload = pickle.load(file)

        if (
            isinstance(payload, dict)
            and payload.get("version") == FIXED_PROP_CACHE_VERSION
            and "prop_database" in payload
        ):
            return payload["prop_database"]

    return save_fixed_velocity_3d_prop_database()