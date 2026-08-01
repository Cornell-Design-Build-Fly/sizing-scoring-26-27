from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pickle

import numpy as np

from src.prop.prop_classes import MPS_TO_MPH
from src.prop.prop_database import ContinuousPropDatabase, load_default_prop_database


FIXED_GRID_VELOCITIES_MPS = np.array([0.01, 9.5, 19.0, 28.35], dtype=float)
FIXED_GRID_VELOCITIES_MPH = FIXED_GRID_VELOCITIES_MPS * MPS_TO_MPH

FIXED_GRID_MIN_RPM = 3000
FIXED_GRID_MAX_RPM = 16000
FIXED_GRID_RPM_STEP = 100
FIXED_GRID_RPMS = np.arange(
    FIXED_GRID_MIN_RPM,
    FIXED_GRID_MAX_RPM + FIXED_GRID_RPM_STEP,
    FIXED_GRID_RPM_STEP,
    dtype=float,
)

DIAMETER_GRID_STEP_IN = 0.1
PITCH_GRID_STEP_IN = 0.1

FIXED_GRID_CACHE_VERSION = 1

DEFAULT_FIXED_GRID_PICKLE_PATH = (
    Path(__file__).resolve().parent / "data" / "prop_data_fixed_v_rpm_grid.pkl"
)


def make_axis_grid(low: float, high: float, step: float) -> np.ndarray:
    """Creates a rounded inclusive grid."""
    return np.round(np.arange(low, high + 0.5 * step, step), 10)


def axis_indices_and_weights(grid: np.ndarray, values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds lower/upper grid indices and interpolation weight for each value.
    Values outside the grid are clipped to the nearest edge.
    """
    values = np.asarray(values, dtype=float)
    clipped = np.clip(values, grid[0], grid[-1])

    upper = np.searchsorted(grid, clipped, side="right")
    upper = np.clip(upper, 1, len(grid) - 1)
    lower = upper - 1

    denom = grid[upper] - grid[lower]
    weight = np.where(denom > 0.0, (clipped - grid[lower]) / denom, 0.0)

    return lower, upper, weight


class FixedVelocityRpmGridPropDatabase:
    """
    Fast prop database.

    Stored table dimensions:
        fixed velocity index, fixed RPM index, diameter grid index, pitch grid index

    Runtime interpolation only happens over:
        diameter, pitch
    """

    def __init__(
        self,
        fixed_velocities_mps: np.ndarray,
        rpm_grid: np.ndarray,
        diameter_grid: np.ndarray,
        pitch_grid: np.ndarray,
        thrust_table: np.ndarray,
        torque_table: np.ndarray,
    ):
        self.fixed_velocities_mps = np.asarray(fixed_velocities_mps, dtype=float)
        self.fixed_velocities_mph = self.fixed_velocities_mps * MPS_TO_MPH

        self.rpm_grid = np.asarray(rpm_grid, dtype=float)
        self.diameter_grid = np.asarray(diameter_grid, dtype=float)
        self.pitch_grid = np.asarray(pitch_grid, dtype=float)

        self.thrust_table = np.asarray(thrust_table)
        self.torque_table = np.asarray(torque_table)

        self.diameter_bounds_in = (
            float(self.diameter_grid[0]),
            float(self.diameter_grid[-1]),
        )
        self.pitch_bounds_in = (
            float(self.pitch_grid[0]),
            float(self.pitch_grid[-1]),
        )
        self.rpm_bounds = (
            float(self.rpm_grid[0]),
            float(self.rpm_grid[-1]),
        )
        self.velocity_bounds_mph = (
            float(self.fixed_velocities_mph[0]),
            float(self.fixed_velocities_mph[-1]),
        )

    def _velocity_indices(self, velocity_mph) -> np.ndarray:
        velocity_mph = np.asarray(velocity_mph, dtype=float)
        diffs = np.abs(velocity_mph[..., None] - self.fixed_velocities_mph)
        indices = np.argmin(diffs, axis=-1)

        matched = np.isclose(
            velocity_mph,
            self.fixed_velocities_mph[indices],
            rtol=0.0,
            atol=1e-5,
        )

        if not np.all(matched):
            bad = np.unique(velocity_mph[~matched])
            raise ValueError(
                "FixedVelocityRpmGridPropDatabase only supports these velocities "
                f"[mph]: {self.fixed_velocities_mph}. Got unsupported: {bad}"
            )

        return indices

    def _rpm_indices(self, rpm) -> np.ndarray:
        rpm = np.asarray(rpm, dtype=float)
        indices = np.rint(
            (rpm - self.rpm_grid[0]) / FIXED_GRID_RPM_STEP
        ).astype(int)

        in_bounds = (indices >= 0) & (indices < len(self.rpm_grid))

        safe_indices = np.clip(indices, 0, len(self.rpm_grid) - 1)
        matched = in_bounds & np.isclose(
            rpm,
            self.rpm_grid[safe_indices],
            rtol=0.0,
            atol=1e-8,
        )

        if not np.all(matched):
            bad = np.unique(rpm[~matched])
            raise ValueError(
                "FixedVelocityRpmGridPropDatabase only supports RPMs on this grid: "
                f"{self.rpm_grid[0]} to {self.rpm_grid[-1]} step "
                f"{FIXED_GRID_RPM_STEP}. Got unsupported: {bad}"
            )

        return indices

    def _evaluate_table(self, table: np.ndarray, diameter_in, pitch_in, velocity_mph, rpm):
        d, p, v, r = np.broadcast_arrays(
            diameter_in,
            pitch_in,
            velocity_mph,
            rpm,
        )

        v_idx = self._velocity_indices(v)
        r_idx = self._rpm_indices(r)

        d0, d1, wd = axis_indices_and_weights(self.diameter_grid, d)
        p0, p1, wp = axis_indices_and_weights(self.pitch_grid, p)

        q00 = table[v_idx, r_idx, d0, p0]
        q10 = table[v_idx, r_idx, d1, p0]
        q01 = table[v_idx, r_idx, d0, p1]
        q11 = table[v_idx, r_idx, d1, p1]

        return (
            (1.0 - wd) * (1.0 - wp) * q00
            + wd * (1.0 - wp) * q10
            + (1.0 - wd) * wp * q01
            + wd * wp * q11
        )

    def thrust_many(self, diameter_in, pitch_in, velocity_mph, rpm) -> np.ndarray:
        return self._evaluate_table(
            self.thrust_table,
            diameter_in,
            pitch_in,
            velocity_mph,
            rpm,
        )

    def torque_many(self, diameter_in, pitch_in, velocity_mph, rpm) -> np.ndarray:
        return self._evaluate_table(
            self.torque_table,
            diameter_in,
            pitch_in,
            velocity_mph,
            rpm,
        )

    def thrust(self, diameter_in: float, pitch_in: float, velocity_mph: float, rpm: float) -> float:
        return float(
            np.asarray(
                self.thrust_many(diameter_in, pitch_in, velocity_mph, rpm)
            ).reshape(-1)[0]
        )

    def torque(self, diameter_in: float, pitch_in: float, velocity_mph: float, rpm: float) -> float:
        return float(
            np.asarray(
                self.torque_many(diameter_in, pitch_in, velocity_mph, rpm)
            ).reshape(-1)[0]
        )


def build_fixed_velocity_rpm_grid_database(
    source_database: ContinuousPropDatabase | None = None,
    table_dtype=np.float32,
    rpm_block_size: int = 4,
) -> FixedVelocityRpmGridPropDatabase:
    """
    Builds a fast fixed-velocity/fixed-RPM grid database from the current 4D database.

    This is a one-time build step:
        4D model: D, P, V, RPM -> thrust/torque

    Runtime after this:
        fixed V, fixed RPM, D, P -> thrust/torque
        with only D/P bilinear interpolation.
    """
    if source_database is None:
        source_database = load_default_prop_database()

    diameter_grid = make_axis_grid(
        source_database.diameter_bounds_in[0],
        source_database.diameter_bounds_in[1],
        DIAMETER_GRID_STEP_IN,
    )
    pitch_grid = make_axis_grid(
        source_database.pitch_bounds_in[0],
        source_database.pitch_bounds_in[1],
        PITCH_GRID_STEP_IN,
    )

    n_v = len(FIXED_GRID_VELOCITIES_MPH)
    n_r = len(FIXED_GRID_RPMS)
    n_d = len(diameter_grid)
    n_p = len(pitch_grid)

    print("Fixed-grid database size:")
    print(f"  velocities: {n_v}")
    print(f"  RPMs:       {n_r}")
    print(f"  diameters:  {n_d}")
    print(f"  pitches:    {n_p}")
    print(f"  total grid points per table: {n_v * n_r * n_d * n_p:,}")

    thrust_table = np.empty((n_v, n_r, n_d, n_p), dtype=table_dtype)
    torque_table = np.empty((n_v, n_r, n_d, n_p), dtype=table_dtype)

    diameter_mesh, pitch_mesh = np.meshgrid(
        diameter_grid,
        pitch_grid,
        indexing="ij",
    )

    diameter_flat = diameter_mesh.ravel()
    pitch_flat = pitch_mesh.ravel()
    n_dp = diameter_flat.size

    for v_i, velocity_mph in enumerate(FIXED_GRID_VELOCITIES_MPH):
        velocity_mps = FIXED_GRID_VELOCITIES_MPS[v_i]
        print(f"\nSampling 4D database at {velocity_mps:.2f} m/s...")

        for r_start in range(0, n_r, rpm_block_size):
            r_end = min(r_start + rpm_block_size, n_r)
            rpm_block = FIXED_GRID_RPMS[r_start:r_end]
            block_count = len(rpm_block)

            d_query = np.broadcast_to(
                diameter_flat[:, None],
                (n_dp, block_count),
            )
            p_query = np.broadcast_to(
                pitch_flat[:, None],
                (n_dp, block_count),
            )
            v_query = np.full(
                (n_dp, block_count),
                velocity_mph,
                dtype=float,
            )
            r_query = np.broadcast_to(
                rpm_block[None, :],
                (n_dp, block_count),
            )

            thrust_block = source_database.thrust_many(
                d_query,
                p_query,
                v_query,
                r_query,
            )
            torque_block = source_database.torque_many(
                d_query,
                p_query,
                v_query,
                r_query,
            )

            thrust_table[v_i, r_start:r_end, :, :] = (
                thrust_block.T.reshape(block_count, n_d, n_p)
            )
            torque_table[v_i, r_start:r_end, :, :] = (
                torque_block.T.reshape(block_count, n_d, n_p)
            )

            print(
                f"  RPM block {r_start + 1:3d}-{r_end:3d} "
                f"of {n_r}",
                end="\r",
            )

        print()

    return FixedVelocityRpmGridPropDatabase(
        fixed_velocities_mps=FIXED_GRID_VELOCITIES_MPS,
        rpm_grid=FIXED_GRID_RPMS,
        diameter_grid=diameter_grid,
        pitch_grid=pitch_grid,
        thrust_table=thrust_table,
        torque_table=torque_table,
    )


def save_fixed_velocity_rpm_grid_database(
    cache_path: Path = DEFAULT_FIXED_GRID_PICKLE_PATH,
) -> FixedVelocityRpmGridPropDatabase:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    prop_database = build_fixed_velocity_rpm_grid_database()

    payload = {
        "version": FIXED_GRID_CACHE_VERSION,
        "prop_database": prop_database,
    }

    with cache_path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved fixed-grid prop database to:\n  {cache_path}")
    return prop_database


@lru_cache(maxsize=1)
def load_fixed_velocity_rpm_grid_database() -> FixedVelocityRpmGridPropDatabase:
    cache_path = DEFAULT_FIXED_GRID_PICKLE_PATH

    if cache_path.exists():
        with cache_path.open("rb") as file:
            payload = pickle.load(file)

        if (
            isinstance(payload, dict)
            and payload.get("version") == FIXED_GRID_CACHE_VERSION
            and "prop_database" in payload
        ):
            return payload["prop_database"]

    return save_fixed_velocity_rpm_grid_database()