from functools import lru_cache
from pathlib import Path
import hashlib
import json
import pickle
import re

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


DEFAULT_PROP_DATA_PATH = Path(__file__).resolve().parent / "data" / "prop_data.json"
DEFAULT_PROP_CACHE_PATH = Path(__file__).resolve().parent / "data" / "prop_data_continuous.pkl"

PROP_CACHE_VERSION = 1


def parse_prop_key(key: str) -> tuple[float, float]:
    """
    Parses prop names like:
        x14x10E
        14x10
        14.5x10

    Returns:
        diameter_in, pitch_in
    """

    match = re.search(r"(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)", key)

    if match is None:
        raise ValueError(f"Could not parse prop diameter/pitch from key: {key}")

    diameter_in = float(match.group(1))
    pitch_in = float(match.group(2))

    return diameter_in, pitch_in


def parse_rpm_key(key: str) -> float | None:
    """
    Extracts RPM from keys like:
        RPM_10000
        10000
        rpm10000
    """

    match = re.search(r"(\d+(?:\.\d+)?)", key)

    if match is None:
        return None

    return float(match.group(1))


def as_1d_float_array(values) -> np.ndarray:
    """
    Converts JSON list data to a 1D NumPy float array.
    """

    return np.asarray(values, dtype=float).reshape(-1)


def load_prop_data_points(json_path: str | Path):
    """
    Temporary/simple prop data loader.

    This does NOT build interpolation yet.
    It only loads the raw prop data into arrays:
        diameter, pitch, velocity, rpm, thrust, torque
    """

    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Could not find prop data file: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    diameter_list = []
    pitch_list = []
    velocity_list = []
    rpm_list = []
    thrust_list = []
    torque_list = []

    for prop_key, prop_entry in raw_data.items():
        try:
            diameter_in, pitch_in = parse_prop_key(prop_key)
        except ValueError:
            continue

        if not isinstance(prop_entry, dict):
            continue

        for rpm_key, rpm_entry in prop_entry.items():
            rpm = parse_rpm_key(rpm_key)

            if rpm is None:
                continue

            if not isinstance(rpm_entry, dict):
                continue

            if "V" not in rpm_entry:
                continue

            if "Thrust_2" not in rpm_entry:
                continue

            if "Torque_2" not in rpm_entry:
                continue

            velocity = as_1d_float_array(rpm_entry["V"])
            thrust = as_1d_float_array(rpm_entry["Thrust_2"])
            torque = as_1d_float_array(rpm_entry["Torque_2"])

            n = min(len(velocity), len(thrust), len(torque))

            velocity = velocity[:n]
            thrust = thrust[:n]
            torque = torque[:n]

            valid = (
                np.isfinite(velocity)
                & np.isfinite(thrust)
                & np.isfinite(torque)
            )

            for v, t, q in zip(velocity[valid], thrust[valid], torque[valid]):
                diameter_list.append(diameter_in)
                pitch_list.append(pitch_in)
                velocity_list.append(v)
                rpm_list.append(rpm)
                thrust_list.append(t)
                torque_list.append(q)

    return {
        "diameter_in": np.array(diameter_list),
        "pitch_in": np.array(pitch_list),
        "velocity_mph": np.array(velocity_list),
        "rpm": np.array(rpm_list),
        "thrust_n": np.array(thrust_list),
        "torque_nm": np.array(torque_list),
    }

'''INTERPOLATION FOR DIAMETER, PITCH, ASPD, RPM'''

def deduplicate_prop_points(points: np.ndarray, thrust_n: np.ndarray, torque_nm: np.ndarray):
    """
    Removes duplicate interpolation points by averaging their thrust/torque values.
    This helps scipy avoid errors from repeated points.
    """

    unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
    counts = np.bincount(inverse)

    thrust_sum = np.bincount(inverse, weights=thrust_n)
    torque_sum = np.bincount(inverse, weights=torque_nm)

    thrust_avg = thrust_sum / counts
    torque_avg = torque_sum / counts

    return unique_points, thrust_avg, torque_avg


class ContinuousPropDatabase:
    """
    Continuous propeller database.

    This lets us ask:

        thrust = f(diameter, pitch, velocity, rpm)
        torque = f(diameter, pitch, velocity, rpm)

    where:
        diameter is in inches
        pitch is in inches
        velocity is in mph
        rpm is in RPM
    """

    def __init__(self, data: dict):
        points = np.column_stack(
            [
                data["diameter_in"],
                data["pitch_in"],
                data["velocity_mph"],
                data["rpm"],
            ]
        )

        thrust_n = np.asarray(data["thrust_n"], dtype=float)
        torque_nm = np.asarray(data["torque_nm"], dtype=float)

        valid = (
            np.all(np.isfinite(points), axis=1)
            & np.isfinite(thrust_n)
            & np.isfinite(torque_nm)
        )

        points = points[valid]
        thrust_n = thrust_n[valid]
        torque_nm = torque_nm[valid]

        if len(points) == 0:
            raise ValueError("No valid prop data points found.")

        points, thrust_n, torque_nm = deduplicate_prop_points(
            points=points,
            thrust_n=thrust_n,
            torque_nm=torque_nm,
        )

        self.points = points
        self.thrust_n = thrust_n
        self.torque_nm = torque_nm

        self.diameter_bounds_in = (float(points[:, 0].min()), float(points[:, 0].max()))
        self.pitch_bounds_in = (float(points[:, 1].min()), float(points[:, 1].max()))
        self.velocity_bounds_mph = (float(points[:, 2].min()), float(points[:, 2].max()))
        self.rpm_bounds = (float(points[:, 3].min()), float(points[:, 3].max()))

        print("Building thrust interpolator...")
        self.thrust_linear = LinearNDInterpolator(
            points,
            thrust_n,
            fill_value=np.nan,
            rescale=True,
        )

        print("Building torque interpolator...")
        self.torque_linear = LinearNDInterpolator(
            points,
            torque_nm,
            fill_value=np.nan,
            rescale=True,
        )

        print("Building nearest-neighbor fallback...")
        self.thrust_nearest = NearestNDInterpolator(
            points,
            thrust_n,
            rescale=True,
        )

        self.torque_nearest = NearestNDInterpolator(
            points,
            torque_nm,
            rescale=True,
        )

        print("Prop interpolators built.")

    def thrust(self, diameter_in: float, pitch_in: float, velocity_mph: float, rpm: float) -> float:
        query = np.array([[diameter_in, pitch_in, velocity_mph, rpm]], dtype=float)

        value = self.thrust_linear(query)[0]

        if np.isnan(value):
            value = self.thrust_nearest(query)[0]

        return float(value)

    def torque(self, diameter_in: float, pitch_in: float, velocity_mph: float, rpm: float) -> float:
        query = np.array([[diameter_in, pitch_in, velocity_mph, rpm]], dtype=float)

        value = self.torque_linear(query)[0]

        if np.isnan(value):
            value = self.torque_nearest(query)[0]

        return float(value)


def load_continuous_prop_database(json_path=DEFAULT_PROP_DATA_PATH) -> ContinuousPropDatabase:
    """
    Loads prop_data.json and builds the continuous interpolation database.
    """

    data = load_prop_data_points(json_path)
    return ContinuousPropDatabase(data)




# @lru_cache(maxsize=1)
# def load_default_prop_database() -> ContinuousPropDatabase:
#     """
#     Load the prebuilt prop interpolator from .pkl if available.
#     If not available/outdated, rebuild from JSON and save a new .pkl.
#     """
#     json_path = DEFAULT_PROP_DATA_PATH
#     cache_path = DEFAULT_PROP_CACHE_PATH

#     if cache_path.exists():
#         with cache_path.open("rb") as file:
#             payload = pickle.load(file)

#         if (
#             isinstance(payload, dict)
#             and payload.get("version") == PROP_CACHE_VERSION
#             and "prop_database" in payload
#         ):
#             return payload["prop_database"]

#     prop_database = load_continuous_prop_database(json_path)

#     payload = {
#         "version": PROP_CACHE_VERSION,
#         "prop_database": prop_database,
#     }

#     with cache_path.open("wb") as file:
#         pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

#     return prop_database

@lru_cache(maxsize=1)
def load_default_prop_database() -> ContinuousPropDatabase:
    import time

    json_path = DEFAULT_PROP_DATA_PATH
    cache_path = DEFAULT_PROP_CACHE_PATH

    print("load_default_prop_database() called")

    if cache_path.exists():
        print(f"Found cache file: {cache_path}")
        print(f"Cache file size: {cache_path.stat().st_size / 1_000_000:.2f} MB")

        start = time.perf_counter()

        try:
            with cache_path.open("rb") as file:
                payload = pickle.load(file)

            print(f"pickle.load() took {time.perf_counter() - start:.2f} seconds")

            if (
                isinstance(payload, dict)
                and payload.get("version") == PROP_CACHE_VERSION
                and "prop_database" in payload
            ):
                print("Using prop database from .pkl cache")
                return payload["prop_database"]

            print("Cache exists but is invalid/outdated. Rebuilding...")

        except Exception as error:
            print(f"Could not load cache. Rebuilding. Reason: {error}")

    start = time.perf_counter()
    print("Building prop database from JSON...")
    prop_database = load_continuous_prop_database(json_path)
    print(f"Building prop database took {time.perf_counter() - start:.2f} seconds")

    start = time.perf_counter()
    print("Saving prop database to .pkl...")
    payload = {
        "version": PROP_CACHE_VERSION,
        "prop_database": prop_database,
    }

    with cache_path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saving .pkl took {time.perf_counter() - start:.2f} seconds")

    return prop_database