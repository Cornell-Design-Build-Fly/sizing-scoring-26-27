from __future__ import annotations

from pathlib import Path
import json
import math
from functools import lru_cache

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from src.vectors import DesignVector, ParameterVector
from src.prop.prop_classes import (
    Battery,
    Motor,
    PropulsionCurveFit,
    MPS_TO_MPH,
    DEFAULT_VELOCITIES_MPS,
)

from src.prop.main_prop import (
    DEFAULT_PROP_DATA_PATH,
    parse_prop_key,
    parse_rpm_key,
    as_1d_float_array,
    motor_check,
    make_motor_from_design,
    make_battery_from_design,
)


def deduplicate_2d_points(
    points: np.ndarray,
    thrust_n: np.ndarray,
    torque_nm: np.ndarray,
):
    """
    Removes duplicate (velocity, rpm) points by averaging thrust/torque.
    """

    unique_points, inverse = np.unique(points, axis=0, return_inverse=True)

    counts = np.bincount(inverse)
    thrust_sum = np.bincount(inverse, weights=thrust_n)
    torque_sum = np.bincount(inverse, weights=torque_nm)

    thrust_avg = thrust_sum / counts
    torque_avg = torque_sum / counts

    return unique_points, thrust_avg, torque_avg


class Prop2DModel:
    """
    A prop-specific interpolation model.

    This is the optimized replacement for repeatedly querying the big 4D interpolator.

    For one real prop, such as 14x10, this builds:

        thrust = f(velocity_mph, rpm)
        torque = f(velocity_mph, rpm)

    So diameter and pitch are not interpolation dimensions inside this object.
    """

    def __init__(
        self,
        prop_key: str,
        diameter_in: float,
        pitch_in: float,
        points: np.ndarray,
        thrust_n: np.ndarray,
        torque_nm: np.ndarray,
    ):
        self.prop_key = prop_key
        self.diameter_in = float(diameter_in)
        self.pitch_in = float(pitch_in)

        self.points = np.asarray(points, dtype=float)
        self.thrust_n = np.asarray(thrust_n, dtype=float)
        self.torque_nm = np.asarray(torque_nm, dtype=float)

        self._built = False
        self.thrust_linear = None
        self.torque_linear = None
        self.thrust_nearest = None
        self.torque_nearest = None

    def _build(self) -> None:
        """
        Lazily builds the 2D interpolators only when this prop is actually used.
        """

        if self._built:
            return

        valid = (
            np.all(np.isfinite(self.points), axis=1)
            & np.isfinite(self.thrust_n)
            & np.isfinite(self.torque_nm)
        )

        points = self.points[valid]
        thrust_n = self.thrust_n[valid]
        torque_nm = self.torque_nm[valid]

        if len(points) == 0:
            raise ValueError(f"No valid data points for prop {self.prop_key}")

        points, thrust_n, torque_nm = deduplicate_2d_points(
            points=points,
            thrust_n=thrust_n,
            torque_nm=torque_nm,
        )

        self.points = points
        self.thrust_n = thrust_n
        self.torque_nm = torque_nm

        self.velocity_bounds_mph = (
            float(points[:, 0].min()),
            float(points[:, 0].max()),
        )
        self.rpm_bounds = (
            float(points[:, 1].min()),
            float(points[:, 1].max()),
        )

        # Nearest-neighbor fallback should almost always work.
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

        # Linear interpolation is better when valid, but can fail for odd data.
        try:
            self.thrust_linear = LinearNDInterpolator(
                points,
                thrust_n,
                fill_value=np.nan,
                rescale=True,
            )
            self.torque_linear = LinearNDInterpolator(
                points,
                torque_nm,
                fill_value=np.nan,
                rescale=True,
            )
        except Exception:
            self.thrust_linear = None
            self.torque_linear = None

        self._built = True

    def thrust(self, velocity_mph: float, rpm: float) -> float:
        self._build()

        query = np.array([[velocity_mph, rpm]], dtype=float)

        value = np.nan

        if self.thrust_linear is not None:
            value = float(np.asarray(self.thrust_linear(query)).reshape(-1)[0])

        if not math.isfinite(value):
            value = float(np.asarray(self.thrust_nearest(query)).reshape(-1)[0])

        return value

    def torque(self, velocity_mph: float, rpm: float) -> float:
        self._build()

        query = np.array([[velocity_mph, rpm]], dtype=float)

        value = np.nan

        if self.torque_linear is not None:
            value = float(np.asarray(self.torque_linear(query)).reshape(-1)[0])

        if not math.isfinite(value):
            value = float(np.asarray(self.torque_nearest(query)).reshape(-1)[0])

        return value

    def summary(self) -> str:
        return (
            f"Exact prop model: {self.prop_key} "
            f"({self.diameter_in:g}x{self.pitch_in:g})"
        )


class LocalBlendedPropModel:
    """
    A local model for a requested diameter/pitch that does not exactly exist.

    Example:
        requested prop = 14.8x10.2

    This object finds nearby real props, such as:
        14x10
        15x10
        15x11

    Then for every velocity/RPM query:
        1. evaluate each nearby prop's 2D interpolator
        2. blend those thrust/torque values using distance-based weights
    """

    def __init__(
        self,
        requested_diameter_in: float,
        requested_pitch_in: float,
        prop_models: list[Prop2DModel],
        weights: np.ndarray,
    ):
        self.requested_diameter_in = float(requested_diameter_in)
        self.requested_pitch_in = float(requested_pitch_in)
        self.prop_models = list(prop_models)
        self.weights = np.asarray(weights, dtype=float)

        if len(self.prop_models) == 0:
            raise ValueError("LocalBlendedPropModel needs at least one prop model.")

        if len(self.prop_models) != len(self.weights):
            raise ValueError("Number of prop models must match number of weights.")

        weight_sum = float(np.sum(self.weights))

        if weight_sum <= 0:
            raise ValueError("Prop blending weights must sum to a positive number.")

        self.weights = self.weights / weight_sum

    def thrust(self, velocity_mph: float, rpm: float) -> float:
        values = []
        weights = []

        for model, weight in zip(self.prop_models, self.weights):
            value = model.thrust(velocity_mph, rpm)

            if math.isfinite(value):
                values.append(value)
                weights.append(weight)

        if len(values) == 0:
            return math.nan

        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)

        return float(np.sum(weights * values))

    def torque(self, velocity_mph: float, rpm: float) -> float:
        values = []
        weights = []

        for model, weight in zip(self.prop_models, self.weights):
            value = model.torque(velocity_mph, rpm)

            if math.isfinite(value):
                values.append(value)
                weights.append(weight)

        if len(values) == 0:
            return math.nan

        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)

        return float(np.sum(weights * values))

    def summary(self) -> str:
        lines = [
            (
                "Local blended prop model for "
                f"{self.requested_diameter_in:g}x{self.requested_pitch_in:g}"
            ),
            "Nearby props used:",
        ]

        for model, weight in zip(self.prop_models, self.weights):
            lines.append(
                f"  {model.prop_key:20s} "
                f"({model.diameter_in:g}x{model.pitch_in:g}) "
                f"weight = {weight:.3f}"
            )

        return "\n".join(lines)


class OptimizedPropDatabase:
    """
    Optimized prop database.

    Unlike ContinuousPropDatabase, this does not build one giant 4D interpolator.

    It stores each real prop separately. Then, given a requested diameter/pitch,
    it returns either:

        1. one exact 2D prop model, if that prop exists
        2. a local blended model using nearby props, if not
    """

    def __init__(self, prop_models: list[Prop2DModel]):
        if len(prop_models) == 0:
            raise ValueError("No prop models were loaded.")

        self.prop_models = list(prop_models)

        self.geometry = np.array(
            [
                [model.diameter_in, model.pitch_in]
                for model in self.prop_models
            ],
            dtype=float,
        )

        self.diameter_bounds_in = (
            float(self.geometry[:, 0].min()),
            float(self.geometry[:, 0].max()),
        )
        self.pitch_bounds_in = (
            float(self.geometry[:, 1].min()),
            float(self.geometry[:, 1].max()),
        )

    def get_prop_model(
        self,
        diameter_in: float,
        pitch_in: float,
        exact_tol: float = 1e-9,
        k_nearest: int = 6,
        distance_power: float = 2.0,
    ):
        """
        Returns a prop model for the requested diameter and pitch.

        Exact match:
            returns one Prop2DModel

        Non-exact match:
            returns a LocalBlendedPropModel using nearby props
        """

        diameter_in = float(diameter_in)
        pitch_in = float(pitch_in)

        if diameter_in <= 0:
            raise ValueError("Propeller diameter must be positive.")

        if pitch_in <= 0:
            raise ValueError("Propeller pitch must be positive.")

        diffs = self.geometry - np.array([diameter_in, pitch_in], dtype=float)
        distances = np.sqrt(np.sum(diffs**2, axis=1))

        exact_indices = np.where(distances <= exact_tol)[0]

        if len(exact_indices) > 0:
            return self.prop_models[int(exact_indices[0])]

        k = min(int(k_nearest), len(self.prop_models))

        nearest_indices = np.argsort(distances)[:k]
        nearest_distances = distances[nearest_indices]

        # Inverse-distance weights.
        # Nearby props matter more than farther props.
        eps = 1e-12
        weights = 1.0 / (nearest_distances + eps) ** distance_power

        nearby_models = [
            self.prop_models[int(index)]
            for index in nearest_indices
        ]

        return LocalBlendedPropModel(
            requested_diameter_in=diameter_in,
            requested_pitch_in=pitch_in,
            prop_models=nearby_models,
            weights=weights,
        )


def load_optimized_prop_database(
    json_path: str | Path = DEFAULT_PROP_DATA_PATH,
) -> OptimizedPropDatabase:
    """
    Loads prop_data.json into prop-specific 2D interpolation models.
    """

    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Could not find prop data file: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    prop_models = []

    for prop_key, prop_entry in raw_data.items():
        try:
            diameter_in, pitch_in = parse_prop_key(prop_key)
        except ValueError:
            continue

        if not isinstance(prop_entry, dict):
            continue

        velocity_list = []
        rpm_list = []
        thrust_list = []
        torque_list = []

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
                velocity_list.append(v)
                rpm_list.append(rpm)
                thrust_list.append(t)
                torque_list.append(q)

        if len(velocity_list) < 3:
            continue

        points = np.column_stack(
            [
                np.asarray(velocity_list, dtype=float),
                np.asarray(rpm_list, dtype=float),
            ]
        )

        prop_models.append(
            Prop2DModel(
                prop_key=prop_key,
                diameter_in=diameter_in,
                pitch_in=pitch_in,
                points=points,
                thrust_n=np.asarray(thrust_list, dtype=float),
                torque_nm=np.asarray(torque_list, dtype=float),
            )
        )

    return OptimizedPropDatabase(prop_models)


@lru_cache(maxsize=1)
def load_default_optimized_prop_database() -> OptimizedPropDatabase:
    """
    Cached optimized prop database.

    This loads the JSON and organizes each prop separately.
    Individual 2D interpolators are still built lazily only when used.
    """

    return load_optimized_prop_database(DEFAULT_PROP_DATA_PATH)


def cruise_values_optimized(
    velocity_mph: float,
    motor: Motor,
    battery: Battery,
    max_current_a: float,
    cruise_throttle: float,
    prop_model,
    min_rpm: int = 3000,
    max_rpm: int = 16000,
    rpm_step: int = 100,
) -> tuple[float, float]:
    """
    Optimized cruise_values.

    Instead of asking:
        thrust(diameter, pitch, velocity, rpm)

    this asks:
        thrust(velocity, rpm)

    because the prop model has already been chosen for the input diameter/pitch.
    """

    if velocity_mph < 0:
        raise ValueError("Velocity cannot be negative.")

    if max_current_a <= 0:
        raise ValueError("Max current must be positive.")

    if cruise_throttle <= 0:
        return 0.0, 0.0

    cruise_throttle = min(float(cruise_throttle), 1.0)

    best_thrust_n = -math.inf
    best_flight_time_s = math.inf

    rpm_low = int(min_rpm)
    rpm_high = int(max_rpm)

    while (rpm_high - rpm_low) >= rpm_step:
        rpm_mid = int(round((rpm_low + rpm_high) / 2))

        thrust_n = prop_model.thrust(
            velocity_mph=velocity_mph,
            rpm=rpm_mid,
        )

        torque_nm = prop_model.torque(
            velocity_mph=velocity_mph,
            rpm=rpm_mid,
        )

        if not math.isfinite(thrust_n) or not math.isfinite(torque_nm):
            rpm_low = rpm_mid + 1
            continue

        check = motor_check(
            torque=torque_nm,
            rpm=rpm_mid,
            motor=motor,
            battery=battery,
        )

        within_limits = (
            check.passed
            and check.throttle <= cruise_throttle
            and check.power_w <= motor.max_power
            and check.current_a <= max_current_a
        )

        if within_limits:
            if thrust_n > best_thrust_n:
                best_thrust_n = thrust_n
                best_flight_time_s = check.flight_time_s

            rpm_low = rpm_mid + 1

        else:
            rpm_high = rpm_mid - 1

    if best_thrust_n == -math.inf:
        return 0.0, 0.0

    return float(best_thrust_n), float(best_flight_time_s)


def _get_value(obj, name: str, default):
    return getattr(obj, name, default)


def prop_main_optimized(
    design_vector: DesignVector,
    parameter_vector: ParameterVector = ParameterVector,
    mission: int = 1,
    prop_database: OptimizedPropDatabase | None = None,
    velocities_mps: np.ndarray | None = None,
    disp_res: bool = False,
) -> PropulsionCurveFit:
    """
    Optimized main propulsion model.

    Main difference from prop_main:
        It chooses/builds a local prop model once for the input diameter/pitch,
        then cruise_values only interpolates over velocity and RPM.
    """

    if mission not in (1, 2, 3):
        raise ValueError("mission must be 1, 2, or 3.")

    if prop_database is None:
        prop_database = load_default_optimized_prop_database()

    if velocities_mps is None:
        velocities_mps = DEFAULT_VELOCITIES_MPS.copy()
    else:
        velocities_mps = np.asarray(velocities_mps, dtype=float).reshape(-1)

    if len(velocities_mps) < 3:
        raise ValueError("Need at least 3 velocity samples for quadratic polyfit.")

    diameter_in = float(_get_value(design_vector, "prop_diameter_in", 14.0))
    pitch_in = float(_get_value(design_vector, "prop_pitch_in", 10.0))

    prop_model = prop_database.get_prop_model(
        diameter_in=diameter_in,
        pitch_in=pitch_in,
    )

    if mission in (1, 2):
        cruise_throttle = float(_get_value(design_vector, "cruise_throttle", 0.90))
    else:
        cruise_throttle = float(
            _get_value(design_vector, "mission3_cruise_throttle", 0.85)
        )

    motor = make_motor_from_design(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
    )

    battery = make_battery_from_design(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
    )

    max_thrust_samples = np.zeros_like(velocities_mps, dtype=float)
    throttled_thrust_samples = np.zeros_like(velocities_mps, dtype=float)
    max_time_samples = np.zeros_like(velocities_mps, dtype=float)
    throttled_time_samples = np.zeros_like(velocities_mps, dtype=float)

    for i, velocity_mps in enumerate(velocities_mps):
        velocity_mph = float(velocity_mps * MPS_TO_MPH)

        max_thrust, max_time = cruise_values_optimized(
            velocity_mph=velocity_mph,
            motor=motor,
            battery=battery,
            max_current_a=motor.max_current,
            cruise_throttle=1.0,
            prop_model=prop_model,
        )

        throttled_thrust, throttled_time = cruise_values_optimized(
            velocity_mph=velocity_mph,
            motor=motor,
            battery=battery,
            max_current_a=motor.max_current,
            cruise_throttle=cruise_throttle,
            prop_model=prop_model,
        )

        # Preserve the same behavior as your current prop_main.
        if i > 0 and max_thrust_samples[i - 1] == 0.0:
            max_thrust_samples[i] = 0.0
            max_time_samples[i] = 0.0
        else:
            max_thrust_samples[i] = max_thrust
            max_time_samples[i] = max_time

        if i > 0 and throttled_thrust_samples[i - 1] == 0.0:
            throttled_thrust_samples[i] = 0.0
            throttled_time_samples[i] = 0.0
        else:
            throttled_thrust_samples[i] = throttled_thrust
            throttled_time_samples[i] = throttled_time

    max_time_samples = np.nan_to_num(
        max_time_samples,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    throttled_time_samples = np.nan_to_num(
        throttled_time_samples,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    max_thrust_fit = np.polyfit(velocities_mps, max_thrust_samples, 2)
    throttled_thrust_fit = np.polyfit(velocities_mps, throttled_thrust_samples, 2)

    max_time_fit = np.polyfit(velocities_mps, max_time_samples, 2)
    throttled_time_fit = np.polyfit(velocities_mps, throttled_time_samples, 2)

    result = PropulsionCurveFit(
        throttled_thrust=throttled_thrust_fit,
        max_thrust=max_thrust_fit,
        throttled_time=throttled_time_fit,
        max_time=max_time_fit,
        sample_velocities_mps=velocities_mps,
        throttled_thrust_samples=throttled_thrust_samples,
        max_thrust_samples=max_thrust_samples,
        throttled_time_samples=throttled_time_samples,
        max_time_samples=max_time_samples,
    )

    if disp_res:
        plot_propulsion_result_optimized(result)

    return result


def prop_main_interp_optimized(
    design_vector: DesignVector,
    parameter_vector: ParameterVector = ParameterVector,
    mission: int = 1,
    prop_database: OptimizedPropDatabase | None = None,
    velocities_mps: np.ndarray | None = None,
    disp_res: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB-style wrapper for the optimized prop main.

    Returns:
        p_throttled_thrust, p_max_thrust, p_throttled_t, p_max_t
    """

    result = prop_main_optimized(
        design_vector=design_vector,
        parameter_vector=parameter_vector,
        mission=mission,
        prop_database=prop_database,
        velocities_mps=velocities_mps,
        disp_res=disp_res,
    )

    return (
        result.throttled_thrust,
        result.max_thrust,
        result.throttled_time,
        result.max_time,
    )


def evaluate_curve_optimized(coefficients: np.ndarray, velocity_mps):
    return np.polyval(coefficients, velocity_mps)


def plot_propulsion_result_optimized(result: PropulsionCurveFit) -> None:
    import matplotlib.pyplot as plt

    velocities = result.sample_velocities_mps

    plt.figure()
    plt.scatter(velocities, result.throttled_thrust_samples, label="Cruise samples")
    plt.scatter(velocities, result.max_thrust_samples, label="Max samples")
    plt.plot(
        velocities,
        evaluate_curve_optimized(result.throttled_thrust, velocities),
        label="Cruise fit",
    )
    plt.plot(
        velocities,
        evaluate_curve_optimized(result.max_thrust, velocities),
        label="Max fit",
    )
    plt.xlabel("Velocity [m/s]")
    plt.ylabel("Thrust [N]")
    plt.title("Optimized propulsion thrust curve")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.scatter(velocities, result.throttled_time_samples, label="Cruise samples")
    plt.scatter(velocities, result.max_time_samples, label="Max samples")
    plt.plot(
        velocities,
        evaluate_curve_optimized(result.throttled_time, velocities),
        label="Cruise fit",
    )
    plt.plot(
        velocities,
        evaluate_curve_optimized(result.max_time, velocities),
        label="Max fit",
    )
    plt.xlabel("Velocity [m/s]")
    plt.ylabel("Flight time [s]")
    plt.title("Optimized propulsion flight-time curve")
    plt.grid(True)
    plt.legend()

    plt.show()