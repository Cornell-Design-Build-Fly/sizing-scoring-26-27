# and here I am, catching you slacking, looking at the codebase for the first time...
from __future__ import annotations
from pathlib import Path
import re
import json
import math
from functools import lru_cache


import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from src.vectors import DesignVector, ParameterVector
from src.prop.prop_classes import (
    Battery,
    Motor,
    MotorCheckResult,
    PropInterpolants,
    PropulsionCurveFit,
    MPS_TO_MPH,
    DEFAULT_VELOCITIES_MPS,
)


''' MATLAB HELPER FOLDER'''
def battery_resistance(capacity_ah: float, num_cells: int) -> float:
    '''Calculates the internal resistance of a battery based on its capacity (ah) and number of cells.'''
    '''if capacity_ah <= 0:
        raise ValueError("Battery capacity must be positive.")
    if num_cells <= 0:
        raise ValueError("Number of cells must be positive.")'''
    return (0.013/capacity_ah)*num_cells

def motor_properties(kv: float, max_power_w: float) -> tuple[float, float]:
    '''Calcualtes the motor resistance and no-load current based on its KV and maximum power.'''
    if kv <= 0:
        raise ValueError("Motor KV must be positive.")
    if max_power_w <= 0:
        raise ValueError("Motor maximum power must be positive.")
    c_R = np.array([0.3517732388, -0.0005385476, -0.0001855504, 0.0000002999, 0.0000000776, 0.0000000380,])
    c_I = np.array([-0.5621009279, 0.0005335965, 0.0016292435, 0.0000005495, 0.0000006015, -0.0000004552])

    Rm = c_R[0] + c_R[1]*kv + c_R[2]*max_power_w + c_R[3]*kv**2 + c_R[4]*kv*max_power_w + c_R[5]*max_power_w**2;
    Inot = c_I[0] + c_I[1]*kv + c_I[2]*max_power_w + c_I[3]*kv**2 + c_I[4]*kv*max_power_w + c_I[5]*max_power_w**2;
    
    return Rm, Inot



'''MOTOR CHECK FUNCTION'''
def motor_check(torque: float, rpm: float, motor: Motor, battery: Battery):
    if rpm <= 0:
        raise ValueError("RPM must be positive.")
    passed = True
    current = (torque / motor.get_kt()) + motor.I0 #(A) Current needed to sustain torque

    V_sag = battery.vnom - current*(battery.Rb) #(V) Voltage drop in battery under load
    V_req = rpm/motor.kv+current*motor.Rm #Voltage required due to EMF
    power = current*V_sag #(W) Power consumed by motor

    #Battery flight time calculation

    if current <= 1e-6:  #Avoid division by zero or very small power; use a small threshold
        t_flight = np.inf; #Effectively infinite time if no significant power drawn
    else:
        #E_battery is in Wh. P is in W. (Wh / W) = hours.
        #Convert hours to seconds by multiplying by 3600.
        t_flight = battery.capacity/current*3600.0

    #Throttle Required
    if V_sag <= 1e-6:  # Avoid division by zero or negative V_sag
        throttle = np.inf  # Effectively infinite throttle required if V_sag is non-positive
    else:
        throttle = V_req / V_sag

    # --- Start of Failure conditions ---

    # Motor Power Limit Check
    # Motor Power will later set all values to zero if overshot
    # if P > P_max
    #     # fprintf('MOTOR CHECK FAIL: Power overload. RPM: %.0f, P_electrical: %.2f W > P_max_motor: %.2f W\n', RPM, P, P_max);
    #     pass = false;
    # end

    # Voltage Required vs Nominal Voltage
    if V_req > battery.vnom and passed:
        # fprintf('MOTOR CHECK FAIL: Insufficient nominal voltage. RPM: %.0f, V_req: %.2f V > V_nom: %.2f V\n', RPM, V_req, V_nom);
        passed = False

    # Throttle Limit Check (This is often the key RPM limiting factor)
    if throttle > 1.0 and passed:
        # fprintf('MOTOR CHECK FAIL: Throttle overload. RPM: %.0f, Throttle: %.3f > 1. (V_req: %.2f V, V_sag: %.2f V, I: %.2f A)\n', RPM, throttle, V_req, V_sag, I);
        passed = False
    
    # Check for V_sag becoming non-positive (battery completely depleted or calculation issue)
    if V_sag <= 0 and passed:
        # fprintf('MOTOR CHECK FAIL: Battery voltage sagged too low. RPM: %.0f, V_sag: %.2f V, I: %.2f A\n', RPM, V_sag, I);
        passed = False
    
    return MotorCheckResult(
        passed=passed,
        current_a=current,
        voltage_sag_v=V_sag,
        voltage_required_v=V_req,
        throttle=throttle,
        power_w=power,
        flight_time_s=t_flight
    )



'''DATA MANAGEMENT'''

DEFAULT_PROP_DATA_PATH = Path(__file__).resolve().parent / "data" / "prop_data.json"


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


@lru_cache(maxsize=1)
def load_default_prop_database() -> ContinuousPropDatabase:
    """
    Cached default prop database.

    Important for efficiency: the interpolators are expensive to build,
    so we only want to build them once.
    """

    return load_continuous_prop_database(DEFAULT_PROP_DATA_PATH)




'''Cruise Values'''

def cruise_values(
    diameter_in: float,
    pitch_in: float,
    velocity_mph: float,
    motor: Motor,
    battery: Battery,
    max_current_a: float,
    cruise_throttle: float,
    prop_database: ContinuousPropDatabase,
    min_rpm: int = 3000,
    max_rpm: int = 16000,
    rpm_step: int = 100,
) -> tuple[float, float]:
    """
    Finds the highest valid thrust at a given airspeed and throttle limit.

    Inputs:
        diameter_in:
            Propeller diameter [in]

        pitch_in:
            Propeller pitch [in]

        velocity_mph:
            Aircraft forward speed [mph]

        motor:
            Motor object

        battery:
            Battery object

        max_current_a:
            Current limit [A]

        cruise_throttle:
            Maximum allowed throttle for this condition.
            Use 1.0 for max-throttle thrust.
            Use something like 0.7 or 0.9 for cruise-throttle thrust.

        prop_database:
            ContinuousPropDatabase object with thrust/torque interpolation.

    Returns:
        best_thrust_n:
            Highest valid thrust found [N]

        best_flight_time_s:
            Estimated flight time at that operating point [s]
    """

    if diameter_in <= 0:
        raise ValueError("Propeller diameter must be positive.")

    if pitch_in <= 0:
        raise ValueError("Propeller pitch must be positive.")

    if velocity_mph < 0:
        raise ValueError("Velocity cannot be negative.")

    if max_current_a <= 0:
        raise ValueError("Max current must be positive.")

    if cruise_throttle <= 0:
        return 0.0, 0.0

    # Do not allow throttle limit above 1.
    cruise_throttle = min(float(cruise_throttle), 1.0)

    best_thrust_n = -math.inf
    best_flight_time_s = math.inf

    rpm_low = int(min_rpm)
    rpm_high = int(max_rpm)

    while (rpm_high - rpm_low) >= rpm_step:
        rpm_mid = int(round((rpm_low + rpm_high) / 2))

        thrust_n = prop_database.thrust(
            diameter_in=diameter_in,
            pitch_in=pitch_in,
            velocity_mph=velocity_mph,
            rpm=rpm_mid,
        )

        torque_nm = prop_database.torque(
            diameter_in=diameter_in,
            pitch_in=pitch_in,
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

            # This RPM works, so try a higher RPM.
            rpm_low = rpm_mid + 1

        else:
            # This RPM does not work, so try a lower RPM.
            rpm_high = rpm_mid - 1

    if best_thrust_n == -math.inf:
        return 0.0, 0.0

    return float(best_thrust_n), float(best_flight_time_s)

'''PROP MAIN BLOCK'''

def _get_value(obj, name: str, default):
    """
    Gets an attribute from DesignVector or ParameterVector.
    Uses default if the field does not exist.
    """
    return getattr(obj, name, default)


def make_motor_from_design(
    design_vector: DesignVector,
    parameter_vector: ParameterVector = ParameterVector,
) -> Motor:
    """
    Creates a Motor object from the DesignVector and ParameterVector.

        Motor(kv, Rm, max_power, I0, max_current, mass=None)
    """

    kv = float(_get_value(design_vector, "motor_kv", 335.0))
    max_power = float(_get_value(design_vector, "motor_max_power", 2200.0))
    max_current = float(_get_value(parameter_vector, "max_current", 100.0))

    Rm, I0 = motor_properties(
        kv=kv,
        max_power_w=max_power,
    )

    return Motor(
        kv=kv,
        Rm=Rm,
        max_power=max_power,
        I0=I0,
        max_current=max_current,
    )


def make_battery_from_design(
    design_vector: DesignVector,
    parameter_vector: ParameterVector = ParameterVector,
) -> Battery:
    """
    Creates a Battery object from the DesignVector and ParameterVector.

    Battery(vnom, cells, Rb, Crat, capacity, mass=None, useable_fraction=...)
    """

    capacity_ah = float(_get_value(design_vector, "batt_capacity", 4.5))
    vnom = float(_get_value(parameter_vector, "voltage", 22.2))

    cells_default = max(1, int(round(vnom / 3.7)))
    cells = int(_get_value(parameter_vector, "num_battery_cells", cells_default))

    useable_fraction = float(_get_value(parameter_vector, "usable_battery_fraction", 0.85))

    Rb = battery_resistance(capacity_ah=capacity_ah,num_cells=cells,)

    return Battery(
        vnom=vnom,
        cells=cells,
        Rb=Rb,
        Crat=0.0,  # Placeholder until C-rating is actually modeled
        capacity=capacity_ah * useable_fraction,
        useable_fraction=useable_fraction,
    )


def prop_main(
    design_vector: DesignVector,
    parameter_vector: ParameterVector = ParameterVector,
    mission: int = 1,
    prop_database: ContinuousPropDatabase | None = None,
    velocities_mps: np.ndarray | None = None,
    disp_res: bool = False,
) -> PropulsionCurveFit:
    """
    Main propulsion model.

    Continuous diameter/pitch replacement for old MATLAB propMainInterp.m.
    """

    if mission not in (1, 2, 3):
        raise ValueError("mission must be 1, 2, or 3.")

    if prop_database is None:
        prop_database = load_default_prop_database()

    if velocities_mps is None:
        velocities_mps = DEFAULT_VELOCITIES_MPS.copy()
    else:
        velocities_mps = np.asarray(velocities_mps, dtype=float).reshape(-1)

    if len(velocities_mps) < 3:
        raise ValueError("Need at least 3 velocity samples for quadratic polyfit.")

    diameter_in = float(_get_value(design_vector, "prop_diameter_in", 14.0))
    pitch_in = float(_get_value(design_vector, "prop_pitch_in", 10.0))

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

        max_thrust, max_time = cruise_values(
            diameter_in=diameter_in,
            pitch_in=pitch_in,
            velocity_mph=velocity_mph,
            motor=motor,
            battery=battery,
            max_current_a=motor.max_current,
            cruise_throttle=1.0,
            prop_database=prop_database,
        )

        throttled_thrust, throttled_time = cruise_values(
            diameter_in=diameter_in,
            pitch_in=pitch_in,
            velocity_mph=velocity_mph,
            motor=motor,
            battery=battery,
            max_current_a=motor.max_current,
            cruise_throttle=cruise_throttle,
            prop_database=prop_database,
        )

        # Match old MATLAB behavior:
        # once thrust becomes zero at a lower speed, keep later speeds at zero.
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
        plot_propulsion_result(result)

    return result


def prop_main_interp(
    design_vector: DesignVector,
    parameter_vector: ParameterVector = ParameterVector,
    mission: int = 1,
    prop_database: ContinuousPropDatabase | None = None,
    velocities_mps: np.ndarray | None = None,
    disp_res: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB-style wrapper.

    Returns:
        p_throttled_thrust, p_max_thrust, p_throttled_t, p_max_t
    """

    result = prop_main(
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


def evaluate_curve(coefficients: np.ndarray, velocity_mps):
    """
    Evaluates a polynomial curve fit.
    """
    return np.polyval(coefficients, velocity_mps)


def plot_propulsion_result(result: PropulsionCurveFit) -> None:
    """
    Optional debug plotting helper.
    """

    import matplotlib.pyplot as plt

    velocities = result.sample_velocities_mps

    plt.figure()
    plt.scatter(velocities, result.throttled_thrust_samples, label="Cruise samples")
    plt.scatter(velocities, result.max_thrust_samples, label="Max samples")
    plt.plot(
        velocities,
        evaluate_curve(result.throttled_thrust, velocities),
        label="Cruise fit",
    )
    plt.plot(
        velocities,
        evaluate_curve(result.max_thrust, velocities),
        label="Max fit",
    )
    plt.xlabel("Velocity [m/s]")
    plt.ylabel("Thrust [N]")
    plt.title("Propulsion thrust curve")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.scatter(velocities, result.throttled_time_samples, label="Cruise samples")
    plt.scatter(velocities, result.max_time_samples, label="Max samples")
    plt.plot(
        velocities,
        evaluate_curve(result.throttled_time, velocities),
        label="Cruise fit",
    )
    plt.plot(
        velocities,
        evaluate_curve(result.max_time, velocities),
        label="Max fit",
    )
    plt.xlabel("Velocity [m/s]")
    plt.ylabel("Flight time [s]")
    plt.title("Propulsion flight-time curve")
    plt.grid(True)
    plt.legend()

    plt.show()
