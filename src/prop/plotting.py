"""
Debug plotting utilities for the propulsion module.

Provides optional visualization of propulsion curve fits and sampled
propulsion data for validation and debugging.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.prop.prop_classes import PropulsionCurveFit

def evaluate_curve(coefficients: np.ndarray, velocity_mps: np.ndarray) -> np.ndarray:
    """
    Evaluates a quadratic propulsion curve fit at the specified velocity.
    """
    return np.polyval(coefficients, velocity_mps)

def plot_propulsion_result(result: PropulsionCurveFit) -> None:
    """
    Plots sampled propulsion data and the corresponding quadratic curve fits.

    Intended for debugging and validation of the propulsion model.
    """
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
