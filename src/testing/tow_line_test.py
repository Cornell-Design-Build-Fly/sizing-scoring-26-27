import numpy as np

from src.aero.drag_model import sensor_drag_force
from src.aero.tow_line_model import (
    DEFAULT_ROPE_REST_LENGTH_M,
    DEFAULT_ROPE_STIFFNESS_N_M,
    steady_tow_line_load,
    tow_line_force_components,
)
from src.vectors import DesignVector, ParameterVector


def test_steady_tow_line_force_balance() -> None:
    design = DesignVector()
    parameters = ParameterVector()
    velocity = 20.0

    backward, downward, tension = tow_line_force_components(
        design, parameters, velocity
    )

    assert np.isclose(
        backward,
        sensor_drag_force(design, parameters, velocity),
    )
    assert np.isclose(
        downward,
        design.sensor_weight_kg * parameters.gravity,
    )
    assert np.isclose(tension, np.hypot(backward, downward))


def test_steady_tow_line_geometry_and_elastic_extension() -> None:
    load = steady_tow_line_load(DesignVector(), ParameterVector(), 20.0)

    assert load.tension_n > 0.0
    assert np.isclose(
        load.extension_m,
        load.tension_n / DEFAULT_ROPE_STIFFNESS_N_M,
    )
    assert np.isclose(
        load.length_m,
        DEFAULT_ROPE_REST_LENGTH_M + load.extension_m,
    )
    assert 0.0 < load.angle_from_vertical_rad < 0.5 * np.pi
