"""Fast regression checks for battery-cell-count plumbing."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.mech.main_mech import evaluate_mechanical_module
from src.opt.topline_opt import (
    ToplineConfig,
    _design_vector_from_optimizer,
    _integrality_mask,
    _optimizer_bounds,
    _optimizer_variable_names,
)
from src.prop.prop_helper_functions import make_battery_from_design
from src.vectors import ASBDesignVector, DesignVector, ParameterVector


def _battery_mass_kg(design: DesignVector) -> float:
    result = evaluate_mechanical_module(design, parameter_vector=ParameterVector())
    return next(
        item.mass_kg
        for item in result.for_mission("M1").items
        if item.name == "Battery"
    )


def main() -> None:
    design_6s = DesignVector(batt_capacity=3.0, battery_cell_count=6)
    design_8s = replace(design_6s, battery_cell_count=8)
    battery_6s = make_battery_from_design(design_6s, ParameterVector())
    battery_8s = make_battery_from_design(design_8s, ParameterVector())

    assert battery_6s.cells == 6
    assert battery_8s.cells == 8
    assert np.isclose(battery_6s.vnom, 22.2)
    assert np.isclose(battery_8s.vnom, 29.6)
    assert np.isclose(design_6s.batt_energy, 66.6)
    assert np.isclose(design_8s.batt_energy, 88.8)
    assert np.isclose(battery_6s.get_Rb(), 0.026)
    assert np.isclose(battery_8s.get_Rb(), 0.034666666666666665)
    assert np.isclose(_battery_mass_kg(design_6s), 0.51498)
    assert np.isclose(_battery_mass_kg(design_8s), 0.68664)
    assert ASBDesignVector.from_design_vector(design_8s).battery_cell_count == 8

    optimizer_vector = design_6s.to_array()
    rebuilt_8s = _design_vector_from_optimizer(
        optimizer_vector,
        ToplineConfig(battery_cell_count=8),
    )
    assert rebuilt_8s.battery_cell_count == 8
    assert np.isclose(rebuilt_8s.battery_nominal_voltage_v, 29.6)

    joint_config = ToplineConfig(
        optimize_battery_cell_count=True,
        battery_cell_count_bounds=(6, 8),
    )
    joint_vector = np.append(optimizer_vector, 7.0)
    rebuilt_7s = _design_vector_from_optimizer(joint_vector, joint_config)
    assert rebuilt_7s.battery_cell_count == 7
    assert _optimizer_variable_names(joint_config)[-1] == "battery_cell_count"
    assert _optimizer_bounds(joint_config)[-1] == (6.0, 8.0)
    assert _integrality_mask(joint_config)[-1]

    for invalid_count in (0, -1, 6.5, np.nan, True):
        try:
            DesignVector(battery_cell_count=invalid_count)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid cell count: {invalid_count!r}")

    print("Battery cell-count checks passed for fixed 6S and 8S designs.")


if __name__ == "__main__":
    main()
