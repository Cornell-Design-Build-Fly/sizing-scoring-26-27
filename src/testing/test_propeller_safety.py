from src.prop.catalog_selection import resolve_catalog_propellers
from src.prop.continuous_prop_database import load_default_continuous_prop_database
from src.prop.mission_performance import PropulsionRequirements
from src.prop.prop_data import load_prop_data, propeller_blade_count
from src.prop.prop_cruise_values import evaluate_cruise_grid
from src.prop.prop_helper_functions import (
    make_battery_from_design,
    make_motor_from_design,
)
from src.vectors import DesignVector, ParameterVector


def test_multiblade_propellers_are_excluded_from_the_database() -> None:
    propellers = load_prop_data()
    assert propellers
    assert all(propeller_blade_count(propeller.key) == 2 for propeller in propellers)
    assert "12x7E-3" not in {propeller.key for propeller in propellers}


def test_optimizer_requests_resolve_to_exact_two_blade_catalog_props() -> None:
    database = load_default_continuous_prop_database()
    requested = DesignVector(
        prop_diameter_in=11.698817919480165,
        prop_pitch_in=7.67856592707368,
        mission3_prop_diameter_in=12.179511815875086,
        mission3_prop_pitch_in=6.906122494041282,
    )
    resolved = resolve_catalog_propellers(
        requested,
        database,
        minimum_diameter_in=10.0,
        maximum_diameter_in=25.0,
        minimum_pitch_in=4.0,
        maximum_pitch_in=18.0,
        minimum_pitch_diameter_ratio=0.4,
        maximum_pitch_diameter_ratio=0.8,
    )
    for mission in (1, 3):
        diameter, pitch = resolved.propeller_for_mission(mission)
        surface = database.catalog.get_by_geometry(diameter, pitch)
        assert propeller_blade_count(surface.key) == 2
    assert resolved.propeller_for_mission(1) == (12.0, 8.0)
    assert resolved.propeller_for_mission(3) == (13.0, 6.5)


def test_thin_electric_rpm_limit_has_design_margin() -> None:
    requirements = PropulsionRequirements()
    manufacturer_limit = (
        requirements.maximum_propeller_rpm_diameter_product / 12.0
    )
    operating_limit = (
        requirements.propeller_rpm_limit_safety_factor * manufacturer_limit
    )
    assert manufacturer_limit == 12_500.0
    assert operating_limit == 11_250.0


def test_flight_grid_rejects_source_data_extrapolation() -> None:
    database = load_default_continuous_prop_database()
    design = DesignVector(prop_diameter_in=14.0, prop_pitch_in=10.0)
    parameters = ParameterVector()
    grid = evaluate_cruise_grid(
        14.0,
        10.0,
        (300.0,),
        make_motor_from_design(design, parameters),
        make_battery_from_design(design, parameters),
        database,
        min_rpm=3_000,
        max_rpm=20_000,
        rpm_step=100,
    )
    assert not grid.base_valid_mask.any()


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"PASS {name}")
