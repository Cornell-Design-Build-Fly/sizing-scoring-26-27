"""Continuous-payload duplicate of the three-case mechanical design sweep.

From the repository root::

    python -m src.testing.mech_test_design_sweep_continuous

This script first checks the floor-placement/fractional-mass contract, then
prints full outputs and saves CSV ledgers plus 2-D layouts for three fractional
design vectors.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from math import floor
from pathlib import Path

import numpy as np

from src.mech import MechanicalModuleConfig
from src.mech.main_mech import evaluate_mechanical_module as evaluate_discrete
from src.mech.main_mech_continuous import (
    evaluate_mechanical_module_continuous,
)
from src.testing.mech_test_design_sweep import (
    REPOSITORY_ROOT,
    DesignCase,
    run_design_cases,
)
from src.vectors import DesignVector


DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data_dump" / "mech_design_sweep_continuous"


CONTINUOUS_DESIGN_CASES = (
    DesignCase(
        slug="small_payload_continuous",
        label="Small continuous payload",
        design_vector=DesignVector(
            ducks_num=3.25,
            pucks_num=1.50,
            banner_length=0.50,
        ),
    ),
    DesignCase(
        slug="medium_payload_continuous",
        label="Medium continuous payload",
        design_vector=DesignVector(
            ducks_num=7.50,
            pucks_num=5.25,
            banner_length=2.75,
        ),
    ),
    DesignCase(
        slug="large_payload_continuous",
        label="Large continuous payload",
        design_vector=DesignVector(
            ducks_num=9.75,
            pucks_num=9.50,
            banner_length=5.00,
        ),
    ),
)


def _whole_payload_items(result, label: str):
    return tuple(
        item
        for item in result.for_mission("M2").items
        if item.category == "mission_2_payload" and item.name.startswith(f"{label} ")
    )


def _fractional_payload_items(result):
    return tuple(
        item
        for item in result.for_mission("M2").items
        if item.category == "mission_2_fractional_payload"
    )


def _assert_continuous_case(
    design: DesignVector,
    config: MechanicalModuleConfig | None = None,
) -> None:
    """Verify one continuous result against its floor-count discrete result."""

    resolved_config = config or MechanicalModuleConfig()
    duck_count = floor(float(design.ducks_num))
    puck_count = floor(float(design.pucks_num))
    duck_fraction = float(design.ducks_num) - duck_count
    puck_fraction = float(design.pucks_num) - puck_count
    floor_design = replace(
        design,
        ducks_num=float(duck_count),
        pucks_num=float(puck_count),
    )
    floor_result = evaluate_discrete(floor_design, resolved_config)
    continuous_result = evaluate_mechanical_module_continuous(
        design,
        resolved_config,
    )
    floor_m2 = floor_result.for_mission("M2")
    continuous_m2 = continuous_result.for_mission("M2")

    assert len(_whole_payload_items(continuous_result, "Duck")) == duck_count
    assert len(_whole_payload_items(continuous_result, "Puck")) == puck_count
    residual_items = _fractional_payload_items(continuous_result)
    expected_residual_count = int(duck_fraction > 0.0) + int(puck_fraction > 0.0)
    assert len(residual_items) == expected_residual_count

    expected_residual_mass = (
        duck_fraction * resolved_config.mission2.duck.mass_kg
        + puck_fraction * resolved_config.mission2.puck.mass_kg
    )
    assert np.isclose(
        continuous_m2.total_mass_kg - floor_m2.total_mass_kg,
        expected_residual_mass,
        rtol=0.0,
        atol=1e-12,
    )
    expected_payload_mass = (
        float(design.ducks_num) * resolved_config.mission2.duck.mass_kg
        + float(design.pucks_num) * resolved_config.mission2.puck.mass_kg
    )
    modeled_payload_mass = sum(
        item.mass_kg
        for item in continuous_m2.items
        if item.category.startswith("mission_2_")
    )
    assert np.isclose(modeled_payload_mass, expected_payload_mass, atol=1e-12)

    # A point mass placed at the floor-layout CG changes mass and weight only.
    assert np.allclose(continuous_m2.cg_m, floor_m2.cg_m, atol=1e-12)
    assert np.allclose(
        continuous_m2.inertia_tensor_kg_m2,
        floor_m2.inertia_tensor_kg_m2,
        atol=1e-12,
    )
    assert np.isclose(continuous_m2.static_margin, floor_m2.static_margin, atol=1e-12)
    for item in residual_items:
        assert np.allclose(item.position_m, continuous_m2.cg_m, atol=1e-12)
        assert np.array_equal(item.dimensions_m, np.zeros(3))


def _assert_integer_parity() -> None:
    design = DesignVector(ducks_num=4.0, pucks_num=2.0)
    discrete = evaluate_discrete(design)
    continuous = evaluate_mechanical_module_continuous(design)
    assert len(_fractional_payload_items(continuous)) == 0
    assert [item.name for item in continuous.all_items] == [
        item.name for item in discrete.all_items
    ]
    for mission in ("M1", "M2", "M3"):
        discrete_mission = discrete.for_mission(mission)
        continuous_mission = continuous.for_mission(mission)
        assert np.isclose(
            continuous_mission.total_mass_kg,
            discrete_mission.total_mass_kg,
            rtol=0.0,
            atol=0.0,
        )
        assert np.array_equal(continuous_mission.cg_m, discrete_mission.cg_m)
        assert np.array_equal(
            continuous_mission.inertia_tensor_kg_m2,
            discrete_mission.inertia_tensor_kg_m2,
        )


def _assert_invalid_amounts_rejected() -> None:
    for field_name in ("ducks_num", "pucks_num"):
        for invalid_value in (-0.1, np.nan, np.inf):
            design = DesignVector(**{field_name: invalid_value})
            try:
                evaluate_mechanical_module_continuous(design)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"{field_name}={invalid_value!r} should have been rejected"
                )


def _run_continuous_regressions() -> None:
    if not __debug__:
        raise RuntimeError(
            "This executable test uses assertions; run it without Python's -O flag."
        )
    for case in CONTINUOUS_DESIGN_CASES:
        _assert_continuous_case(case.design_vector)

    # Direct contract example: 3.75 ducks and 1.25 pucks place 3 and 1 whole
    # items while retaining the remaining 0.75- and 0.25-item masses.
    contract_design = DesignVector(ducks_num=3.75, pucks_num=1.25)
    _assert_continuous_case(contract_design)
    contract_result = evaluate_mechanical_module_continuous(contract_design)
    contract_payload_mass = sum(
        item.mass_kg
        for item in contract_result.for_mission("M2").items
        if item.category.startswith("mission_2_")
    )
    assert np.isclose(contract_payload_mass, 0.280, atol=1e-12)

    # Fraction-only loads and configured unit masses use the same contract.
    _assert_continuous_case(DesignVector(ducks_num=0.75, pucks_num=0.25))
    base_config = MechanicalModuleConfig()
    custom_config = replace(
        base_config,
        mission2=replace(
            base_config.mission2,
            duck=replace(base_config.mission2.duck, mass_kg=0.020),
            puck=replace(base_config.mission2.puck, mass_kg=0.200),
        ),
    )
    _assert_continuous_case(
        DesignVector(ducks_num=3.75, pucks_num=1.25),
        custom_config,
    )

    # Packing capacity is governed by floor counts, and flooring is strict even
    # for a representable value immediately below an integer.
    _assert_continuous_case(DesignVector(ducks_num=10.99, pucks_num=9.99))
    just_below_three = float(np.nextafter(3.0, 0.0))
    boundary = evaluate_mechanical_module_continuous(
        DesignVector(ducks_num=just_below_three, pucks_num=0.0)
    )
    assert len(_whole_payload_items(boundary, "Duck")) == 2

    _assert_integer_parity()
    _assert_invalid_amounts_rejected()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV and PNG outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    arguments = parser.parse_args()
    _run_continuous_regressions()
    print("Continuous-payload regression checks passed.")
    run_design_cases(
        CONTINUOUS_DESIGN_CASES,
        evaluator=evaluate_mechanical_module_continuous,
        output_dir=arguments.output_dir,
    )


if __name__ == "__main__":
    main()
