"""Run three editable discrete mechanical-design cases and save 2-D layouts.

Edit the three ``DesignVector`` blocks below to compare complete aircraft
designs. Every configurable design-vector value is written explicitly in each
case, including the fuselage cross-section.

From the repository root::

    python -m src.testing.mech_test_custom_design
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.testing.mech_test_design_sweep import (
    REPOSITORY_ROOT,
    DesignCase,
    run_design_cases,
)
from src.vectors import DesignVector


DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data_dump" / "mech_custom_design"


# Change any values in these three complete vectors. Payload counts are kept as
# integers because this script runs the discrete evaluator.
DESIGN_CASES = (
    DesignCase(
        slug="custom_design_1",
        label="Custom design 1",
        design_vector=DesignVector(
            wing_span=1.181,
            wing_chord=0.307,
            tail_arm=0.845,
            nose_length=0.254,
            ducks_num=3,
            pucks_num=1,
            banner_length=0.50,
            batt_capacity=4.5,
            fuselage_width=0.13,
            fuselage_height=0.13,
        ),
    ),
    DesignCase(
        slug="custom_design_2",
        label="Custom design 2",
        design_vector=DesignVector(
            wing_span=1.181,
            wing_chord=0.307,
            tail_arm=0.845,
            nose_length=0.254,
            ducks_num=7,
            pucks_num=5,
            banner_length=2.75,
            batt_capacity=4.5,
            fuselage_width=0.13,
            fuselage_height=0.13,
        ),
    ),
    DesignCase(
        slug="custom_design_3",
        label="Custom design 3",
        design_vector=DesignVector(
            wing_span=1.181,
            wing_chord=0.307,
            tail_arm=0.845,
            nose_length=0.254,
            ducks_num=10,
            pucks_num=9,
            banner_length=5.00,
            batt_capacity=4.5,
            fuselage_width=0.13,
            fuselage_height=0.13,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV and PNG outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    arguments = parser.parse_args()
    run_design_cases(DESIGN_CASES, output_dir=arguments.output_dir)


if __name__ == "__main__":
    main()
