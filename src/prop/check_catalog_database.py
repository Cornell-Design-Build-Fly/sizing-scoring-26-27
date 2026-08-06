from __future__ import annotations

import numpy as np

from src.prop.prop_database import (
    CatalogPropSurface,
    load_default_catalog_prop_database,
)


MPS_TO_MPH = 2.2369

FIXED_VELOCITIES_MPS = np.array(
    [0.01, 9.5, 19.0, 28.35],
    dtype=float,
)

FIXED_RPMS = np.arange(
    3000.0,
    16000.0 + 100.0,
    100.0,
)


def check_raw_point_reproduction(
    surface: CatalogPropSurface,
) -> tuple[float, float]:
    """
    Return maximum interpolation error at the original raw samples.
    """

    velocity_mph = surface.points[:, 0]
    rpm = surface.points[:, 1]

    interpolated_thrust, interpolated_torque = (
        surface.evaluate(
            velocity_mph,
            rpm,
        )
    )

    interpolated_thrust = np.asarray(
        interpolated_thrust,
        dtype=float,
    )

    interpolated_torque = np.asarray(
        interpolated_torque,
        dtype=float,
    )

    maximum_thrust_error = float(
        np.max(
            np.abs(
                interpolated_thrust
                - surface.thrust_values
            )
        )
    )

    maximum_torque_error = float(
        np.max(
            np.abs(
                interpolated_torque
                - surface.torque_values
            )
        )
    )

    return (
        maximum_thrust_error,
        maximum_torque_error,
    )


def count_outside_solver_grid(
    surface: CatalogPropSurface,
) -> int:
    """
    Count fixed-grid queries outside this propeller's convex hull.
    """

    velocities_mph = (
        FIXED_VELOCITIES_MPS * MPS_TO_MPH
    )

    velocity_grid, rpm_grid = np.meshgrid(
        velocities_mph,
        FIXED_RPMS,
        indexing="ij",
    )

    inside = np.asarray(
        surface.contains(
            velocity_grid,
            rpm_grid,
        ),
        dtype=bool,
    )

    return int(np.count_nonzero(~inside))

def count_nonfinite_solver_outputs(
    surface: CatalogPropSurface,
) -> int:
    """Count NaN or infinite outputs on the fixed solver grid."""

    velocities_mph = (
        FIXED_VELOCITIES_MPS * MPS_TO_MPH
    )

    velocity_grid, rpm_grid = np.meshgrid(
        velocities_mph,
        FIXED_RPMS,
        indexing="ij",
    )

    thrust, torque = surface.evaluate(
        velocity_grid,
        rpm_grid,
    )

    thrust = np.asarray(thrust, dtype=float)
    torque = np.asarray(torque, dtype=float)

    invalid = (
        ~np.isfinite(thrust)
        | ~np.isfinite(torque)
    )

    return int(np.count_nonzero(invalid))


def main() -> None:
    database = load_default_catalog_prop_database()

    maximum_thrust_error = 0.0
    maximum_torque_error = 0.0

    surfaces_with_outside_points: list[
        tuple[str, int]
    ] = []

    nonfinite_output_count = 0

    for surface in database.surfaces:

        thrust_error, torque_error = (
            check_raw_point_reproduction(surface)
        )

        maximum_thrust_error = max(
            maximum_thrust_error,
            thrust_error,
        )

        maximum_torque_error = max(
            maximum_torque_error,
            torque_error,
        )

        outside_count = count_outside_solver_grid(
            surface
        )

        if outside_count > 0:
            surfaces_with_outside_points.append(
                (surface.key, outside_count)
            )

        nonfinite_output_count += (count_nonfinite_solver_outputs(surface))   

    total_fixed_queries_per_prop = (
        len(FIXED_VELOCITIES_MPS)
        * len(FIXED_RPMS)
    )

    print(
        f"Catalog propellers built: "
        f"{database.propeller_count}"
    )

    print(
        f"Fixed-grid queries per propeller: "
        f"{total_fixed_queries_per_prop}"
    )

    print()
    print("Raw-point reproduction:")

    print(
        f"  Maximum thrust error: "
        f"{maximum_thrust_error:.16g} N"
    )

    print(
        f"  Maximum torque error: "
        f"{maximum_torque_error:.16g} N·m"
    )

    print()
    print("Fixed solver-grid coverage:")

    fully_covered_count = (
        database.propeller_count
        - len(surfaces_with_outside_points)
    )

    print(
        f"  Fully covered propellers: "
        f"{fully_covered_count}"
    )

    print(
        f"  Propellers requiring extrapolation: "
        f"{len(surfaces_with_outside_points)}"
    )

    if surfaces_with_outside_points:
        print()
        print(
            "First propellers requiring extrapolation:"
        )

        for key, outside_count in (
            surfaces_with_outside_points[:20]
        ):
            print(
                f"  {key}: "
                f"{outside_count} of "
                f"{total_fixed_queries_per_prop} points"
            )

        remaining_count = (
            len(surfaces_with_outside_points) - 20
        )

        if remaining_count > 0:
            print(
                f"  ... and {remaining_count} more"
            )

    print()
    print("Extrapolated output validation:")
    print(
        f"  Non-finite thrust/torque outputs: "
        f"{nonfinite_output_count}"
    )


if __name__ == "__main__":
    main()