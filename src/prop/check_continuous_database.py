from __future__ import annotations

import numpy as np

from src.prop.continuous_prop_database import (
    ContinuousPropDatabase,
    load_default_continuous_prop_database,
)


MPS_TO_MPH = 2.2369

VELOCITIES_MPS = np.array(
    [0.01, 9.5, 19.0, 28.35],
    dtype=float,
)

RPMS = np.arange(
    3000.0,
    16000.0 + 100.0,
    100.0,
)

OFF_GRID_GEOMETRIES = (
    (14.8, 10.2),
    (13.5, 10.2),
    (20.3, 11.7),
    (4.2, 3.5),
    (26.5, 14.0),
)


def find_geometry_outside_convex_hull(
    database: ContinuousPropDatabase,
    grid_size: int = 201,
) -> tuple[float, float]:
    """
    Find a diameter/pitch point that is inside the database's
    rectangular bounds but outside the catalog convex hull.
    """

    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")

    diameters = np.linspace(
        database.diameter_min,
        database.diameter_max,
        grid_size,
    )

    pitches = np.linspace(
        database.pitch_min,
        database.pitch_max,
        grid_size,
    )

    # Skip the outermost rectangle boundary so the test point is
    # clearly inside the allowed diameter and pitch bounds.
    for diameter_in in diameters[1:-1]:
        for pitch_in in pitches[1:-1]:
            geometry = np.array(
                [diameter_in, pitch_in],
                dtype=float,
            )

            simplex_index = (
                database.geometry_triangulation.find_simplex(
                    geometry
                )
            )

            if simplex_index < 0:
                return (
                    float(diameter_in),
                    float(pitch_in),
                )

    raise RuntimeError(
        "Could not find a point inside the rectangular bounds "
        "and outside the catalog convex hull."
    )


def count_nonfinite(
    thrust: np.ndarray | float,
    torque: np.ndarray | float,
) -> int:
    """Count query points with invalid thrust or torque."""

    thrust_array = np.asarray(
        thrust,
        dtype=float,
    )

    torque_array = np.asarray(
        torque,
        dtype=float,
    )

    invalid = (
        ~np.isfinite(thrust_array)
        | ~np.isfinite(torque_array)
    )

    return int(np.count_nonzero(invalid))


def main() -> None:
    database = (
        load_default_continuous_prop_database()
    )

    velocities_mph = (
        VELOCITIES_MPS * MPS_TO_MPH
    )

    velocity_grid, rpm_grid = np.meshgrid(
        velocities_mph,
        RPMS,
        indexing="ij",
    )

    maximum_thrust_error = 0.0
    maximum_torque_error = 0.0
    nonfinite_exact_outputs = 0

    # Confirm that the continuous geometry layer adds no error
    # when an exact catalog diameter and pitch are requested.
    for surface in database.catalog.surfaces:
        direct_thrust, direct_torque = (
            surface.evaluate(
                velocity_grid,
                rpm_grid,
            )
        )

        continuous_thrust, continuous_torque = (
            database.evaluate(
                surface.diameter_in,
                surface.pitch_in,
                velocity_grid,
                rpm_grid,
            )
        )

        thrust_error = np.max(
            np.abs(
                np.asarray(continuous_thrust)
                - np.asarray(direct_thrust)
            )
        )

        torque_error = np.max(
            np.abs(
                np.asarray(continuous_torque)
                - np.asarray(direct_torque)
            )
        )

        maximum_thrust_error = max(
            maximum_thrust_error,
            float(thrust_error),
        )

        maximum_torque_error = max(
            maximum_torque_error,
            float(torque_error),
        )

        nonfinite_exact_outputs += count_nonfinite(
            continuous_thrust,
            continuous_torque,
        )

    print("Exact catalog geometry validation:")

    print(
        f"  Maximum added thrust error: "
        f"{maximum_thrust_error:.16g} N"
    )

    print(
        f"  Maximum added torque error: "
        f"{maximum_torque_error:.16g} N·m"
    )

    print(
        f"  Non-finite outputs: "
        f"{nonfinite_exact_outputs}"
    )

    print()
    print("Off-grid interpolation validation:")

    for diameter_in, pitch_in in (
        OFF_GRID_GEOMETRIES
    ):
        blend = database.geometry_blend(
            diameter_in,
            pitch_in,
        )

        thrust, torque = database.evaluate(
            diameter_in,
            pitch_in,
            velocity_grid,
            rpm_grid,
        )

        invalid_count = count_nonfinite(
            thrust,
            torque,
        )

        print(
            f"  {diameter_in:g}x{pitch_in:g}: "
            f"method={blend.method}, "
            f"catalog props={len(blend.surfaces)}, "
            f"weight sum={np.sum(blend.weights):.12g}, "
            f"non-finite outputs={invalid_count}"
        )

    # Automatically find and test a point that activates the
    # diameter/pitch extrapolation branch.
    outside_diameter, outside_pitch = (
        find_geometry_outside_convex_hull(
            database
        )
    )

    outside_blend = database.geometry_blend(
        outside_diameter,
        outside_pitch,
    )

    outside_thrust, outside_torque = (
        database.evaluate(
            outside_diameter,
            outside_pitch,
            velocity_grid,
            rpm_grid,
        )
    )

    outside_invalid_count = count_nonfinite(
        outside_thrust,
        outside_torque,
    )

    print()
    print("Geometry extrapolation validation:")

    print(
        f"  Test geometry: "
        f"{outside_diameter:.4f}x"
        f"{outside_pitch:.4f}"
    )

    print(
        f"  Method: {outside_blend.method}"
    )

    print(
        f"  Catalog props used: "
        f"{len(outside_blend.surfaces)}"
    )

    print(
        f"  Weight sum: "
        f"{np.sum(outside_blend.weights):.12g}"
    )

    print(
        f"  Non-finite outputs: "
        f"{outside_invalid_count}"
    )

    if (
        outside_blend.method
        != "local-linear-extrapolation"
    ):
        raise AssertionError(
            "The automatically selected geometry did not "
            "activate geometry extrapolation."
        )

    if outside_invalid_count != 0:
        raise AssertionError(
            "Geometry extrapolation produced invalid outputs."
        )


if __name__ == "__main__":
    main()