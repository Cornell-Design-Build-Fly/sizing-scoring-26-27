from src.prop.fixed_grid_prop_database import (
    FIXED_GRID_VELOCITIES_MPS,
    FIXED_GRID_RPMS,
    DIAMETER_GRID_STEP_IN,
    PITCH_GRID_STEP_IN,
    save_fixed_velocity_rpm_grid_database,
)


def main() -> None:
    print("Building fixed velocity/RPM prop grid database...")
    print(f"Fixed velocities [m/s]: {FIXED_GRID_VELOCITIES_MPS}")
    print(f"RPM range: {FIXED_GRID_RPMS[0]} to {FIXED_GRID_RPMS[-1]}")
    print(f"Diameter step [in]: {DIAMETER_GRID_STEP_IN}")
    print(f"Pitch step [in]:    {PITCH_GRID_STEP_IN}")

    prop_database = save_fixed_velocity_rpm_grid_database()

    print("\nDone.")
    print(f"Diameter bounds [in]: {prop_database.diameter_bounds_in}")
    print(f"Pitch bounds [in]:    {prop_database.pitch_bounds_in}")
    print(f"RPM bounds:           {prop_database.rpm_bounds}")


if __name__ == "__main__":
    main()