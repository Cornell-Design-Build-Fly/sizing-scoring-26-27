from src.prop.main_prop import load_default_prop_database


def main():
    print("=== Loading continuous prop database ===")
    prop_database = load_default_prop_database()

    print("\n=== Database bounds ===")
    print("Diameter bounds [in]:", prop_database.diameter_bounds_in)
    print("Pitch bounds [in]:", prop_database.pitch_bounds_in)
    print("Velocity bounds [mph]:", prop_database.velocity_bounds_mph)
    print("RPM bounds:", prop_database.rpm_bounds)

    print("\n=== Testing one interpolation query ===")

    diameter_in = 14.0
    pitch_in = 10.0
    velocity_mph = 20.0
    rpm = 10000.0

    thrust = prop_database.thrust(
        diameter_in=diameter_in,
        pitch_in=pitch_in,
        velocity_mph=velocity_mph,
        rpm=rpm,
    )

    torque = prop_database.torque(
        diameter_in=diameter_in,
        pitch_in=pitch_in,
        velocity_mph=velocity_mph,
        rpm=rpm,
    )

    print(f"Diameter: {diameter_in} in")
    print(f"Pitch: {pitch_in} in")
    print(f"Velocity: {velocity_mph} mph")
    print(f"RPM: {rpm}")

    print(f"Interpolated thrust: {thrust:.4f} N")
    print(f"Interpolated torque: {torque:.6f} N*m")


if __name__ == "__main__":
    main()