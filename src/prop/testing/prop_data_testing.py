from src.prop.main_prop import (
    DEFAULT_PROP_DATA_PATH,
    parse_prop_key,
    parse_rpm_key,
    load_prop_data_points,
)


def main():
    print("=== Testing key parsing ===")

    diameter, pitch = parse_prop_key("x14x10E")
    print("x14x10E ->", diameter, pitch)

    rpm = parse_rpm_key("RPM_10000")
    print("RPM_10000 ->", rpm)

    print("\n=== Loading prop data ===")
    print("Looking for file at:")
    print(DEFAULT_PROP_DATA_PATH)

    data = load_prop_data_points(DEFAULT_PROP_DATA_PATH)

    print("\n=== Loaded successfully ===")
    print("Number of data points:", len(data["diameter_in"]))

    print("Diameter range:", data["diameter_in"].min(), "to", data["diameter_in"].max())
    print("Pitch range:", data["pitch_in"].min(), "to", data["pitch_in"].max())
    print("Velocity range:", data["velocity_mph"].min(), "to", data["velocity_mph"].max())
    print("RPM range:", data["rpm"].min(), "to", data["rpm"].max())

    print("\nFirst data point:")
    print("Diameter:", data["diameter_in"][0])
    print("Pitch:", data["pitch_in"][0])
    print("Velocity:", data["velocity_mph"][0])
    print("RPM:", data["rpm"][0])
    print("Thrust:", data["thrust_n"][0])
    print("Torque:", data["torque_nm"][0])


if __name__ == "__main__":
    main()