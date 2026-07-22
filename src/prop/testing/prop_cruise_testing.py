from src.prop.main_prop import (
    Battery,
    Motor,
    battery_resistance,
    motor_properties,
    load_default_prop_database,
    cruise_values,
)


def main():
    print("=== Loading prop database ===")
    prop_database = load_default_prop_database()

    print("\n=== Creating battery ===")
    capacity_ah = 4.5
    num_cells = 6
    nominal_voltage_v = 22.2

    battery = Battery(
        vnom=nominal_voltage_v,
        capacity=capacity_ah,
        cells=num_cells,
        Rb=battery_resistance(
            capacity_ah=capacity_ah,
            num_cells=num_cells,
        ),
        useable_fraction=0.85,
        Crat = 30,  # C rating
    )

    print("Battery:", battery)

    print("\n=== Creating motor ===")
    kv = 335.0
    max_power_w = 2200.0

    resistance_ohm, no_load_current_a = motor_properties(
        kv=kv,
        max_power_w=max_power_w,
    )

    motor = Motor(
        kv=kv,
        Rm=resistance_ohm,
        I0=no_load_current_a,
        max_power=max_power_w,
        max_current=100.0,
    )

    print("Motor:", motor)

    print("\n=== Testing cruise_values ===")

    diameter_in = 14.0
    pitch_in = 10.0
    velocity_mph = 20.0
    max_current_a = 100.0

    for throttle in [0.5, 0.7, 0.9, 1.0]:
        thrust_n, flight_time_s = cruise_values(
            diameter_in=diameter_in,
            pitch_in=pitch_in,
            velocity_mph=velocity_mph,
            motor=motor,
            battery=battery,
            max_current_a=max_current_a,
            cruise_throttle=throttle,
            prop_database=prop_database,
        )

        print(f"\nThrottle limit: {throttle:.2f}")
        print(f"Best thrust: {thrust_n:.3f} N")
        print(f"Estimated flight time: {flight_time_s:.1f} s")


if __name__ == "__main__":
    main()