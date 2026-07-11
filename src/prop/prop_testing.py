from src.prop.main_prop import (
    Battery,
    Motor,
    battery_resistance,
    motor_properties,
    motor_check
)


def main():
    # Basic battery setup
    capacity_ah = 4.5
    num_cells = 6
    nominal_voltage_v = 22.2


    pack_resistance = battery_resistance(capacity_ah, num_cells)

    battery = Battery(
        vnom=nominal_voltage_v,
        capacity=capacity_ah,
        cells=num_cells,
        Rb=pack_resistance,
        useable_fraction=0.85,
        Crat = 30,  # C rating
    )

    # Basic motor setup
    kv = 335.0
    max_power_w = 2200.0

    motor_resistance, no_load_current = motor_properties(
        kv=kv,
        max_power_w=max_power_w,
    )

    motor = Motor(
        kv=kv,
        Rm=motor_resistance,
        I0=no_load_current,
        max_power=max_power_w,
        max_current=100.0,
    )

    # Fake prop operating point.
    # This is NOT from real prop data yet.
    # It is just used to test motor_check().
    torque_nm = 0.2
    rpm = 4000.0

    result = motor_check(
        torque=torque_nm,
        rpm=rpm,
        motor=motor,
        battery=battery,
    )

    print("=== Battery ===")
    print(f"Capacity: {capacity_ah:.2f} Ah")
    print(f"Cells: {num_cells}S")
    print(f"Nominal voltage: {nominal_voltage_v:.2f} V")
    #print(f"Cell resistance: {cell_resistance:.6f} ohm")
    print(f"Pack resistance: {pack_resistance:.6f} ohm")

    print("\n=== Motor ===")
    print(f"Kv: {motor.kv:.1f} RPM/V")
    print(f"Motor resistance: {motor.Rm:.6f} ohm")
    print(f"No-load current: {motor.I0:.3f} A")
    print(f"Max power: {motor.max_power:.1f} W")

    print("\n=== Motor Check ===")
    print(f"Input torque: {torque_nm:.3f} N*m")
    print(f"Input RPM: {rpm:.1f}")
    print(f"Passed: {result[0]}")
    print(f"Current: {result[1]:.3f} A")
    print(f"Voltage post sag: {result[2]:.3f} V")
    print(f"Required voltage: {result[3]:.3f} V")
    print(f"Throttle: {result[4]:.3f}")
    print(f"Power: {result[5]:.3f} W")
    print(f"Estimated flight time: {result[6]:.1f} s")


if __name__ == "__main__":
    main()