from src.prop.main_prop import (
    battery,
    motor,
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
        nominal_voltage_v=nominal_voltage_v,
        capacity_ah=capacity_ah,
        num_cells=num_cells,
        pack_resistance_ohm=pack_resistance,
        usable_fraction=0.85,
    )

    # Basic motor setup
    kv = 335.0
    max_power_w = 2200.0

    motor_resistance, no_load_current = motor_properties(
        kv=kv,
        max_power_w=max_power_w,
    )

    motor = motor(
        kv=kv,
        resistance_ohm=motor_resistance,
        no_load_current_a=no_load_current,
        max_power_w=max_power_w,
    )

    # Fake prop operating point.
    # This is NOT from real prop data yet.
    # It is just used to test motor_check().
    torque_nm = 0.2
    rpm = 5000.0

    result = motor_check(
        torque_nm=torque_nm,
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
    print(f"Motor resistance: {motor.resistance_ohm:.6f} ohm")
    print(f"No-load current: {motor.no_load_current_a:.3f} A")
    print(f"Max power: {motor.max_power_w:.1f} W")

    print("\n=== Motor Check ===")
    print(f"Input torque: {torque_nm:.3f} N*m")
    print(f"Input RPM: {rpm:.1f}")
    print(f"Passed: {result.passed}")
    print(f"Current: {result.current_a:.3f} A")
    print(f"Terminal voltage: {result.terminal_voltage_v:.3f} V")
    print(f"Required voltage: {result.voltage_required_v:.3f} V")
    print(f"Throttle: {result.throttle:.3f}")
    print(f"Power: {result.power_w:.3f} W")
    print(f"Estimated flight time: {result.flight_time_s:.1f} s")


if __name__ == "__main__":
    main()