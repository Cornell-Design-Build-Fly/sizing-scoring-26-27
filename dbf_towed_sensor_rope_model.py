import math
import csv
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# DBF TOWED-SENSOR DYNAMICS MODEL -- TENSION-ONLY ROPE
# ============================================================
#
# Model:
#   - airplane = prescribed point-mass trajectory at constant speed
#   - sensor   = point mass
#   - connection = massless, tension-only ELASTIC rope with free
#                  ball-joint-like rotation at both ends
#   - sensor sees gravity + quadratic aerodynamic drag
#   - airplane follows:
#       straight -> 180 right -> straight -> 360 left
#       -> straight -> 180 right -> straight
#
# Why use an elastic rope instead of a perfectly inextensible rope?
# ---------------------------------------------------------------
# A real rope can go slack and cannot carry compression. If a perfectly
# inextensible rope goes slack and then suddenly becomes taut, the ideal
# model produces an impulsive (formally unbounded) catch load. A small
# amount of axial elasticity gives a finite, physically interpretable
# peak tension. Peak re-tension loads therefore depend strongly on the
# rope axial stiffness and damping values below.
#
# Rope law:
#       extension = distance - ROPE_REST_LENGTH_FT
#
#       T = max(0, k*extension + c*extension_rate)
#
# when extension > 0. When extension <= 0, T = 0 and the rope is slack.
# The rope never pushes on the airplane or sensor.
#
# The airplane path is prescribed rather than dynamically solved. This
# lets the model answer: "What extra tether loads must the airplane
# overcome to remain on the DBF course?"
#
# IMPORTANT:
#   - Rope aerodynamic drag and rope mass are neglected.
#   - Sensor pitch/yaw/roll are not modeled; the sensor is a point mass.
#   - The nominal bank command defines course curvature. The real airplane
#     may need additional lift/control authority because of tether loads.
# ============================================================


# ---------------- USER INPUTS ----------------

G = 32.174                         # ft/s^2
AIR_DENSITY_SLUG_FT3 = 0.002377   # sea-level standard density

AIRCRAFT_WEIGHT_LBF = 20.0         # airplane weight EXCLUDING sensor
SENSOR_WEIGHT_LBF = 30.0

AIRSPEED_FPS = 70.0                # 70 ft/s = 47.7 mph
BANK_ANGLE_DEG = 35.0              # nominal max bank defining turn radius
ROLL_TIME_S = 1.5                  # smooth roll-in / roll-out time

STRAIGHT_LENGTH_FT = 500.0         # each of the four straight segments

# Rope properties ------------------------------------------------
ROPE_REST_LENGTH_FT = 9.0          # unstretched rope length

# Axial spring stiffness of the WHOLE rope [lbf/ft].
# If you know rope EA, use k = EA / L.
# Example: EA = 9000 lbf and L = 9 ft -> k = 1000 lbf/ft.
ROPE_STIFFNESS_LBF_PER_FT = 1000.0

# Dimensionless viscous damping ratio for axial rope stretch.
# c = 2*zeta*sqrt(k*m_sensor)
# Keep this small unless you have test data.
ROPE_DAMPING_RATIO = 0.03

# Sensor drag -----------------------------------------------------
SENSOR_DIAMETER_IN = 3.0
SENSOR_CD = 0.40                   # EDIT if you have CFD/test data
                                   # frontal area = 3-in diameter circle

WIND_FPS = np.array([0.0, 0.0, 0.0])  # inertial x,y,z wind velocity

INITIAL_HEADING_DEG = 180.0        # starts heading toward left marker

# A stiff rope creates a short axial vibration period, so use a smaller
# step than the old rigid-link model. 0.001-0.002 s is a good starting
# point for k around 1000 lbf/ft and a ~30 lbf sensor.
DT = 0.002                         # integration step [s]

SAVE_CSV = True
CSV_NAME = "dbf_towed_sensor_rope_results.csv"


# ---------------- DERIVED VALUES ----------------

aircraft_mass = AIRCRAFT_WEIGHT_LBF / G   # slugs
sensor_mass = SENSOR_WEIGHT_LBF / G       # slugs

sensor_diameter_ft = SENSOR_DIAMETER_IN / 12.0
sensor_area_ft2 = math.pi * sensor_diameter_ft**2 / 4.0

bank_max = math.radians(BANK_ANGLE_DEG)
initial_heading = math.radians(INITIAL_HEADING_DEG)

if AIRSPEED_FPS <= 0:
    raise ValueError("AIRSPEED_FPS must be positive.")
if ROPE_REST_LENGTH_FT <= 0:
    raise ValueError("ROPE_REST_LENGTH_FT must be positive.")
if ROPE_STIFFNESS_LBF_PER_FT <= 0:
    raise ValueError("ROPE_STIFFNESS_LBF_PER_FT must be positive.")
if ROPE_DAMPING_RATIO < 0:
    raise ValueError("ROPE_DAMPING_RATIO cannot be negative.")
if ROLL_TIME_S <= 0:
    raise ValueError("ROLL_TIME_S must be positive.")
if DT <= 0:
    raise ValueError("DT must be positive.")

rope_damping_lbf_s_per_ft = (
    2.0
    * ROPE_DAMPING_RATIO
    * math.sqrt(ROPE_STIFFNESS_LBF_PER_FT * sensor_mass)
)

rope_axial_natural_freq_rad_s = math.sqrt(
    ROPE_STIFFNESS_LBF_PER_FT / sensor_mass
)
rope_axial_period_s = 2.0 * math.pi / rope_axial_natural_freq_rad_s


# ============================================================
# COURSE / BANK COMMAND
# ============================================================
#
# Coordinated-turn heading rate used to prescribe the path:
#
#       psi_dot = g * tan(bank) / V
#
# The bank ramps in/out with a half-cosine so the prescribed aircraft
# lateral acceleration does not jump instantaneously.
# ============================================================

def ramp_bank_magnitude(tau):
    """0 -> bank_max over ROLL_TIME_S using a half-cosine."""
    return bank_max * 0.5 * (1.0 - math.cos(math.pi * tau / ROLL_TIME_S))


# Heading change accumulated during one roll ramp.
ramp_t = np.linspace(0.0, ROLL_TIME_S, 20001)
ramp_phi = bank_max * 0.5 * (1.0 - np.cos(math.pi * ramp_t / ROLL_TIME_S))
ramp_yaw_rate = G * np.tan(ramp_phi) / AIRSPEED_FPS
ramp_heading_change = np.trapezoid(ramp_yaw_rate, ramp_t)

max_yaw_rate = G * math.tan(bank_max) / AIRSPEED_FPS


def turn_hold_time(total_heading_change):
    """Full-bank hold time required to produce requested heading change."""
    remaining = total_heading_change - 2.0 * ramp_heading_change
    if remaining < 0:
        raise ValueError(
            "ROLL_TIME_S is too long for the requested turn angle. "
            "Reduce ROLL_TIME_S or increase BANK_ANGLE_DEG."
        )
    return remaining / max_yaw_rate


# right turn = negative bank / heading rate
# left turn  = positive bank / heading rate
course_spec = [
    ("Straight 1", "straight", 0.0, 0),
    ("180 right",  "turn", math.pi, -1),
    ("Straight 2", "straight", 0.0, 0),
    ("360 left",   "turn", 2.0 * math.pi, +1),
    ("Straight 3", "straight", 0.0, 0),
    ("180 right",  "turn", math.pi, -1),
    ("Straight 4", "straight", 0.0, 0),
]

schedule = []
t_cursor = 0.0

for name, kind, angle, sign in course_spec:
    if kind == "straight":
        hold = 0.0
        duration = STRAIGHT_LENGTH_FT / AIRSPEED_FPS
    else:
        hold = turn_hold_time(angle)
        duration = 2.0 * ROLL_TIME_S + hold

    schedule.append({
        "name": name,
        "kind": kind,
        "angle": angle,
        "sign": sign,
        "hold": hold,
        "t0": t_cursor,
        "t1": t_cursor + duration,
    })
    t_cursor += duration

TOTAL_TIME = t_cursor


def get_segment(t):
    if t <= 0.0:
        return schedule[0], 0.0

    for seg in schedule:
        if t < seg["t1"]:
            return seg, t - seg["t0"]

    seg = schedule[-1]
    return seg, seg["t1"] - seg["t0"]


def bank_command(t):
    """Nominal bank angle [rad] and current course-segment name."""
    seg, tau = get_segment(t)

    if seg["kind"] == "straight":
        return 0.0, seg["name"]

    sign = seg["sign"]

    if tau < ROLL_TIME_S:
        mag = ramp_bank_magnitude(tau)

    elif tau < ROLL_TIME_S + seg["hold"]:
        mag = bank_max

    else:
        tau_out = tau - (ROLL_TIME_S + seg["hold"])
        tau_out = min(max(tau_out, 0.0), ROLL_TIME_S)

        mag = bank_max * 0.5 * (
            1.0 + math.cos(math.pi * tau_out / ROLL_TIME_S)
        )

    return sign * mag, seg["name"]


# ============================================================
# AIRPLANE KINEMATICS
# ============================================================

def plane_kinematics(heading, bank):
    """Returns plane velocity, acceleration, and heading rate."""
    heading_rate = G * math.tan(bank) / AIRSPEED_FPS

    forward = np.array([
        math.cos(heading),
        math.sin(heading),
        0.0,
    ])

    left = np.array([
        -math.sin(heading),
        math.cos(heading),
        0.0,
    ])

    velocity = AIRSPEED_FPS * forward
    acceleration = AIRSPEED_FPS * heading_rate * left

    return velocity, acceleration, heading_rate


# ============================================================
# SENSOR AERODYNAMIC DRAG
# ============================================================

def sensor_drag_force(v_air_relative):
    """
    Quadratic drag on the sensor.

        Fd = -0.5 * rho * Cd * A * |V| * V
    """
    speed = np.linalg.norm(v_air_relative)

    if speed < 1e-12:
        return np.zeros(3)

    return (
        -0.5
        * AIR_DENSITY_SLUG_FT3
        * SENSOR_CD
        * sensor_area_ft2
        * speed
        * v_air_relative
    )


# ============================================================
# TENSION-ONLY ROPE MODEL
# ============================================================

def rope_force(r_plane, v_plane, r_sensor, v_sensor):
    """
    Returns:
        F_rope_on_sensor [lbf]
        tension [lbf]
        distance [ft]
        extension [ft]
        extension_rate [ft/s]
        is_taut [bool]

    Rope only carries tension. It is slack whenever its endpoint
    separation is <= the unstretched length.
    """
    rel = r_sensor - r_plane
    distance = np.linalg.norm(rel)

    if distance < 1e-12:
        return np.zeros(3), 0.0, distance, 0.0, 0.0, False

    u = rel / distance
    extension = distance - ROPE_REST_LENGTH_FT
    rel_velocity = v_sensor - v_plane
    extension_rate = np.dot(rel_velocity, u)

    if extension <= 0.0:
        return np.zeros(3), 0.0, distance, extension, extension_rate, False

    # Kelvin-Voigt axial spring-damper. Clamp at zero because a rope
    # cannot push if damping would otherwise make the axial force negative.
    tension = (
        ROPE_STIFFNESS_LBF_PER_FT * extension
        + rope_damping_lbf_s_per_ft * extension_rate
    )
    tension = max(0.0, tension)

    if tension <= 0.0:
        return np.zeros(3), 0.0, distance, extension, extension_rate, False

    # Rope pulls sensor toward airplane.
    F_rope_on_sensor = -tension * u

    return (
        F_rope_on_sensor,
        tension,
        distance,
        extension,
        extension_rate,
        True,
    )


# ============================================================
# STATE AND EQUATIONS OF MOTION
# ============================================================
#
# State:
#   [x_plane, y_plane, heading,
#    x_sensor, y_sensor, z_sensor,
#    vx_sensor, vy_sensor, vz_sensor]
#
# The plane trajectory is prescribed. The sensor is completely free
# whenever the rope is slack. When stretched past the rest length, the
# tension-only spring-damper pulls it toward the airplane.
# ============================================================

def rhs(t, state):
    x_plane, y_plane, heading = state[:3]
    r_sensor = state[3:6]
    v_sensor = state[6:9]

    bank, _ = bank_command(t)
    v_plane, a_plane, heading_rate = plane_kinematics(heading, bank)

    r_plane = np.array([x_plane, y_plane, 0.0])

    drag_force = sensor_drag_force(v_sensor - WIND_FPS)
    gravity_force = np.array([0.0, 0.0, -SENSOR_WEIGHT_LBF])

    F_rope, _, _, _, _, _ = rope_force(
        r_plane, v_plane, r_sensor, v_sensor
    )

    a_sensor = (gravity_force + drag_force + F_rope) / sensor_mass

    return np.concatenate((
        v_plane[:2],
        np.array([heading_rate]),
        v_sensor,
        a_sensor,
    ))


# ============================================================
# INITIAL CONDITION
# ============================================================
#
# Start in straight-flight equilibrium with the rope already taut.
# For steady straight flight:
#
#       F_gravity + F_drag + F_rope = 0
#
# so the rope points along the resultant gravity+drag direction and is
# stretched by T/k. This avoids an artificial "drop and catch" transient
# at t = 0.
# ============================================================

bank0, _ = bank_command(0.0)
v_plane0, _, _ = plane_kinematics(initial_heading, bank0)
r_plane0 = np.array([0.0, 0.0, 0.0])

initial_drag = sensor_drag_force(v_plane0 - WIND_FPS)
initial_external_force = (
    np.array([0.0, 0.0, -SENSOR_WEIGHT_LBF])
    + initial_drag
)

initial_tension = np.linalg.norm(initial_external_force)
u0 = initial_external_force / initial_tension
initial_extension = initial_tension / ROPE_STIFFNESS_LBF_PER_FT

r_sensor0 = (
    r_plane0
    + (ROPE_REST_LENGTH_FT + initial_extension) * u0
)
v_sensor0 = v_plane0.copy()

state0 = np.concatenate((
    np.array([0.0, 0.0, initial_heading]),
    r_sensor0,
    v_sensor0,
))


# ============================================================
# RK4 INTEGRATION
# ============================================================

n_steps = int(math.ceil(TOTAL_TIME / DT)) + 1
time = np.linspace(0.0, TOTAL_TIME, n_steps)

state = np.zeros((n_steps, len(state0)))
state[0] = state0

for i in range(n_steps - 1):
    t = time[i]
    h = time[i + 1] - time[i]
    y = state[i]

    k1 = rhs(t, y)
    k2 = rhs(t + 0.5*h, y + 0.5*h*k1)
    k3 = rhs(t + 0.5*h, y + 0.5*h*k2)
    k4 = rhs(t + h, y + h*k3)

    state[i + 1] = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0


# ============================================================
# POST-PROCESS LOADS
# ============================================================

plane_position = np.zeros((n_steps, 3))
sensor_position = np.zeros((n_steps, 3))

rope_tension = np.zeros(n_steps)
rope_distance = np.zeros(n_steps)
rope_extension = np.zeros(n_steps)
rope_extension_rate = np.zeros(n_steps)
rope_taut = np.zeros(n_steps, dtype=bool)

tow_backward = np.zeros(n_steps)       # + pulls airplane backward
tow_left = np.zeros(n_steps)           # + pulls airplane left
tow_down = np.zeros(n_steps)           # + pulls airplane downward

aft_offset = np.zeros(n_steps)
left_offset = np.zeros(n_steps)
below_offset = np.zeros(n_steps)
swing_from_vertical_deg = np.zeros(n_steps)

internal_payload_reference = np.zeros(n_steps)
required_aircraft_forward_force = np.zeros(n_steps)
required_aircraft_perp_force = np.zeros(n_steps)
required_aircraft_total_nonweight = np.zeros(n_steps)

bank_history_deg = np.zeros(n_steps)
segment_name = []

g_vector = np.array([0.0, 0.0, -G])
plane_weight_force = np.array([0.0, 0.0, -AIRCRAFT_WEIGHT_LBF])

for i, t in enumerate(time):
    x_plane, y_plane, heading = state[i, :3]
    r_sensor = state[i, 3:6]
    v_sensor = state[i, 6:9]

    bank, name = bank_command(t)
    segment_name.append(name)
    bank_history_deg[i] = math.degrees(bank)

    v_plane, a_plane, _ = plane_kinematics(heading, bank)
    r_plane = np.array([x_plane, y_plane, 0.0])

    forward = v_plane / np.linalg.norm(v_plane)
    left = np.array([-forward[1], forward[0], 0.0])

    plane_position[i] = r_plane
    sensor_position[i] = r_sensor

    (
        F_rope_on_sensor,
        T,
        distance,
        extension,
        extension_rate,
        is_taut,
    ) = rope_force(r_plane, v_plane, r_sensor, v_sensor)

    rope_tension[i] = T
    rope_distance[i] = distance
    rope_extension[i] = max(0.0, extension)
    rope_extension_rate[i] = extension_rate
    rope_taut[i] = is_taut

    # Equal-and-opposite force of rope ON airplane.
    F_rope_on_plane = -F_rope_on_sensor

    tow_backward[i] = -np.dot(F_rope_on_plane, forward)
    tow_left[i] = np.dot(F_rope_on_plane, left)
    tow_down[i] = -F_rope_on_plane[2]

    rel = r_sensor - r_plane
    aft_offset[i] = -np.dot(rel, forward)
    left_offset[i] = np.dot(rel, left)
    below_offset[i] = -rel[2]

    if distance > 1e-12:
        u = rel / distance
        swing_from_vertical_deg[i] = math.degrees(
            math.acos(np.clip(-u[2], -1.0, 1.0))
        )

    # Reference load for same sensor mass rigidly carried in airplane.
    F_internal_on_sensor = sensor_mass * (a_plane - g_vector)
    internal_payload_reference[i] = np.linalg.norm(F_internal_on_sensor)

    # Non-weight force the AIRPLANE must create to stay on prescribed path.
    # This excludes the airplane's own aerodynamic drag model because one
    # was not supplied. Forward component is therefore the extra thrust-like
    # force needed to overcome the tether contribution.
    F_required_aircraft = (
        aircraft_mass * a_plane
        - plane_weight_force
        - F_rope_on_plane
    )

    required_aircraft_forward_force[i] = np.dot(
        F_required_aircraft, forward
    )

    F_perp = (
        F_required_aircraft
        - required_aircraft_forward_force[i] * forward
    )
    required_aircraft_perp_force[i] = np.linalg.norm(F_perp)
    required_aircraft_total_nonweight[i] = np.linalg.norm(F_required_aircraft)


# ============================================================
# SUMMARY
# ============================================================

nominal_sensor_drag = np.linalg.norm(
    sensor_drag_force(v_plane0 - WIND_FPS)
)

max_internal_ref = np.max(internal_payload_reference)

ratio = np.divide(
    rope_tension,
    internal_payload_reference,
    out=np.zeros_like(rope_tension),
    where=internal_payload_reference > 1e-12,
)

slack_fraction = 1.0 - np.mean(rope_taut)
max_extension = np.max(rope_extension)
max_strain = max_extension / ROPE_REST_LENGTH_FT

# Simple stability check for time step relative to axial vibration period.
steps_per_axial_period = rope_axial_period_s / DT

print()
print("========== DBF TOWED SENSOR -- ROPE MODEL ==========")
print(f"Total simulated course time:        {TOTAL_TIME:8.3f} s")
print(f"Nominal sensor drag:                {nominal_sensor_drag:8.3f} lbf")
print()
print(f"Rope rest length:                   {ROPE_REST_LENGTH_FT:8.3f} ft")
print(f"Rope axial stiffness:               {ROPE_STIFFNESS_LBF_PER_FT:8.3f} lbf/ft")
print(f"Rope axial damping:                 {rope_damping_lbf_s_per_ft:8.3f} lbf*s/ft")
print(f"Rope axial natural period:          {rope_axial_period_s:8.4f} s")
print(f"Integration steps / axial period:   {steps_per_axial_period:8.1f}")
print()
print(f"Maximum rope tension:               {np.max(rope_tension):8.3f} lbf")
print(f"Maximum rope extension:             {max_extension:8.5f} ft")
print(f"Maximum rope axial strain:          {100.0*max_strain:8.3f} %")
print(f"Fraction of simulation rope slack:  {100.0*slack_fraction:8.3f} %")
print()
print(f"Maximum backward tow force:         {np.max(tow_backward):8.3f} lbf")
print(f"Maximum |side tow force|:           {np.max(np.abs(tow_left)):8.3f} lbf")
print(f"Maximum downward tow force:         {np.max(tow_down):8.3f} lbf")
print()
print(f"Maximum aft sensor offset:          {np.max(aft_offset):8.3f} ft")
print(f"Maximum |side sensor offset|:       {np.max(np.abs(left_offset)):8.3f} ft")
print(f"Maximum sensor swing from vertical: {np.max(swing_from_vertical_deg):8.3f} deg")
print()
print(f"Max internal-payload reference load:{max_internal_ref:8.3f} lbf")
print(f"Max rope tension / internal ref:    {np.max(ratio):8.3f} x")
print()
print(
    "Max required aircraft forward force\n"
    "(tether contribution + prescribed acceleration,\n"
    f" excluding airplane aerodynamic drag): "
    f"{np.max(required_aircraft_forward_force):.3f} lbf"
)
print(
    f"Max required aircraft perpendicular force: "
    f"{np.max(required_aircraft_perp_force):.3f} lbf"
)

if steps_per_axial_period < 30.0:
    print()
    print("WARNING:")
    print(
        "The integration time step may be too large for the selected "
        "rope stiffness. Reduce DT until there are preferably at least "
        "~30-50 steps per rope axial vibration period."
    )

print()
print("NOTE:")
print(
    "Peak catch tension is sensitive to ROPE_STIFFNESS_LBF_PER_FT and "
    "ROPE_DAMPING_RATIO. Use measured or manufacturer-derived rope "
    "properties before treating the peak loads as design values."
)


# ============================================================
# SAVE CSV
# ============================================================

if SAVE_CSV:
    with open(CSV_NAME, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "time_s",
            "segment",
            "plane_x_ft",
            "plane_y_ft",
            "sensor_x_ft",
            "sensor_y_ft",
            "sensor_z_ft",
            "bank_deg",
            "rope_tension_lbf",
            "rope_distance_ft",
            "rope_extension_ft",
            "rope_extension_rate_fps",
            "rope_taut",
            "tow_backward_lbf",
            "tow_left_lbf",
            "tow_down_lbf",
            "aft_offset_ft",
            "left_offset_ft",
            "below_offset_ft",
            "swing_from_vertical_deg",
            "internal_payload_reference_lbf",
            "required_aircraft_forward_force_lbf",
            "required_aircraft_perp_force_lbf",
            "required_aircraft_total_nonweight_lbf",
        ])

        for i in range(n_steps):
            writer.writerow([
                time[i],
                segment_name[i],
                plane_position[i, 0],
                plane_position[i, 1],
                sensor_position[i, 0],
                sensor_position[i, 1],
                sensor_position[i, 2],
                bank_history_deg[i],
                rope_tension[i],
                rope_distance[i],
                rope_extension[i],
                rope_extension_rate[i],
                int(rope_taut[i]),
                tow_backward[i],
                tow_left[i],
                tow_down[i],
                aft_offset[i],
                left_offset[i],
                below_offset[i],
                swing_from_vertical_deg[i],
                internal_payload_reference[i],
                required_aircraft_forward_force[i],
                required_aircraft_perp_force[i],
                required_aircraft_total_nonweight[i],
            ])

    print()
    print(f"Saved: {CSV_NAME}")


# ============================================================
# PLOTS
# ============================================================

segment_boundaries = [seg["t1"] for seg in schedule[:-1]]


# 1) Top-down trajectory
plt.figure()
plt.plot(plane_position[:, 0], plane_position[:, 1], label="Airplane")
plt.plot(sensor_position[:, 0], sensor_position[:, 1], label="Sensor")
plt.xlabel("x [ft]")
plt.ylabel("y [ft]")
plt.title("Top-Down DBF Course")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()


# 2) Sensor location relative to airplane
plt.figure()
plt.plot(time, aft_offset, label="Aft (+)")
plt.plot(time, left_offset, label="Left (+)")
plt.plot(time, below_offset, label="Below (+)")
for tb in segment_boundaries:
    plt.axvline(tb, linewidth=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Relative offset [ft]")
plt.title("Sensor Position Relative to Airplane")
plt.grid(True)
plt.legend()
plt.tight_layout()


# 3) Rope tension vs same mass carried internally
plt.figure()
plt.plot(time, rope_tension, label="Rope tension")
plt.plot(time, internal_payload_reference, label="Same mass carried internally")
for tb in segment_boundaries:
    plt.axvline(tb, linewidth=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Force [lbf]")
plt.title("Rope Tension vs Internal-Payload Reference")
plt.grid(True)
plt.legend()
plt.tight_layout()


# 4) Tow force components on airplane
plt.figure()
plt.plot(time, tow_backward, label="Backward (+)")
plt.plot(time, tow_left, label="Left (+)")
plt.plot(time, tow_down, label="Downward (+)")
for tb in segment_boundaries:
    plt.axvline(tb, linewidth=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Force on airplane [lbf]")
plt.title("Tow-Load Components on Airplane")
plt.grid(True)
plt.legend()
plt.tight_layout()


# 5) Rope extension and taut/slack state
plt.figure()
plt.plot(time, 12.0 * rope_extension, label="Rope extension")
for tb in segment_boundaries:
    plt.axvline(tb, linewidth=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Extension [in]")
plt.title("Rope Stretch (zero means slack or just taut)")
plt.grid(True)
plt.tight_layout()


# 6) Sensor swing angle
plt.figure()
plt.plot(time, swing_from_vertical_deg)
for tb in segment_boundaries:
    plt.axvline(tb, linewidth=0.7)
plt.xlabel("Time [s]")
plt.ylabel("Swing from downward vertical [deg]")
plt.title("Sensor Pendulum Motion")
plt.grid(True)
plt.tight_layout()

plt.show()
