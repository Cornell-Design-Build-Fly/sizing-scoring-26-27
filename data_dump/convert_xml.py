import json
import aerosandbox as asb

RESULT_FILE = r"C:\Users\charl\Documents\sizing-scoring-26-27\data_dump\opt_topline\run_20260908_234056\best_design_report.json"
OUTPUT_FILE = r"C:\Users\charl\Documents\sizing-scoring-26-27\data_dump\opt_topline\run_20260908_234056\BEST_aircraft.xml"
WING_AIRFOIL = "naca2415"
TAIL_AIRFOIL = "naca0012"

with open(RESULT_FILE, "r") as f:
    data = json.load(f)

# Use the resolved geometry, not the raw optimizer variables
d = data["resolved_vector"]

# M2 tensor is about the CG, in the same geometry axes as the exported plane.
m2 = data["mechanical"]["missions"]["M2"]
cg = m2["cg_m"]
inertia = m2["inertia_tensor_kg_m2"]
mass_props = asb.MassProperties(
    mass=m2["total_mass_kg"],
    x_cg=cg[0], y_cg=cg[1], z_cg=cg[2],
    Ixx=inertia[0][0], Iyy=inertia[1][1], Izz=inertia[2][2],
    Ixy=inertia[0][1], Ixz=inertia[0][2], Iyz=inertia[1][2],
)

# Main wing
wing = asb.Wing(
    name="Main Wing",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=[0, 0, 0],
            chord=d["wing_chord"],
            airfoil=asb.Airfoil(WING_AIRFOIL),
        ),
        asb.WingXSec(
            xyz_le=[0, d["wing_span"] / 2, 0],
            chord=d["wing_chord"],
            airfoil=asb.Airfoil(WING_AIRFOIL),
        ),
    ],
)

# Horizontal stabilizer
hstab = asb.Wing(
    name="Horizontal Stabilizer",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=[d["tail_arm"], 0, 0],
            chord=d["hstab_chord"],
            airfoil=asb.Airfoil(TAIL_AIRFOIL),
        ),
        asb.WingXSec(
            xyz_le=[d["tail_arm"], d["hstab_span"] / 2, 0],
            chord=d["hstab_chord"],
            airfoil=asb.Airfoil(TAIL_AIRFOIL),
        ),
    ],
)

# Vertical stabilizer
vstab = asb.Wing(
    name="Vertical Stabilizer",
    symmetric=False,
    xsecs=[
        asb.WingXSec(
            xyz_le=[d["tail_arm"], 0, 0],
            chord=d["vstab_chord"],
            airfoil=asb.Airfoil(TAIL_AIRFOIL),
        ),
        asb.WingXSec(
            xyz_le=[d["tail_arm"], 0, d["vstab_span"]],
            chord=d["vstab_chord"],
            airfoil=asb.Airfoil(TAIL_AIRFOIL),
        ),
    ],
)

# Build airplane
airplane = asb.Airplane(
    name="DBF Optimized Aircraft M2",
    wings=[wing, hstab, vstab],
)

# Export to XFLR5 XML
airplane.export_XFLR5_xml(
    filename=OUTPUT_FILE,
    mass_props=mass_props,
    mainwing=wing,
    elevator=hstab,
    fin=vstab,
    include_fuselages=False,
)

print(f"Exported XFLR5 model to: {OUTPUT_FILE}")
