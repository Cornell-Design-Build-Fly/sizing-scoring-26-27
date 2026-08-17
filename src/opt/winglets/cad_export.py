from __future__ import annotations

import json
from pathlib import Path

import aerosandbox as asb
import aerosandbox.numpy as np
import trimesh

from src.opt.winglets.geometry import WingletGeometry, make_winglet_airplane
from src.vectors import DesignVector


SINGLE_WINGLET_STEP_FILENAME = "optimized_single_winglet.step"
SINGLE_WINGLET_STL_FILENAME = "optimized_single_winglet_mm.stl"
SINGLE_WINGLET_MESH_REPORT_FILENAME = "optimized_single_winglet_mesh_report.json"
MAIN_WING_STEP_FILENAME = "optimized_main_wing_with_winglets.step"
FULL_AIRPLANE_STEP_FILENAME = "optimized_winglet_airplane.step"


def make_single_winglet_airplane(
    design_vector: DesignVector,
    winglet: WingletGeometry,
    *,
    winglet_airfoil: str,
) -> asb.Airplane:
    airplane = make_winglet_airplane(
        design_vector,
        winglet,
        name="Optimized Single Winglet CAD Export",
        winglet_airfoil=winglet_airfoil,
    )
    main_wing = next(
        wing
        for wing in airplane.wings
        if wing.name == "Main Wing with Remorphed Tips"
    )
    winglet_root_y = design_vector.wing_span / 2.0 - winglet.blend_length_m
    winglet_xsecs = [
        xsec
        for xsec in main_wing.xsecs
        if float(xsec.xyz_le[1]) >= winglet_root_y - 1e-9
    ]
    root_xyz_le = np.array(winglet_xsecs[0].xyz_le)
    local_xsecs = [
        xsec.translate(-root_xyz_le)
        for xsec in winglet_xsecs
    ]
    local_winglet = asb.Wing(
        name="Single Right Winglet",
        symmetric=False,
        xsecs=local_xsecs,
    )
    return asb.Airplane(
        name="Optimized Single Right Winglet",
        wings=[local_winglet],
        s_ref=max(local_winglet.area(), 1e-6),
        c_ref=design_vector.wing_chord,
        b_ref=max(local_winglet.span(), 1e-6),
    )


def make_cad_export_airplane(
    design_vector: DesignVector,
    winglet: WingletGeometry,
    *,
    winglet_airfoil: str,
    single_winglet: bool = True,
    full_airplane: bool = False,
) -> asb.Airplane:
    if single_winglet:
        return make_single_winglet_airplane(
            design_vector,
            winglet,
            winglet_airfoil=winglet_airfoil,
        )

    airplane = make_winglet_airplane(
        design_vector,
        winglet,
        name="Optimized Winglet CAD Export",
        winglet_airfoil=winglet_airfoil,
    )
    if full_airplane:
        return airplane

    main_wing = next(
        wing
        for wing in airplane.wings
        if wing.name == "Main Wing with Remorphed Tips"
    )
    return asb.Airplane(
        name="Optimized Main Wing with Winglets",
        wings=[main_wing],
        s_ref=airplane.s_ref,
        c_ref=airplane.c_ref,
        b_ref=airplane.b_ref,
    )


def export_step_geometry(
    design_vector: DesignVector,
    winglet: WingletGeometry,
    output_dir: Path,
    *,
    winglet_airfoil: str,
    single_winglet: bool = True,
    full_airplane: bool = False,
    minimum_airfoil_te_thickness: float = 0.001,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if single_winglet:
        filename = SINGLE_WINGLET_STEP_FILENAME
    elif full_airplane:
        filename = FULL_AIRPLANE_STEP_FILENAME
    else:
        filename = MAIN_WING_STEP_FILENAME
    path = output_dir / filename
    cad_airplane = make_cad_export_airplane(
        design_vector,
        winglet,
        winglet_airfoil=winglet_airfoil,
        single_winglet=single_winglet,
        full_airplane=full_airplane,
    )
    cad_airplane.export_cadquery_geometry(
        str(path),
        minimum_airfoil_TE_thickness=minimum_airfoil_te_thickness,
    )
    return path


def export_print_mesh_geometry(
    design_vector: DesignVector,
    winglet: WingletGeometry,
    output_dir: Path,
    *,
    winglet_airfoil: str,
    chordwise_resolution: int = 96,
    minimum_airfoil_te_thickness: float = 0.003,
    scale_to_mm: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cad_airplane = make_single_winglet_airplane(
        design_vector,
        winglet,
        winglet_airfoil=winglet_airfoil,
    )
    wing = cad_airplane.wings[0]
    print_xsecs = []
    for xsec in wing.xsecs:
        airfoil = xsec.airfoil
        if airfoil.TE_thickness() < minimum_airfoil_te_thickness:
            airfoil = airfoil.set_TE_thickness(
                thickness=minimum_airfoil_te_thickness
            )
        print_xsecs.append(
            asb.WingXSec(
                xyz_le=xsec.xyz_le,
                chord=xsec.chord,
                twist=xsec.twist,
                airfoil=airfoil,
                control_surfaces=xsec.control_surfaces,
                analysis_specific_options=xsec.analysis_specific_options,
            )
        )
    wing = asb.Wing(
        name=wing.name,
        symmetric=False,
        xsecs=print_xsecs,
    )
    points, faces = wing.mesh_body(
        method="tri",
        chordwise_resolution=chordwise_resolution,
        mesh_symmetric=False,
    )
    vertices = np.array(points, dtype=float)
    if scale_to_mm:
        vertices = vertices * 1000.0

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.array(faces, dtype=int),
        process=True,
    )
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    try:
        mesh.fix_normals()
    except ModuleNotFoundError:
        pass

    path = output_dir / SINGLE_WINGLET_STL_FILENAME
    mesh.export(path)

    report = {
        "path": str(path),
        "units": "millimeters" if scale_to_mm else "meters",
        "minimum_airfoil_te_thickness_chord_fraction": minimum_airfoil_te_thickness,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "bounds": mesh.bounds.tolist(),
    }
    (output_dir / SINGLE_WINGLET_MESH_REPORT_FILENAME).write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
