"""Size the propulsion system for an aircraft supplied as XFLR5 XML."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from io import StringIO
import json
import math
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import differential_evolution

from src.aero.aero_score import _compute_lap_time
from src.aero.cruise_analysis_continuous import cruise_analysis_continuous
from src.prop.catalog_selection import resolve_catalog_propellers
from src.prop.continuous_prop_database import (
    ContinuousPropDatabase,
    load_default_continuous_prop_database,
)
from src.prop.prop_helper_functions import make_battery_from_design, make_motor_from_design
from src.prop.mission_performance import evaluate_mission_propulsion
from src.vectors import DesignVector, ParameterVector


SG6042_NAME = "sg6042"
DEFAULT_OUTPUT = Path("data_dump") / "propulsion_sizing.json"
BAD_OBJECTIVE = 1.0e6
DEFAULT_MISSION3_MASS_REDUCTION_KG = 5.9


@dataclass(frozen=True)
class SurfaceSection:
    span_station_m: float
    chord_m: float
    x_offset_m: float
    dihedral_deg: float
    twist_deg: float
    left_airfoil: str
    right_airfoil: str


@dataclass(frozen=True)
class SurfaceGeometry:
    name: str
    surface_type: str
    span_m: float
    mean_chord_m: float
    area_m2: float
    leading_edge_x_m: float
    symmetric: bool
    position_m: tuple[float, float, float]
    tilt_deg: float
    sections: tuple[SurfaceSection, ...]


@dataclass(frozen=True)
class PointMass:
    tag: str
    mass_kg: float
    coordinates_m: tuple[float, float, float]


@dataclass(frozen=True)
class FixedMassProperties:
    total_mass_kg: float
    cg_m: tuple[float, float, float]
    inertia_tensor_kg_m2: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    point_masses: tuple[PointMass, ...]


@dataclass(frozen=True)
class ImportedAircraftGeometry:
    name: str
    main_wing: SurfaceGeometry
    horizontal_tail: SurfaceGeometry
    vertical_tail: SurfaceGeometry
    mass_properties: FixedMassProperties
    length_unit_to_m: float
    mass_unit_to_kg: float
    warnings: tuple[str, ...]

    def to_design_vector(self) -> DesignVector:
        """Build the immutable aerodynamic view used by propulsion sizing."""

        source = DesignVector()
        tail_arm_m = self.horizontal_tail.leading_edge_x_m - self.main_wing.leading_edge_x_m
        if tail_arm_m <= 0.0:
            raise ValueError("The horizontal-tail leading edge must be aft of the main wing.")
        return replace(
            source,
            wing_span=self.main_wing.span_m,
            wing_chord=self.main_wing.mean_chord_m,
            tail_arm=tail_arm_m,
            hstab_span_override_m=self.horizontal_tail.span_m,
            hstab_chord_override_m=self.horizontal_tail.mean_chord_m,
            vstab_span_override_m=self.vertical_tail.span_m,
            vstab_chord_override_m=self.vertical_tail.mean_chord_m,
            mission3_sensor_drag_enabled=False,
            fuselage_drag_enabled=False,
            # Propulsion sizing always uses the supplied SG6042 coordinates,
            # regardless of a foil label embedded in the XML.
            wing_airfoil=SG6042_NAME,
        )


def _local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    target = name.casefold()
    return [child for child in element if _local_name(child).casefold() == target]


def _descendants(element: ET.Element, name: str) -> list[ET.Element]:
    target = name.casefold()
    return [child for child in element.iter() if _local_name(child).casefold() == target]


def _text(element: ET.Element, name: str, default: str | None = None) -> str:
    matches = _children(element, name)
    if not matches or matches[0].text is None:
        if default is not None:
            return default
        raise ValueError(f"Missing <{name}> inside <{_local_name(element)}>.")
    return matches[0].text.strip()


def _float(element: ET.Element, name: str, default: float | None = None) -> float:
    raw = _text(element, name, None if default is None else str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"<{name}> must contain a number; got {raw!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"<{name}> must be finite.")
    return value


def _truthy(raw: str) -> bool:
    return raw.strip().casefold() in {"true", "1", "yes"}


def _coordinates(raw: str, *, label: str, scale: float) -> tuple[float, float, float]:
    try:
        coordinates = [float(value.strip()) for value in raw.split(",")]
    except ValueError as exc:
        raise ValueError(f"Invalid {label} value {raw!r}.") from exc
    if len(coordinates) != 3 or not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{label} must contain three finite coordinates; got {raw!r}.")
    return tuple(float(value * scale) for value in coordinates)


def _surface_geometry(wing: ET.Element, scale: float) -> tuple[SurfaceGeometry, list[str]]:
    name = _text(wing, "Name", "Unnamed surface")
    surface_type = _text(wing, "Type", "UNKNOWN").upper()
    symmetric = _truthy(_text(wing, "Symetric", "false"))
    sections_parent = _children(wing, "Sections")
    sections = _children(sections_parent[0], "Section") if sections_parent else []
    if len(sections) < 2:
        raise ValueError(f"Wing {name!r} needs at least two XML sections.")

    parsed_sections = tuple(
        sorted(
            (
                SurfaceSection(
                    span_station_m=_float(section, "y_position") * scale,
                    chord_m=_float(section, "Chord") * scale,
                    x_offset_m=_float(section, "xOffset", 0.0) * scale,
                    dihedral_deg=_float(section, "Dihedral", 0.0),
                    twist_deg=_float(section, "Twist", 0.0),
                    left_airfoil=_text(section, "Left_Side_FoilName", ""),
                    right_airfoil=_text(section, "Right_Side_FoilName", ""),
                )
                for section in sections
            ),
            key=lambda section: section.span_station_m,
        )
    )
    station_positions = np.asarray(
        [section.span_station_m for section in parsed_sections], dtype=float
    )
    chords = np.asarray([section.chord_m for section in parsed_sections], dtype=float)
    if np.any(chords <= 0.0) or station_positions[-1] <= station_positions[0]:
        raise ValueError(f"Wing {name!r} has nonpositive chord or span.")

    half_or_full_span = float(station_positions[-1] - station_positions[0])
    multiplier = 2.0 if symmetric and surface_type != "FIN" else 1.0
    span_m = multiplier * half_or_full_span
    area_m2 = multiplier * float(np.trapezoid(chords, station_positions))
    mean_chord_m = area_m2 / span_m
    position_m = _coordinates(
        _text(wing, "Position", "0,0,0"), label="wing <Position>", scale=scale
    )
    leading_edge_x_m = position_m[0] + parsed_sections[0].x_offset_m

    warnings: list[str] = []
    offsets = np.asarray([section.x_offset_m for section in parsed_sections], dtype=float)
    twists = np.asarray([section.twist_deg for section in parsed_sections], dtype=float)
    if not np.allclose(chords, chords[0]):
        warnings.append(f"{name}: taper was converted to an equal-area rectangular surface.")
    if not np.allclose(offsets, offsets[0]):
        warnings.append(f"{name}: sweep/section offsets are not represented by the sizing model.")
    if not np.allclose(twists, 0.0):
        warnings.append(f"{name}: section twist is not represented by the sizing model.")

    return (
        SurfaceGeometry(
            name=name,
            surface_type=surface_type,
            span_m=span_m,
            mean_chord_m=mean_chord_m,
            area_m2=area_m2,
            leading_edge_x_m=leading_edge_x_m,
            symmetric=symmetric,
            position_m=position_m,
            tilt_deg=_float(wing, "Tilt_angle", 0.0),
            sections=parsed_sections,
        ),
        warnings,
    )


def _mass_properties(
    plane: ET.Element,
    *,
    length_scale: float,
    mass_scale: float,
) -> FixedMassProperties:
    inertia_elements = _children(plane, "Inertia")
    if not inertia_elements:
        raise ValueError("The XML must contain a <Plane><Inertia> block.")
    point_mass_elements = _children(inertia_elements[0], "Point_Mass")
    if not point_mass_elements:
        raise ValueError(
            "The XML must contain at least one <Point_Mass>; its mass is treated "
            "as the complete, fixed aircraft mass."
        )
    point_masses = tuple(
        PointMass(
            tag=_text(element, "Tag", f"pm{index}"),
            mass_kg=_float(element, "Mass") * mass_scale,
            coordinates_m=_coordinates(
                _text(element, "coordinates"),
                label="point-mass <coordinates>",
                scale=length_scale,
            ),
        )
        for index, element in enumerate(point_mass_elements)
    )
    masses = np.asarray([point.mass_kg for point in point_masses], dtype=float)
    coordinates = np.asarray(
        [point.coordinates_m for point in point_masses], dtype=float
    )
    if np.any(masses <= 0.0):
        raise ValueError("Every XML point mass must be positive.")
    total_mass = float(np.sum(masses))
    cg = np.sum(masses[:, None] * coordinates, axis=0) / total_mass
    tensor = np.zeros((3, 3), dtype=float)
    identity = np.eye(3)
    for mass, coordinate in zip(masses, coordinates):
        displacement = coordinate - cg
        tensor += mass * (
            float(np.dot(displacement, displacement)) * identity
            - np.outer(displacement, displacement)
        )
    return FixedMassProperties(
        total_mass_kg=total_mass,
        cg_m=tuple(float(value) for value in cg),
        inertia_tensor_kg_m2=tuple(
            tuple(float(value) for value in row) for row in tensor
        ),
        point_masses=point_masses,
    )


def _pick_surface(
    surfaces: Iterable[SurfaceGeometry],
    types: set[str],
    name_words: tuple[str, ...],
) -> SurfaceGeometry:
    choices = list(surfaces)
    for surface in choices:
        if surface.surface_type in types:
            return surface
    for surface in choices:
        folded_name = surface.name.casefold()
        if any(word in folded_name for word in name_words):
            return surface
    raise ValueError(f"The XML does not contain a required {'/'.join(sorted(types))} surface.")


def import_xflr5_geometry(xml_path: str | Path) -> ImportedAircraftGeometry:
    """Read XFLR5 aircraft XML and return geometry in SI units."""

    path = Path(xml_path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse XML file {path}: {exc}.") from exc

    units = _descendants(root, "Units")
    scale = _float(units[0], "length_unit_to_meter", 1.0) if units else 1.0
    mass_scale = _float(units[0], "mass_unit_to_kg", 1.0) if units else 1.0
    if scale <= 0.0 or mass_scale <= 0.0:
        raise ValueError("XML length and mass unit scales must be positive.")
    planes = _descendants(root, "Plane")
    if not planes:
        raise ValueError("Expected an XFLR5 <Plane> element.")
    plane = planes[0]
    plane_name = _text(plane, "Name", path.stem)
    if _truthy(_text(plane, "has_body", "false")):
        raise ValueError(
            "This propulsion-only importer cannot yet reproduce XFLR5 body "
            "geometry; export the plane with <has_body>false</has_body>."
        )
    mass_properties = _mass_properties(
        plane, length_scale=scale, mass_scale=mass_scale
    )

    parsed = [_surface_geometry(element, scale) for element in _children(plane, "wing")]
    surfaces = [item[0] for item in parsed]
    warnings = [warning for _, item_warnings in parsed for warning in item_warnings]
    main_wing = _pick_surface(surfaces, {"MAINWING"}, ("main",))
    horizontal_tail = _pick_surface(
        (surface for surface in surfaces if surface is not main_wing),
        {"ELEVATOR", "SECONDWING"},
        ("horizontal", "stabilizer", "elevator"),
    )
    vertical_tail = _pick_surface(
        (surface for surface in surfaces if surface not in {main_wing, horizontal_tail}),
        {"FIN"},
        ("vertical", "fin"),
    )
    if not math.isclose(
        horizontal_tail.leading_edge_x_m,
        vertical_tail.leading_edge_x_m,
        rel_tol=0.0,
        abs_tol=1.0e-3,
    ):
        warnings.append(
            "Horizontal and vertical tails have different leading-edge stations; "
            "the horizontal-tail station is used for both."
        )

    return ImportedAircraftGeometry(
        name=plane_name,
        main_wing=main_wing,
        horizontal_tail=horizontal_tail,
        vertical_tail=vertical_tail,
        mass_properties=mass_properties,
        length_unit_to_m=scale,
        mass_unit_to_kg=mass_scale,
        warnings=tuple(warnings),
    )


def _eligible_propellers(database: ContinuousPropDatabase):
    bounds = dict(zip(DesignVector.opt_names(), DesignVector.bounds()))
    return tuple(
        surface
        for surface in database.catalog.surfaces
        if bounds["prop_diameter_in"][0] <= surface.diameter_in <= bounds["prop_diameter_in"][1]
        and bounds["prop_pitch_in"][0] <= surface.pitch_in <= bounds["prop_pitch_in"][1]
        and 0.4 <= surface.pitch_in / surface.diameter_in <= 0.8
    )


def _candidate_design(base: DesignVector, x: np.ndarray, propellers) -> DesignVector:
    mission12 = propellers[int(round(float(x[1])))]
    mission3 = propellers[int(round(float(x[2])))]
    return replace(
        base,
        batt_capacity=float(x[0]),
        prop_diameter_in=float(mission12.diameter_in),
        prop_pitch_in=float(mission12.pitch_in),
        mission3_prop_diameter_in=float(mission3.diameter_in),
        mission3_prop_pitch_in=float(mission3.pitch_in),
        motor_kv=float(x[3]),
        motor_max_power=float(x[4]),
    )


def _evaluate_fixed_aircraft(
    design: DesignVector,
    geometry: ImportedAircraftGeometry,
    parameters: ParameterVector,
    database: ContinuousPropDatabase,
    mission3_mass_reduction_kg: float,
) -> dict:
    """Evaluate propulsion while holding XML mass and geometry unchanged."""

    xml_mass = geometry.mass_properties.total_mass_kg
    mission3_mass = xml_mass - mission3_mass_reduction_kg
    if mission3_mass <= 0.0:
        raise ValueError(
            "Mission-3 mass reduction must be smaller than the XML mass."
        )
    xml_cg = np.asarray(geometry.mass_properties.cg_m, dtype=float)
    # The legacy aerodynamic equations place the main-wing leading edge at
    # x=0, while XFLR5 stores point masses in the plane's global coordinates.
    analysis_cg = xml_cg.copy()
    analysis_cg[0] -= geometry.main_wing.leading_edge_x_m

    mission_results: dict[str, dict] = {}
    missing_missions = 0
    for mission in (1, 2, 3):
        mass = mission3_mass if mission == 3 else xml_mass
        condition = cruise_analysis_continuous(
            design,
            parameters,
            tuple(float(value) for value in analysis_cg),
            mass,
            mission,
            database,
        )
        if not condition.converged or condition.stall_speed is None:
            missing_missions += 1
            continue
        cruise_speed = float(condition.operating_point.velocity)
        stall_speed = float(condition.stall_speed)
        lap_time = float(
            _compute_lap_time(cruise_speed, stall_speed, parameters)
        )
        performance = evaluate_mission_propulsion(
            design,
            parameters,
            mission=mission,
            mass_kg=mass,
            supported_mass_kg=mass,
            cruise_speed_mps=cruise_speed,
            stall_speed_mps=stall_speed,
            lap_time_s=lap_time,
            prop_database=database,
        )
        mission_results[f"M{mission}"] = performance.to_dict()

    penalties = sum(
        float(result["penalty"]) for result in mission_results.values()
    )
    all_feasible = (
        len(mission_results) == 3
        and all(bool(result["feasible"]) for result in mission_results.values())
    )
    # Once all hard requirements pass, shorter propulsion-limited lap time is
    # the only merit. No hardware-mass reward or payload score is introduced.
    speed_merit = sum(
        100.0 / max(float(result["modeled_lap_time_s"]), 1.0e-9)
        for result in mission_results.values()
    )
    merit = speed_merit - 100.0 * penalties - 10_000.0 * missing_missions
    return {
        "merit": float(merit),
        "all_missions_feasible": all_feasible,
        "mission_performance": mission_results,
        "mission_masses_kg": {
            "M1": xml_mass,
            "M2": xml_mass,
            "M3": mission3_mass,
        },
        "mission3_derived_mass_properties": {
            "total_mass_kg": mission3_mass,
            "point_mass_scale_from_xml": mission3_mass / xml_mass,
            "cg_m": geometry.mass_properties.cg_m,
            "inertia_tensor_kg_m2": tuple(
                tuple(value * mission3_mass / xml_mass for value in row)
                for row in geometry.mass_properties.inertia_tensor_kg_m2
            ),
        },
    }


def size_propulsion(
    geometry: ImportedAircraftGeometry,
    *,
    generations: int = 3,
    popsize: int = 1,
    seed: int = 20260911,
    mission3_mass_reduction_kg: float = DEFAULT_MISSION3_MASS_REDUCTION_KG,
) -> dict:
    """Optimize battery, motor, and mission propellers for imported geometry."""

    if generations < 1 or popsize < 1:
        raise ValueError("generations and popsize must be positive integers.")
    if (
        not math.isfinite(mission3_mass_reduction_kg)
        or mission3_mass_reduction_kg < 0.0
        or mission3_mass_reduction_kg >= geometry.mass_properties.total_mass_kg
    ):
        raise ValueError(
            "mission3_mass_reduction_kg must be finite, nonnegative, and "
            "smaller than the XML mass."
        )
    parameters = ParameterVector()
    imported_design = geometry.to_design_vector()
    with redirect_stdout(StringIO()):
        database = load_default_continuous_prop_database()
    propellers = _eligible_propellers(database)
    if not propellers:
        raise RuntimeError("No two-blade catalog propellers satisfy the sizing bounds.")

    design_bounds = dict(zip(DesignVector.opt_names(), DesignVector.bounds()))
    bounds = (
        design_bounds["batt_capacity"],
        (0.0, float(len(propellers) - 1)),
        (0.0, float(len(propellers) - 1)),
        design_bounds["motor_kv"],
        design_bounds["motor_max_power"],
    )

    def objective(x: np.ndarray) -> float:
        try:
            design = _candidate_design(imported_design, x, propellers)
            evaluation = _evaluate_fixed_aircraft(
                design,
                geometry,
                parameters,
                database,
                mission3_mass_reduction_kg,
            )
        except Exception:
            return BAD_OBJECTIVE
        merit = float(evaluation["merit"])
        return -merit if math.isfinite(merit) else BAD_OBJECTIVE

    result = differential_evolution(
        objective,
        bounds,
        integrality=(False, True, True, False, False),
        maxiter=generations,
        popsize=popsize,
        seed=seed,
        init="sobol",
        polish=False,
        workers=1,
        updating="immediate",
        tol=1.0e-4,
    )
    selected = resolve_catalog_propellers(
        _candidate_design(imported_design, result.x, propellers), database
    )
    evaluation = _evaluate_fixed_aircraft(
        selected,
        geometry,
        parameters,
        database,
        mission3_mass_reduction_kg,
    )
    battery = make_battery_from_design(selected, parameters)
    motor = make_motor_from_design(selected, parameters)
    mission_performance = evaluation["mission_performance"]

    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_aircraft": {
            "name": geometry.name,
            "geometry": asdict(geometry),
            "modeling_warnings": list(geometry.warnings),
        },
        "assumptions": {
            "wing_airfoil": "SG6042 (repository coordinate file)",
            "tail_airfoil": "NACA 0012",
            "xml_format": "XFLR5 aircraft XML",
            "fixed_mass": (
                "The XML point masses define M1/M2. M3 removes the configured "
                "mass while preserving the XML CG and normalized mass "
                "distribution. No propulsion hardware mass is added."
            ),
            "fixed_geometry": (
                "All XML surface dimensions remain fixed during the search."
            ),
            "propellers": "Real two-blade catalog entries; M1/M2 share one and M3 may use another.",
        },
        "selected_propulsion_system": {
            "motor": {
                "kv_rpm_per_v": selected.motor_kv,
                "rated_power_w": selected.motor_max_power,
                "modeled_max_current_a": motor.max_current,
            },
            "esc": {
                "minimum_continuous_current_a": min(
                    float(motor.max_current), float(battery.get_max_current())
                ),
                "note": "Choose the next available ESC rating above this modeled current limit.",
            },
            "battery": {
                "series_cells": battery.cells,
                "nominal_voltage_v": battery.vnom,
                "capacity_ah": battery.capacity,
                "nominal_energy_wh": selected.batt_energy,
                "usable_energy_wh": selected.batt_energy * battery.useable_fraction,
                "c_rating": battery.Crat,
                "maximum_current_a": battery.get_max_current(),
            },
            "mission_1_and_2_propeller": mission_performance.get("M1", {}).get(
                "propeller_key"
            ),
            "mission_3_propeller": mission_performance.get("M3", {}).get(
                "propeller_key"
            ),
        },
        "result": {
            "propulsion_feasible": bool(evaluation["all_missions_feasible"]),
            "propulsion_sizing_merit": float(evaluation["merit"]),
            "fixed_aircraft_mass_kg": geometry.mass_properties.total_mass_kg,
            "mission_masses_kg": evaluation["mission_masses_kg"],
            "mission3_mass_reduction_kg": mission3_mass_reduction_kg,
            "mission3_derived_mass_properties": evaluation[
                "mission3_derived_mass_properties"
            ],
            "mass_added_by_search_kg": 0.0,
            "limiting_constraint_by_mission": {
                mission: values["limiting_constraint"]
                for mission, values in mission_performance.items()
            },
        },
        "mission_performance": mission_performance,
        "varied_parameters": {
            "battery_capacity_ah": selected.batt_capacity,
            "motor_kv_rpm_per_v": selected.motor_kv,
            "motor_rated_power_w": selected.motor_max_power,
            "mission_1_and_2_propeller_diameter_in": selected.prop_diameter_in,
            "mission_1_and_2_propeller_pitch_in": selected.prop_pitch_in,
            "mission_3_propeller_diameter_in": selected.mission3_prop_diameter_in,
            "mission_3_propeller_pitch_in": selected.mission3_prop_pitch_in,
        },
        "optimization": {
            "success": bool(result.success),
            "message": str(result.message),
            "generations": int(result.nit),
            "evaluations": int(result.nfev),
            "seed": seed,
            "eligible_catalog_propellers": len(propellers),
        },
    }


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def _json_safe(value):
    """Convert solver infinities and NumPy scalars into strict JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Size an electric propulsion system for an XFLR5 aircraft XML file."
    )
    parser.add_argument("aircraft_xml", type=Path, help="XFLR5 aircraft XML input")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report path")
    parser.add_argument("--generations", type=_positive_int, default=3)
    parser.add_argument("--popsize", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--mission3-mass-reduction-kg",
        type=float,
        default=DEFAULT_MISSION3_MASS_REDUCTION_KG,
        metavar="KG",
        help="Mass removed from the XML aircraft for M3 (default: 5.9 kg)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    geometry = import_xflr5_geometry(args.aircraft_xml)
    print(
        f"Sizing propulsion for {geometry.name!r} with SG6042 "
        f"({args.generations} generations)...",
        flush=True,
    )
    report = size_propulsion(
        geometry,
        generations=args.generations,
        popsize=args.popsize,
        seed=args.seed,
        mission3_mass_reduction_kg=args.mission3_mass_reduction_kg,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    system = report["selected_propulsion_system"]
    print(f"Propulsion sizing complete: {geometry.name}")
    print(f"  Airfoil: SG6042")
    print(
        f"  Fixed masses: M1/M2 "
        f"{report['result']['mission_masses_kg']['M2']:.3f} kg, M3 "
        f"{report['result']['mission_masses_kg']['M3']:.3f} kg "
        f"(added mass: 0 kg)"
    )
    print(
        f"  Motor: {system['motor']['kv_rpm_per_v']:.0f} kV, "
        f"{system['motor']['rated_power_w']:.0f} W"
    )
    print(
        f"  Battery: {system['battery']['series_cells']}S "
        f"{system['battery']['capacity_ah']:.3f} Ah"
    )
    print(f"  M1/M2 propeller: {system['mission_1_and_2_propeller']}")
    print(f"  M3 propeller: {system['mission_3_propeller']}")
    print(f"  Feasible: {report['result']['propulsion_feasible']}")
    print(f"  Full report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
