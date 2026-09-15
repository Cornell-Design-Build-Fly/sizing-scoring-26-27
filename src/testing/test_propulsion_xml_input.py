from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from src.airfoils import SG6042_DATA_PATH, load_airfoil
from src.prop.size_from_xml import _candidate_design, import_xflr5_geometry
from src.vectors import ASBDesignVector, DesignVector


def test_xflr5_geometry_import_preserves_area_and_tail_dimensions(tmp_path) -> None:
    xml = """<?xml version="1.0"?>
<explane>
  <Units>
    <length_unit_to_meter>0.001</length_unit_to_meter>
    <mass_unit_to_kg>0.5</mass_unit_to_kg>
  </Units>
  <Plane>
    <Name>Tapered test plane</Name>
    <Inertia>
      <Point_Mass><Tag>left</Tag><Mass>2</Mass><coordinates>0,0,0</coordinates></Point_Mass>
      <Point_Mass><Tag>right</Tag><Mass>2</Mass><coordinates>1000,0,0</coordinates></Point_Mass>
    </Inertia>
    <has_body>false</has_body>
    <wing>
      <Name>Main Wing</Name><Type>MAINWING</Type><Position>100,0,0</Position><Symetric>true</Symetric>
      <Sections>
        <Section><y_position>0</y_position><Chord>300</Chord><xOffset>0</xOffset><Twist>0</Twist></Section>
        <Section><y_position>500</y_position><Chord>200</Chord><xOffset>50</xOffset><Twist>-2</Twist></Section>
      </Sections>
    </wing>
    <wing>
      <Name>Horizontal Tail</Name><Type>ELEVATOR</Type><Position>800,0,0</Position><Symetric>true</Symetric>
      <Sections>
        <Section><y_position>0</y_position><Chord>150</Chord></Section>
        <Section><y_position>200</y_position><Chord>150</Chord></Section>
      </Sections>
    </wing>
    <wing>
      <Name>Vertical Tail</Name><Type>FIN</Type><Position>800,0,0</Position><Symetric>true</Symetric>
      <Sections>
        <Section><y_position>0</y_position><Chord>160</Chord></Section>
        <Section><y_position>180</y_position><Chord>160</Chord></Section>
      </Sections>
    </wing>
  </Plane>
</explane>
"""
    path = tmp_path / "plane.xml"
    path.write_text(xml, encoding="utf-8")

    geometry = import_xflr5_geometry(path)
    design = geometry.to_design_vector()

    assert geometry.name == "Tapered test plane"
    assert math.isclose(design.wing_span, 1.0)
    assert math.isclose(design.wing_chord, 0.25)
    assert math.isclose(design.wing_area, 0.25)
    assert math.isclose(design.tail_arm, 0.7)
    assert math.isclose(design.hstab_span, 0.4)
    assert math.isclose(design.hstab_chord, 0.15)
    assert math.isclose(design.vstab_span, 0.18)
    assert math.isclose(design.vstab_chord, 0.16)
    assert design.wing_airfoil == "sg6042"
    assert not design.mission3_sensor_drag_enabled
    assert not design.fuselage_drag_enabled
    assert math.isclose(geometry.mass_properties.total_mass_kg, 2.0)
    assert geometry.mass_properties.cg_m == (0.5, 0.0, 0.0)
    assert geometry.mass_properties.inertia_tensor_kg_m2 == (
        (0.0, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
    )
    assert geometry.main_wing.sections[1].x_offset_m == 0.05
    assert geometry.main_wing.sections[1].twist_deg == -2.0
    assert len(geometry.warnings) == 3


def test_sg6042_is_checked_in_and_used_by_default() -> None:
    assert SG6042_DATA_PATH.is_file()
    airfoil = load_airfoil("sg6042")
    assert airfoil.name == "sg6042"
    assert len(airfoil.coordinates) == 81

    design = DesignVector()
    airplane = ASBDesignVector.from_design_vector(design).make_airplane()
    assert design.wing_airfoil == "sg6042"
    assert airplane.wings[0].xsecs[0].airfoil.name == "sg6042"


def test_tail_overrides_are_optional_and_validated() -> None:
    baseline = DesignVector()
    assert baseline.hstab_span_override_m is None
    assert baseline.mission3_sensor_drag_enabled
    assert baseline.fuselage_drag_enabled

    overridden = DesignVector(
        hstab_span_override_m=0.5,
        hstab_chord_override_m=0.12,
        vstab_span_override_m=0.2,
        vstab_chord_override_m=0.14,
    )
    assert overridden.hstab_area == 0.06
    assert math.isclose(overridden.vstab_area, 0.028)

    try:
        DesignVector(hstab_span_override_m=0.5)
    except ValueError as exc:
        assert "provided together" in str(exc)
    else:
        raise AssertionError("A partial tail override should be rejected.")


def test_propulsion_candidate_cannot_change_fixed_geometry() -> None:
    base = DesignVector(
        wing_span=1.7,
        wing_chord=0.31,
        tail_arm=0.72,
        hstab_span_override_m=0.55,
        hstab_chord_override_m=0.13,
        vstab_span_override_m=0.21,
        vstab_chord_override_m=0.15,
        mission3_sensor_drag_enabled=False,
        fuselage_drag_enabled=False,
    )
    props = (
        SimpleNamespace(diameter_in=12.0, pitch_in=6.0),
        SimpleNamespace(diameter_in=18.0, pitch_in=10.0),
    )
    candidate = _candidate_design(
        base,
        np.asarray([3.1, 0.0, 1.0, 410.0, 2400.0]),
        props,
    )

    fixed_names = (
        "wing_span",
        "wing_chord",
        "tail_arm",
        "hstab_span",
        "hstab_chord",
        "vstab_span",
        "vstab_chord",
        "wing_airfoil",
        "mission3_sensor_drag_enabled",
        "fuselage_drag_enabled",
    )
    assert all(getattr(candidate, name) == getattr(base, name) for name in fixed_names)
    assert candidate.batt_capacity == 3.1
    assert candidate.motor_kv == 410.0
    assert candidate.motor_max_power == 2400.0
