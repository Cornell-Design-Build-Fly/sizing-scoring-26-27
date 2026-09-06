"""Resolve design requests to real, two-blade catalog propellers."""

from __future__ import annotations

from dataclasses import replace

from src.prop.continuous_prop_database import ContinuousPropDatabase
from src.vectors import DesignVector


def resolve_catalog_propellers(
    design: DesignVector,
    database: ContinuousPropDatabase,
    *,
    minimum_diameter_in: float | None = None,
    maximum_diameter_in: float | None = None,
    minimum_pitch_in: float | None = None,
    maximum_pitch_in: float | None = None,
    minimum_pitch_diameter_ratio: float | None = None,
    maximum_pitch_diameter_ratio: float | None = None,
) -> DesignVector:
    """Snap both mission propellers to actual catalog geometries.

    The database loader excludes every encoded three-/four-blade propeller, so
    a surface returned here is necessarily a two-blade choice.  Exact inputs
    remain unchanged; arbitrary optimizer requests resolve deterministically
    to their nearest catalog products.
    """

    common = dict(
        minimum_diameter_in=minimum_diameter_in,
        maximum_diameter_in=maximum_diameter_in,
        minimum_pitch_in=minimum_pitch_in,
        maximum_pitch_in=maximum_pitch_in,
        minimum_pitch_diameter_ratio=minimum_pitch_diameter_ratio,
        maximum_pitch_diameter_ratio=maximum_pitch_diameter_ratio,
    )
    mission12 = database.nearest_catalog_surface(
        design.prop_diameter_in,
        design.prop_pitch_in,
        **common,
    )
    mission3 = database.nearest_catalog_surface(
        float(design.mission3_prop_diameter_in),
        float(design.mission3_prop_pitch_in),
        **common,
    )
    return replace(
        design,
        prop_diameter_in=float(mission12.diameter_in),
        prop_pitch_in=float(mission12.pitch_in),
        mission3_prop_diameter_in=float(mission3.diameter_in),
        mission3_prop_pitch_in=float(mission3.pitch_in),
    )


def catalog_propeller_keys(
    design: DesignVector,
    database: ContinuousPropDatabase,
) -> dict[int, str]:
    """Return exact product keys for an already-resolved design."""

    mission12 = database.catalog.get_by_geometry(
        design.prop_diameter_in,
        design.prop_pitch_in,
    )
    mission3 = database.catalog.get_by_geometry(
        float(design.mission3_prop_diameter_in),
        float(design.mission3_prop_pitch_in),
    )
    return {1: mission12.key, 2: mission12.key, 3: mission3.key}
