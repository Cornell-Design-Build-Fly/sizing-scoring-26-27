"""Continuous-payload counterpart to :mod:`src.mech.main_mech`.

This module floors Mission-2 duck and puck amounts for physical placement, then
represents each fractional remainder as a point mass at the resulting M2 center
of gravity. The payload-derived fuselage therefore encloses the whole payloads;
fractional point-mass equivalents do not extend its envelope. Import this module
directly when continuous optimizer values are required.
"""

from __future__ import annotations

from dataclasses import replace
from math import floor

import numpy as np

from src.mech.main_mech import evaluate_mechanical_module as _evaluate_discrete
from src.mech.mass_properties import inertia_tensor_about_cg, static_margin
from src.mech.models import (
    MassItem,
    MechanicalModuleConfig,
    MechanicalResult,
)
from src.vectors import DesignVector, ParameterVector


def _split_payload_amount(value: float, name: str) -> tuple[int, float]:
    """Return the strict floor count and fractional remainder for one amount."""

    amount = float(value)
    if not np.isfinite(amount) or amount < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    whole_count = floor(amount)
    return whole_count, amount - whole_count


def _fractional_payload_item(
    *,
    label: str,
    fraction: float,
    unit_mass_kg: float,
    position_m: np.ndarray,
) -> MassItem | None:
    """Build one auditable point-mass row for a fractional payload remainder."""

    if fraction == 0.0:
        return None
    return MassItem(
        name=f"Fractional {label} equivalent ({fraction:.6g})",
        mass_kg=fraction * unit_mass_kg,
        position_m=position_m,
        dimensions_m=(0.0, 0.0, 0.0),
        missions=frozenset({"M2"}),
        category="mission_2_fractional_payload",
        notes=(
            f"Unplaced {fraction:.12g} fraction of one {label}; modeled as a "
            "point mass at the floor-count Mission-2 CG."
        ),
    )


def evaluate_mechanical_module(
    design_vector: DesignVector,
    config: MechanicalModuleConfig | None = None,
) -> MechanicalResult:
    """Evaluate all missions while accepting continuous M2 payload amounts.

    Whole ducks and pucks are placed by the deterministic center-out process.
    Fractional remainders add their exact mass at the CG of that
    floor-count arrangement, so they change total mass and weight without
    changing CG, static margin, or inertia about the CG.
    """

    duck_count, duck_fraction = _split_payload_amount(
        design_vector.ducks_num, "ducks_num"
    )
    puck_count, puck_fraction = _split_payload_amount(
        design_vector.pucks_num, "pucks_num"
    )
    resolved_config = config or MechanicalModuleConfig()
    floor_design = replace(
        design_vector,
        ducks_num=float(duck_count),
        pucks_num=float(puck_count),
    )
    result = _evaluate_discrete(floor_design, resolved_config)

    if duck_fraction == 0.0 and puck_fraction == 0.0:
        return result

    floor_m2 = result.for_mission("M2")
    floor_cg = floor_m2.cg_m.copy()
    residual_items = tuple(
        item
        for item in (
            _fractional_payload_item(
                label=resolved_config.mission2.duck.label,
                fraction=duck_fraction,
                unit_mass_kg=resolved_config.mission2.duck.mass_kg,
                position_m=floor_cg,
            ),
            _fractional_payload_item(
                label=resolved_config.mission2.puck.label,
                fraction=puck_fraction,
                unit_mass_kg=resolved_config.mission2.puck.mass_kg,
                position_m=floor_cg,
            ),
        )
        if item is not None
    )
    m2_items = floor_m2.items + residual_items
    m2_cg, m2_inertia = inertia_tensor_about_cg(m2_items)
    m2_mass = float(sum(item.mass_kg for item in m2_items))
    m2_static_margin = static_margin(
        result.neutral_point_x_m,
        m2_cg[0],
        design_vector.wing_chord,
    )
    margin_config = resolved_config.static_margin
    tolerance = 1e-12

    continuous_notes: list[str] = []
    if duck_fraction:
        continuous_notes.append(
            f"Continuous M2 ducks_num={float(design_vector.ducks_num):.12g}: "
            f"placed {duck_count} whole Duck item(s) and added "
            f"{duck_fraction * resolved_config.mission2.duck.mass_kg:.12g} kg "
            "at the floor-count M2 CG."
        )
    if puck_fraction:
        continuous_notes.append(
            f"Continuous M2 pucks_num={float(design_vector.pucks_num):.12g}: "
            f"placed {puck_count} whole Puck item(s) and added "
            f"{puck_fraction * resolved_config.mission2.puck.mass_kg:.12g} kg "
            "at the floor-count M2 CG."
        )

    m2 = replace(
        floor_m2,
        items=m2_items,
        total_mass_kg=m2_mass,
        weight_n=m2_mass * ParameterVector.gravity,
        cg_m=m2_cg,
        inertia_tensor_kg_m2=m2_inertia,
        static_margin=m2_static_margin,
        static_margin_feasible=(
            margin_config.minimum - tolerance
            <= m2_static_margin
            <= margin_config.maximum + tolerance
        ),
        warnings=tuple(dict.fromkeys(floor_m2.warnings + tuple(continuous_notes))),
    )
    missions = dict(result.missions)
    missions["M2"] = m2
    return replace(
        result,
        all_items=result.all_items + residual_items,
        missions=missions,
        warnings=tuple(dict.fromkeys(result.warnings + tuple(continuous_notes))),
    )


def mech_main(
    design_vector: DesignVector,
    mission: str = "M1",
    config: MechanicalModuleConfig | None = None,
) -> tuple[tuple[float, float, float], np.ndarray, float]:
    """Continuous-payload counterpart to the standard compatibility entry point."""

    result = evaluate_mechanical_module(design_vector, config)
    mission_result = result.for_mission(mission)
    return (
        tuple(float(value) for value in mission_result.cg_m),
        mission_result.inertia_tensor_kg_m2.copy(),
        mission_result.weight_n,
    )


# Explicit aliases make the alternate behavior easy to discover at call sites.
evaluate_mechanical_module_continuous = evaluate_mechanical_module
mech_main_continuous = mech_main


__all__ = [
    "evaluate_mechanical_module",
    "evaluate_mechanical_module_continuous",
    "mech_main",
    "mech_main_continuous",
]
