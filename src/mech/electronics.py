"""Electronics packaging geometry and lightweight linear mass models.

The mechanical model only knows the equivalent center of mass of the installed
electronics.  Until component-level installation coordinates are measured, the
battery, propulsion, ESC, and miscellaneous electronics share that equivalent
location in the mass ledger.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


INCH_M = 0.0254


@dataclass(frozen=True)
class LinearMassModel:
    """A one-variable linear mass relationship with an optional mass floor.

    ``sizing_value`` can be any caller-selected quantity, such as battery
    capacity, motor power, or propeller diameter.  The units are documented by
    ``input_name`` and must be consistent with ``slope_kg_per_unit``.

    Use :meth:`from_points` when two measured catalogue entries are available.
    The resulting line is deliberately tiny and dependency-free so it can be
    evaluated many times inside the aircraft optimizer.
    """

    reference_input: float
    reference_mass_kg: float
    slope_kg_per_unit: float = 0.0
    minimum_mass_kg: float = 0.0
    input_name: str = "sizing value"

    def __post_init__(self) -> None:
        values = (
            self.reference_input,
            self.reference_mass_kg,
            self.slope_kg_per_unit,
            self.minimum_mass_kg,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Linear mass-model values must be finite.")
        if self.reference_mass_kg < 0 or self.minimum_mass_kg < 0:
            raise ValueError("Linear mass-model masses cannot be negative.")
        if not self.input_name:
            raise ValueError("Linear mass-model input_name cannot be empty.")

    @classmethod
    def from_points(
        cls,
        first_input: float,
        first_mass_kg: float,
        second_input: float,
        second_mass_kg: float,
        *,
        minimum_mass_kg: float = 0.0,
        input_name: str = "sizing value",
    ) -> "LinearMassModel":
        """Create the line through two measured ``(input, mass)`` points."""

        if not np.all(
            np.isfinite((first_input, first_mass_kg, second_input, second_mass_kg))
        ):
            raise ValueError("Interpolation points must be finite.")
        if np.isclose(first_input, second_input):
            raise ValueError("Interpolation input values must be distinct.")
        slope = (second_mass_kg - first_mass_kg) / (second_input - first_input)
        return cls(
            reference_input=float(first_input),
            reference_mass_kg=float(first_mass_kg),
            slope_kg_per_unit=float(slope),
            minimum_mass_kg=minimum_mass_kg,
            input_name=input_name,
        )

    def mass_kg(self, sizing_value: float) -> float:
        if not np.isfinite(sizing_value):
            raise ValueError(f"{self.input_name} must be finite.")
        mass = self.reference_mass_kg + self.slope_kg_per_unit * (
            sizing_value - self.reference_input
        )
        return max(float(mass), self.minimum_mass_kg)


@dataclass(frozen=True)
class ElectronicsPackagingConfig:
    """Skinny/fat electronics-area definitions supplied by the design team."""

    skinny_width_limit_m: float = 0.127
    skinny_height_limit_m: float = 0.127
    skinny_length_m: float = 0.254
    skinny_cg_from_front_m: float = 0.135
    fat_length_m: float = 0.228
    fat_cg_from_front_m: float = 0.119
    cg_below_wing_m: float = 3.0 * INCH_M

    def __post_init__(self) -> None:
        values = (
            self.skinny_width_limit_m,
            self.skinny_height_limit_m,
            self.skinny_length_m,
            self.skinny_cg_from_front_m,
            self.fat_length_m,
            self.fat_cg_from_front_m,
            self.cg_below_wing_m,
        )
        if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0):
            raise ValueError("Electronics packaging dimensions must be finite and positive.")
        if not 0 < self.skinny_cg_from_front_m < self.skinny_length_m:
            raise ValueError("Skinny electronics CG must lie inside its area.")
        if not 0 < self.fat_cg_from_front_m < self.fat_length_m:
            raise ValueError("Fat electronics CG must lie inside its area.")


@dataclass(frozen=True)
class ElectronicsLayout:
    """Resolved longitudinal electronics area for one airplane geometry."""

    profile: str
    length_m: float
    cg_from_front_m: float
    front_edge_x_m: float
    cg_x_m: float
    back_edge_x_m: float
    cg_y_m: float
    cg_z_m: float

    @property
    def position_m(self) -> np.ndarray:
        return np.array((self.cg_x_m, self.cg_y_m, self.cg_z_m), dtype=float)


def resolve_electronics_layout(
    *,
    cg_x_m: float,
    fuselage_width_m: float,
    fuselage_height_m: float,
    config: ElectronicsPackagingConfig,
    cg_y_m: float = 0.0,
) -> ElectronicsLayout:
    """Resolve the electronics front/back edges around the required CM.

    A plane is skinny only when *both* fuselage cross-section dimensions are
    strictly less than 0.127 m.  A dimension exactly on the limit is therefore
    classified as fat, matching the stated "less than" rule.
    """

    values = (cg_x_m, cg_y_m, fuselage_width_m, fuselage_height_m)
    if not np.all(np.isfinite(values)):
        raise ValueError("Electronics layout inputs must be finite.")
    if fuselage_width_m <= 0 or fuselage_height_m <= 0:
        raise ValueError("Fuselage width and height must be positive.")

    skinny = (
        fuselage_width_m < config.skinny_width_limit_m
        and fuselage_height_m < config.skinny_height_limit_m
    )
    if skinny:
        profile = "skinny"
        length = config.skinny_length_m
        cg_from_front = config.skinny_cg_from_front_m
    else:
        profile = "fat"
        length = config.fat_length_m
        cg_from_front = config.fat_cg_from_front_m

    front_edge = float(cg_x_m - cg_from_front)
    return ElectronicsLayout(
        profile=profile,
        length_m=length,
        cg_from_front_m=cg_from_front,
        front_edge_x_m=front_edge,
        cg_x_m=float(cg_x_m),
        back_edge_x_m=float(front_edge + length),
        cg_y_m=float(cg_y_m),
        cg_z_m=-config.cg_below_wing_m,
    )
