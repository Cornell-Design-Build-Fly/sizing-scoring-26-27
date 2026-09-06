"""Plain-flap lift and drag increments.

Flaps are modelled as coefficient increments, not as geometry. Nothing here
touches ``make_airplane``: adding a real control surface would pull the whole
stability derivative chain along with it, and the effect that matters for
sizing is entirely captured by a shifted CLmax and an added CD0.

Configuration matters more than the increments themselves. The same wing has
three different maximum lift coefficients depending on what the flaps are
doing, and the model must not mix them up:

  clean    cruise, and every turn on the course
  takeoff  the ground roll, the acceleration to climb speed, and the climb;
           the flaps come up on reaching cruise altitude
  landing  the approach, reported only -- nothing constrains the landing

Charging the turn envelope with flapped CLmax would let the aircraft pull 2.5 g
on lift it does not have, which is why ``cl_max_for`` takes the configuration
explicitly instead of defaulting to one number.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


# Section maximum lift coefficient assumed for the wing airfoil, with the
# standard finite-wing correction. Previously duplicated in three modules.
SECTION_CL_MAX = 1.45


def clean_cl_max(wing_aspect_ratio: float) -> float:
    """Return the clean-wing CLmax for an aspect ratio."""

    if not math.isfinite(wing_aspect_ratio) or wing_aspect_ratio <= 0.0:
        raise ValueError("Wing aspect ratio must be finite and positive.")
    return SECTION_CL_MAX * wing_aspect_ratio / (wing_aspect_ratio + 2.0)


@dataclass(frozen=True)
class FlapConfig:
    """Plain flap, sized by chord and span fraction of the main wing.

    Defaults are a plain 25%-chord flap over the inboard 60% of the span: a
    simple hinge and a pair of servos, which is what the team is building.

    ``reference_delta_cl_max_2d`` is the two-dimensional section lift increment
    at ``reference_deflection_deg`` (Raymer, *Aircraft Design*, Table 12.2:
    0.9 for a plain flap at its usual 60 deg maximum). Deflection is scaled by
    ``sin(delta)``, the conventional form, which reproduces the flattening of
    the plain-flap lift curve as the flow separates over the deflected surface.
    """

    chord_fraction: float = 0.25
    span_fraction: float = 0.60
    reference_delta_cl_max_2d: float = 0.90
    reference_deflection_deg: float = 60.0
    # Takeoff is a real compromise now that the flaps stay down through the
    # climb: deflection shortens the ground roll but costs climb rate and climb
    # energy. Sweeping the archived optimum puts the best score at 20 deg, which
    # also sits in the usual 10-25 deg light-aircraft band. The optimum moves
    # with the design; this is a constant by the team's choice, not a search.
    takeoff_deflection_deg: float = 20.0
    # Landing wants maximum CLmax. Nothing scores on it -- the team removed the
    # landing-speed constraint -- so it only feeds the reported stall speed.
    landing_deflection_deg: float = 40.0
    # Three-dimensional correction on the section increment for an unswept
    # wing (Raymer Eq. 12.21: 0.9 * delta_cl_max * S_flapped/S_ref * cos(sweep);
    # the wing has no sweep, so the cosine is one).
    three_dimensional_factor: float = 0.90
    # Plain-flap profile-drag increment, Roskam/Torenbeek form
    #   delta_CD0 = k * (cf/c)^1.38 * (S_flapped/S_ref) * sin^2(delta).
    drag_increment_coefficient: float = 1.70
    drag_chord_exponent: float = 1.38

    def __post_init__(self) -> None:
        fractions = (self.chord_fraction, self.span_fraction)
        if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in fractions):
            raise ValueError("Flap chord and span fractions must lie in (0, 1].")
        deflections = (
            self.reference_deflection_deg,
            self.takeoff_deflection_deg,
            self.landing_deflection_deg,
        )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 90.0
            for value in deflections
        ):
            raise ValueError("Flap deflections must lie in [0, 90] degrees.")
        if self.reference_deflection_deg <= 0.0:
            raise ValueError("Reference flap deflection must be positive.")
        positive = (
            self.reference_delta_cl_max_2d,
            self.three_dimensional_factor,
            self.drag_increment_coefficient,
            self.drag_chord_exponent,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Flap model coefficients must be finite and positive.")

    def delta_cl_max(self, deflection_deg: float) -> float:
        """Aircraft CLmax increment at a flap deflection."""

        if not math.isfinite(deflection_deg) or deflection_deg <= 0.0:
            return 0.0
        section_increment = self.reference_delta_cl_max_2d * (
            math.sin(math.radians(deflection_deg))
            / math.sin(math.radians(self.reference_deflection_deg))
        )
        return (
            self.three_dimensional_factor * section_increment * self.span_fraction
        )

    def delta_cd0(self, deflection_deg: float) -> float:
        """Profile-drag increment at a flap deflection, on wing reference area."""

        if not math.isfinite(deflection_deg) or deflection_deg <= 0.0:
            return 0.0
        return (
            self.drag_increment_coefficient
            * self.chord_fraction**self.drag_chord_exponent
            * self.span_fraction
            * math.sin(math.radians(deflection_deg)) ** 2
        )

    def deflection_for(self, configuration: str) -> float:
        """Flap deflection for ``"clean"``, ``"takeoff"`` or ``"landing"``."""

        if configuration == "clean":
            return 0.0
        if configuration == "takeoff":
            return self.takeoff_deflection_deg
        if configuration == "landing":
            return self.landing_deflection_deg
        raise ValueError(
            "configuration must be 'clean', 'takeoff' or 'landing'; "
            f"got {configuration!r}."
        )

    def cl_max_for(self, wing_aspect_ratio: float, configuration: str) -> float:
        """CLmax in a named flap configuration."""

        return clean_cl_max(wing_aspect_ratio) + self.delta_cl_max(
            self.deflection_for(configuration)
        )

    def stall_speed_for(
        self,
        clean_stall_speed_mps: float,
        wing_aspect_ratio: float,
        configuration: str,
    ) -> float:
        """Rescale a clean stall speed into another flap configuration.

        Stall speed goes as ``1 / sqrt(CLmax)`` at fixed weight and area, so the
        flapped speeds follow from the clean one without re-solving lift.
        """

        if not math.isfinite(clean_stall_speed_mps) or clean_stall_speed_mps <= 0.0:
            raise ValueError("Clean stall speed must be finite and positive.")
        clean = clean_cl_max(wing_aspect_ratio)
        flapped = self.cl_max_for(wing_aspect_ratio, configuration)
        return clean_stall_speed_mps * math.sqrt(clean / flapped)


DEFAULT_FLAPS = FlapConfig()


__all__ = [
    "DEFAULT_FLAPS",
    "SECTION_CL_MAX",
    "FlapConfig",
    "clean_cl_max",
]
