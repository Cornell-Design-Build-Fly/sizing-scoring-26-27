from __future__ import annotations

"""
Apply static-test thrust correction factors to raw propeller data.

This module is meant to be used at BUILD TIME, before creating a corrected
ContinuousPropDatabase .pkl file.

It does not add runtime work inside prop_main(). It modifies the raw thrust
array before the corrected interpolator is built and pickled.

Inputs:
    src/prop/data/static_thrust_correction_factors.csv

Expected correction CSV columns:
    prop_diameter_in
    prop_pitch_in
    estimated_rpm
    correction_factor

Main function:
    apply_static_thrust_correction(data)

where data is the dictionary returned by load_prop_data_points().
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Literal

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CORRECTION_CSV_PATH = DATA_DIR / "static_thrust_correction_factors.csv"

UntestedPropMode = Literal["nearest", "none"]


@dataclass(frozen=True, slots=True)
class CorrectionCurve:
    prop_diameter_in: float
    prop_pitch_in: float
    rpms: np.ndarray
    factors: np.ndarray

    def factor_at_rpm(self, rpm: float) -> float:
        """
        Returns an interpolated correction factor at this RPM.

        np.interp uses constant extrapolation at the ends, so below the lowest
        tested RPM it uses the lowest-RPM factor, and above the highest tested
        RPM it uses the highest-RPM factor.
        """
        if len(self.rpms) == 0:
            return 1.0

        if len(self.rpms) == 1:
            return float(self.factors[0])

        return float(
            np.interp(
                float(rpm),
                self.rpms,
                self.factors,
                left=self.factors[0],
                right=self.factors[-1],
            )
        )


class StaticThrustCorrectionModel:
    """
    Stores separate RPM -> correction_factor curves for each tested prop.

    The static-test throttle is NOT used here. Throttle was only used upstream
    to estimate RPM in static_thrust_correction.py.

    This model applies correction based on:
        prop diameter, prop pitch, RPM
    """

    def __init__(
        self,
        curves: dict[tuple[float, float], CorrectionCurve],
        untested_prop_mode: UntestedPropMode = "nearest",
    ) -> None:
        if not curves:
            raise ValueError("No correction curves were provided.")

        if untested_prop_mode not in ("nearest", "none"):
            raise ValueError('untested_prop_mode must be "nearest" or "none".')

        self.curves = curves
        self.untested_prop_mode = untested_prop_mode

        self._keys = np.array(list(curves.keys()), dtype=float)
        self._diameter_span = max(float(np.ptp(self._keys[:, 0])), 1.0)
        self._pitch_span = max(float(np.ptp(self._keys[:, 1])), 1.0)

    def _exact_key(self, diameter_in: float, pitch_in: float) -> tuple[float, float] | None:
        for key in self.curves:
            if math.isclose(key[0], diameter_in, abs_tol=1e-6) and math.isclose(
                key[1],
                pitch_in,
                abs_tol=1e-6,
            ):
                return key

        return None

    def _nearest_key(self, diameter_in: float, pitch_in: float) -> tuple[float, float]:
        normalized_diameter_error = (self._keys[:, 0] - diameter_in) / self._diameter_span
        normalized_pitch_error = (self._keys[:, 1] - pitch_in) / self._pitch_span
        distances = normalized_diameter_error**2 + normalized_pitch_error**2

        nearest_index = int(np.argmin(distances))
        nearest = self._keys[nearest_index]

        return float(nearest[0]), float(nearest[1])

    def factor(
        self,
        diameter_in: float,
        pitch_in: float,
        rpm: float,
    ) -> float:
        exact_key = self._exact_key(diameter_in, pitch_in)

        if exact_key is not None:
            return self.curves[exact_key].factor_at_rpm(rpm)

        if self.untested_prop_mode == "none":
            return 1.0

        nearest_key = self._nearest_key(diameter_in, pitch_in)
        return self.curves[nearest_key].factor_at_rpm(rpm)

    def factors_for_arrays(
        self,
        diameter_in: np.ndarray,
        pitch_in: np.ndarray,
        rpm: np.ndarray,
    ) -> np.ndarray:
        factors = np.empty_like(rpm, dtype=float)

        for i in range(len(factors)):
            factors[i] = self.factor(
                diameter_in=float(diameter_in[i]),
                pitch_in=float(pitch_in[i]),
                rpm=float(rpm[i]),
            )

        return factors


def _float_from_row(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")

    if value is None or value == "":
        raise ValueError(f"Missing required correction CSV value: {key}")

    return float(value)


def _build_curve_from_rows(
    prop_diameter_in: float,
    prop_pitch_in: float,
    rows: list[tuple[float, float]],
) -> CorrectionCurve:
    """
    Builds one RPM -> correction_factor curve for one prop.

    Duplicate/near-duplicate RPM rows are binned to the nearest 50 RPM and averaged.
    That helps combine multiple static test passes for the same prop.
    """
    rpm_to_factors: dict[float, list[float]] = {}

    for rpm, factor in rows:
        if not math.isfinite(rpm) or not math.isfinite(factor):
            continue

        if rpm <= 0.0 or factor <= 0.0:
            continue

        binned_rpm = round(rpm / 50.0) * 50.0
        rpm_to_factors.setdefault(binned_rpm, []).append(factor)

    if not rpm_to_factors:
        raise ValueError(
            f"No valid correction factors for {prop_diameter_in}x{prop_pitch_in}."
        )

    sorted_rpms = np.array(sorted(rpm_to_factors), dtype=float)
    sorted_factors = np.array(
        [float(np.mean(rpm_to_factors[rpm])) for rpm in sorted_rpms],
        dtype=float,
    )

    return CorrectionCurve(
        prop_diameter_in=prop_diameter_in,
        prop_pitch_in=prop_pitch_in,
        rpms=sorted_rpms,
        factors=sorted_factors,
    )


def load_static_thrust_correction_model(
    correction_csv_path: Path = DEFAULT_CORRECTION_CSV_PATH,
    untested_prop_mode: UntestedPropMode = "nearest",
    allow_thrust_increase: bool = False,
) -> StaticThrustCorrectionModel:
    """
    Loads correction factors and creates one correction curve per tested prop.

    Args:
        correction_csv_path:
            CSV created by static_thrust_correction.py.

        untested_prop_mode:
            "nearest" means untested props use the nearest tested prop's correction
            curve. "none" means untested props keep factor 1.0.

        allow_thrust_increase:
            False means correction factors are capped at 1.0. This makes the method
            a knockdown only. True allows factors above 1.0 if the test data says
            measured thrust exceeded predicted thrust.
    """
    if not correction_csv_path.exists():
        raise FileNotFoundError(
            f"Could not find correction factor CSV: {correction_csv_path}\n"
            "Run python -m src.prop.static_thrust_correction first."
        )

    grouped_rows: dict[tuple[float, float], list[tuple[float, float]]] = {}

    with correction_csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            diameter = _float_from_row(row, "prop_diameter_in")
            pitch = _float_from_row(row, "prop_pitch_in")
            rpm = _float_from_row(row, "estimated_rpm")
            factor = _float_from_row(row, "correction_factor")

            if not allow_thrust_increase:
                factor = min(factor, 1.0)

            if not math.isfinite(factor) or factor <= 0.0:
                continue

            grouped_rows.setdefault((diameter, pitch), []).append((rpm, factor))

    curves = {
        key: _build_curve_from_rows(
            prop_diameter_in=key[0],
            prop_pitch_in=key[1],
            rows=rows,
        )
        for key, rows in grouped_rows.items()
    }

    return StaticThrustCorrectionModel(
        curves=curves,
        untested_prop_mode=untested_prop_mode,
    )


def apply_static_thrust_correction(
    data: dict[str, np.ndarray],
    correction_csv_path: Path = DEFAULT_CORRECTION_CSV_PATH,
    untested_prop_mode: UntestedPropMode = "nearest",
    allow_thrust_increase: bool = False,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Applies prop-specific, RPM-dependent static thrust knockdown factors.

    Args:
        data:
            Dictionary returned by load_prop_data_points().

        correction_csv_path:
            static_thrust_correction_factors.csv.

        untested_prop_mode:
            "nearest" applies the nearest tested prop's correction curve to untested
            props. "none" leaves untested props uncorrected.

        allow_thrust_increase:
            False caps correction factors at 1.0, making this a knockdown-only
            correction.

    Returns:
        A copy of data with corrected thrust_n.
        torque_nm is intentionally unchanged.
    """
    required_keys = ("diameter_in", "pitch_in", "rpm", "thrust_n", "torque_nm")

    for key in required_keys:
        if key not in data:
            raise KeyError(f"Prop data missing required key: {key}")

    model = load_static_thrust_correction_model(
        correction_csv_path=correction_csv_path,
        untested_prop_mode=untested_prop_mode,
        allow_thrust_increase=allow_thrust_increase,
    )

    diameter = np.asarray(data["diameter_in"], dtype=float)
    pitch = np.asarray(data["pitch_in"], dtype=float)
    rpm = np.asarray(data["rpm"], dtype=float)
    raw_thrust = np.asarray(data["thrust_n"], dtype=float)

    factors = model.factors_for_arrays(
        diameter_in=diameter,
        pitch_in=pitch,
        rpm=rpm,
    )

    corrected_data: dict[str, np.ndarray] = {
        key: np.asarray(value).copy()
        for key, value in data.items()
    }

    corrected_data["thrust_n"] = raw_thrust * factors
    corrected_data["thrust_correction_factor"] = factors

    if verbose:
        tested_props = ", ".join(
            f"{key[0]:g}x{key[1]:g}" for key in sorted(model.curves)
        )
        print("Applied static thrust correction.")
        print(f"  Correction CSV: {correction_csv_path}")
        print(f"  Tested correction curves: {tested_props}")
        print(f"  Untested prop mode: {untested_prop_mode}")
        print(f"  Allow thrust increase: {allow_thrust_increase}")
        print(f"  Mean factor applied: {float(np.mean(factors)):.3f}")
        print(f"  Min factor applied:  {float(np.min(factors)):.3f}")
        print(f"  Max factor applied:  {float(np.max(factors)):.3f}")

    return corrected_data
