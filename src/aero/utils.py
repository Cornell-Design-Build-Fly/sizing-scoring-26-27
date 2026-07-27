from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.aero.custom_classes import ModeResult


def require_scalar(value) -> float:
    """Convert an AeroSandbox scalar-like output into a plain float."""
    array_value = np.asarray(value)
    if array_value.ndim == 0:
        return float(array_value)
    if array_value.size == 1:
        return float(array_value.reshape(-1)[0])
    raise TypeError(f"Expected scalar-like output, got shape {array_value.shape}.")


def optional_scalar(value) -> float | None:
    """Convert a scalar-like value to a float, preserving None."""
    if value is None:
        return None
    return require_scalar(value)


def dict_to_mode_result(mode_dict: dict) -> "ModeResult":
    """Convert a dictionary of mode results to a ModeResult object."""
    # Imported here to keep utils independent at module import time.
    from src.aero.custom_classes import ModeResult

    return ModeResult(
        eigenvalue_real=require_scalar(mode_dict["eigenvalue_real"]),
        eigenvalue_imag=require_scalar(mode_dict["eigenvalue_imag"]),
        damping_ratio=require_scalar(mode_dict["damping_ratio"]),
        eigenvalue_imag_approx=optional_scalar(mode_dict.get("eigenvalue_imag_approx")),
        damping_ratio_approx=optional_scalar(mode_dict.get("damping_ratio_approx")),
    )
