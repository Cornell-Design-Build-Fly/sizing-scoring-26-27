"""Mechanical mass-properties module exports."""

from src.mech.electronics import (
    ElectronicsLayout,
    ElectronicsPackagingConfig,
    LinearMassModel,
    PiecewiseLinearMassModel,
    resolve_electronics_layout,
)
from src.mech.main_mech import evaluate_mechanical_module, mech_main
from src.mech.mass_properties import (
    center_of_gravity,
    estimate_neutral_point_x,
    geometry_stations,
    inertia_tensor_about_cg,
    inertia_tensor_about_point,
    static_margin,
)
from src.mech.models import (
    AirframeMassConfig,
    BatteryMassModel,
    MassItem,
    MechanicalModuleConfig,
    MechanicalResult,
    Mission2Config,
    Mission3Config,
    MissionMassProperties,
    NeutralPointConfig,
    PayloadTypeConfig,
    PlacementRules,
    RelativePayloadRules,
    StaticMarginConfig,
)
from src.mech.payload_placement import PayloadPlacementError, place_mission2_payload

__all__ = [
    "AirframeMassConfig",
    "BatteryMassModel",
    "ElectronicsLayout",
    "ElectronicsPackagingConfig",
    "LinearMassModel",
    "MassItem",
    "MechanicalModuleConfig",
    "MechanicalResult",
    "Mission2Config",
    "Mission3Config",
    "MissionMassProperties",
    "NeutralPointConfig",
    "PayloadPlacementError",
    "PayloadTypeConfig",
    "PiecewiseLinearMassModel",
    "PlacementRules",
    "RelativePayloadRules",
    "StaticMarginConfig",
    "center_of_gravity",
    "estimate_neutral_point_x",
    "evaluate_mechanical_module",
    "geometry_stations",
    "inertia_tensor_about_cg",
    "inertia_tensor_about_point",
    "mech_main",
    "place_mission2_payload",
    "resolve_electronics_layout",
    "static_margin",
]
