from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

DEFAULT_PROP_DATA_PATH = (
    Path(__file__).resolve().parent / "data" / "prop_data.json"
)

_PROP_SIZE_PATTERN = re.compile(
    r"(?P<diameter>\d+(?:\.\d+)?)\s*[xX]\s*"
    r"(?P<pitch>\d+(?:\.\d+)?)"
)

_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")

# For duplicate diameter/pitch groups that do not have a normal
# key ending exactly in "E", explicitly choose which entry to keep.
DUPLICATE_GEOMETRY_OVERRIDES: dict[
    tuple[float, float],
    str,
] = {
    (13.0, 4.5): "13x4.5EP",
}

@dataclass(frozen=True, slots=True)
class PropRpmData:
    """Performance data for one propeller at one RPM."""

    rpm: float
    velocity_mph: FloatArray
    thrust_n: FloatArray
    torque_nm: FloatArray
    removed_velocity_count: int

    @property
    def sample_count(self) -> int:
        return len(self.velocity_mph)


@dataclass(frozen=True, slots=True)
class PropellerData:
    """All raw performance data belonging to one JSON propeller key."""

    key: str
    diameter_in: float
    pitch_in: float
    rpm_data: tuple[PropRpmData, ...]

    @property
    def sample_count(self) -> int:
        return sum(table.sample_count for table in self.rpm_data)


def parse_prop_geometry(prop_key: str) -> tuple[float, float]:
    """Extract diameter and pitch from a JSON propeller key."""

    matches = list(_PROP_SIZE_PATTERN.finditer(prop_key))

    if len(matches) != 1:
        raise ValueError(
            f'Expected exactly one diameter/pitch pair in prop key '
            f'"{prop_key}", but found {len(matches)}.'
        )

    match = matches[0]

    diameter_in = float(match.group("diameter"))
    pitch_in = float(match.group("pitch"))

    if diameter_in <= 0 or pitch_in <= 0:
        raise ValueError(
            f'Prop key "{prop_key}" contains nonpositive geometry.'
        )

    return diameter_in, pitch_in


def parse_rpm(rpm_key: str) -> float:
    """Extract RPM from a JSON RPM key."""

    matches = _NUMBER_PATTERN.findall(rpm_key)

    if len(matches) != 1:
        raise ValueError(
            f'Expected exactly one numeric RPM in key "{rpm_key}", '
            f"but found {len(matches)}."
        )

    rpm = float(matches[0])

    if rpm <= 0:
        raise ValueError(f'RPM key "{rpm_key}" must be positive.')

    return rpm


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """
    Build a JSON object while rejecting duplicate keys.

    Normal json.load() silently keeps only the final occurrence of a
    duplicate key, which would hide a source-data problem.
    """

    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key found: "{key}"')

        result[key] = value

    return result


def _read_float_array(values: Any, *, context: str) -> FloatArray:
    """Convert one JSON array to a finite, one-dimensional float array."""

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{context} could not be converted to floats."
        ) from error

    if array.ndim == 0:
        raise ValueError(f"{context} must be an array, not a scalar.")

    array = array.reshape(-1)

    if array.size == 0:
        raise ValueError(f"{context} is empty.")

    if not np.all(np.isfinite(array)):
        invalid_count = int(np.count_nonzero(~np.isfinite(array)))
        raise ValueError(
            f"{context} contains {invalid_count} non-finite values."
        )

    # Prevent accidental modification of source data after loading.
    array.setflags(write=False)

    return array



def resolve_duplicate_geometries(
    propellers: list[PropellerData],
) -> tuple[list[PropellerData], tuple[str, ...]]:
    """
    Resolve propellers that share the same diameter and pitch.

    Rules:
    1. Unique diameter/pitch entries are always kept.
    2. A manually specified override is used when one exists.
    3. Otherwise, the plain propeller key ending exactly in "E" is kept.
    4. If no plain E propeller exists, every entry in that duplicate
       group is removed.
    """

    geometry_groups: dict[
        tuple[float, float],
        list[PropellerData],
    ] = defaultdict(list)

    # Group propellers by diameter and pitch.
    for propeller in propellers:
        geometry = (
            propeller.diameter_in,
            propeller.pitch_in,
        )
        geometry_groups[geometry].append(propeller)

    kept_propellers: list[PropellerData] = []
    removed_keys: list[str] = []

    # Matches plain keys such as 14x8E, but not 14x8WE,
    # 14x8EP, 14x8E-3, or 14x8E(F2B).
    plain_e_pattern = re.compile(
        r"^\d+(?:\.\d+)?[xX]\d+(?:\.\d+)?E$"
    )

    for geometry, group in geometry_groups.items():
        # Nothing needs to be resolved when the geometry is unique.
        if len(group) == 1:
            kept_propellers.append(group[0])
            continue

        # First check whether this geometry has a manual override.
        override_key = DUPLICATE_GEOMETRY_OVERRIDES.get(geometry)

        if override_key is not None:
            override_matches = [
                propeller
                for propeller in group
                if propeller.key == override_key
            ]

            if len(override_matches) != 1:
                diameter, pitch = geometry
                group_keys = [
                    propeller.key
                    for propeller in group
                ]

                raise ValueError(
                    f"Duplicate geometry {diameter:g}x{pitch:g} "
                    f'requires override key "{override_key}", '
                    f"but it was not found exactly once. "
                    f"Available keys: {group_keys}"
                )

            kept_propeller = override_matches[0]
            kept_propellers.append(kept_propeller)

            removed_keys.extend(
                propeller.key
                for propeller in group
                if propeller is not kept_propeller
            )

            continue

        # If there is no override, look for a normal key ending
        # exactly in "E".
        plain_e_propellers = [
            propeller
            for propeller in group
            if plain_e_pattern.fullmatch(propeller.key)
        ]

        if len(plain_e_propellers) > 1:
            diameter, pitch = geometry
            plain_e_keys = [
                propeller.key
                for propeller in plain_e_propellers
            ]

            raise ValueError(
                f"Duplicate geometry {diameter:g}x{pitch:g} "
                "contains more than one plain E propeller: "
                f"{plain_e_keys}"
            )

        if len(plain_e_propellers) == 1:
            kept_propeller = plain_e_propellers[0]
            kept_propellers.append(kept_propeller)

            removed_keys.extend(
                propeller.key
                for propeller in group
                if propeller is not kept_propeller
            )

            continue

        # No override and no plain E entry exist.
        removed_keys.extend(
            propeller.key
            for propeller in group
        )

    kept_propellers.sort(
        key=lambda propeller: (
            propeller.diameter_in,
            propeller.pitch_in,
            propeller.key,
        )
    )

    removed_keys.sort()

    return kept_propellers, tuple(removed_keys)



def load_prop_data(
    json_path: str | Path = DEFAULT_PROP_DATA_PATH,
) -> tuple[PropellerData, ...]:
    """
    Load and strictly validate the raw propeller JSON database.

    No entries are skipped, no arrays are truncated, and no duplicate
    points are averaged.
    """

    path = Path(json_path)

    if not path.is_file():
        raise FileNotFoundError(f"Prop data file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw_data = json.load(
            file,
            object_pairs_hook=_reject_duplicate_json_keys,
        )

    if not isinstance(raw_data, dict):
        raise ValueError("Top-level JSON value must be an object.")

    if not raw_data:
        raise ValueError("Prop data JSON contains no propellers.")

    propellers: list[PropellerData] = []

    for prop_key, prop_entry in raw_data.items():
        if not isinstance(prop_key, str):
            raise ValueError("Every propeller key must be a string.")

        if not isinstance(prop_entry, dict) or not prop_entry:
            raise ValueError(
                f'Propeller "{prop_key}" must contain RPM entries.'
            )

        diameter_in, pitch_in = parse_prop_geometry(prop_key)

        rpm_tables: list[PropRpmData] = []
        seen_rpms: set[float] = set()

        for rpm_key, rpm_entry in prop_entry.items():
            if not isinstance(rpm_entry, dict):
                raise ValueError(
                    f'Entry "{prop_key}" -> "{rpm_key}" '
                    f"must be an object."
                )

            rpm = parse_rpm(rpm_key)

            if rpm in seen_rpms:
                raise ValueError(
                    f'Propeller "{prop_key}" contains duplicate '
                    f"numeric RPM {rpm:g}."
                )

            seen_rpms.add(rpm)

            context = f'"{prop_key}" -> "{rpm_key}"'

            required_fields = ("V", "Thrust_2", "Torque_2")
            missing_fields = [
                field
                for field in required_fields
                if field not in rpm_entry
            ]

            if missing_fields:
                raise ValueError(
                    f"{context} is missing fields: "
                    f"{', '.join(missing_fields)}"
                )

            velocity_mph = _read_float_array(
                rpm_entry["V"],
                context=f"{context} V",
            )
            thrust_n = _read_float_array(
                rpm_entry["Thrust_2"],
                context=f"{context} Thrust_2",
            )
            torque_nm = _read_float_array(
                rpm_entry["Torque_2"],
                context=f"{context} Torque_2",
            )

            if len(thrust_n) != len(torque_nm):
                raise ValueError(
                    f"{context} has different thrust and torque lengths: "
                    f"Thrust_2={len(thrust_n)}, "
                    f"Torque_2={len(torque_nm)}."
                )

            sample_count = len(thrust_n)

            if len(velocity_mph) < sample_count:
                raise ValueError(
                    f"{context} has fewer velocity values than output values: "
                    f"V={len(velocity_mph)}, "
                    f"Thrust_2={len(thrust_n)}, "
                    f"Torque_2={len(torque_nm)}."
                )

            removed_velocity_count = len(velocity_mph) - sample_count
            velocity_mph = velocity_mph[:sample_count]

            rpm_tables.append(
                PropRpmData(
                    rpm=rpm,
                    velocity_mph=velocity_mph,
                    thrust_n=thrust_n,
                    torque_nm=torque_nm,
                    removed_velocity_count=removed_velocity_count,
                )
            )

        rpm_tables.sort(key=lambda table: table.rpm)

        propellers.append(
            PropellerData(
                key=prop_key,
                diameter_in=diameter_in,
                pitch_in=pitch_in,
                rpm_data=tuple(rpm_tables),
            )
        )

    propellers, removed_duplicate_keys = (
        resolve_duplicate_geometries(propellers)
    )

    if removed_duplicate_keys:
        print("Removed duplicate-geometry propeller variants:")

        for key in removed_duplicate_keys:
            print(f"  {key}")

    return tuple(propellers)


def print_prop_data_summary(
    propellers: tuple[PropellerData, ...],
) -> None:
    """Print the information needed before choosing interpolation rules."""

    if not propellers:
        raise ValueError("Cannot summarize an empty propeller dataset.")

    geometry_to_keys: dict[
        tuple[float, float], list[str]
    ] = defaultdict(list)

    rpm_table_count = 0
    sample_count = 0
    truncated_entry_count = 0
    removed_velocity_count = 0

    velocity_min = np.inf
    velocity_max = -np.inf
    rpm_min = np.inf
    rpm_max = -np.inf

    duplicate_velocity_entries: list[str] = []

    for propeller in propellers:
        geometry = (
            propeller.diameter_in,
            propeller.pitch_in,
        )
        geometry_to_keys[geometry].append(propeller.key)

        for table in propeller.rpm_data:
            rpm_table_count += 1
            sample_count += table.sample_count

            if table.removed_velocity_count > 0:
                truncated_entry_count += 1
                removed_velocity_count += table.removed_velocity_count

            velocity_min = min(
                velocity_min,
                float(np.min(table.velocity_mph)),
            )
            velocity_max = max(
                velocity_max,
                float(np.max(table.velocity_mph)),
            )
            rpm_min = min(rpm_min, table.rpm)
            rpm_max = max(rpm_max, table.rpm)

            _, counts = np.unique(
                table.velocity_mph,
                return_counts=True,
            )

            if np.any(counts > 1):
                duplicate_velocity_entries.append(
                    f"{propeller.key} at {table.rpm:g} RPM"
                )

    duplicate_geometries = {
        geometry: keys
        for geometry, keys in geometry_to_keys.items()
        if len(keys) > 1
    }

    diameters = np.array(
        [prop.diameter_in for prop in propellers],
        dtype=float,
    )
    pitches = np.array(
        [prop.pitch_in for prop in propellers],
        dtype=float,
    )

    print(f"JSON path: {DEFAULT_PROP_DATA_PATH}")
    print(f"Propeller keys: {len(propellers)}")
    print(f"Unique diameter/pitch pairs: {len(geometry_to_keys)}")
    print(f"RPM tables: {rpm_table_count}")
    print(f"Total samples: {sample_count}")
    print(f"Truncated RPM entries: {truncated_entry_count}")
    print(
        f"Unmatched velocity values removed: "
        f"{removed_velocity_count}"
    )
    print(
        f"Diameter bounds: "
        f"{diameters.min():g} to {diameters.max():g} in"
    )
    print(
        f"Pitch bounds: "
        f"{pitches.min():g} to {pitches.max():g} in"
    )
    print(
        f"Velocity bounds: "
        f"{velocity_min:g} to {velocity_max:g} mph"
    )
    print(f"RPM bounds: {rpm_min:g} to {rpm_max:g}")

    print()
    print(
        "Duplicate diameter/pitch pairs: "
        f"{len(duplicate_geometries)}"
    )

    for geometry, keys in sorted(duplicate_geometries.items()):
        diameter, pitch = geometry
        print(f"  {diameter:g}x{pitch:g}: {', '.join(keys)}")

    print()
    print(
        "Prop/RPM entries with repeated velocity values: "
        f"{len(duplicate_velocity_entries)}"
    )

    for entry in duplicate_velocity_entries[:20]:
        print(f"  {entry}")

    if len(duplicate_velocity_entries) > 20:
        remaining = len(duplicate_velocity_entries) - 20
        print(f"  ... and {remaining} more")

def find_array_length_mismatches(
    json_path: str | Path = DEFAULT_PROP_DATA_PATH,
) -> tuple[str, ...]:
    """Return every JSON entry whose data arrays have different lengths."""

    path = Path(json_path)

    with path.open("r", encoding="utf-8") as file:
        raw_data = json.load(
            file,
            object_pairs_hook=_reject_duplicate_json_keys,
        )

    mismatches: list[str] = []

    for prop_key, prop_entry in raw_data.items():
        if not isinstance(prop_entry, dict):
            continue

        for rpm_key, rpm_entry in prop_entry.items():
            if not isinstance(rpm_entry, dict):
                continue

            required_fields = ("V", "Thrust_2", "Torque_2")

            if any(field not in rpm_entry for field in required_fields):
                continue

            velocity_count = len(rpm_entry["V"])
            thrust_count = len(rpm_entry["Thrust_2"])
            torque_count = len(rpm_entry["Torque_2"])

            if len(
                {velocity_count, thrust_count, torque_count}
            ) != 1:
                mismatches.append(
                    f'"{prop_key}" -> "{rpm_key}": '
                    f"V={velocity_count}, "
                    f"Thrust_2={thrust_count}, "
                    f"Torque_2={torque_count}"
                )

    return tuple(mismatches)

def main() -> None:
    propellers = load_prop_data()
    print_prop_data_summary(propellers)


if __name__ == "__main__":
    main()