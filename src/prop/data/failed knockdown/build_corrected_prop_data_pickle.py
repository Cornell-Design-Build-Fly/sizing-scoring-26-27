from __future__ import annotations

"""
Build a SEPARATE corrected prop database pickle.

This keeps your existing raw cache:
    src/prop/data/prop_data_continuous.pkl

and creates a new corrected cache:
    src/prop/data/prop_data_continuous_corrected.pkl

Run from the repo root:
    python -m src.prop.build_corrected_prop_data_pickle

This does not modify prop_main(). For comparisons, load the raw database and the
corrected database separately, then pass them into prop_main(..., prop_database=...).
"""

from pathlib import Path
import hashlib
import pickle
import time
from typing import Any

from src.prop.prop_correction import (
    DEFAULT_CORRECTION_CSV_PATH,
    apply_static_thrust_correction,
)

# Prefer the separated prop database file. Fall back to main_prop for compatibility
# with older branches where the database code still lived in main_prop.py.
try:
    from src.prop.prop_database import (
        DEFAULT_PROP_DATA_PATH,
        ContinuousPropDatabase,
        load_prop_data_points,
    )
except ImportError:
    from src.prop.main_prop import (  # type: ignore
        DEFAULT_PROP_DATA_PATH,
        ContinuousPropDatabase,
        load_prop_data_points,
    )


DATA_DIR = Path(__file__).resolve().parent / "data"

CORRECTED_PROP_CACHE_PATH = DATA_DIR / "prop_data_continuous_corrected.pkl"

CORRECTED_CACHE_VERSION = 1

# "nearest" means untested props use the nearest tested prop's correction curve.
# "none" means untested props are left uncorrected with factor 1.0.
UNTESTED_PROP_MODE = "nearest"

# False makes the correction a knockdown only.
# Any correction_factor above 1.0 is capped at 1.0.
ALLOW_THRUST_INCREASE = False


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def build_corrected_prop_database(
    json_path: Path = DEFAULT_PROP_DATA_PATH,
    correction_csv_path: Path = DEFAULT_CORRECTION_CSV_PATH,
    untested_prop_mode: str = UNTESTED_PROP_MODE,
    allow_thrust_increase: bool = ALLOW_THRUST_INCREASE,
) -> ContinuousPropDatabase:
    print("Loading raw prop data from JSON...")
    raw_data = load_prop_data_points(json_path)

    print("Applying static thrust correction to raw thrust data...")
    corrected_data = apply_static_thrust_correction(
        data=raw_data,
        correction_csv_path=correction_csv_path,
        untested_prop_mode=untested_prop_mode,  # type: ignore[arg-type]
        allow_thrust_increase=allow_thrust_increase,
        verbose=True,
    )

    print("Building corrected ContinuousPropDatabase...")
    return ContinuousPropDatabase(corrected_data)


def save_corrected_prop_database(
    prop_database: ContinuousPropDatabase,
    cache_path: Path = CORRECTED_PROP_CACHE_PATH,
    json_path: Path = DEFAULT_PROP_DATA_PATH,
    correction_csv_path: Path = DEFAULT_CORRECTION_CSV_PATH,
    untested_prop_mode: str = UNTESTED_PROP_MODE,
    allow_thrust_increase: bool = ALLOW_THRUST_INCREASE,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "version": CORRECTED_CACHE_VERSION,
        "kind": "corrected_continuous_prop_database",
        "prop_database": prop_database,
        "raw_prop_json_path": str(json_path),
        "raw_prop_json_sha256": file_sha256(json_path),
        "correction_csv_path": str(correction_csv_path),
        "correction_csv_sha256": file_sha256(correction_csv_path),
        "untested_prop_mode": untested_prop_mode,
        "allow_thrust_increase": allow_thrust_increase,
        "notes": (
            "Thrust has been corrected using static-thrust correction factors. "
            "Torque data is unchanged. The raw prop_data_continuous.pkl file is not modified."
        ),
    }

    print(f"Saving corrected prop database to: {cache_path}")
    with cache_path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_corrected_prop_database(
    cache_path: Path = CORRECTED_PROP_CACHE_PATH,
) -> ContinuousPropDatabase:
    """
    Loads the corrected .pkl file.

    Use this for comparison scripts:
        corrected_db = load_corrected_prop_database()
        result = prop_main(..., prop_database=corrected_db)
    """
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Could not find corrected prop database cache: {cache_path}\\n"
            "Run python -m src.prop.build_corrected_prop_data_pickle first."
        )

    with cache_path.open("rb") as file:
        payload = pickle.load(file)

    if isinstance(payload, dict) and "prop_database" in payload:
        return payload["prop_database"]

    # Compatibility fallback in case someone pickles only the object later.
    return payload


def main() -> None:
    start = time.perf_counter()

    print("Building corrected prop database cache.")
    print(f"Raw JSON:        {DEFAULT_PROP_DATA_PATH}")
    print(f"Correction CSV:  {DEFAULT_CORRECTION_CSV_PATH}")
    print(f"Output .pkl:     {CORRECTED_PROP_CACHE_PATH}")
    print(f"Untested mode:   {UNTESTED_PROP_MODE}")
    print(f"Allow increase:  {ALLOW_THRUST_INCREASE}")
    print()

    corrected_database = build_corrected_prop_database(
        json_path=DEFAULT_PROP_DATA_PATH,
        correction_csv_path=DEFAULT_CORRECTION_CSV_PATH,
        untested_prop_mode=UNTESTED_PROP_MODE,
        allow_thrust_increase=ALLOW_THRUST_INCREASE,
    )

    save_corrected_prop_database(
        prop_database=corrected_database,
        cache_path=CORRECTED_PROP_CACHE_PATH,
        json_path=DEFAULT_PROP_DATA_PATH,
        correction_csv_path=DEFAULT_CORRECTION_CSV_PATH,
        untested_prop_mode=UNTESTED_PROP_MODE,
        allow_thrust_increase=ALLOW_THRUST_INCREASE,
    )

    elapsed = time.perf_counter() - start
    print()
    print("Corrected prop database cache complete.")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Raw cache was not changed: {DATA_DIR / 'prop_data_continuous.pkl'}")
    print(f"Corrected cache created:  {CORRECTED_PROP_CACHE_PATH}")


if __name__ == "__main__":
    main()
