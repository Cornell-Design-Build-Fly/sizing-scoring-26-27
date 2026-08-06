"""Generate deterministic design sets for coarse/full accuracy studies."""

import json
from pathlib import Path

import numpy as np

from src.vectors import OPT_VARS


COUNT = 150
SEED = 26027
OUTPUT_DIR = Path("data_dump/accuracy_designs")


def randomized(rng: np.random.Generator) -> list[dict[str, float]]:
    bounds = dict(OPT_VARS)
    rows = []
    for _ in range(COUNT):
        row = {name: float(rng.uniform(*limit)) for name, limit in bounds.items()}
        row["ducks_num"] = int(rng.integers(3, 11))
        row["pucks_num"] = int(rng.integers(1, 11))
        rows.append(row)
    return rows


def warmed(rng: np.random.Generator) -> list[dict[str, float]]:
    rows = []
    for _ in range(COUNT):
        span = rng.uniform(1.10, 1.60)
        chord = np.clip(span / rng.uniform(4.2, 6.0), 0.20, 0.33)
        ducks = int(rng.integers(3, 8))
        pucks = int(rng.integers(1, 6))
        rows.append({
            "wing_span": float(span),
            "wing_chord": float(chord),
            "tail_arm": float(np.clip(rng.uniform(2.3, 3.1) * chord, 0.55, 0.90)),
            "nose_length": float(np.clip(rng.uniform(0.65, 0.95) * chord, 0.14, 0.29)),
            "ducks_num": ducks,
            "pucks_num": pucks,
            "banner_length": float(np.clip(2.2 + 1.2 * (span - 1.1) + rng.normal(0, 0.35), 1.5, 4.2)),
            "batt_capacity": float(np.clip(3.0 + 0.35 * ducks + 0.18 * pucks + rng.normal(0, 0.5), 3.5, 7.5)),
        })
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {
        "randomized_design_vectors.json": randomized(rng),
        "warmed_design_vectors.json": warmed(rng),
    }
    for name, rows in datasets.items():
        path = OUTPUT_DIR / name
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(rows)} designs to {path}")


if __name__ == "__main__":
    main()
