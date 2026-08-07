"""Run configurable design batches through main or aero_main."""

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from src.aero.main_aero import aero_main
from src.main import main as run_main
from src.mech.main_mech import evaluate_mechanical_module
from src.prop.main_prop import prop_main
from src.prop.prop_database import ContinuousPropDatabase, load_default_prop_database
from src.vectors import DesignVector, ParameterVector


# Change these switches only.
DATASET = "warmed"                 # "warmed" or "randomized"
RUN_TARGET = "aero_main"           # "aero_main" or "main"
MODEL_LABEL = "coarse"             # Used only in the output filename.
AERO_MISSION = 1
MAX_CASES: int | None = None        # Set small while testing.
RESULT_FIELDS: tuple[str, ...] | None = None  # None stores every result field.
STORE_DESIGN_FIELDS = True
DATA_DIR = Path("data_dump/accuracy_designs")
OUTPUT_DIR = Path("data_dump/accuracy_results")
OUTPUT_PATH: Path | None = None


def _constant_aero_inputs(pv: ParameterVector) -> dict:
    """Cache the baseline inputs calculated by main's mech/prop path."""
    cache = DATA_DIR / f"aero_main_constant_inputs_m{AERO_MISSION}.json"
    if cache.exists():
        inputs = json.loads(cache.read_text(encoding="utf-8"))
        if "flight_time_fit" in inputs:
            return inputs

    baseline = DesignVector()
    mech = evaluate_mechanical_module(baseline, parameter_vector=pv)
    resolved = replace(
        baseline,
        fuselage_width=mech.resolved_fuselage_width_m,
        fuselage_height=mech.resolved_fuselage_height_m,
    )
    props = mech.for_mission(f"M{AERO_MISSION}")
    thrust, flight_time_fit = prop_main(resolved, pv, mission=AERO_MISSION)
    inputs = {
        "thrust_velocity": list(map(float, thrust)),
        "flight_time_fit": list(map(float, flight_time_fit)),
        "cg": list(map(float, props.cg_m)),
        "inertia_matrix": np.asarray(props.inertia_tensor_kg_m2, dtype=float).tolist(),
        "mass": float(props.total_mass_kg),
    }
    cache.write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    return inputs


def _result(
    design: DesignVector,
    pv: ParameterVector,
    aero_inputs: dict | None,
    prop_database: ContinuousPropDatabase | None,
) -> dict:
    if RUN_TARGET == "main":
        score, breakdown = run_main(
            design, pv, disp_res=False, prop_database=prop_database
        )
        return {"total_score": float(score), **{f"breakdown_{i}": float(v) for i, v in enumerate(breakdown)}}
    if RUN_TARGET == "aero_main" and aero_inputs is not None:
        score = aero_main(
            design_vector=design,
            parameter_vector=pv,
            thrust_velocity=tuple(aero_inputs["thrust_velocity"]),
            flight_time_fit=tuple(aero_inputs["flight_time_fit"]),
            mission=AERO_MISSION,
            cg=tuple(aero_inputs["cg"]),
            inertia_matrix=np.asarray(aero_inputs["inertia_matrix"], dtype=float),
            mass=float(aero_inputs["mass"]),
            disp_res=False,
            debug=False,
        )
        return asdict(score)
    raise ValueError("RUN_TARGET must be 'main' or 'aero_main'.")


def main() -> None:
    dataset_path = DATA_DIR / f"{DATASET}_design_vectors.json"
    designs = json.loads(dataset_path.read_text(encoding="utf-8"))
    if MAX_CASES is not None:
        designs = designs[:MAX_CASES]

    pv = ParameterVector()
    aero_inputs = _constant_aero_inputs(pv) if RUN_TARGET == "aero_main" else None
    prop_database = load_default_prop_database() if RUN_TARGET == "main" else None
    rows = []
    for index, values in enumerate(designs):
        start = perf_counter()
        row = {"case_id": index, "status": "ok"}
        if STORE_DESIGN_FIELDS:
            row.update({f"design_{key}": value for key, value in values.items()})
        try:
            result = _result(DesignVector(**values), pv, aero_inputs, prop_database)
            if RESULT_FIELDS is not None:
                result = {key: result[key] for key in RESULT_FIELDS}
            row.update(result)
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        row["elapsed_seconds"] = perf_counter() - start
        rows.append(row)
        print(f"[{index + 1}/{len(designs)}] {row['status']} ({row['elapsed_seconds']:.4f} s)", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_PATH or OUTPUT_DIR / f"{DATASET}_{RUN_TARGET}_{MODEL_LABEL}.csv"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Stored {len(rows)} results in {output}")


if __name__ == "__main__":
    main()
