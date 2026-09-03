from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from src.opt.winglets.cad_export import export_print_mesh_geometry, export_step_geometry
from src.opt.winglets.geometry import WingletGeometry, make_winglet_airplane
from src.vectors import DesignVector


RUN_DIR: Path | None = None
BASE_OUTPUT_DIR = Path("data_dump") / "opt_winglets"

# Options: "plotly", "matplotlib", "three_view", "pyvista", "trimesh".
# "plotly" gives the most reliable browser-based interactive 3D model.
VIEW_MODE = "plotly"

THIN_WINGS = False
EXPORT_STEP = False
EXPORT_SINGLE_WINGLET_STEP = True
EXPORT_FULL_AIRPLANE_STEP = False
EXPORT_STL = False


def _latest_run_dir(base_output_dir: Path) -> Path:
    run_dirs = [
        path
        for path in base_output_dir.glob("run_*")
        if path.is_dir() and (path / "winglet_optimization_report.json").exists()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No winglet optimization runs found in {base_output_dir}.")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def _design_vector_from_report(report: dict[str, Any]) -> DesignVector:
    design_vector_data = report["config"]["design_vector"]
    init_field_names = {field.name for field in fields(DesignVector) if field.init}
    return DesignVector(
        **{
            name: value
            for name, value in design_vector_data.items()
            if name in init_field_names
        }
    )


def _winglet_from_report(report: dict[str, Any]) -> WingletGeometry:
    return WingletGeometry(**report["result"]["winglet"])


def load_run_airplane(run_dir: Path):
    report_path = run_dir / "winglet_optimization_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = report["config"]
    airplane = make_winglet_airplane(
        _design_vector_from_report(report),
        _winglet_from_report(report),
        name=f"Optimized Winglet Candidate ({run_dir.name})",
        winglet_airfoil=config["winglet_airfoil"],
    )
    return airplane, report


def main() -> None:
    run_dir = RUN_DIR or _latest_run_dir(BASE_OUTPUT_DIR)
    airplane, report = load_run_airplane(run_dir)
    winglet = report["result"]["winglet"]
    print(f"Viewing: {run_dir}")
    print(f"CD: {report['result']['baseline_cd']:.5f} -> {report['result']['optimized_cd']:.5f}")
    print(f"Drag reduction: {report['result']['drag_reduction_percent']:.2f}%")
    print(f"Winglet: {winglet}")

    if EXPORT_STEP:
        step_path = export_step_geometry(
            _design_vector_from_report(report),
            _winglet_from_report(report),
            run_dir,
            winglet_airfoil=report["config"]["winglet_airfoil"],
            single_winglet=EXPORT_SINGLE_WINGLET_STEP,
            full_airplane=EXPORT_FULL_AIRPLANE_STEP,
        )
        print(f"STEP: {step_path}")
    if EXPORT_STL:
        stl_path = export_print_mesh_geometry(
            _design_vector_from_report(report),
            _winglet_from_report(report),
            run_dir,
            winglet_airfoil=report["config"]["winglet_airfoil"],
        )
        print(f"STL: {stl_path}")

    if VIEW_MODE == "three_view":
        import matplotlib.pyplot as plt

        airplane.draw_three_view(style="shaded", show=False)
        plt.show()
    else:
        airplane.draw(backend=VIEW_MODE, thin_wings=THIN_WINGS, show=True)


if __name__ == "__main__":
    main()
