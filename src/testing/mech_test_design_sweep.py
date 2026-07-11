"""Run three auditable mechanical-module design cases and save 2-D layouts.

From the repository root::

    python -m src.testing.mech_test_design_sweep

The script prints the normal mission mass-property outputs, prints and saves a
unique mass-element ledger for each design, and writes top/side M2 layout PNGs
under ``data_dump/mech_design_sweep``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from src.mech import MechanicalModuleConfig, MechanicalResult, evaluate_mechanical_module
from src.mech.mass_properties import geometry_stations
from src.vectors import DesignVector


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data_dump" / "mech_design_sweep"


@dataclass(frozen=True)
class DesignCase:
    """One named design-vector case for the report."""

    slug: str
    label: str
    design_vector: DesignVector


# Geometry and battery are intentionally held at their baseline values so the
# three reports isolate the effects of small, medium, and large payloads.
DESIGN_CASES = (
    DesignCase(
        slug="small_payload",
        label="Small payload",
        design_vector=DesignVector(
            ducks_num=3,
            pucks_num=1,
            banner_length=0.50,
        ),
    ),
    DesignCase(
        slug="medium_payload",
        label="Medium payload",
        design_vector=DesignVector(
            ducks_num=7,
            pucks_num=5,
            banner_length=2.75,
        ),
    ),
    DesignCase(
        slug="large_payload",
        label="Large payload",
        design_vector=DesignVector(
            ducks_num=10,
            pucks_num=9,
            banner_length=5.00,
        ),
    ),
)


Evaluator = Callable[[DesignVector, MechanicalModuleConfig | None], MechanicalResult]


def _validate_result(case: DesignCase, result: MechanicalResult) -> None:
    """Catch broken or incomplete results before writing a reassuring report."""

    assert set(result.missions) == {"M1", "M2", "M3"}
    assert len(result.all_items) > 0
    for mission in ("M1", "M2", "M3"):
        properties = result.for_mission(mission)
        assert properties.placement_feasible, f"{case.label} {mission} placement failed"
        assert properties.static_margin_feasible, (
            f"{case.label} {mission} static margin is outside its configured range"
        )
        assert properties.total_mass_kg > 0.0
        assert np.isclose(
            properties.total_mass_kg,
            sum(item.mass_kg for item in properties.items),
        )
        assert np.all(np.isfinite(properties.cg_m))
        assert np.all(np.isfinite(properties.inertia_tensor_kg_m2))
        assert np.allclose(
            properties.inertia_tensor_kg_m2,
            properties.inertia_tensor_kg_m2.T,
            atol=1e-12,
        )


def _require_assertions_enabled() -> None:
    if not __debug__:
        raise RuntimeError(
            "This executable test uses assertions; run it without Python's -O flag."
        )


def _write_mass_ledger(path: Path, result: MechanicalResult) -> None:
    """Save every unique mass element, including its location and dimensions."""

    fieldnames = (
        "element_id",
        "name",
        "category",
        "missions",
        "mass_kg",
        "x_m",
        "y_m",
        "z_m",
        "length_x_m",
        "width_y_m",
        "height_z_m",
        "notes",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for element_id, item in enumerate(result.all_items, start=1):
            writer.writerow(
                {
                    "element_id": element_id,
                    "name": item.name,
                    "category": item.category,
                    "missions": ",".join(sorted(item.missions)),
                    "mass_kg": f"{item.mass_kg:.12g}",
                    "x_m": f"{item.position_m[0]:.12g}",
                    "y_m": f"{item.position_m[1]:.12g}",
                    "z_m": f"{item.position_m[2]:.12g}",
                    "length_x_m": f"{item.dimensions_m[0]:.12g}",
                    "width_y_m": f"{item.dimensions_m[1]:.12g}",
                    "height_z_m": f"{item.dimensions_m[2]:.12g}",
                    "notes": item.notes,
                }
            )


def _write_mission_summary(path: Path, result: MechanicalResult) -> None:
    """Save the normal mission outputs in a spreadsheet-friendly form."""

    fieldnames = (
        "mission",
        "total_mass_kg",
        "weight_n",
        "cg_x_m",
        "cg_y_m",
        "cg_z_m",
        "static_margin",
        "static_margin_percent",
        "static_margin_feasible",
        "placement_feasible",
        "inertia_xx_kg_m2",
        "inertia_xy_kg_m2",
        "inertia_xz_kg_m2",
        "inertia_yy_kg_m2",
        "inertia_yz_kg_m2",
        "inertia_zz_kg_m2",
        "warnings",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for mission in ("M1", "M2", "M3"):
            properties = result.for_mission(mission)
            inertia = properties.inertia_tensor_kg_m2
            writer.writerow(
                {
                    "mission": mission,
                    "total_mass_kg": f"{properties.total_mass_kg:.12g}",
                    "weight_n": f"{properties.weight_n:.12g}",
                    "cg_x_m": f"{properties.cg_m[0]:.12g}",
                    "cg_y_m": f"{properties.cg_m[1]:.12g}",
                    "cg_z_m": f"{properties.cg_m[2]:.12g}",
                    "static_margin": f"{properties.static_margin:.12g}",
                    "static_margin_percent": f"{100 * properties.static_margin:.12g}",
                    "static_margin_feasible": properties.static_margin_feasible,
                    "placement_feasible": properties.placement_feasible,
                    "inertia_xx_kg_m2": f"{inertia[0, 0]:.12g}",
                    "inertia_xy_kg_m2": f"{inertia[0, 1]:.12g}",
                    "inertia_xz_kg_m2": f"{inertia[0, 2]:.12g}",
                    "inertia_yy_kg_m2": f"{inertia[1, 1]:.12g}",
                    "inertia_yz_kg_m2": f"{inertia[1, 2]:.12g}",
                    "inertia_zz_kg_m2": f"{inertia[2, 2]:.12g}",
                    "warnings": " | ".join(properties.warnings),
                }
            )


def _print_report(case: DesignCase, result: MechanicalResult) -> None:
    """Print mission outputs and one unique component ledger for manual checks."""

    design = case.design_vector
    print("\n" + "=" * 118)
    print(f"{case.label} ({case.slug})")
    print("Design vector [optimizer order]")
    for name, value in zip(design.opt_names(), design.to_array()):
        print(f"  {name:<16} {value:.6g}")
    print(f"Neutral point x:             {result.neutral_point_x_m:.6f} m")
    print(f"Target CG x:                 {result.target_cg_x_m:.6f} m")
    print(
        "Acceptable CG x range:      "
        f"[{result.acceptable_cg_x_range_m[0]:.6f}, "
        f"{result.acceptable_cg_x_range_m[1]:.6f}] m"
    )
    print(f"Electronics CM:              {result.electronics_position_m} m")
    print(
        f"Electronics envelope x:      [{result.electronics_layout.front_edge_x_m:.6f}, "
        f"{result.electronics_layout.back_edge_x_m:.6f}] m"
    )

    for mission in ("M1", "M2", "M3"):
        properties = result.for_mission(mission)
        print(
            f"\n{mission}: mass={properties.total_mass_kg:.6f} kg, "
            f"weight={properties.weight_n:.6f} N"
        )
        print(f"    CG={np.array2string(properties.cg_m, precision=8)} m")
        print(
            f"    static margin={100 * properties.static_margin:.4f}% "
            f"(feasible={properties.static_margin_feasible}), "
            f"placement feasible={properties.placement_feasible}"
        )
        print("    inertia about CG [kg m^2]:")
        for row in properties.inertia_tensor_kg_m2:
            print("      " + " ".join(f"{value: .9f}" for value in row))
        for warning in properties.warnings:
            print(f"    warning: {warning}")

    print("\nUnique mass-element ledger (IDs match the 2-D M2 plot labels)")
    print(
        f"{'ID':>3}  {'Name':<42} {'Mass [kg]':>11} "
        f"{'x [m]':>10} {'y [m]':>10} {'z [m]':>10}  {'Missions':<8} Category"
    )
    print("-" * 118)
    for element_id, item in enumerate(result.all_items, start=1):
        print(
            f"{element_id:>3}  {item.name[:42]:<42} {item.mass_kg:>11.6f} "
            f"{item.position_m[0]:>10.6f} {item.position_m[1]:>10.6f} "
            f"{item.position_m[2]:>10.6f}  "
            f"{','.join(sorted(item.missions)):<8} {item.category}"
        )
    if result.warnings:
        print("\nResult warnings")
        for warning in result.warnings:
            print(f"  - {warning}")


def _item_appearance(item) -> tuple[str, float]:
    if item.category == "mission_2_fractional_payload":
        return "black", 1.0
    if item.name.startswith("Duck "):
        return "#e6a700", 0.65
    if item.name.startswith("Puck "):
        return "#2878b5", 0.65
    if item.category == "propulsion_and_electronics":
        return "#d9534f", 0.55
    if item.category == "controls":
        return "#4daf4a", 0.40
    if item.category == "integration":
        return "#9467bd", 0.35
    return "#7f8c8d", 0.22


def _item_hatch(item) -> str | None:
    if item.name.startswith("Duck "):
        return "///"
    if item.name.startswith("Puck "):
        return "xx"
    return None


def _draw_projection(
    *,
    axis,
    items,
    element_ids: dict[int, int],
    vertical_coordinate: int,
    vertical_label: str,
) -> None:
    """Draw axis-aligned component boxes in either x-y or x-z."""

    co_located_ids: dict[tuple[float, float], list[int]] = {}
    for item in items:
        color, alpha = _item_appearance(item)
        hatch = _item_hatch(item)
        x_position = float(item.position_m[0])
        vertical_position = float(item.position_m[vertical_coordinate])
        location_key = (round(x_position, 12), round(vertical_position, 12))
        co_located_ids.setdefault(location_key, []).append(element_ids[id(item)])
        length = float(item.dimensions_m[0])
        height = float(item.dimensions_m[vertical_coordinate])
        if length > 0.0 and height > 0.0:
            axis.add_patch(
                Rectangle(
                    (x_position - 0.5 * length, vertical_position - 0.5 * height),
                    length,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.8,
                    alpha=alpha,
                    hatch=hatch,
                )
            )
        else:
            axis.scatter(
                [x_position],
                [vertical_position],
                marker="*",
                s=60,
                color=color,
                zorder=5,
            )

    for (x_position, vertical_position), ids in co_located_ids.items():
        axis.annotate(
            "/".join(str(element_id) for element_id in ids),
            (x_position, vertical_position),
            fontsize=5.5,
            ha="center",
            va="center",
            color="black",
            zorder=6,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.35,
                "pad": 0.3,
            },
        )

    axis.set_xlabel("x [m] (positive aft)")
    axis.set_ylabel(vertical_label)
    axis.grid(True, linewidth=0.4, alpha=0.4)
    axis.autoscale_view()
    axis.margins(x=0.04, y=0.08)
    axis.set_aspect("equal", adjustable="box")


def _save_m2_layout(
    path: Path,
    case: DesignCase,
    result: MechanicalResult,
) -> None:
    """Save top and side projections of every M2 mass element."""

    m2 = result.for_mission("M2")
    stations = geometry_stations(case.design_vector)
    tail_le_x = min(
        stations.horizontal_tail_le_x_m,
        stations.vertical_tail_le_x_m,
    )
    element_ids = {id(item): index for index, item in enumerate(result.all_items, 1)}
    figure, axes = plt.subplots(2, 1, figsize=(12, 11), constrained_layout=True)
    _draw_projection(
        axis=axes[0],
        items=m2.items,
        element_ids=element_ids,
        vertical_coordinate=1,
        vertical_label="y [m] (top view)",
    )
    _draw_projection(
        axis=axes[1],
        items=m2.items,
        element_ids=element_ids,
        vertical_coordinate=2,
        vertical_label="z [m] (side view)",
    )

    for axis in axes:
        axis.axvline(
            m2.cg_m[0],
            color="#2ca02c",
            linestyle="--",
            linewidth=1.5,
            label="M2 CG",
        )
        axis.axvline(
            result.neutral_point_x_m,
            color="#6f42c1",
            linestyle=":",
            linewidth=1.5,
            label="Neutral point",
        )
        axis.axvline(
            result.electronics_layout.back_edge_x_m,
            color="#d9534f",
            linestyle="-.",
            linewidth=1.0,
            label="Electronics back edge",
        )
        axis.axvline(
            tail_le_x,
            color="#333333",
            linestyle="-.",
            linewidth=1.0,
            label="Tail leading edge",
        )

    axes[0].set_title("Top projection (x-y)")
    axes[1].set_title("Side projection (x-z)")
    legend_handles = [
        Patch(facecolor="#e6a700", hatch="///", label="Whole ducks"),
        Patch(facecolor="#2878b5", hatch="xx", label="Whole pucks"),
        Patch(facecolor="#d9534f", alpha=0.55, label="Electronics/propulsion"),
        Patch(facecolor="#7f8c8d", alpha=0.22, label="Other permanent mass"),
        Line2D([], [], color="#2ca02c", linestyle="--", label="M2 CG"),
        Line2D([], [], color="#6f42c1", linestyle=":", label="Neutral point"),
        Line2D([], [], color="#d9534f", linestyle="-.", label="Electronics back edge"),
        Line2D([], [], color="#333333", linestyle="-.", label="Tail leading edge"),
    ]
    if any(
        item.category == "mission_2_fractional_payload" for item in m2.items
    ):
        legend_handles.insert(
            4,
            Line2D(
                [],
                [],
                color="black",
                marker="*",
                linestyle="None",
                label="Fractional point mass",
            ),
        )
    axes[0].legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=5,
        fontsize=8,
    )
    figure.suptitle(
        f"{case.label}: Mission 2 mass-element layout\n"
        f"mass={m2.total_mass_kg:.4f} kg, CG={np.array2string(m2.cg_m, precision=4)}, "
        f"static margin={100 * m2.static_margin:.2f}%",
        fontsize=12,
    )
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def run_design_cases(
    cases: Sequence[DesignCase],
    *,
    evaluator: Evaluator = evaluate_mechanical_module,
    config: MechanicalModuleConfig | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, MechanicalResult]:
    """Evaluate, verify, print, and visualize a collection of design cases."""

    _require_assertions_enabled()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, MechanicalResult] = {}
    for case in cases:
        result = evaluator(case.design_vector, config)
        _validate_result(case, result)
        _print_report(case, result)
        ledger_path = output_dir / f"{case.slug}_mass_elements.csv"
        summary_path = output_dir / f"{case.slug}_mission_summary.csv"
        plot_path = output_dir / f"{case.slug}_m2_layout.png"
        _write_mass_ledger(ledger_path, result)
        _write_mission_summary(summary_path, result)
        _save_m2_layout(plot_path, case, result)
        print(f"\nSaved mass ledger:    {ledger_path}")
        print(f"Saved mission output: {summary_path}")
        print(f"Saved 2-D layout:     {plot_path}")
        results[case.slug] = result
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV and PNG outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    arguments = parser.parse_args()
    run_design_cases(DESIGN_CASES, output_dir=arguments.output_dir)


if __name__ == "__main__":
    main()
