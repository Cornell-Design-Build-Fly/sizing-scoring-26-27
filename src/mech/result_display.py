"""CSV and image output for mechanical mass placements."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.mech.models import MechanicalResult, MissionMassProperties


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data_dump" / "mech_results"


def _write_placement_csv(path: Path, mission: "MissionMassProperties") -> None:
    """Write every mass element used by one mission."""

    fieldnames = (
        "element_id",
        "name",
        "category",
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
        for element_id, item in enumerate(mission.items, start=1):
            writer.writerow(
                {
                    "element_id": element_id,
                    "name": item.name,
                    "category": item.category,
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


def _item_color(item) -> str:
    """Return a consistent color for each mass-element category."""

    if item.category == "mission_2_payload":
        return "#e6a700"
    if item.category == "mission_3_payload":
        return "#17a589"
    if item.category == "release_mechanism":
        return "#2878b5"
    if item.category == "propulsion_and_electronics":
        return "#d9534f"
    if item.category == "controls":
        return "#4daf4a"
    if item.category == "integration":
        return "#9467bd"
    return "#7f8c8d"


def _draw_projection(axis, mission, vertical_index: int, vertical_label: str) -> None:
    """Draw the axis-aligned mass boxes in an x-y or x-z projection."""

    from matplotlib.patches import Rectangle

    for element_id, item in enumerate(mission.items, start=1):
        x = float(item.position_m[0])
        vertical = float(item.position_m[vertical_index])
        length = float(item.dimensions_m[0])
        height = float(item.dimensions_m[vertical_index])
        color = _item_color(item)

        if length > 0.0 and height > 0.0:
            axis.add_patch(
                Rectangle(
                    (x - length / 2.0, vertical - height / 2.0),
                    length,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.8,
                    alpha=0.45,
                )
            )
        else:
            axis.scatter([x], [vertical], marker="*", s=55, color=color, zorder=5)

        axis.annotate(
            str(element_id),
            (x, vertical),
            fontsize=5.5,
            ha="center",
            va="center",
            color="black",
            zorder=6,
        )

    axis.axvline(
        float(mission.cg_m[0]),
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        label="CG",
    )
    axis.set_xlabel("x [m] (positive aft)")
    axis.set_ylabel(vertical_label)
    axis.grid(True, linewidth=0.4, alpha=0.4)
    axis.autoscale_view()
    axis.margins(x=0.04, y=0.08)
    axis.set_aspect("equal", adjustable="box")


def _write_placement_image(path: Path, mission_name: str, mission) -> None:
    """Write top and side mass-placement projections for one mission."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(12, 11), constrained_layout=True)
    _draw_projection(axes[0], mission, 1, "y [m] (top view)")
    _draw_projection(axes[1], mission, 2, "z [m] (side view)")
    axes[0].set_title("Top projection (x-y)")
    axes[1].set_title("Side projection (x-z)")
    axes[0].legend(loc="best")
    figure.suptitle(
        f"{mission_name} mass placements\n"
        f"mass={mission.total_mass_kg:.4f} kg, "
        f"CG=({mission.cg_m[0]:.4f}, {mission.cg_m[1]:.4f}, "
        f"{mission.cg_m[2]:.4f}) m",
        fontsize=12,
    )
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_mass_placement_results(
    result: "MechanicalResult",
    output_dir: Path | None = None,
) -> tuple[Path, ...]:
    """Save M2/M3 placement CSVs and images and return their paths."""

    destination = output_dir or DEFAULT_OUTPUT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    for mission_name in ("M2", "M3"):
        mission = result.for_mission(mission_name)
        stem = mission_name.lower() + "_mass_placements"
        csv_path = destination / f"{stem}.csv"
        image_path = destination / f"{stem}.png"
        _write_placement_csv(csv_path, mission)
        _write_placement_image(image_path, mission_name, mission)
        written_paths.extend((csv_path, image_path))

    print(f"[mech] Saved M2/M3 mass-placement results to {destination}", flush=True)
    return tuple(written_paths)


__all__ = ["DEFAULT_OUTPUT_DIR", "write_mass_placement_results"]
