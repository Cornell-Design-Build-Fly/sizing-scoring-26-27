from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from src.opt.score import (
    DEFAULT_SCORING_REFERENCES,
    ScoringReferences,
    scoring_reference_values,
)

if TYPE_CHECKING:
    from src.opt.topline_opt import ToplineConfig


DEFAULT_OUTPUT_DIR = Path("data_dump") / "opt_sensitivity"
SENSITIVITY_PERCENTAGES = (-0.25, -0.10, 0.10, 0.25)
SENSITIVITY_MAXITER = 40
SENSITIVITY_POPSIZE = 12
SENSITIVITY_WORKERS = 12
SENSITIVITY_SEED = 20260810

SENSITIVITY_FIELDS: dict[str, tuple[str, ...]] = {
    "mission_time_limit": ("seconds_per_mission",),
    "ground_best_time": ("best_gm_time_s",),
    "m2_best_profit": ("best_m2_profit",),
    "m3_best_lap_time": ("best_m3_lap_time_s",),
    "m3_best_banner": ("best_banner_length_in",),
}

PRIMARY_REFERENCE_FIELDS = (
    "seconds_per_mission",
    "best_gm_time_s",
    "best_m2_profit",
    "best_m3_lap_time_s",
    "best_banner_length_in",
)


@dataclass(frozen=True)
class SensitivityCase:
    """One scoring-reference perturbation to optimize."""

    name: str
    label: str
    changed_group: str
    percent_change: float
    scoring_references: ScoringReferences


def _slug_percent(percent_change: float) -> str:
    signed_percent = int(round(percent_change * 100.0))
    if signed_percent == 0:
        return "base"
    prefix = "p" if signed_percent > 0 else "m"
    return f"{prefix}{abs(signed_percent)}"


def _scale_fields(
    refs: ScoringReferences,
    fields: Iterable[str],
    percent_change: float,
) -> ScoringReferences:
    values = asdict(refs)
    multiplier = 1.0 + percent_change
    if multiplier <= 0.0:
        raise ValueError("percent_change must keep all references positive.")
    for field in fields:
        values[field] = float(values[field]) * multiplier
    return ScoringReferences(**values)


def build_reference_sensitivity_cases(
    percentages: Iterable[float] = (-0.25, -0.10, 0.10, 0.25),
    *,
    base_refs: ScoringReferences = DEFAULT_SCORING_REFERENCES,
    include_baseline: bool = True,
    include_all_references: bool = True,
) -> list[SensitivityCase]:
    """Build one-at-a-time reference shifts for the topline optimizer.

    Percentages are fractional changes, so ``0.10`` means a 10% increase.
    """

    cases: list[SensitivityCase] = []
    if include_baseline:
        cases.append(
            SensitivityCase(
                name="baseline",
                label="baseline",
                changed_group="baseline",
                percent_change=0.0,
                scoring_references=base_refs,
            )
        )

    for group, fields in SENSITIVITY_FIELDS.items():
        for percent_change in percentages:
            cases.append(
                SensitivityCase(
                    name=f"{group}_{_slug_percent(percent_change)}",
                    label=f"{group} {percent_change:+.0%}",
                    changed_group=group,
                    percent_change=float(percent_change),
                    scoring_references=_scale_fields(
                        base_refs,
                        fields,
                        float(percent_change),
                    ),
                )
            )

    if include_all_references:
        for percent_change in percentages:
            cases.append(
                SensitivityCase(
                    name=f"all_refs_{_slug_percent(percent_change)}",
                    label=f"all_refs {percent_change:+.0%}",
                    changed_group="all_refs",
                    percent_change=float(percent_change),
                    scoring_references=_scale_fields(
                        base_refs,
                        PRIMARY_REFERENCE_FIELDS,
                        float(percent_change),
                    ),
                )
            )

    return cases


def _latest_run_dir(output_dir: Path) -> Path:
    run_dirs = [path for path in output_dir.glob("run_*") if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directory found under {output_dir}.")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def _classify_regime(row: dict) -> str:
    payload_count = row["ducks_num"] + row["pucks_num"]
    if payload_count >= 40 or row["resolved_fuselage_width_m"] >= 0.18:
        return "cargo"
    if payload_count <= 10 and row["banner_length_m"] >= 3.0:
        return "banner"
    return "mixed"


def summarize_sensitivity_run(case: SensitivityCase, run_dir: Path) -> dict:
    """Read one completed optimization run into a flat summary row."""

    report_path = run_dir / "best_design_report.json"
    summary_path = run_dir / "run_summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    vector = report["resolved_vector"]
    mechanical = report["mechanical"]["missions"]
    aero = report["aero"]
    row = {
        "case": case.name,
        "label": case.label,
        "changed_group": case.changed_group,
        "percent_change": case.percent_change,
        "run_dir": str(run_dir),
        "score": report["score"],
        "ground_score": report["breakdown"]["ground"],
        "m1_score": report["breakdown"]["m1"],
        "m2_score": report["breakdown"]["m2"],
        "m3_score": report["breakdown"]["m3"],
        "ducks_num": int(vector["ducks_num"]),
        "pucks_num": int(vector["pucks_num"]),
        "payload_count": int(vector["ducks_num"]) + int(vector["pucks_num"]),
        "banner_length_m": vector["banner_length"],
        "wing_span_m": vector["wing_span"],
        "wing_chord_m": vector["wing_chord"],
        "wing_area_m2": vector["wing_area"],
        "resolved_fuselage_width_m": vector["fuselage_width"],
        "m1_lap_time_s": aero["M1"]["lap_time"],
        "m2_lap_time_s": aero["M2"]["lap_time"],
        "m3_lap_time_s": aero["M3"]["lap_time"],
        "m1_mass_kg": mechanical["M1"]["total_mass_kg"],
        "m2_mass_kg": mechanical["M2"]["total_mass_kg"],
        "m3_mass_kg": mechanical["M3"]["total_mass_kg"],
        "nfev": run_summary["nfev"],
        "nit": run_summary["nit"],
        "success": run_summary["success"],
    }
    row["regime"] = _classify_regime(row)
    return row


def write_summary_csv(rows: list[dict], path: Path) -> Path:
    """Write flat sensitivity results."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_sensitivity_summary(rows: list[dict], path: Path) -> Path:
    """Create a first-pass plot showing cargo/banner regime movement."""

    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("No sensitivity rows to plot.")

    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "baseline": "black",
        "mission_time_limit": "tab:blue",
        "ground_best_time": "tab:green",
        "m2_best_profit": "tab:orange",
        "m3_best_lap_time": "tab:red",
        "m3_best_banner": "tab:purple",
        "all_refs": "tab:gray",
    }
    markers = {"cargo": "s", "banner": "^", "mixed": "o"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for row in rows:
        color = colors.get(row["changed_group"], "tab:gray")
        marker = markers.get(row["regime"], "o")
        axes[0, 0].scatter(
            row["payload_count"],
            row["banner_length_m"],
            s=70,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.85,
        )
        if row["changed_group"] == "baseline":
            axes[0, 0].annotate(
                "baseline",
                (row["payload_count"], row["banner_length_m"]),
                xytext=(6, 6),
                textcoords="offset points",
            )

    axes[0, 0].set_xlabel("Payload count (ducks + pucks)")
    axes[0, 0].set_ylabel("Banner length [m]")
    axes[0, 0].set_title("Optimized Regime Map")
    axes[0, 0].grid(True, alpha=0.25)

    for group in sorted({row["changed_group"] for row in rows} - {"baseline"}):
        group_rows = sorted(
            [row for row in rows if row["changed_group"] == group],
            key=lambda row: row["percent_change"],
        )
        x = [row["percent_change"] * 100.0 for row in group_rows]
        color = colors.get(group, "tab:gray")
        axes[0, 1].plot(
            x,
            [row["payload_count"] for row in group_rows],
            marker="o",
            label=group,
            color=color,
        )
        axes[1, 0].plot(
            x,
            [row["banner_length_m"] for row in group_rows],
            marker="o",
            label=group,
            color=color,
        )
        axes[1, 1].plot(
            x,
            [row["wing_span_m"] for row in group_rows],
            marker="o",
            label=group,
            color=color,
        )

    axes[0, 1].set_title("Payload Response")
    axes[0, 1].set_ylabel("Payload count")
    axes[1, 0].set_title("Banner Response")
    axes[1, 0].set_ylabel("Banner length [m]")
    axes[1, 1].set_title("Airframe Size Response")
    axes[1, 1].set_ylabel("Wing span [m]")
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.set_xlabel("Reference change [%]")
        axis.grid(True, alpha=0.25)

    axes[1, 1].legend(loc="best", fontsize="small")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_reference_sensitivity(
    *,
    percentages: Iterable[float] = (-0.25, -0.10, 0.10, 0.25),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config_template: ToplineConfig | None = None,
    include_baseline: bool = True,
    include_all_references: bool = True,
) -> list[dict]:
    """Run topline optimizations across shifted scoring references."""

    from src.opt.topline_opt import ToplineConfig, run_topline_optimization

    output_dir.mkdir(parents=True, exist_ok=True)
    config_template = config_template or ToplineConfig(
        output_dir=output_dir,
        maxiter=40,
        popsize=12,
        workers=1,
        seed=20260810,
    )
    cases = build_reference_sensitivity_cases(
        percentages,
        base_refs=config_template.scoring_references,
        include_baseline=include_baseline,
        include_all_references=include_all_references,
    )

    rows: list[dict] = []
    for index, case in enumerate(cases, start=1):
        case_dir = output_dir / case.name
        config = replace(
            config_template,
            output_dir=case_dir,
            scoring_references=case.scoring_references,
            seed=config_template.seed + index,
        )
        (case_dir / "sensitivity_case.json").parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        (case_dir / "sensitivity_case.json").write_text(
            json.dumps(
                {
                    "case": asdict(case),
                    "scoring_references": scoring_reference_values(
                        case.scoring_references
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        run_topline_optimization(config)
        run_dir = _latest_run_dir(case_dir)
        rows.append(summarize_sensitivity_run(case, run_dir))
        write_summary_csv(rows, output_dir / "sensitivity_summary.csv")

    plot_sensitivity_summary(rows, output_dir / "sensitivity_regime_map.png")
    return rows


def main() -> None:
    from src.opt.topline_opt import ToplineConfig

    config = ToplineConfig(
        output_dir=DEFAULT_OUTPUT_DIR,
        maxiter=SENSITIVITY_MAXITER,
        popsize=SENSITIVITY_POPSIZE,
        workers=SENSITIVITY_WORKERS,
        seed=SENSITIVITY_SEED,
    )
    run_reference_sensitivity(
        percentages=SENSITIVITY_PERCENTAGES,
        output_dir=DEFAULT_OUTPUT_DIR,
        config_template=config,
    )


if __name__ == "__main__":
    main()
