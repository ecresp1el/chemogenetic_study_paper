#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemogenetic_analysis import ShollDataProcessor


TECH_ORDER = ["DREADD", "PSAM", "LMO7", "EYFP"]
GROUPS = ["Group I (Activation)", "Group II (Expression only)"]
GROUP_MARKERS = {
    "Group I (Activation)": "o",
    "Group II (Expression only)": "s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 1x4 panel figure (one panel per technology) showing "
            "Activation vs Expression mean (dots) and SEM (error bars)."
        )
    )
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"),
        help="Path to the raw Sholl CSV file.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "output" / "plots" / "activation_vs_expression_1x4.png"),
        help="Path for output figure.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(
            REPO_ROOT / "output" / "plots" / "activation_vs_expression_mean_sem.csv"
        ),
        help="Path for filtered mean/SEM summary CSV.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output DPI for saved figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = ShollDataProcessor(args.input)

    recoded_df = processor.recode_conditions(split_shared_control=True)
    summary_df = processor.summarize_mean_sem_by_technology(recoded_df=recoded_df)
    summary_df = summary_df.loc[summary_df["analysis_group"].isin(GROUPS)].copy()

    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output, index=False)

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
    group_offsets = {
        "Group I (Activation)": -1.0,
        "Group II (Expression only)": 1.0,
    }

    for idx, technology in enumerate(TECH_ORDER):
        ax = axes[idx]
        tech_df = summary_df.loc[summary_df["technology"] == technology].copy()
        tech_map = processor.TECHNOLOGY_CONDITIONS.get(technology, {})

        for group_name in GROUPS:
            condition_name = tech_map.get(group_name)
            if condition_name is None:
                continue
            line_df = tech_df.loc[
                (tech_df["analysis_group"] == group_name)
                & (tech_df["condition"] == condition_name)
            ].sort_values("radius_um")
            if line_df.empty:
                continue

            color = processor.GROUP_COLORS.get(group_name, "#4c4c4c")
            x = line_df["radius_um"] + group_offsets.get(group_name, 0.0)
            y = line_df["mean_intersections"]
            sem = line_df["sem_intersections"]

            ax.errorbar(
                x,
                y,
                yerr=sem,
                fmt=GROUP_MARKERS.get(group_name, "o"),
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.5,
                markersize=3.2,
                alpha=0.9,
            )

        ax.set_title(technology)
        activation_cond = tech_map.get("Group I (Activation)", "NA")
        expression_cond = tech_map.get("Group II (Expression only)", "NA")
        ax.text(
            0.02,
            0.98,
            f"A: {activation_cond}\nE: {expression_cond}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
        ax.grid(alpha=0.2)
        ax.set_xlabel("Radius from Soma (um)")
        if idx == 0:
            ax.set_ylabel("Intersections")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=GROUP_MARKERS.get(group, "o"),
            color=processor.GROUP_COLORS.get(group, "#4c4c4c"),
            linestyle="None",
            markersize=6,
            label=group,
        )
        for group in GROUPS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Activation vs Expression: Mean (dots) +/- SEM (bars)", y=1.08)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Filtered summary saved to: {summary_output}")
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
