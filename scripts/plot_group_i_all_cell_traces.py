#!/usr/bin/env python3
"""Render every Group I Sholl trace as a labeled small multiple."""
from __future__ import annotations

from pathlib import Path
import sys
from math import ceil

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemogenetic_analysis import ShollDataProcessor, ShollStatsAnalyzer


CONDITIONS = (
    ("DREADD_CNO", "hM3Dq + CNO", "#6AA84F"),
    ("LMO7_hCTZ", "LMO7 + hCTZ", "#46B3C3"),
    ("PSAM_uPSEM", "PSAM4-5HT3 + uPSEM", "#8E7CC3"),
    ("EYFP_Control_Media", "EYFP + media", "#9AA0A6"),
)
# The removal set that yielded two nominal Dunnett-significant Group I contrasts.
FOCUS_CELL_IDS = {
    "DREADD/CNO__r12__sNA",
    "CONTROL/MEDIA__r5__sNA",
    "CONTROL/MEDIA__r7__sNA",
    "CONTROL/MEDIA__r8__sNA",
    "CONTROL/MEDIA__r10__sNA",
    "LMO7/hCTZ__r3__sNA",
    "LMO7/hCTZ__r7__sNA",
    "PSAM/uPSEM__r1__sNA",
}
FOCUS_BORDER = "#C9A227"


def cell_id(frame: pd.DataFrame) -> pd.Series:
    sample = frame["sample_id"].where(frame["sample_id"].notna(), "NA").astype(str)
    return frame["source_condition"].astype(str) + "__r" + frame["replicate"].astype(int).astype(str) + "__s" + sample


def main() -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.55})
    processor = ShollDataProcessor(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv")
    traces = processor.recode_conditions(split_shared_control=True)
    traces["cell_id"] = cell_id(traces)
    traces = traces.loc[traces["condition"].isin([item[0] for item in CONDITIONS])].copy()
    auc = ShollStatsAnalyzer(traces).build_auc_per_neuron()[["cell_id", "auc"]]
    auc_by_cell = auc.set_index("cell_id")["auc"].to_dict()
    cells_by_condition = {
        condition: [
            (identifier, frame.sort_values("radius_um"))
            for identifier, frame in traces.loc[traces["condition"] == condition].groupby("cell_id", sort=False)
        ]
        for condition, _, _ in CONDITIONS
    }
    x_limits = (0, float(traces["radius_um"].max()) * 1.02)
    y_limits = (0, max(1.0, float(traces["intersections"].max()) * 1.08))
    output_dir = REPO_ROOT / "output" / "plots"
    for condition, label, color in CONDITIONS:
        cells = sorted(cells_by_condition[condition], key=lambda item: (item[0] not in FOCUS_CELL_IDS, item[0]))
        n_columns = 3
        n_rows = ceil(len(cells) / n_columns)
        figure, axes = plt.subplots(n_rows, n_columns, figsize=(16, 4.35 * n_rows), sharex=True, sharey=True, squeeze=False)
        for index, axis in enumerate(axes.flat):
            if index >= len(cells):
                axis.axis("off")
                continue
            identifier, trace = cells[index]
            axis.plot(trace["radius_um"], trace["intersections"], color=color, linewidth=1.55)
            state = "SENSITIVITY SET" if identifier in FOCUS_CELL_IDS else "other cell"
            axis.set_title(f"{identifier}\nAUC = {auc_by_cell[identifier]:.1f}  |  {state}", fontsize=9.2, pad=7)
            if identifier in FOCUS_CELL_IDS:
                for spine in axis.spines.values():
                    spine.set_color(FOCUS_BORDER)
                    spine.set_linewidth(2.0)
            axis.set_xlim(x_limits)
            axis.set_ylim(y_limits)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=8, length=2.5, pad=2)
            axis.grid(False)
            axis.set_xlabel("Distance from soma (µm)", fontsize=8)
            axis.set_ylabel("Intersections", fontsize=8)
        figure.suptitle(f"Group I — {label}: Every Individual-Cell Sholl Curve", fontsize=16, fontweight="bold", color=color, y=0.995)
        figure.text(0.5, 0.01, "Gold border = cell included in the two-treatment nominal Dunnett sensitivity set; all other panels are non-focus cells.", ha="center", fontsize=8)
        figure.tight_layout(rect=(0.02, 0.035, 0.995, 0.965), h_pad=3.0, w_pad=1.7)
        figure.savefig(output_dir / f"group_i_{condition.lower()}_cell_traces_labeled.png", dpi=300, facecolor="white")
        plt.close(figure)

    focus_cells = [
        (condition, label, color, identifier, trace)
        for condition, label, color in CONDITIONS
        for identifier, trace in cells_by_condition[condition]
        if identifier in FOCUS_CELL_IDS
    ]
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    for axis, (condition, label, color, identifier, trace) in zip(axes.flat, focus_cells):
        axis.plot(trace["radius_um"], trace["intersections"], color=color, linewidth=2.0)
        for spine in axis.spines.values():
            spine.set_color(FOCUS_BORDER)
            spine.set_linewidth(2.2)
        axis.set_title(f"{label}\n{identifier}\nAUC = {auc_by_cell[identifier]:.1f}", fontsize=9.5, pad=8)
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8, length=2.5, pad=2)
        axis.grid(False)
        axis.set_xlabel("Distance from soma (µm)", fontsize=8)
        axis.set_ylabel("Intersections", fontsize=8)
    figure.suptitle("Group I — Sensitivity-Set Cells: Individual Sholl Curves", fontsize=16, fontweight="bold", y=0.985)
    figure.tight_layout(rect=(0.02, 0.02, 0.995, 0.94), h_pad=2.5, w_pad=1.6)
    figure.savefig(output_dir / "group_i_sensitivity_set_cell_traces.png", dpi=300, facecolor="white")
    plt.close(figure)
    mapping = pd.DataFrame([
        {"cell_id": identifier, "condition": condition, "auc": auc_by_cell[identifier], "focus_sensitivity_set": identifier in FOCUS_CELL_IDS}
        for condition, cells in cells_by_condition.items() for identifier, _ in cells
    ])
    mapping.to_csv(output_dir / "group_i_all_cell_sholl_traces_labeled_cells.csv", index=False)
    print(f"Saved readable labeled Group I cell traces to: {output_dir}")


if __name__ == "__main__":
    main()
