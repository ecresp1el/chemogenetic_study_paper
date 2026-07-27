#!/usr/bin/env python3
"""Render Group I/II AUC comparisons against one pooled EYFP control distribution."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
STATS_DIR = REPO_ROOT / "output" / "stats" / "pooled_eyfp_control_kruskal_dunn"
POOLED_CONTROL = "Pooled_EYFP_Control"
COLORS = {
    "DREADD_CNO": "#6AA84F", "LMO7_hCTZ": "#46B3C3", "PSAM_uPSEM": "#8E7CC3",
    "DREADD_Vehicle": "#9AA0A6", "LMO7_Vehicle": "#9AA0A6", "PSAM_Vehicle": "#9AA0A6",
    POOLED_CONTROL: "#9AA0A6",
}
LABELS = {
    "DREADD_CNO": "hM3Dq + CNO", "LMO7_hCTZ": "LMO7 + hCTZ", "PSAM_uPSEM": "PSAM4-5HT3 + uPSEM",
    "DREADD_Vehicle": "hM3Dq + vehicle", "LMO7_Vehicle": "LMO7 + vehicle", "PSAM_Vehicle": "PSAM4-5HT3 + vehicle",
    POOLED_CONTROL: "Pooled EYFP control",
}
PANELS = (
    ("Group I: treatment condition vs pooled EYFP", ("DREADD_CNO", "LMO7_hCTZ", "PSAM_uPSEM", POOLED_CONTROL)),
    ("Group II: vehicle condition vs pooled EYFP", ("DREADD_Vehicle", "LMO7_Vehicle", "PSAM_Vehicle", POOLED_CONTROL)),
)


def p_text(value: float) -> str:
    return "p < 0.0001" if value < 0.0001 else f"p = {value:.4f}"


def main() -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.65})
    auc = pd.read_csv(STATS_DIR / "auc_per_cell_with_pooled_eyfp_control.csv")
    kw = pd.read_csv(STATS_DIR / "kruskal_wallis_by_major_group.csv")
    dunn = pd.read_csv(STATS_DIR / "dunn_pooled_eyfp_control_contrasts.csv")
    figure = plt.figure(figsize=(10.6, 6.0))
    grid = figure.add_gridspec(2, 2, height_ratios=(4.2, 0.95), hspace=0.38, wspace=0.44)

    for column, (group, conditions) in enumerate(PANELS):
        axis = figure.add_subplot(grid[0, column])
        panel = auc.loc[auc["major_group"] == group]
        values = [panel.loc[panel["condition"] == condition, "auc"].to_numpy() for condition in conditions]
        positions = np.arange(len(conditions))
        boxes = axis.boxplot(values, vert=False, positions=positions, widths=0.52, patch_artist=True, showfliers=False,
                             medianprops={"color": "#303030", "linewidth": 1.0}, boxprops={"linewidth": 0.8},
                             whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8})
        rng = np.random.default_rng(20260725 + column)
        for index, (condition, value) in enumerate(zip(conditions, values)):
            boxes["boxes"][index].set_facecolor(COLORS[condition])
            boxes["boxes"][index].set_alpha(0.22)
            axis.scatter(value, rng.normal(index, 0.055, len(value)), s=30, color=COLORS[condition], alpha=0.88, linewidths=0, zorder=3)

        x_max = max(np.max(value) for value in values) * 1.04
        contrast = dunn.loc[dunn["major_group"] == group].set_index("treatment_condition")
        control_index = len(conditions) - 1
        for index, condition in enumerate(conditions[:-1]):
            x = x_max * (1.04 + 0.072 * index)
            tick = x * 0.015
            axis.plot([x, x], [index, control_index], color="#303030", linewidth=0.75)
            axis.plot([x - tick, x], [index, index], color="#303030", linewidth=0.75)
            axis.plot([x - tick, x], [control_index, control_index], color="#303030", linewidth=0.75)
            axis.text(x + tick * 0.18, (index + control_index) / 2, p_text(float(contrast.loc[condition, "dunn_pvalue_holm_adjusted"])), va="center", fontsize=6.6)

        labels = [f"{LABELS[condition]} (n={len(value)})" for condition, value in zip(conditions, values)]
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlim(0, x_max * 1.31)
        axis.set_xlabel("Sholl AUC")
        axis.set_title(group.replace(" vs pooled EYFP", ""), fontsize=10, fontweight="bold", pad=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=7, length=2.5, pad=3)

        text_axis = figure.add_subplot(grid[1, column])
        text_axis.axis("off")
        kw_row = kw.loc[kw["major_group"] == group].iloc[0]
        text_axis.text(0.5, 0.96,
                       f"Pooled EYFP control: n=23 (EYFP control + EYFP + media)\n"
                       f"Kruskal–Wallis: H(3) = {kw_row['kruskal_wallis_h_statistic']:.3f}, {p_text(float(kw_row['kruskal_wallis_pvalue']))}; Dunn vs pooled EYFP, Holm-adjusted.",
                       ha="center", va="top", fontsize=7.1)

    figure.suptitle("Sholl AUC: Comparisons Against the Pooled EYFP Control Distribution", fontsize=13.5, fontweight="bold", y=0.98)
    figure.text(0.5, 0.015, "Points are individual cells; boxes show the interquartile range and median. Brackets: Dunn comparisons versus the pooled EYFP control distribution.", ha="center", fontsize=7.5)
    output = REPO_ROOT / "output" / "plots" / "auc_pooled_eyfp_control_kruskal_dunn.png"
    figure.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"Pooled-EYFP-control comparison figure saved to: {output}")


if __name__ == "__main__":
    main()
