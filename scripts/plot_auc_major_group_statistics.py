#!/usr/bin/env python3
"""Plot raw and matched-control-normalized AUC comparisons by major group."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemogenetic_analysis import ConditionAnalysisReport, ReportCondition, ShollDataProcessor


CONTROL_GRAY = "#9AA0A6"
STATS_DIR = REPO_ROOT / "output" / "stats" / "major_group_kruskal_dunn"


@dataclass(frozen=True)
class AUCStatsPanel:
    title: str
    control: str
    conditions: tuple[ReportCondition, ...]


PANELS = (
    AUCStatsPanel(
        "Group III — Ligand/media-only control",
        "None_Vehicle",
        (
            ReportCondition("None_CNO", "CNO only", "#B7D7A8"),
            ReportCondition("None_hCTZ", "hCTZ only", "#A9DCE5"),
            ReportCondition("None_uPSEM", "uPSEM only", "#C9C2E4"),
            ReportCondition("None_Vehicle", "Media only", CONTROL_GRAY),
        ),
    ),
    AUCStatsPanel(
        "Group II — Vehicle condition",
        "EYFP_Control",
        (
            ReportCondition("DREADD_Vehicle", "hM3Dq + vehicle", CONTROL_GRAY),
            ReportCondition("LMO7_Vehicle", "LMO7 + vehicle", CONTROL_GRAY),
            ReportCondition("PSAM_Vehicle", "PSAM4-5HT3 + vehicle", CONTROL_GRAY),
            ReportCondition("EYFP_Control", "EYFP control", CONTROL_GRAY),
        ),
    ),
    AUCStatsPanel(
        "Group I — Treatment condition",
        "EYFP_Control_Media",
        (
            ReportCondition("DREADD_CNO", "hM3Dq + CNO", "#6AA84F"),
            ReportCondition("LMO7_hCTZ", "LMO7 + hCTZ", "#46B3C3"),
            ReportCondition("PSAM_uPSEM", "PSAM4-5HT3 + uPSEM", "#8E7CC3"),
            ReportCondition("EYFP_Control_Media", "EYFP + media", CONTROL_GRAY),
        ),
    ),
)


def _format_pvalue(value: float) -> str:
    if value < 0.0001:
        return "p < 0.0001"
    return f"p = {value:.4f}"


def _axis_label(spec: ReportCondition, n: int) -> str:
    """Wrap actuator-plus-compound labels so brackets remain visually separate."""
    label = spec.label.replace("PSAM4-5HT3 + ", "PSAM4-5HT3\n+ ")
    label = label.replace("hM3Dq + ", "hM3Dq\n+ ")
    label = label.replace("LMO7 + ", "LMO7\n+ ")
    label = label.replace("EYFP + ", "EYFP\n+ ")
    return f"{label} (n={n})"


def _report_metrics(grouped_df: pd.DataFrame, panel: AUCStatsPanel) -> pd.DataFrame:
    report = ConditionAnalysisReport(
        grouped_df,
        conditions=panel.conditions,
        reference_condition=panel.control,
        title=panel.title,
    )
    return report.build_cell_metrics()


def _draw_comparison(axis, y_a: float, y_b: float, x: float, label: str) -> None:
    tick = x * 0.015
    axis.plot([x, x], [y_a, y_b], color="#303030", linewidth=0.75, clip_on=False)
    axis.plot([x - tick, x], [y_a, y_a], color="#303030", linewidth=0.75, clip_on=False)
    axis.plot([x - tick, x], [y_b, y_b], color="#303030", linewidth=0.75, clip_on=False)
    axis.text(x + tick * 0.18, (y_a + y_b) / 2, label, va="center", ha="left", fontsize=6.6)


def _plot_panel(
    axis,
    panel: AUCStatsPanel,
    metrics: pd.DataFrame,
    dunn: pd.DataFrame,
    metric_column: str,
    x_label: str,
) -> None:
    values = [
        metrics.loc[metrics["condition"] == spec.condition, metric_column].to_numpy()
        for spec in panel.conditions
    ]
    positions = np.arange(len(panel.conditions))
    boxes = axis.boxplot(
        values,
        vert=False,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#303030", "linewidth": 1.0},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    jitter = np.random.default_rng(20260725)
    for index, (spec, value) in enumerate(zip(panel.conditions, values)):
        boxes["boxes"][index].set_facecolor(spec.color)
        boxes["boxes"][index].set_alpha(0.22)
        axis.scatter(value, jitter.normal(index, 0.055, len(value)), s=30, color=spec.color, alpha=0.88, linewidths=0, zorder=3)

    control_index = len(panel.conditions) - 1
    contrast_rows = dunn.loc[dunn["control_condition"] == panel.control].set_index("treatment_condition")
    x_data_max = max(np.max(value) for value in values) * 1.04
    annotation_start = x_data_max * 1.035
    annotation_step = x_data_max * 0.072
    for index, spec in enumerate(panel.conditions[:-1]):
        pvalue = float(contrast_rows.loc[spec.condition, "dunn_pvalue_holm_adjusted"])
        _draw_comparison(axis, index, control_index, annotation_start + index * annotation_step, _format_pvalue(pvalue))

    labels = [_axis_label(spec, len(value)) for spec, value in zip(panel.conditions, values)]
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, x_data_max * 1.31)
    if metric_column != "auc":
        axis.axvline(1.0, color="#666666", linestyle="--", linewidth=0.75, zorder=0)
    axis.set_xlabel(x_label)
    axis.set_title(panel.title, fontsize=10, fontweight="bold", pad=9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", labelsize=7, length=2.5, pad=3)
    axis.tick_params(axis="y", labelsize=6.6, length=2.5, pad=3)
    axis.grid(False)


def _summary_text(panel: AUCStatsPanel, metrics: pd.DataFrame, kw: pd.DataFrame, dunn: pd.DataFrame) -> str:
    kw_row = kw.loc[kw["control_condition"] == panel.control].iloc[0]
    counts = ", ".join(
        f"{spec.label}: n={int((metrics['condition'] == spec.condition).sum())}"
        for spec in panel.conditions
    )
    return (
        f"{counts}\n"
        f"Kruskal–Wallis: H(3) = {kw_row['kruskal_wallis_h_statistic']:.3f}, "
        f"{_format_pvalue(float(kw_row['kruskal_wallis_pvalue']))}; "
        "planned Dunn tests vs shared control, Holm-adjusted."
    )


def _render_figure(
    metrics: list[pd.DataFrame],
    kw: pd.DataFrame,
    dunn: pd.DataFrame,
    metric_column: str,
    x_label: str,
    title: str,
    output: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.65})
    figure = plt.figure(figsize=(15.2, 6.4))
    grid = figure.add_gridspec(2, 3, height_ratios=(4.5, 1.0), hspace=0.38, wspace=0.36)
    for column, (panel, frame) in enumerate(zip(PANELS, metrics)):
        axis = figure.add_subplot(grid[0, column])
        _plot_panel(axis, panel, frame, dunn, metric_column, x_label)
        report_axis = figure.add_subplot(grid[1, column])
        report_axis.axis("off")
        report_axis.text(0.5, 0.95, _summary_text(panel, frame, kw, dunn), va="top", ha="center", fontsize=7.1, wrap=True)

    figure.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    figure.text(0.5, 0.015, "Points are individual cells; boxes show the interquartile range and median. Brackets: Dunn test versus the group-specific shared control (Holm-adjusted).", ha="center", fontsize=7.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"Statistical AUC comparison figure saved to: {output}")


def main() -> None:
    grouped_df = ShollDataProcessor(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv").recode_conditions(split_shared_control=True)
    metrics = [_report_metrics(grouped_df, panel) for panel in PANELS]
    pooled_eyfp_mean = pd.concat(metrics, ignore_index=True).loc[
        lambda frame: frame["condition"].isin(["EYFP_Control", "EYFP_Control_Media"]), "auc"
    ].mean()
    for panel, frame in zip(PANELS, metrics):
        control_median = frame.loc[frame["condition"] == panel.control, "auc"].median()
        frame["fold_change_vs_median_reference"] = frame["auc"] / control_median
        pooled_reference = pooled_eyfp_mean if panel.control.startswith("EYFP_") else frame.loc[
            frame["condition"] == panel.control, "auc"
        ].mean()
        frame["fold_change_vs_pooled_eyfp_reference"] = frame["auc"] / pooled_reference
    _render_figure(
        metrics,
        pd.read_csv(STATS_DIR / "kruskal_wallis_raw_auc_by_major_group.csv"),
        pd.read_csv(STATS_DIR / "dunn_raw_auc_shared_control_contrasts_by_major_group.csv"),
        "auc",
        "Sholl AUC",
        "Raw Sholl AUC: Planned Comparisons Within Major Study Groups",
        REPO_ROOT / "output" / "plots" / "auc_major_groups_kruskal_dunn_raw.png",
    )
    _render_figure(
        metrics,
        pd.read_csv(STATS_DIR / "kruskal_wallis_mean_normalized_auc_by_major_group.csv"),
        pd.read_csv(STATS_DIR / "dunn_mean_normalized_auc_shared_control_contrasts_by_major_group.csv"),
        "fold_change_vs_reference",
        "Fold change (AUC / matched-control mean)",
        "Mean-Normalized Sholl AUC: Planned Comparisons Within Major Study Groups",
        REPO_ROOT / "output" / "plots" / "auc_major_groups_kruskal_dunn_mean_normalized.png",
    )
    _render_figure(
        metrics,
        pd.read_csv(STATS_DIR / "kruskal_wallis_median_normalized_auc_by_major_group.csv"),
        pd.read_csv(STATS_DIR / "dunn_median_normalized_auc_shared_control_contrasts_by_major_group.csv"),
        "fold_change_vs_median_reference",
        "Fold change (AUC / matched-control median)",
        "Median-Normalized Sholl AUC: Planned Comparisons Within Major Study Groups",
        REPO_ROOT / "output" / "plots" / "auc_major_groups_kruskal_dunn_median_normalized.png",
    )
    _render_figure(
        metrics,
        pd.read_csv(STATS_DIR / "kruskal_wallis_pooled_eyfp_normalized_auc_by_major_group.csv"),
        pd.read_csv(STATS_DIR / "dunn_pooled_eyfp_normalized_auc_shared_control_contrasts_by_major_group.csv"),
        "fold_change_vs_pooled_eyfp_reference",
        "Fold change (Groups I/II: pooled EYFP mean; Group III: media mean)",
        "Pooled-EYFP-Normalized Sholl AUC: Planned Comparisons Within Major Study Groups",
        REPO_ROOT / "output" / "plots" / "auc_major_groups_kruskal_dunn_pooled_eyfp_normalized.png",
    )


if __name__ == "__main__":
    main()
