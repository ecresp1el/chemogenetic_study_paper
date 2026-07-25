#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemogenetic_analysis import ConditionAnalysisReport, ReportCondition, ShollDataProcessor


CONTROL_GRAY = "#9AA0A6"


@dataclass(frozen=True)
class AUCPanelSpec:
    title: str
    conditions: tuple[ReportCondition, ...]


class MajorGroupAUCFigure:
    """Render matched AUC dot-and-box panels for Groups I, II, and III."""

    PANELS = (
        AUCPanelSpec(
            "Group III — Ligand/media-only control",
            (
                ReportCondition("None_CNO", "CNO only", "#B7D7A8"),
                ReportCondition("None_hCTZ", "hCTZ only", "#A9DCE5"),
                ReportCondition("None_uPSEM", "uPSEM only", "#C9C2E4"),
                ReportCondition("None_Vehicle", "Media only", CONTROL_GRAY),
            ),
        ),
        AUCPanelSpec(
            "Group II — Vehicle condition",
            (
                ReportCondition("DREADD_Vehicle", "hM3Dq + vehicle", CONTROL_GRAY),
                ReportCondition("LMO7_Vehicle", "LMO7 + vehicle", CONTROL_GRAY),
                ReportCondition("PSAM_Vehicle", "PSAM4-5HT3 + vehicle", CONTROL_GRAY),
                ReportCondition("EYFP_Control", "EYFP control", CONTROL_GRAY),
            ),
        ),
        AUCPanelSpec(
            "Group I — Treatment condition",
            (
                ReportCondition("DREADD_CNO", "hM3Dq + CNO", "#6AA84F"),
                ReportCondition("LMO7_hCTZ", "LMO7 + hCTZ", "#46B3C3"),
                ReportCondition("PSAM_uPSEM", "PSAM4-5HT3 + uPSEM", "#8E7CC3"),
                ReportCondition("EYFP_Control_Media", "EYFP + media", CONTROL_GRAY),
            ),
        ),
    )

    def __init__(self, grouped_df):
        self.grouped_df = grouped_df.copy()

    @classmethod
    def from_raw_csv(cls, raw_csv_path: str | Path) -> "MajorGroupAUCFigure":
        processor = ShollDataProcessor(raw_csv_path)
        return cls(processor.recode_conditions(split_shared_control=True))

    def render(self, output_path: str | Path, dpi: int = 300) -> Path:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.65})
        panel_metrics = [self._metrics_for_panel(panel) for panel in self.PANELS]
        x_max = max(metrics["auc"].max() for metrics in panel_metrics) * 1.08

        figure, axes = plt.subplots(
            3,
            3,
            figsize=(15.0, 12.3),
            gridspec_kw={"height_ratios": [1.0, 1.1, 1.0]},
        )
        for column, (panel, metrics) in enumerate(zip(self.PANELS, panel_metrics)):
            self._plot_box_panel(axes[0, column], panel, metrics, x_max)
            self._plot_bar_panel(axes[1, column], panel, metrics, x_max)
            self._plot_normalized_box_panel(
                axes[2, column], panel, metrics
            )

        figure.suptitle("Area Under the Curve Across Major Study Groups", fontsize=14, fontweight="bold", y=0.97)
        figure.text(
            0.5,
            0.02,
            "Top: individual-cell AUC with interquartile-range boxes and median. "
            "Middle: mean AUC ± SEM. Bottom: individual-cell fold change relative to the matched-control mean.",
            ha="center",
            fontsize=8,
        )
        figure.tight_layout(rect=(0.02, 0.06, 0.99, 0.92), w_pad=3.4, h_pad=3.5)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, facecolor="white")
        plt.close(figure)
        return output

    def _metrics_for_panel(self, panel: AUCPanelSpec):
        report = ConditionAnalysisReport(
            self.grouped_df,
            conditions=panel.conditions,
            reference_condition=panel.conditions[-1].condition,
            title=panel.title,
        )
        return report.build_cell_metrics()

    @staticmethod
    def _plot_box_panel(axis, panel: AUCPanelSpec, metrics, x_max: float) -> None:
        values = [
            metrics.loc[metrics["condition"] == spec.condition, "auc"].to_numpy()
            for spec in panel.conditions
        ]
        positions = np.arange(len(panel.conditions))
        boxplot = axis.boxplot(
            values,
            vert=False,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#333333", "linewidth": 1.0},
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        rng = np.random.default_rng(20260725)
        y_labels = []
        for index, (spec, value) in enumerate(zip(panel.conditions, values)):
            boxplot["boxes"][index].set_facecolor(spec.color)
            boxplot["boxes"][index].set_alpha(0.22)
            axis.scatter(
                value,
                rng.normal(index, 0.06, len(value)),
                s=15,
                color=spec.color,
                alpha=0.9,
                linewidths=0,
                zorder=3,
            )
            y_labels.append(f"{spec.label} (n={len(value)})")

        axis.set_yticks(positions, y_labels)
        axis.invert_yaxis()
        axis.set_xlim(0, x_max)
        axis.set_xlabel("AUC")
        axis.set_title(panel.title, fontsize=10, fontweight="bold", pad=10)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=7, length=2.5, pad=3)
        axis.grid(False)

    @staticmethod
    def _plot_bar_panel(axis, panel: AUCPanelSpec, metrics, y_max: float) -> None:
        values = [
            metrics.loc[metrics["condition"] == spec.condition, "auc"].to_numpy()
            for spec in panel.conditions
        ]
        positions = np.arange(len(panel.conditions))
        means = [float(np.mean(value)) for value in values]
        sems = [float(np.std(value, ddof=1) / np.sqrt(len(value))) for value in values]
        colors = [spec.color for spec in panel.conditions]
        axis.barh(
            positions,
            means,
            xerr=sems,
            capsize=3,
            color=colors,
            alpha=0.45,
            edgecolor=colors,
            linewidth=0.8,
            error_kw={"elinewidth": 0.9, "ecolor": "#333333"},
            zorder=2,
        )
        rng = np.random.default_rng(20260726)
        for position, value, color in zip(positions, values, colors):
            axis.scatter(
                value,
                rng.normal(position, 0.07, len(value)),
                s=15,
                color=color,
                alpha=0.9,
                linewidths=0,
                zorder=3,
            )

        y_labels = []
        for spec, value in zip(panel.conditions, values):
            display_label = spec.label.replace(" + ", "\n+ ")
            y_labels.append(f"{display_label}\n(n={len(value)})")
        axis.set_yticks(positions, y_labels)
        axis.invert_yaxis()
        axis.set_xlim(0, y_max)
        axis.set_xlabel("Mean AUC ± SEM")
        axis.set_title("Mean AUC ± SEM", fontsize=9, pad=8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelsize=7, length=2.5, pad=3)
        axis.tick_params(axis="y", labelsize=6.5, length=2.5, pad=3)
        axis.grid(False)

    @staticmethod
    def _normalized_values(panel: AUCPanelSpec, metrics):
        control_condition = panel.conditions[-1].condition
        control_mean = float(
            metrics.loc[metrics["condition"] == control_condition, "auc"].mean()
        )
        if control_mean <= 0:
            raise ValueError(f"Matched-control mean AUC must be positive for {panel.title}.")
        return metrics["auc"] / control_mean

    @classmethod
    def _plot_normalized_box_panel(cls, axis, panel: AUCPanelSpec, metrics) -> None:
        normalized = metrics.copy()
        normalized["normalized_auc"] = cls._normalized_values(panel, metrics)
        values = [
            normalized.loc[normalized["condition"] == spec.condition, "normalized_auc"].to_numpy()
            for spec in panel.conditions
        ]
        positions = np.arange(len(panel.conditions))
        boxplot = axis.boxplot(
            values,
            vert=False,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#333333", "linewidth": 1.0},
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        rng = np.random.default_rng(20260728)
        y_labels = []
        for index, (spec, value) in enumerate(zip(panel.conditions, values)):
            boxplot["boxes"][index].set_facecolor(spec.color)
            boxplot["boxes"][index].set_alpha(0.22)
            axis.scatter(
                value,
                rng.normal(index, 0.06, len(value)),
                s=15,
                color=spec.color,
                alpha=0.9,
                linewidths=0,
                zorder=3,
            )
            y_labels.append(f"{spec.label} (n={len(value)})")

        axis.axvline(1.0, color="#555555", linestyle="--", linewidth=0.8, zorder=0)
        axis.set_yticks(positions, y_labels)
        axis.invert_yaxis()
        axis.set_xlim(0, 3)
        axis.set_xlabel("Fold change (AUC / matched-control mean)")
        axis.set_title("Fold change vs matched control", fontsize=9, pad=8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=7, length=2.5, pad=3)
        axis.grid(False)


def main() -> None:
    report = MajorGroupAUCFigure.from_raw_csv(
        REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"
    )
    output = report.render(REPO_ROOT / "output" / "plots" / "auc_major_groups.png")
    print(f"AUC comparison figure saved to: {output}")


if __name__ == "__main__":
    main()
