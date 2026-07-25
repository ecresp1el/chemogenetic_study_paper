from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .sholl_processor import ShollDataProcessor


@dataclass(frozen=True)
class ReportCondition:
    """Display configuration for one condition in an analysis-report figure."""

    condition: str
    label: str
    color: str


class ConditionAnalysisReport:
    """Build a four-condition, multi-level Sholl analysis-report figure."""

    DEFAULT_CONDITIONS = (
        ReportCondition("None_CNO", "CNO", "#B7D7A8"),
        ReportCondition("None_Vehicle", "Media", "#9AA0A6"),
        ReportCondition("None_hCTZ", "hCTZ", "#A9DCE5"),
        ReportCondition("None_uPSEM", "uPSEM", "#C9C2E4"),
    )

    def __init__(
        self,
        grouped_df: pd.DataFrame,
        conditions: Sequence[ReportCondition] | None = None,
        reference_condition: str = "None_Vehicle",
        title: str = "Single-Condition Sholl Analysis Report",
    ):
        self.grouped_df = grouped_df.copy()
        self.conditions = tuple(conditions or self.DEFAULT_CONDITIONS)
        self.reference_condition = reference_condition
        self.title = title
        if len(self.conditions) != 4:
            raise ValueError("Analysis reports require exactly four conditions.")

        available = set(self.grouped_df["condition"].dropna().unique())
        missing = [spec.condition for spec in self.conditions if spec.condition not in available]
        if missing:
            raise ValueError(f"Conditions not found in grouped data: {', '.join(missing)}")
        if self.reference_condition not in {spec.condition for spec in self.conditions}:
            raise ValueError("The reference condition must be one of the four report conditions.")

    @classmethod
    def from_raw_csv(
        cls,
        raw_csv_path: str | Path,
        conditions: Sequence[ReportCondition] | None = None,
        reference_condition: str = "None_Vehicle",
        title: str = "Single-Condition Sholl Analysis Report",
    ) -> "ConditionAnalysisReport":
        processor = ShollDataProcessor(raw_csv_path)
        grouped_df = processor.recode_conditions(split_shared_control=True)
        return cls(
            grouped_df,
            conditions=conditions,
            reference_condition=reference_condition,
            title=title,
        )

    def build_cell_metrics(self) -> pd.DataFrame:
        """Return one AUC and fold-change value per cell for the four report conditions."""
        selected_conditions = [spec.condition for spec in self.conditions]
        subset = self.grouped_df.loc[
            self.grouped_df["condition"].isin(selected_conditions)
        ].copy()
        subset["radius_um"] = pd.to_numeric(subset["radius_um"], errors="coerce")
        subset["intersections"] = pd.to_numeric(subset["intersections"], errors="coerce")

        rows: list[dict[str, object]] = []
        for (condition, replicate), cell_df in subset.groupby(["condition", "replicate"], sort=False):
            cell_df = cell_df.sort_values("radius_um")
            x = cell_df["radius_um"].to_numpy(dtype=float)
            y = cell_df["intersections"].to_numpy(dtype=float)
            rows.append(
                {
                    "condition": condition,
                    "replicate": int(replicate),
                    "cell_label": f"Cell {int(replicate)}",
                    "auc": float(np.trapezoid(y, x)),
                }
            )

        metrics = pd.DataFrame(rows)
        reference_auc = metrics.loc[
            metrics["condition"] == self.reference_condition, "auc"
        ].median()
        if not np.isfinite(reference_auc) or reference_auc == 0:
            raise ValueError("Reference-condition median AUC must be finite and non-zero.")
        metrics["fold_change_vs_reference"] = metrics["auc"] / reference_auc
        return metrics

    def render(
        self,
        output_path: str | Path,
        metrics_output_path: str | Path | None = None,
        representative_cells: int = 9,
        dpi: int = 300,
    ) -> Path:
        """Render the complete four-row analysis report and optional metrics table."""
        if representative_cells < 1 or representative_cells > 12:
            raise ValueError("representative_cells must be between 1 and 12.")

        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.rcParams.update(
            {
                "font.family": "Arial",
                "font.size": 7,
                "axes.linewidth": 0.6,
                "xtick.major.width": 0.6,
                "ytick.major.width": 0.6,
            }
        )

        data = self._selected_data()
        metrics = self.build_cell_metrics()
        x_limits = self._padded_limits(data["radius_um"])
        y_limits = self._padded_limits(data["intersections"], lower=0.0)

        figure = plt.figure(figsize=(15.5, 12.0), constrained_layout=False)
        grid = figure.add_gridspec(
            4,
            5,
            width_ratios=[1, 1, 1, 1, 1.25],
            height_ratios=[1.25, 1, 1, 1],
            left=0.055,
            right=0.97,
            bottom=0.065,
            top=0.92,
            hspace=0.38,
            wspace=0.22,
        )

        figure.suptitle(
            self.title,
            fontsize=14,
            fontweight="bold",
            y=0.975,
        )
        figure.text(0.055, 0.942, "Representative examples", fontsize=9, fontweight="bold")

        for col, spec in enumerate(self.conditions):
            condition_df = data.loc[data["condition"] == spec.condition]
            figure.text(
                0.055 + (col + 0.48) * 0.1825,
                0.928,
                f"{spec.label} (n={condition_df['replicate'].nunique()})",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color=spec.color,
            )
            self._plot_representative_grid(
                figure,
                grid[0, col],
                condition_df,
                spec,
                x_limits,
                y_limits,
                representative_cells,
            )

            axes = [figure.add_subplot(grid[row, col]) for row in range(1, 4)]
            self._plot_individual_cells(axes[0], condition_df, spec, x_limits, y_limits)
            self._plot_treatment_overlay(axes[1], condition_df, spec, x_limits, y_limits)
            self._plot_mean_sem(axes[2], condition_df, spec, x_limits, y_limits)
            for row, axis in zip(range(1, 4), axes):
                if col == 0:
                    axis.set_ylabel("Intersections")
                if row == 3:
                    axis.set_xlabel("Radius from soma (µm)")

        figure.text(0.785, 0.942, "Quantitative summaries", fontsize=9, fontweight="bold")
        figure.text(0.016, 0.625, "Individual traces", fontsize=8, rotation=90, va="center")
        figure.text(0.016, 0.415, "Treatment overlay", fontsize=8, rotation=90, va="center")
        figure.text(0.016, 0.205, "Mean ± SEM", fontsize=8, rotation=90, va="center")
        fold_axis = figure.add_subplot(grid[0, 4])
        auc_axis = figure.add_subplot(grid[1, 4])
        spacer_axis = figure.add_subplot(grid[2, 4])
        grand_axis = figure.add_subplot(grid[3, 4])
        spacer_axis.axis("off")
        self._plot_distribution(
            fold_axis,
            metrics,
            metric="fold_change_vs_reference",
            title=f"Fold change vs {self._reference_label()}",
            reference=1.0,
        )
        self._plot_distribution(auc_axis, metrics, metric="auc", title="Area under curve")
        self._plot_grand_overlay(grand_axis, data, x_limits, y_limits)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, facecolor="white")
        plt.close(figure)
        if metrics_output_path is not None:
            metrics_output = Path(metrics_output_path)
            metrics_output.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(metrics_output, index=False)
        return output

    def _selected_data(self) -> pd.DataFrame:
        condition_order = [spec.condition for spec in self.conditions]
        selected = self.grouped_df.loc[self.grouped_df["condition"].isin(condition_order)].copy()
        selected["radius_um"] = pd.to_numeric(selected["radius_um"], errors="coerce")
        selected["intersections"] = pd.to_numeric(selected["intersections"], errors="coerce")
        return selected.dropna(subset=["radius_um", "intersections"])

    def _reference_label(self) -> str:
        return next(
            spec.label for spec in self.conditions if spec.condition == self.reference_condition
        )

    @staticmethod
    def _padded_limits(values: pd.Series, lower: float | None = None) -> tuple[float, float]:
        value_min, value_max = float(values.min()), float(values.max())
        padding = max((value_max - value_min) * 0.04, 1.0)
        return (lower if lower is not None else value_min - padding, value_max + padding)

    @staticmethod
    def _style_axis(axis: object, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> None:
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=6, length=2.5, pad=2)
        axis.grid(False)

    def _plot_representative_grid(
        self,
        figure: object,
        slot: object,
        condition_df: pd.DataFrame,
        spec: ReportCondition,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        representative_cells: int,
    ) -> None:
        replicates = sorted(condition_df["replicate"].unique())
        selected_indices = np.linspace(0, len(replicates) - 1, min(representative_cells, len(replicates)), dtype=int)
        selected_replicates = [replicates[index] for index in selected_indices]
        nested = slot.subgridspec(3, 3, wspace=0.08, hspace=0.08)
        for index in range(9):
            axis = figure.add_subplot(nested[index // 3, index % 3])
            if index < len(selected_replicates):
                replicate = selected_replicates[index]
                trace = condition_df.loc[condition_df["replicate"] == replicate].sort_values("radius_um")
                axis.plot(trace["radius_um"], trace["intersections"], color=spec.color, linewidth=0.8)
                axis.text(0.04, 0.88, f"Cell {replicate}", transform=axis.transAxes, fontsize=4.5)
            axis.set_xlim(x_limits)
            axis.set_ylim(y_limits)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines[["top", "right", "bottom", "left"]].set_linewidth(0.35)

    def _plot_individual_cells(
        self,
        axis: object,
        condition_df: pd.DataFrame,
        spec: ReportCondition,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
    ) -> None:
        for _, trace in condition_df.groupby("replicate"):
            trace = trace.sort_values("radius_um")
            axis.plot(trace["radius_um"], trace["intersections"], color="#6E6E6E", alpha=0.5, linewidth=0.6)
        axis.set_title(spec.label, fontsize=8, color=spec.color)
        self._style_axis(axis, x_limits, y_limits)

    def _plot_treatment_overlay(
        self,
        axis: object,
        condition_df: pd.DataFrame,
        spec: ReportCondition,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
    ) -> None:
        for _, trace in condition_df.groupby("replicate"):
            trace = trace.sort_values("radius_um")
            axis.plot(trace["radius_um"], trace["intersections"], color=spec.color, alpha=0.5, linewidth=0.6)
        self._style_axis(axis, x_limits, y_limits)

    def _plot_mean_sem(
        self,
        axis: object,
        condition_df: pd.DataFrame,
        spec: ReportCondition,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
    ) -> None:
        summary = (
            condition_df.groupby("radius_um", as_index=False)["intersections"]
            .agg(mean="mean", sem="sem")
            .sort_values("radius_um")
        )
        summary["sem"] = summary["sem"].fillna(0.0)
        axis.fill_between(
            summary["radius_um"],
            summary["mean"] - summary["sem"],
            summary["mean"] + summary["sem"],
            color=spec.color,
            alpha=0.2,
            linewidth=0,
        )
        axis.plot(summary["radius_um"], summary["mean"], color=spec.color, linewidth=1.7)
        self._style_axis(axis, x_limits, y_limits)

    def _plot_distribution(
        self,
        axis: object,
        metrics: pd.DataFrame,
        metric: str,
        title: str,
        reference: float | None = None,
    ) -> None:
        ordered = [spec.condition for spec in self.conditions]
        labels = [spec.label for spec in self.conditions]
        data = [metrics.loc[metrics["condition"] == condition, metric].to_numpy() for condition in ordered]
        boxplot = axis.boxplot(
            data,
            vert=False,
            positions=np.arange(len(ordered)),
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#2B2B2B", "linewidth": 1.0},
            boxprops={"linewidth": 0.65},
            whiskerprops={"linewidth": 0.65},
            capprops={"linewidth": 0.65},
        )
        rng = np.random.default_rng(20260725)
        for index, (values, spec) in enumerate(zip(data, self.conditions)):
            boxplot["boxes"][index].set_facecolor(spec.color)
            boxplot["boxes"][index].set_alpha(0.28)
            axis.scatter(values, rng.normal(index, 0.065, len(values)), s=14, color=spec.color, alpha=0.85, linewidths=0)
        if reference is not None:
            axis.axvline(reference, color="#555555", linewidth=0.7, linestyle="--", zorder=0)
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=8, loc="left", pad=4)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=6, length=2.5, pad=2)
        axis.grid(False)

    def _plot_grand_overlay(
        self,
        axis: object,
        data: pd.DataFrame,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
    ) -> None:
        for spec in self.conditions:
            summary = (
                data.loc[data["condition"] == spec.condition]
                .groupby("radius_um", as_index=False)["intersections"]
                .agg(mean="mean", sem="sem")
                .sort_values("radius_um")
            )
            summary["sem"] = summary["sem"].fillna(0.0)
            axis.fill_between(
                summary["radius_um"],
                summary["mean"] - summary["sem"],
                summary["mean"] + summary["sem"],
                color=spec.color,
                alpha=0.12,
                linewidth=0,
            )
            axis.plot(summary["radius_um"], summary["mean"], color=spec.color, linewidth=1.2, label=spec.label)
        self._style_axis(axis, x_limits, y_limits)
        axis.set_title("Grand overlay", fontsize=8, loc="left", pad=4)
        axis.set_xlabel("Radius (µm)")
        axis.set_ylabel("Intersections")
        axis.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.14),
            frameon=False,
            fontsize=6,
            ncol=2,
        )
