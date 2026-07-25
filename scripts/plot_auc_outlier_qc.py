#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemogenetic_analysis import ReportCondition, ShollStatsAnalyzer


CONTROL_GRAY = "#9AA0A6"
FLAG_COLOR = "#B23A48"


@dataclass(frozen=True)
class QCPane:
    title: str
    conditions: tuple[ReportCondition, ...]


class AUCOutlierQCDiagnostic:
    """Plot condition-level AUC prediction intervals and candidate outliers."""

    PANES = (
        QCPane(
            "Group III — Ligand/media-only control",
            (
                ReportCondition("None_CNO", "CNO only", "#B7D7A8"),
                ReportCondition("None_hCTZ", "hCTZ only", "#A9DCE5"),
                ReportCondition("None_uPSEM", "uPSEM only", "#C9C2E4"),
                ReportCondition("None_Vehicle", "Media only", CONTROL_GRAY),
            ),
        ),
        QCPane(
            "Group II — Vehicle condition",
            (
                ReportCondition("DREADD_Vehicle", "hM3Dq + vehicle", CONTROL_GRAY),
                ReportCondition("LMO7_Vehicle", "LMO7 + vehicle", CONTROL_GRAY),
                ReportCondition("PSAM_Vehicle", "PSAM4-5HT3 + vehicle", CONTROL_GRAY),
                ReportCondition("EYFP_Control", "EYFP control", CONTROL_GRAY),
            ),
        ),
        QCPane(
            "Group I — Treatment condition",
            (
                ReportCondition("DREADD_CNO", "hM3Dq + CNO", "#6AA84F"),
                ReportCondition("LMO7_hCTZ", "LMO7 + hCTZ", "#46B3C3"),
                ReportCondition("PSAM_uPSEM", "PSAM4-5HT3 + uPSEM", "#8E7CC3"),
                ReportCondition("EYFP_Control_Media", "EYFP + media", CONTROL_GRAY),
            ),
        ),
    )

    def __init__(self, auc_df: pd.DataFrame):
        self.auc_df = auc_df.copy()

    @classmethod
    def from_raw_csv(cls, raw_csv_path: str | Path) -> "AUCOutlierQCDiagnostic":
        analyzer = ShollStatsAnalyzer.from_raw_csv(raw_csv_path)
        return cls(analyzer.build_auc_per_neuron())

    def summarize(self) -> pd.DataFrame:
        """Compute 95% prediction intervals and candidate flags within each condition."""
        rows: list[dict[str, object]] = []
        pane_by_condition = {
            spec.condition: pane.title for pane in self.PANES for spec in pane.conditions
        }
        label_by_condition = {
            spec.condition: spec.label for pane in self.PANES for spec in pane.conditions
        }
        selected = self.auc_df.loc[self.auc_df["condition"].isin(pane_by_condition)].copy()
        for condition, condition_df in selected.groupby("condition", sort=False):
            values = condition_df["auc"].to_numpy(dtype=float)
            n_total = len(values)
            mean_auc = float(np.mean(values))
            sd_auc = float(np.std(values, ddof=1))
            critical_value = float(t.ppf(0.975, n_total - 1))
            half_width = critical_value * sd_auc * np.sqrt(1.0 + 1.0 / n_total)
            lower = mean_auc - half_width
            upper = mean_auc + half_width
            flags = (values < lower) | (values > upper)
            for (_, cell), is_flagged in zip(condition_df.iterrows(), flags):
                rows.append(
                    {
                        "major_group": pane_by_condition[condition],
                        "condition": condition,
                        "condition_label": label_by_condition[condition],
                        "cell_id": cell["cell_id"],
                        "replicate": int(cell["replicate"]),
                        "auc": float(cell["auc"]),
                        "mean_auc": mean_auc,
                        "sd_auc": sd_auc,
                        "prediction_interval_95_low": lower,
                        "prediction_interval_95_high": upper,
                        "candidate_outlier_95_pi": bool(is_flagged),
                    }
                )
        return pd.DataFrame(rows)

    def render(self, output_path: str | Path, qc_table_path: str | Path, dpi: int = 300) -> Path:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.65})
        qc_df = self.summarize()
        x_max = max(float(qc_df["auc"].max()), float(qc_df["prediction_interval_95_high"].max())) * 1.05
        figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.5), sharex=True)
        for axis, pane in zip(axes, self.PANES):
            self._plot_pane(axis, pane, qc_df, x_max)

        figure.suptitle("Condition-Level AUC Outlier QC", fontsize=14, fontweight="bold", y=0.97)
        figure.text(
            0.5,
            0.02,
            "Bars show the 95% prediction interval for an individual AUC within each condition. "
            "Filled points are retained; open red points are candidate outliers.",
            ha="center",
            fontsize=8,
        )
        figure.tight_layout(rect=(0.02, 0.06, 0.99, 0.92), w_pad=3.4)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, facecolor="white")
        plt.close(figure)

        qc_table = Path(qc_table_path)
        qc_table.parent.mkdir(parents=True, exist_ok=True)
        qc_df.to_csv(qc_table, index=False)
        return output

    @staticmethod
    def _plot_pane(axis, pane: QCPane, qc_df: pd.DataFrame, x_max: float) -> None:
        rng = np.random.default_rng(20260727)
        y_labels = []
        for position, spec in enumerate(pane.conditions):
            condition_df = qc_df.loc[qc_df["condition"] == spec.condition].copy()
            if condition_df.empty:
                continue
            lower = float(condition_df["prediction_interval_95_low"].iloc[0])
            upper = float(condition_df["prediction_interval_95_high"].iloc[0])
            mean_auc = float(condition_df["mean_auc"].iloc[0])
            flagged = condition_df["candidate_outlier_95_pi"].to_numpy(dtype=bool)
            retained = condition_df.loc[~flagged]
            excluded = condition_df.loc[flagged]

            axis.hlines(position, lower, upper, color=spec.color, linewidth=7, alpha=0.32, zorder=1)
            axis.vlines([lower, upper], position - 0.12, position + 0.12, color=spec.color, linewidth=0.9, zorder=2)
            axis.vlines(mean_auc, position - 0.15, position + 0.15, color="#333333", linewidth=1.1, zorder=3)
            axis.scatter(
                retained["auc"],
                rng.normal(position, 0.055, len(retained)),
                color=spec.color,
                s=17,
                alpha=0.9,
                linewidths=0,
                zorder=4,
            )
            if not excluded.empty:
                axis.scatter(
                    excluded["auc"],
                    rng.normal(position, 0.055, len(excluded)),
                    facecolors="white",
                    edgecolors=FLAG_COLOR,
                    s=33,
                    linewidths=1.1,
                    zorder=5,
                )
            y_labels.append(f"{spec.label} (plotted {len(retained)}/{len(condition_df)}; excluded {len(excluded)})")

        axis.set_yticks(np.arange(len(y_labels)), y_labels)
        axis.invert_yaxis()
        axis.set_xlim(min(0, float(qc_df["prediction_interval_95_low"].min())), x_max)
        axis.set_xlabel("AUC")
        axis.set_title(pane.title, fontsize=10, fontweight="bold", pad=10)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=7, length=2.5, pad=3)
        axis.grid(False)


def main() -> None:
    diagnostic = AUCOutlierQCDiagnostic.from_raw_csv(
        REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"
    )
    output = diagnostic.render(
        REPO_ROOT / "output" / "plots" / "auc_outlier_qc_95_prediction_interval.png",
        REPO_ROOT / "output" / "stats" / "auc_outlier_qc_95_prediction_interval.csv",
    )
    summary = diagnostic.summarize()
    counts = (
        summary.groupby(["major_group", "condition_label"], as_index=False)
        .agg(n_total=("cell_id", "size"), n_candidate_excluded=("candidate_outlier_95_pi", "sum"))
    )
    counts["n_plotted"] = counts["n_total"] - counts["n_candidate_excluded"]
    print(f"AUC QC figure saved to: {output}")
    print("Candidate outlier summary:")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
