#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemogenetic_analysis import ShollStatsAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess per-condition Sholl AUC normality with Shapiro-Wilk tests and Q-Q plots."
    )
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"),
        help="Path to the raw Sholl CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "output" / "stats" / "distribution_diagnostics"),
        help="Directory for the normality table and Q-Q plot.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Shapiro-Wilk alpha (default: 0.05).")
    return parser.parse_args()


def plot_qq(auc_df, normality_df, output_path: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.65})
    conditions = normality_df["condition"].tolist()
    figure, axes = plt.subplots(3, 4, figsize=(12, 8.5))
    for axis, condition in zip(axes.ravel(), conditions):
        values = auc_df.loc[auc_df["condition"] == condition, "auc"].dropna().to_numpy(dtype=float)
        stats.probplot(values, dist="norm", plot=axis)
        pvalue = float(normality_df.loc[normality_df["condition"] == condition, "shapiro_pvalue"].iloc[0])
        decision = "not rejected" if pvalue >= 0.05 else "possible departure"
        axis.set_title(f"{condition}\nShapiro p={pvalue:.3f} ({decision})", fontsize=7, pad=4)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=6, length=2.5, pad=2)
        axis.grid(False)
    figure.suptitle("AUC Distribution Diagnostics: Normal Q-Q Plots", fontsize=13, fontweight="bold", y=0.98)
    figure.text(0.5, 0.02, "AUC = area under each cell's Sholl curve (intersections versus radius).", ha="center", fontsize=8)
    figure.tight_layout(rect=(0.03, 0.05, 0.99, 0.94), w_pad=2.0, h_pad=2.0)
    figure.savefig(output_path, dpi=300, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    analyzer = ShollStatsAnalyzer.from_raw_csv(args.input)
    auc_df = analyzer.build_auc_per_neuron()
    normality_df = analyzer.assess_auc_normality(auc_df, alpha=args.alpha)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    auc_path = output_dir / "auc_per_cell.csv"
    normality_path = output_dir / "auc_normality_by_condition.csv"
    qq_path = output_dir / "auc_qq_plots_by_condition.png"
    auc_df.to_csv(auc_path, index=False)
    normality_df.to_csv(normality_path, index=False)
    plot_qq(auc_df, normality_df, qq_path)

    print(f"Per-cell AUC: {auc_path}")
    print(f"Normality diagnostics: {normality_path}")
    print(f"Q-Q plots: {qq_path}")
    print(normality_df.to_string(index=False))


if __name__ == "__main__":
    main()
