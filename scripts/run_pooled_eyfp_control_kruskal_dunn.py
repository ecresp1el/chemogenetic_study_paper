#!/usr/bin/env python3
"""Compare Groups I and II treatments against one pooled EYFP control distribution."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemogenetic_analysis import ShollStatsAnalyzer


POOLED_CONTROL = "Pooled_EYFP_Control"


@dataclass(frozen=True)
class PooledEYFPComparison:
    major_group: str
    treatments: tuple[str, ...]


COMPARISONS = (
    PooledEYFPComparison("Group I: treatment condition vs pooled EYFP", ("DREADD_CNO", "LMO7_hCTZ", "PSAM_uPSEM")),
    PooledEYFPComparison("Group II: vehicle condition vs pooled EYFP", ("DREADD_Vehicle", "LMO7_Vehicle", "PSAM_Vehicle")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pooled-EYFP-control Kruskal-Wallis and Dunn contrasts for Groups I and II."
    )
    parser.add_argument("--input", default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"))
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "output" / "stats" / "pooled_eyfp_control_kruskal_dunn"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = ShollStatsAnalyzer.from_raw_csv(args.input)
    auc_df = analyzer.build_auc_per_neuron()
    pooled_control = auc_df.loc[
        auc_df["condition"].isin(["EYFP_Control", "EYFP_Control_Media"])
    ].copy()
    pooled_control["condition"] = POOLED_CONTROL

    kw_frames: list[pd.DataFrame] = []
    dunn_frames: list[pd.DataFrame] = []
    analysis_frames: list[pd.DataFrame] = []
    for comparison in COMPARISONS:
        analysis_df = pd.concat(
            [pooled_control, auc_df.loc[auc_df["condition"].isin(comparison.treatments)]],
            ignore_index=True,
        )
        kw, dunn = analyzer.run_kruskal_dunn(
            analysis_df,
            control_condition=POOLED_CONTROL,
            treatment_conditions=list(comparison.treatments),
        )
        kw.insert(0, "major_group", comparison.major_group)
        dunn.insert(0, "major_group", comparison.major_group)
        analysis_df.insert(0, "major_group", comparison.major_group)
        kw_frames.append(kw)
        dunn_frames.append(dunn)
        analysis_frames.append(analysis_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(analysis_frames, ignore_index=True).to_csv(output_dir / "auc_per_cell_with_pooled_eyfp_control.csv", index=False)
    pd.concat(kw_frames, ignore_index=True).to_csv(output_dir / "kruskal_wallis_by_major_group.csv", index=False)
    pd.concat(dunn_frames, ignore_index=True).to_csv(output_dir / "dunn_pooled_eyfp_control_contrasts.csv", index=False)
    pd.DataFrame([{
        "pooled_control": POOLED_CONTROL,
        "source_conditions": "EYFP_Control | EYFP_Control_Media",
        "n_cells": len(pooled_control),
        "mean_raw_auc": pooled_control["auc"].mean(),
        "median_raw_auc": pooled_control["auc"].median(),
    }]).to_csv(output_dir / "pooled_eyfp_control_summary.csv", index=False)
    print(pd.concat(kw_frames, ignore_index=True).to_string(index=False))
    print("\nDunn shared pooled-EYFP-control contrasts (Holm primary):")
    print(pd.concat(dunn_frames, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
