#!/usr/bin/env python3
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


@dataclass(frozen=True)
class GroupComparison:
    name: str
    control: str
    treatments: list[str]


COMPARISONS = (
    GroupComparison("Group I: treatment condition", "EYFP_Control_Media", ["DREADD_CNO", "LMO7_hCTZ", "PSAM_uPSEM"]),
    GroupComparison("Group II: vehicle condition", "EYFP_Control", ["DREADD_Vehicle", "LMO7_Vehicle", "PSAM_Vehicle"]),
    GroupComparison("Group III: ligand/media-only control", "None_Vehicle", ["None_CNO", "None_hCTZ", "None_uPSEM"]),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kruskal-Wallis and Dunn shared-control contrasts within each major group.")
    parser.add_argument("--input", default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"), help="Path to the raw Sholl CSV file.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "output" / "stats" / "major_group_kruskal_dunn"), help="Directory for result tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = ShollStatsAnalyzer.from_raw_csv(args.input)
    auc_df = analyzer.build_auc_per_neuron()
    kruskal_frames: list[pd.DataFrame] = []
    dunn_frames: list[pd.DataFrame] = []
    for comparison in COMPARISONS:
        kruskal_df, dunn_df = analyzer.run_kruskal_dunn(
            auc_df, control_condition=comparison.control, treatment_conditions=comparison.treatments
        )
        kruskal_df.insert(0, "major_group", comparison.name)
        dunn_df.insert(0, "major_group", comparison.name)
        kruskal_frames.append(kruskal_df)
        dunn_frames.append(dunn_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kruskal_results = pd.concat(kruskal_frames, ignore_index=True)
    dunn_results = pd.concat(dunn_frames, ignore_index=True)
    auc_df.to_csv(output_dir / "auc_per_cell.csv", index=False)
    kruskal_results.to_csv(output_dir / "kruskal_wallis_by_major_group.csv", index=False)
    dunn_results.to_csv(output_dir / "dunn_shared_control_contrasts_by_major_group.csv", index=False)
    print("Kruskal-Wallis results:")
    print(kruskal_results.to_string(index=False))
    print("\nDunn contrasts (Holm primary; Bonferroni also reported):")
    print(dunn_results.to_string(index=False))


if __name__ == "__main__":
    main()
