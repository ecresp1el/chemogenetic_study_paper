#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemogenetic_analysis import ShollStatsAnalyzer


PRIMARY_COMPARISONS = {
    "DREADD: hM3Dq + CNO versus hM3Dq + vehicle": ["DREADD_CNO", "DREADD_Vehicle"],
    "LMO7: hCTZ versus vehicle": ["LMO7_hCTZ", "LMO7_Vehicle"],
    "PSAM4-5HT3: uPSEM versus vehicle": ["PSAM_uPSEM", "PSAM_Vehicle"],
}
MAJOR_GROUP_COMPARISONS = {
    "Group I: treatment condition": ["DREADD_CNO", "PSAM_uPSEM", "LMO7_hCTZ", "EYFP_Control_Media"],
    "Group II: vehicle condition": ["DREADD_Vehicle", "PSAM_Vehicle", "LMO7_Vehicle", "EYFP_Control"],
    "Group III: ligand/media-only control": ["None_CNO", "None_uPSEM", "None_hCTZ", "None_Vehicle"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Brown-Forsythe equal-variance tests on per-cell Sholl AUC values.")
    parser.add_argument("--input", default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"), help="Path to the raw Sholl CSV file.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "output" / "stats" / "variance_homogeneity"), help="Directory for result tables.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Test alpha (default: 0.05).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = ShollStatsAnalyzer.from_raw_csv(args.input)
    auc_df = analyzer.build_auc_per_neuron()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = analyzer.assess_auc_variance_homogeneity(auc_df, PRIMARY_COMPARISONS, alpha=args.alpha)
    major_groups = analyzer.assess_auc_variance_homogeneity(auc_df, MAJOR_GROUP_COMPARISONS, alpha=args.alpha)
    auc_df.to_csv(output_dir / "auc_per_cell.csv", index=False)
    primary.to_csv(output_dir / "brown_forsythe_primary_comparisons.csv", index=False)
    major_groups.to_csv(output_dir / "brown_forsythe_major_groups.csv", index=False)
    print("Primary actuator comparisons:")
    print(primary.to_string(index=False))
    print("\nMajor-group comparisons:")
    print(major_groups.to_string(index=False))


if __name__ == "__main__":
    main()
