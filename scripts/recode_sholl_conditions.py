#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemogenetic_analysis import ShollDataProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recode tidy Sholl condition labels into analysis groups."
    )
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"),
        help="Path to the raw Sholl CSV file.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "output" / "tidy_sholl_analysis_grouped.csv"),
        help="Path to write grouped output CSV.",
    )
    parser.add_argument(
        "--counts-output",
        default=str(REPO_ROOT / "output" / "cell_counts_by_group_condition.csv"),
        help="Path to write total cell counts by group and condition.",
    )
    parser.add_argument(
        "--no-split-shared-control",
        action="store_true",
        help="Keep EYFP_Vehicle as a single shared-control group label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = ShollDataProcessor(args.input)
    grouped_df = processor.recode_conditions(
        split_shared_control=not args.no_split_shared_control
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_df.to_csv(output_path, index=False)

    counts_df = processor.summarize_cell_counts(
        recoded_df=grouped_df,
        split_shared_control=not args.no_split_shared_control,
    )
    counts_output_path = Path(args.counts_output)
    counts_output_path.parent.mkdir(parents=True, exist_ok=True)
    counts_df.to_csv(counts_output_path, index=False)

    print(f"Grouped CSV saved to: {output_path}")
    print(f"Cell count summary saved to: {counts_output_path}")
    print(f"Rows written: {len(grouped_df)}")
    print("Conditions:")
    for condition in sorted(grouped_df["condition"].dropna().unique()):
        print(f"  - {condition}")
    print("Cell totals by group/condition:")
    print(counts_df.to_string(index=False))


if __name__ == "__main__":
    main()
