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
        description=(
            "Generate per-technology overlay plots (mean +/- SEM) for "
            "Activation, Expression, and Effector groups."
        )
    )
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"),
        help="Path to the raw Sholl CSV file.",
    )
    parser.add_argument(
        "--plots-dir",
        default=str(REPO_ROOT / "output" / "plots"),
        help="Directory where overlay plots will be written.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(REPO_ROOT / "output" / "plots" / "mean_sem_by_technology_group.csv"),
        help="CSV path for mean/SEM summary used for plotting.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output DPI for saved PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = ShollDataProcessor(args.input)
    recoded_df = processor.recode_conditions(split_shared_control=True)
    summary_df = processor.summarize_mean_sem_by_technology(recoded_df=recoded_df)

    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output, index=False)

    plot_paths = processor.plot_technology_overlays(
        output_dir=args.plots_dir,
        summary_df=summary_df,
        dpi=args.dpi,
    )

    print(f"Mean/SEM summary saved to: {summary_output}")
    print("Plots saved:")
    for plot_path in plot_paths:
        print(f"  - {plot_path}")


if __name__ == "__main__":
    main()
