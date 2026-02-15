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
            "Analyze radius support and zero-intersection trends by technology. "
            "Writes CSV summaries plus one diagnostic plot per technology."
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
        help="Directory where diagnostic plots will be written.",
    )
    parser.add_argument(
        "--coverage-output",
        default=str(REPO_ROOT / "output" / "plots" / "radius_coverage_summary.csv"),
        help="CSV path for radius coverage summary.",
    )
    parser.add_argument(
        "--majority-output",
        default=str(REPO_ROOT / "output" / "plots" / "radius_majority_windows.csv"),
        help="CSV path for majority-shared radius windows.",
    )
    parser.add_argument(
        "--majority-threshold",
        type=float,
        default=0.5,
        help="Threshold used for majority flags (default: 0.5).",
    )
    parser.add_argument(
        "--highlight-radius",
        type=float,
        default=200.0,
        help="Radius to highlight with a vertical guide line (default: 200).",
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

    coverage_df = processor.summarize_radius_coverage(
        majority_threshold=args.majority_threshold
    )
    coverage_output = Path(args.coverage_output)
    coverage_output.parent.mkdir(parents=True, exist_ok=True)
    coverage_df.to_csv(coverage_output, index=False)

    majority_df = processor.summarize_majority_windows(
        coverage_df=coverage_df,
        majority_threshold=args.majority_threshold,
        radius_min=0.0,
    )
    majority_output = Path(args.majority_output)
    majority_output.parent.mkdir(parents=True, exist_ok=True)
    majority_df.to_csv(majority_output, index=False)

    plot_paths = processor.plot_radius_coverage_by_technology(
        output_dir=args.plots_dir,
        coverage_df=coverage_df,
        majority_threshold=args.majority_threshold,
        highlight_radius=args.highlight_radius,
        dpi=args.dpi,
    )

    print(f"Radius coverage summary saved to: {coverage_output}")
    print(f"Majority radius windows saved to: {majority_output}")
    print("Coverage plots saved:")
    for plot_path in plot_paths:
        print(f"  - {plot_path}")


if __name__ == "__main__":
    main()
