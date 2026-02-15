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
        description="Import and tidy wide-format Sholl analysis CSV data."
    )
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"),
        help="Path to the raw Sholl CSV file.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "output" / "tidy_sholl_analysis.csv"),
        help="Path to write tidy CSV output.",
    )
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="Keep rows with missing intersection values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = ShollDataProcessor(args.input)
    output_path = processor.write_tidy_csv(
        args.output, drop_missing_intersections=not args.keep_missing
    )
    tidy_rows = len(processor.tidy(drop_missing_intersections=not args.keep_missing))
    print(f"Tidy CSV saved to: {output_path}")
    print(f"Rows written: {tidy_rows}")


if __name__ == "__main__":
    main()
