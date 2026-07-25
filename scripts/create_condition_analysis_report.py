#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemogenetic_analysis import ConditionAnalysisReport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a four-condition supplementary analysis report with representative "
            "cells, individual traces, mean/SEM, fold-change, and AUC summaries."
        )
    )
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"),
        help="Path to the raw Sholl CSV file.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "output" / "plots" / "no_actuator_analysis_report.png"),
        help="Path for the report figure.",
    )
    parser.add_argument(
        "--metrics-output",
        default=str(REPO_ROOT / "output" / "plots" / "no_actuator_analysis_report_metrics.csv"),
        help="Path for per-cell AUC and fold-change metrics.",
    )
    parser.add_argument(
        "--representative-cells",
        type=int,
        default=9,
        help="Number of representative cells per condition (1-12; default: 9).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = ConditionAnalysisReport.from_raw_csv(args.input)
    output_path = report.render(
        output_path=args.output,
        metrics_output_path=args.metrics_output,
        representative_cells=args.representative_cells,
        dpi=args.dpi,
    )
    print(f"Analysis report saved to: {output_path}")
    print(f"Per-cell metrics saved to: {args.metrics_output}")


if __name__ == "__main__":
    main()
