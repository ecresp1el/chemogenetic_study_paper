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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AUC-based primary and secondary statistics using mixed models "
            "(with OLS fallback if needed)."
        )
    )
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"),
        help="Path to raw Sholl CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "output" / "stats"),
        help="Directory where stats outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = ShollStatsAnalyzer.from_raw_csv(args.input)
    auc_df = analyzer.build_auc_per_neuron()

    primary_df = analyzer.run_primary_within_actuator(auc_df)
    secondary_coef_df, secondary_key_df = analyzer.run_secondary_across_technologies(
        auc_df
    )

    auc_path = out_dir / "auc_per_neuron.csv"
    primary_path = out_dir / "primary_within_actuator_mixedlm.csv"
    secondary_coef_path = out_dir / "secondary_across_technologies_fixed_effects.csv"
    secondary_key_path = out_dir / "secondary_key_tests.csv"
    notes_path = out_dir / "analysis_notes.md"

    auc_df.to_csv(auc_path, index=False)
    primary_df.to_csv(primary_path, index=False)
    secondary_coef_df.to_csv(secondary_coef_path, index=False)
    secondary_key_df.to_csv(secondary_key_path, index=False)

    notes = """# AUC Statistical Analysis Notes

## Primary analysis
- Per-neuron AUC computed via trapezoidal integration across radius.
- Within each actuator:
  - `DREADD_Vehicle` vs `DREADD_CNO`
  - `LMO7_Vehicle` vs `LMO7_hCTZ`
  - `PSAM_Vehicle` vs `PSAM_uPSEM`
- Model: `auc ~ stimulation_binary + (1 | experiment)`

## Secondary analysis
- Across technologies (DREADD, LMO7, PSAM):
- Model: `auc ~ C(actuator) * stimulation_binary + (1 | experiment)`
- Interaction terms test whether stimulation effect magnitude differs by actuator.

## Experiment variable assumption
- The source data does not include an explicit `Experiment` column.
- This pipeline uses:
  - For primary actuator conditions, `"{actuator}_rep{replicate}"` to align vehicle/stim pairs.
  - Otherwise, `sample_id` when present.
  - Fallback to `"{source_condition}_rep{replicate}"`.

## Fallback behavior
- MixedLM is used by default.
- If MixedLM fails to converge/fit, OLS with cluster-robust SE by `experiment` is used.
"""
    notes_path.write_text(notes)

    print(f"AUC per neuron: {auc_path}")
    print(f"Primary analysis: {primary_path}")
    print(f"Secondary fixed effects: {secondary_coef_path}")
    print(f"Secondary key tests: {secondary_key_path}")
    print(f"Notes: {notes_path}")


if __name__ == "__main__":
    main()
