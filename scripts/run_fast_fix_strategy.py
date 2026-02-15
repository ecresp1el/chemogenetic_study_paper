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
            "Run the fast-fix reanalysis sequence: metadata rebuild, AUC recompute, "
            "QC, within-actuator mixed models, and experiment-level delta cross-check."
        )
    )
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"),
        help="Path to raw Sholl CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "output" / "stats" / "fast_fix"),
        help="Directory for fast-fix outputs.",
    )
    parser.add_argument(
        "--min-radius-points",
        type=int,
        default=20,
        help="QC rule: minimum radius points per neuron.",
    )
    parser.add_argument(
        "--min-radius-max",
        type=float,
        default=150.0,
        help="QC rule: minimum max radius (um) per neuron.",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=3.0,
        help="QC rule: IQR multiplier for AUC outlier fences by condition.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = ShollStatsAnalyzer.from_raw_csv(args.input)

    # 1) Clean metadata table
    auc_raw = analyzer.build_auc_per_neuron()
    metadata = analyzer.build_clean_metadata_table(auc_raw)

    # 2) AUC computed from raw curves in one place (auc_raw above)

    # 3) Predefined QC
    qc_flags = analyzer.apply_qc_rules(
        auc_raw,
        min_radius_points=args.min_radius_points,
        min_radius_max_um=args.min_radius_max,
        iqr_multiplier=args.iqr_multiplier,
    )
    auc_qc = qc_flags.loc[qc_flags["qc_pass"]].copy()

    # 4) Within-actuator comparisons on QC-pass
    primary_qc = analyzer.run_primary_within_actuator(auc_qc)

    # Secondary model on QC-pass as companion output
    secondary_coef_qc, secondary_key_qc = analyzer.run_secondary_across_technologies(
        auc_qc
    )

    # 5) Experiment-level delta cross-check
    exp_delta, exp_delta_summary = analyzer.summarize_experiment_deltas(auc_qc)

    paths = {
        "metadata": out_dir / "1_metadata_per_neuron.csv",
        "auc_raw": out_dir / "2_auc_per_neuron_raw.csv",
        "qc_flags": out_dir / "3_qc_flags_per_neuron.csv",
        "auc_qc": out_dir / "3_auc_per_neuron_qc_pass.csv",
        "primary_qc": out_dir / "4_primary_within_actuator_qc.csv",
        "secondary_coef_qc": out_dir
        / "4_secondary_across_technologies_qc_fixed_effects.csv",
        "secondary_key_qc": out_dir / "4_secondary_across_technologies_qc_key_tests.csv",
        "exp_delta": out_dir / "5_experiment_level_deltas.csv",
        "exp_delta_summary": out_dir / "5_experiment_level_delta_summary.csv",
        "notes": out_dir / "fast_fix_notes.md",
    }

    metadata.to_csv(paths["metadata"], index=False)
    auc_raw.to_csv(paths["auc_raw"], index=False)
    qc_flags.to_csv(paths["qc_flags"], index=False)
    auc_qc.to_csv(paths["auc_qc"], index=False)
    primary_qc.to_csv(paths["primary_qc"], index=False)
    secondary_coef_qc.to_csv(paths["secondary_coef_qc"], index=False)
    secondary_key_qc.to_csv(paths["secondary_key_qc"], index=False)
    exp_delta.to_csv(paths["exp_delta"], index=False)
    exp_delta_summary.to_csv(paths["exp_delta_summary"], index=False)

    notes = f"""# Fast Fix Reanalysis Notes

Pipeline order:
1. Metadata rebuild (`1_metadata_per_neuron.csv`)
2. AUC recomputation from raw curves (`2_auc_per_neuron_raw.csv`)
3. Predefined QC filters (`3_qc_flags_per_neuron.csv`, `3_auc_per_neuron_qc_pass.csv`)
4. Within-actuator comparisons + secondary across-technologies on QC-pass
5. Experiment-level delta cross-check

QC rules:
- min_radius_points >= {args.min_radius_points}
- radius_max >= {args.min_radius_max}
- AUC inlier by condition using IQR fence (multiplier={args.iqr_multiplier})

Experiment id definition:
- for primary actuator conditions: actuator + replicate index
- otherwise sample_id if present
- fallback to source_condition + replicate index

Interpretation check:
- Compare the sign of `coef_stimulation` in step 4 with mean experiment deltas in step 5.
"""
    paths["notes"].write_text(notes)

    print("Fast-fix outputs:")
    for key in [
        "metadata",
        "auc_raw",
        "qc_flags",
        "auc_qc",
        "primary_qc",
        "secondary_coef_qc",
        "secondary_key_qc",
        "exp_delta",
        "exp_delta_summary",
        "notes",
    ]:
        print(f"- {paths[key]}")


if __name__ == "__main__":
    main()
