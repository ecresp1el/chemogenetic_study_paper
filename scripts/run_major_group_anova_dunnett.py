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
    GroupComparison(
        "Group I: treatment condition",
        "EYFP_Control_Media",
        ["DREADD_CNO", "LMO7_hCTZ", "PSAM_uPSEM"],
    ),
    GroupComparison(
        "Group II: vehicle condition",
        "EYFP_Control",
        ["DREADD_Vehicle", "LMO7_Vehicle", "PSAM_Vehicle"],
    ),
    GroupComparison(
        "Group III: ligand/media-only control",
        "None_Vehicle",
        ["None_CNO", "None_hCTZ", "None_uPSEM"],
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-way ANOVA and Dunnett contrasts within each major study group."
    )
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"),
        help="Path to the raw Sholl CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "output" / "stats" / "major_group_anova_dunnett"),
        help="Directory for ANOVA and Dunnett output tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = ShollStatsAnalyzer.from_raw_csv(args.input)
    auc_df = analyzer.build_auc_per_neuron()
    anova_frames: list[pd.DataFrame] = []
    contrast_frames: list[pd.DataFrame] = []
    fold_change_anova_frames: list[pd.DataFrame] = []
    fold_change_contrast_frames: list[pd.DataFrame] = []
    normalization_frames: list[pd.DataFrame] = []
    for comparison in COMPARISONS:
        anova_df, contrast_df = analyzer.run_one_way_anova_dunnett(
            auc_df,
            control_condition=comparison.control,
            treatment_conditions=comparison.treatments,
        )
        anova_df.insert(0, "major_group", comparison.name)
        anova_df.insert(1, "analysis_scale", "raw_auc")
        contrast_df.insert(0, "major_group", comparison.name)
        contrast_df.insert(1, "analysis_scale", "raw_auc")
        anova_frames.append(anova_df)
        contrast_frames.append(contrast_df)

        group_conditions = [comparison.control, *comparison.treatments]
        control_mean_auc = float(
            auc_df.loc[auc_df["condition"] == comparison.control, "auc"].mean()
        )
        fold_change_df = auc_df.copy()
        group_mask = fold_change_df["condition"].isin(group_conditions)
        fold_change_df.loc[group_mask, "auc"] = (
            fold_change_df.loc[group_mask, "auc"] / control_mean_auc
        )
        fold_anova_df, fold_contrast_df = analyzer.run_one_way_anova_dunnett(
            fold_change_df,
            control_condition=comparison.control,
            treatment_conditions=comparison.treatments,
        )
        fold_anova_df.insert(0, "major_group", comparison.name)
        fold_anova_df.insert(1, "analysis_scale", "fold_change_vs_shared_control_mean")
        fold_anova_df.insert(2, "normalization_control_mean_auc", control_mean_auc)
        fold_contrast_df.insert(0, "major_group", comparison.name)
        fold_contrast_df.insert(1, "analysis_scale", "fold_change_vs_shared_control_mean")
        fold_contrast_df.insert(2, "normalization_control_mean_auc", control_mean_auc)
        fold_change_anova_frames.append(fold_anova_df)
        fold_change_contrast_frames.append(fold_contrast_df)
        normalization_frames.append(
            pd.DataFrame(
                [
                    {
                        "major_group": comparison.name,
                        "normalization_control_condition": comparison.control,
                        "normalization_control_mean_auc": control_mean_auc,
                        "formula": "cell_auc / shared_control_mean_auc",
                    }
                ]
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    anova_results = pd.concat(anova_frames, ignore_index=True)
    contrast_results = pd.concat(contrast_frames, ignore_index=True)
    fold_change_anova_results = pd.concat(fold_change_anova_frames, ignore_index=True)
    fold_change_contrast_results = pd.concat(fold_change_contrast_frames, ignore_index=True)
    normalization_results = pd.concat(normalization_frames, ignore_index=True)
    auc_df.to_csv(output_dir / "auc_per_cell.csv", index=False)
    anova_results.to_csv(output_dir / "one_way_anova_raw_auc_by_major_group.csv", index=False)
    contrast_results.to_csv(output_dir / "dunnett_raw_auc_by_major_group.csv", index=False)
    fold_change_anova_results.to_csv(
        output_dir / "one_way_anova_fold_change_by_major_group.csv", index=False
    )
    fold_change_contrast_results.to_csv(
        output_dir / "dunnett_fold_change_by_major_group.csv", index=False
    )
    normalization_results.to_csv(output_dir / "fold_change_normalization_controls.csv", index=False)

    # Backward-compatible aliases for the original raw-AUC outputs.
    anova_results.to_csv(output_dir / "one_way_anova_by_major_group.csv", index=False)
    contrast_results.to_csv(output_dir / "dunnett_contrasts_by_major_group.csv", index=False)
    print("Raw-AUC one-way ANOVA results:")
    print(anova_results.to_string(index=False))
    print("\nRaw-AUC Dunnett-adjusted contrasts:")
    print(contrast_results.to_string(index=False))
    print("\nFold-change one-way ANOVA results:")
    print(fold_change_anova_results.to_string(index=False))
    print("\nFold-change Dunnett-adjusted contrasts:")
    print(fold_change_contrast_results.to_string(index=False))


if __name__ == "__main__":
    main()
