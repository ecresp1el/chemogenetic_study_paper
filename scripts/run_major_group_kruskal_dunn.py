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
    raw_kruskal_frames: list[pd.DataFrame] = []
    raw_dunn_frames: list[pd.DataFrame] = []
    mean_kruskal_frames: list[pd.DataFrame] = []
    mean_dunn_frames: list[pd.DataFrame] = []
    mean_auc_frames: list[pd.DataFrame] = []
    median_kruskal_frames: list[pd.DataFrame] = []
    median_dunn_frames: list[pd.DataFrame] = []
    median_auc_frames: list[pd.DataFrame] = []
    references: list[dict[str, str | float | int]] = []
    for comparison in COMPARISONS:
        raw_kruskal_df, raw_dunn_df = analyzer.run_kruskal_dunn(
            auc_df, control_condition=comparison.control, treatment_conditions=comparison.treatments
        )
        raw_kruskal_df.insert(0, "major_group", comparison.name)
        raw_dunn_df.insert(0, "major_group", comparison.name)
        raw_kruskal_frames.append(raw_kruskal_df)
        raw_dunn_frames.append(raw_dunn_df)

        conditions = [comparison.control, *comparison.treatments]
        group_auc = auc_df.loc[auc_df["condition"].isin(conditions)].copy()
        control_values = group_auc.loc[group_auc["condition"] == comparison.control, "auc"]
        control_mean, control_median = control_values.mean(), control_values.median()
        references.append({
            "major_group": comparison.name,
            "control_condition": comparison.control,
            "control_n": len(control_values),
            "control_mean_raw_auc": control_mean,
            "control_median_raw_auc": control_median,
        })

        for reference_name, reference_value, output_frames in (
            ("matched_control_mean", control_mean, (mean_auc_frames, mean_kruskal_frames, mean_dunn_frames)),
            ("matched_control_median", control_median, (median_auc_frames, median_kruskal_frames, median_dunn_frames)),
        ):
            normalized_auc = group_auc.copy()
            normalized_auc["auc"] = normalized_auc["auc"] / reference_value
            normalized_auc.insert(0, "major_group", comparison.name)
            normalized_auc["normalization_reference"] = reference_name
            normalized_auc["matched_control_reference_raw_auc"] = reference_value
            output_frames[0].append(normalized_auc)
            kruskal_df, dunn_df = analyzer.run_kruskal_dunn(
                normalized_auc.drop(columns=["major_group", "normalization_reference", "matched_control_reference_raw_auc"]),
                control_condition=comparison.control,
                treatment_conditions=comparison.treatments,
            )
            kruskal_df.insert(0, "major_group", comparison.name)
            dunn_df.insert(0, "major_group", comparison.name)
            output_frames[1].append(kruskal_df)
            output_frames[2].append(dunn_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_kruskal_results = pd.concat(raw_kruskal_frames, ignore_index=True)
    raw_dunn_results = pd.concat(raw_dunn_frames, ignore_index=True)
    mean_kruskal_results = pd.concat(mean_kruskal_frames, ignore_index=True)
    mean_dunn_results = pd.concat(mean_dunn_frames, ignore_index=True)
    median_kruskal_results = pd.concat(median_kruskal_frames, ignore_index=True)
    median_dunn_results = pd.concat(median_dunn_frames, ignore_index=True)
    auc_df.to_csv(output_dir / "auc_per_cell.csv", index=False)
    pd.DataFrame(references).to_csv(output_dir / "normalization_references.csv", index=False)
    pd.concat(mean_auc_frames, ignore_index=True).to_csv(output_dir / "mean_normalized_auc_per_cell.csv", index=False)
    pd.concat(median_auc_frames, ignore_index=True).to_csv(output_dir / "median_normalized_auc_per_cell.csv", index=False)
    # Backward-compatible filename: "normalized" formerly meant mean-normalized.
    pd.concat(mean_auc_frames, ignore_index=True).to_csv(output_dir / "normalized_auc_per_cell.csv", index=False)
    raw_kruskal_results.to_csv(output_dir / "kruskal_wallis_raw_auc_by_major_group.csv", index=False)
    raw_dunn_results.to_csv(output_dir / "dunn_raw_auc_shared_control_contrasts_by_major_group.csv", index=False)
    mean_kruskal_results.to_csv(output_dir / "kruskal_wallis_mean_normalized_auc_by_major_group.csv", index=False)
    mean_dunn_results.to_csv(output_dir / "dunn_mean_normalized_auc_shared_control_contrasts_by_major_group.csv", index=False)
    median_kruskal_results.to_csv(output_dir / "kruskal_wallis_median_normalized_auc_by_major_group.csv", index=False)
    median_dunn_results.to_csv(output_dir / "dunn_median_normalized_auc_shared_control_contrasts_by_major_group.csv", index=False)
    # Backward-compatible mean-normalized filenames.
    mean_kruskal_results.to_csv(output_dir / "kruskal_wallis_normalized_auc_by_major_group.csv", index=False)
    mean_dunn_results.to_csv(output_dir / "dunn_normalized_auc_shared_control_contrasts_by_major_group.csv", index=False)
    # Backward-compatible raw-AUC filenames.
    raw_kruskal_results.to_csv(output_dir / "kruskal_wallis_by_major_group.csv", index=False)
    raw_dunn_results.to_csv(output_dir / "dunn_shared_control_contrasts_by_major_group.csv", index=False)
    print("Raw-AUC Kruskal-Wallis results:")
    print(raw_kruskal_results.to_string(index=False))
    print("\nMean-normalized AUC Kruskal-Wallis results:")
    print(mean_kruskal_results.to_string(index=False))
    print("\nMedian-normalized AUC Kruskal-Wallis results:")
    print(median_kruskal_results.to_string(index=False))
    print("\nDunn contrasts (Holm primary; Bonferroni also reported):")
    print(raw_dunn_results.to_string(index=False))


if __name__ == "__main__":
    main()
