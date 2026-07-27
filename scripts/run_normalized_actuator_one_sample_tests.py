#!/usr/bin/env python3
"""Test each normalized actuator-vector condition against fold change = 1."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemogenetic_analysis import ShollStatsAnalyzer


ACTUATOR_GROUPS = {
    "Group I: treatment condition": {
        "control": "EYFP_Control_Media",
        "conditions": ("DREADD_CNO", "LMO7_hCTZ", "PSAM_uPSEM"),
    },
    "Group II: vehicle condition": {
        "control": "EYFP_Control",
        "conditions": ("DREADD_Vehicle", "LMO7_Vehicle", "PSAM_Vehicle"),
    },
}


def holm_adjust(pvalues: list[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-sided Wilcoxon signed-rank tests of normalized actuator conditions against 1."
    )
    parser.add_argument("--input", default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"))
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "output" / "stats" / "normalized_actuator_one_sample_tests"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auc = ShollStatsAnalyzer.from_raw_csv(args.input).build_auc_per_neuron()
    pooled_eyfp_mean = auc.loc[
        auc["condition"].isin(["EYFP_Control", "EYFP_Control_Media"]), "auc"
    ].mean()
    variants = (
        ("matched_control_mean", "AUC / own matched-control mean"),
        ("matched_control_median", "AUC / own matched-control median"),
        ("pooled_EYFP_mean", "AUC / pooled EYFP mean"),
    )
    results: list[dict[str, object]] = []
    values_out: list[pd.DataFrame] = []
    for variant, definition in variants:
        variant_rows: list[dict[str, object]] = []
        for major_group, spec in ACTUATOR_GROUPS.items():
            control_values = auc.loc[auc["condition"] == spec["control"], "auc"]
            denominator = {
                "matched_control_mean": control_values.mean(),
                "matched_control_median": control_values.median(),
                "pooled_EYFP_mean": pooled_eyfp_mean,
            }[variant]
            for condition in spec["conditions"]:
                normalized = auc.loc[auc["condition"] == condition, "auc"] / denominator
                test = stats.wilcoxon(
                    normalized - 1.0,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                variant_rows.append({
                    "normalization": variant,
                    "normalization_definition": definition,
                    "major_group": major_group,
                    "condition": condition,
                    "reference_value": 1.0,
                    "normalization_denominator_raw_auc": denominator,
                    "n_cells": len(normalized),
                    "fold_change_mean": normalized.mean(),
                    "fold_change_median": normalized.median(),
                    "wilcoxon_w_statistic": test.statistic,
                    "wilcoxon_pvalue_unadjusted": test.pvalue,
                })
                values_out.append(pd.DataFrame({
                    "normalization": variant,
                    "major_group": major_group,
                    "condition": condition,
                    "normalized_auc": normalized,
                }))
        adjusted = holm_adjust([float(row["wilcoxon_pvalue_unadjusted"]) for row in variant_rows])
        for row, pvalue in zip(variant_rows, adjusted):
            row["wilcoxon_pvalue_holm_adjusted"] = pvalue
        results.extend(variant_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_dir / "wilcoxon_one_sample_vs_fold_change_one.csv", index=False)
    pd.concat(values_out, ignore_index=True).to_csv(output_dir / "normalized_actuator_values.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
