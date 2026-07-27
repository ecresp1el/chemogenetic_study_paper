#!/usr/bin/env python3
"""Run planned Dunnett post hoc tests for significant Group I removal combinations."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemogenetic_analysis import ShollStatsAnalyzer


CONTROL = "EYFP_Control_Media"
TREATMENTS = ["DREADD_CNO", "LMO7_hCTZ", "PSAM_uPSEM"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen Group I removal combinations with planned Dunnett post hoc tests."
    )
    parser.add_argument("--input", default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"))
    parser.add_argument(
        "--removal-table",
        default=str(REPO_ROOT / "output" / "stats" / "group_i_multi_cell_removal_anova" / "multi_cell_removal_anova_ranked.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "output" / "stats" / "group_i_removal_dunnett_screen"),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = ShollStatsAnalyzer.from_raw_csv(args.input)
    auc = analyzer.build_auc_per_neuron()
    removal_table = pd.read_csv(args.removal_table)
    candidates = removal_table.loc[removal_table["significant_at_alpha"]].copy()
    combination_rows: list[dict[str, object]] = []
    contrast_rows: list[pd.DataFrame] = []
    for row in candidates.itertuples(index=False):
        removed_cell_ids = row.removed_cell_ids.split(" | ")
        reduced = auc.loc[~auc["cell_id"].isin(removed_cell_ids)]
        anova, dunnett = analyzer.run_one_way_anova_dunnett(
            reduced,
            control_condition=CONTROL,
            treatment_conditions=TREATMENTS,
        )
        dunnett.insert(0, "sensitivity_rank", row.sensitivity_rank_lowest_anova_pvalue)
        dunnett.insert(1, "n_removed", row.n_removed)
        dunnett.insert(2, "removed_cell_ids", row.removed_cell_ids)
        dunnett.insert(3, "removed_conditions", row.removed_conditions)
        dunnett["significant_dunnett_adjusted_alpha"] = dunnett["dunnett_pvalue_adjusted"] < args.alpha
        contrast_rows.append(dunnett)
        significant_conditions = dunnett.loc[
            dunnett["significant_dunnett_adjusted_alpha"], "treatment_condition"
        ].tolist()
        combination_rows.append({
            "sensitivity_rank": row.sensitivity_rank_lowest_anova_pvalue,
            "n_removed": row.n_removed,
            "removed_cell_ids": row.removed_cell_ids,
            "removed_conditions": row.removed_conditions,
            "anova_pvalue": float(anova.loc[0, "anova_pvalue"]),
            "minimum_dunnett_adjusted_pvalue": float(dunnett["dunnett_pvalue_adjusted"].min()),
            "significant_dunnett_conditions": " | ".join(significant_conditions),
            "has_significant_dunnett_contrast": bool(significant_conditions),
        })

    combinations = pd.DataFrame(combination_rows).sort_values(
        ["minimum_dunnett_adjusted_pvalue", "n_removed", "sensitivity_rank"], kind="stable"
    ).reset_index(drop=True)
    combinations.insert(0, "posthoc_rank", combinations.index + 1)
    contrasts = pd.concat(contrast_rows, ignore_index=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combinations.to_csv(output_dir / "removal_combinations_ranked_by_dunnett.csv", index=False)
    contrasts.to_csv(output_dir / "dunnett_contrasts_for_nominal_anova_combinations.csv", index=False)
    pd.DataFrame([{
        "n_nominal_anova_combinations_screened": len(combinations),
        "n_combinations_with_significant_dunnett_contrast": int(combinations["has_significant_dunnett_contrast"].sum()),
        "minimum_dunnett_adjusted_pvalue": combinations["minimum_dunnett_adjusted_pvalue"].min(),
        "alpha": args.alpha,
    }]).to_csv(output_dir / "screen_summary.csv", index=False)
    print(combinations.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
