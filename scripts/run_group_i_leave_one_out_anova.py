#!/usr/bin/env python3
"""Leave-one-cell-out ANOVA sensitivity analysis for Group I Sholl AUC."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemogenetic_analysis import ShollStatsAnalyzer


GROUP_I_CONDITIONS = ("DREADD_CNO", "LMO7_hCTZ", "PSAM_uPSEM", "EYFP_Control_Media")


def one_way_anova(frame: pd.DataFrame) -> tuple[float, float]:
    samples = [frame.loc[frame["condition"] == condition, "auc"].to_numpy() for condition in GROUP_I_CONDITIONS]
    result = stats.f_oneway(*samples)
    return float(result.statistic), float(result.pvalue)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leave-one-cell-out one-way ANOVA sensitivity analysis for Group I."
    )
    parser.add_argument("--input", default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"))
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "output" / "stats" / "group_i_leave_one_out_anova"),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auc = ShollStatsAnalyzer.from_raw_csv(args.input).build_auc_per_neuron()
    group_i = auc.loc[auc["condition"].isin(GROUP_I_CONDITIONS)].copy()
    baseline_f, baseline_p = one_way_anova(group_i)
    rows: list[dict[str, object]] = []
    for cell in group_i.itertuples(index=False):
        reduced = group_i.loc[group_i["cell_id"] != cell.cell_id]
        statistic, pvalue = one_way_anova(reduced)
        counts = reduced.groupby("condition").size().reindex(GROUP_I_CONDITIONS)
        rows.append(
            {
                "removed_cell_id": cell.cell_id,
                "removed_condition": cell.condition,
                "removed_source_condition": cell.source_condition,
                "removed_replicate": cell.replicate,
                "removed_experiment": cell.experiment,
                "removed_auc": cell.auc,
                "n_after_removal": len(reduced),
                "n_by_condition_after_removal": " | ".join(f"{name}:{count}" for name, count in counts.items()),
                "anova_f_statistic_after_removal": statistic,
                "anova_pvalue_after_removal": pvalue,
                "pvalue_change_from_full_data": pvalue - baseline_p,
                "would_be_significant_alpha": pvalue < args.alpha,
                "alpha": args.alpha,
            }
        )
    results = pd.DataFrame(rows).sort_values(
        ["anova_pvalue_after_removal", "removed_condition", "removed_cell_id"],
        kind="stable",
    ).reset_index(drop=True)
    results.insert(0, "sensitivity_rank_lowest_anova_pvalue", results.index + 1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "leave_one_cell_out_anova_ranked.csv", index=False)
    pd.DataFrame([{
        "analysis": "Group I one-way ANOVA leave-one-cell-out sensitivity",
        "conditions": " | ".join(GROUP_I_CONDITIONS),
        "n_cells_full_data": len(group_i),
        "anova_f_statistic_full_data": baseline_f,
        "anova_pvalue_full_data": baseline_p,
        "alpha": args.alpha,
        "n_single_cell_removals_with_significant_anova": int(results["would_be_significant_alpha"].sum()),
    }]).to_csv(output_dir / "analysis_summary.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
