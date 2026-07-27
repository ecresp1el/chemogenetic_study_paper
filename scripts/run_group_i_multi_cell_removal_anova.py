#!/usr/bin/env python3
"""Exploratory multi-cell-removal ANOVA sensitivity screen for Group I."""
from __future__ import annotations

import argparse
from itertools import combinations
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


CONDITIONS = ("DREADD_CNO", "LMO7_hCTZ", "PSAM_uPSEM", "EYFP_Control_Media")


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues, dtype=float)
    adjusted[order] = np.minimum(
        1.0,
        np.maximum.accumulate((len(pvalues) - np.arange(len(pvalues))) * pvalues[order]),
    )
    return adjusted


def anova_for_kept(values: np.ndarray, group_codes: np.ndarray, removed: np.ndarray) -> tuple[float, float]:
    keep = np.ones(len(values), dtype=bool)
    keep[removed] = False
    kept_values = values[keep]
    kept_groups = group_codes[keep]
    counts = np.bincount(kept_groups, minlength=len(CONDITIONS))
    if np.any(counts < 2):
        return np.nan, np.nan
    grand_mean = kept_values.mean()
    ss_between = 0.0
    ss_within = 0.0
    for group in range(len(CONDITIONS)):
        group_values = kept_values[kept_groups == group]
        group_mean = group_values.mean()
        ss_between += len(group_values) * (group_mean - grand_mean) ** 2
        ss_within += np.sum((group_values - group_mean) ** 2)
    f_statistic = (ss_between / (len(CONDITIONS) - 1)) / (ss_within / (len(kept_values) - len(CONDITIONS)))
    return float(f_statistic), float(stats.f.sf(f_statistic, len(CONDITIONS) - 1, len(kept_values) - len(CONDITIONS)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exhaustive/random multi-cell-removal ANOVA sensitivity analysis for Group I."
    )
    parser.add_argument("--input", default=str(REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "output" / "stats" / "group_i_multi_cell_removal_anova"))
    parser.add_argument(
        "--draws-per-size",
        type=int,
        default=3000,
        help="Seeded random combinations for each removal size >= 3 (default: 3,000).",
    )
    parser.add_argument("--max-removals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auc = ShollStatsAnalyzer.from_raw_csv(args.input).build_auc_per_neuron()
    group_i = auc.loc[auc["condition"].isin(CONDITIONS)].copy().reset_index(drop=True)
    values = group_i["auc"].to_numpy(float)
    group_codes = pd.Categorical(group_i["condition"], categories=CONDITIONS).codes
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []

    def add_combination(removed_indices: np.ndarray, method: str) -> None:
        statistic, pvalue = anova_for_kept(values, group_codes, removed_indices)
        removed = group_i.iloc[removed_indices]
        rows.append({
            "removal_method": method,
            "n_removed": len(removed_indices),
            "removed_cell_ids": " | ".join(removed["cell_id"]),
            "removed_conditions": " | ".join(removed["condition"]),
            "removed_auc_values": " | ".join(f"{value:g}" for value in removed["auc"]),
            "anova_f_statistic": statistic,
            "anova_pvalue": pvalue,
            "significant_at_alpha": pvalue < args.alpha,
        })

    # Every pair is tested exactly; larger spaces are sampled reproducibly.
    for pair in combinations(range(len(group_i)), 2):
        add_combination(np.asarray(pair), "exhaustive_pair")
    for n_removed in range(3, args.max_removals + 1):
        seen: set[tuple[int, ...]] = set()
        while len(seen) < args.draws_per_size:
            candidate = tuple(sorted(rng.choice(len(group_i), size=n_removed, replace=False).tolist()))
            if candidate in seen:
                continue
            # Never reduce any condition below two cells.
            candidate_counts = np.bincount(group_codes[list(candidate)], minlength=len(CONDITIONS))
            if np.any(np.bincount(group_codes, minlength=len(CONDITIONS)) - candidate_counts < 2):
                continue
            seen.add(candidate)
            add_combination(np.asarray(candidate), "seeded_random")

    results = pd.DataFrame(rows).sort_values(["anova_pvalue", "n_removed"], kind="stable").reset_index(drop=True)
    results.insert(0, "sensitivity_rank_lowest_anova_pvalue", results.index + 1)
    results["anova_pvalue_holm_adjusted_all_removal_sets"] = holm_adjust(
        results["anova_pvalue"].to_numpy(float)
    )
    results["significant_after_holm_all_removal_sets"] = (
        results["anova_pvalue_holm_adjusted_all_removal_sets"] < args.alpha
    )
    significant = results.loc[results["significant_at_alpha"]]
    cell_rows: list[dict[str, object]] = []
    if not significant.empty:
        significant_sets = [set(value.split(" | ")) for value in significant["removed_cell_ids"]]
        all_sets = [set(value.split(" | ")) for value in results["removed_cell_ids"]]
        for cell in group_i.itertuples(index=False):
            observed = sum(cell.cell_id in removed_set for removed_set in significant_sets)
            expected_rate = sum(cell.cell_id in removed_set for removed_set in all_sets) / len(all_sets)
            observed_rate = observed / len(significant_sets)
            cell_rows.append({
                "cell_id": cell.cell_id,
                "condition": cell.condition,
                "auc": cell.auc,
                "n_significant_combinations_containing_cell": observed,
                "proportion_significant_combinations_containing_cell": observed_rate,
                "proportion_all_tested_combinations_containing_cell": expected_rate,
                "enrichment_ratio_vs_all_tested_combinations": observed_rate / expected_rate if expected_rate else np.nan,
            })
    cell_summary = pd.DataFrame(cell_rows).sort_values(
        ["n_significant_combinations_containing_cell", "enrichment_ratio_vs_all_tested_combinations"],
        ascending=False,
        kind="stable",
    ) if cell_rows else pd.DataFrame()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "multi_cell_removal_anova_ranked.csv", index=False)
    cell_summary.to_csv(output_dir / "cell_recurrence_among_significant_combinations.csv", index=False)
    pd.DataFrame([{
        "n_group_i_cells": len(group_i),
        "exhaustive_pair_combinations": int((len(group_i) * (len(group_i) - 1)) / 2),
        "random_draws_per_removal_size": args.draws_per_size,
        "random_removal_sizes": f"3 through {args.max_removals}",
        "seed": args.seed,
        "alpha": args.alpha,
        "n_combinations_tested": len(results),
        "n_significant_combinations": len(significant),
        "n_significant_after_holm_all_removal_sets": int(results["significant_after_holm_all_removal_sets"].sum()),
        "minimum_anova_pvalue": results["anova_pvalue"].min(),
        "minimum_holm_adjusted_anova_pvalue": results["anova_pvalue_holm_adjusted_all_removal_sets"].min(),
    }]).to_csv(output_dir / "analysis_summary.csv", index=False)
    print(results.head(20).to_string(index=False))
    print(f"\nTested {len(results)} combinations; {len(significant)} had ANOVA p < {args.alpha}.")


if __name__ == "__main__":
    main()
