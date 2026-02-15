from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .sholl_processor import ShollDataProcessor


@dataclass
class ModelOutput:
    method: str
    result: Any
    converged: bool


class ShollStatsAnalyzer:
    """
    Statistical analysis on per-neuron Sholl AUC values.

    Primary:
    - Within actuator stimulation effects:
      DREADD_Vehicle vs DREADD_CNO
      LMO7_Vehicle vs LMO7_hCTZ
      PSAM_Vehicle vs PSAM_uPSEM
    - Mixed model: auc ~ stimulation_binary + (1 | experiment)

    Secondary:
    - Across technologies:
      auc ~ C(actuator) * stimulation_binary + (1 | experiment)
    """

    CONDITION_METADATA = {
        "DREADD_CNO": {"actuator": "DREADD", "stimulation": "Stimulated", "ligand": "CNO"},
        "DREADD_Vehicle": {"actuator": "DREADD", "stimulation": "Vehicle", "ligand": "none"},
        "LMO7_hCTZ": {"actuator": "LMO7", "stimulation": "Stimulated", "ligand": "hCTZ"},
        "LMO7_Vehicle": {"actuator": "LMO7", "stimulation": "Vehicle", "ligand": "none"},
        "PSAM_uPSEM": {"actuator": "PSAM", "stimulation": "Stimulated", "ligand": "uPSEM"},
        "PSAM_Vehicle": {"actuator": "PSAM", "stimulation": "Vehicle", "ligand": "none"},
        "EYFP_Control": {"actuator": "EYFP", "stimulation": "Vehicle", "ligand": "none"},
        "EYFP_Control_Media": {"actuator": "EYFP", "stimulation": "Stimulated", "ligand": "none"},
        "None_CNO": {
            "actuator": "NoActuator",
            "stimulation": "Stimulated",
            "ligand": "CNO",
        },
        "None_hCTZ": {
            "actuator": "NoActuator",
            "stimulation": "Stimulated",
            "ligand": "hCTZ",
        },
        "None_uPSEM": {
            "actuator": "NoActuator",
            "stimulation": "Stimulated",
            "ligand": "uPSEM",
        },
        "None_Vehicle": {
            "actuator": "NoActuator",
            "stimulation": "Vehicle",
            "ligand": "none",
        },
    }
    PRIMARY_CONDITION_SET = {
        "DREADD_CNO",
        "DREADD_Vehicle",
        "LMO7_hCTZ",
        "LMO7_Vehicle",
        "PSAM_uPSEM",
        "PSAM_Vehicle",
    }

    ACTUATOR_ORDER = ["DREADD", "LMO7", "PSAM"]

    def __init__(self, grouped_df: pd.DataFrame):
        self.grouped_df = grouped_df.copy()

    @classmethod
    def from_raw_csv(cls, raw_csv_path: str | Path) -> "ShollStatsAnalyzer":
        processor = ShollDataProcessor(raw_csv_path)
        grouped_df = processor.recode_conditions(split_shared_control=True)
        return cls(grouped_df)

    def build_auc_per_neuron(self) -> pd.DataFrame:
        """Compute AUC (trapezoidal) across radius for each neuron/cell."""
        df = self.grouped_df.copy()
        df["radius_um"] = pd.to_numeric(df["radius_um"], errors="coerce")
        df["intersections"] = pd.to_numeric(df["intersections"], errors="coerce")

        sample_series = df["sample_id"].where(df["sample_id"].notna(), "NA").astype(str)
        df["cell_id"] = (
            df["source_condition"].astype(str)
            + "__r"
            + df["replicate"].astype(int).astype(str)
            + "__s"
            + sample_series
        )

        rows: list[dict[str, Any]] = []
        group_cols = [
            "analysis_group",
            "condition",
            "source_condition",
            "replicate",
            "sample_id",
            "cell_id",
        ]
        for keys, g in df.groupby(group_cols, dropna=False):
            g = g.sort_values("radius_um")
            x = g["radius_um"].to_numpy(dtype=float)
            y = g["intersections"].to_numpy(dtype=float)
            if len(x) == 0:
                continue
            auc = float(np.trapezoid(y, x)) if len(x) > 1 else 0.0
            (
                analysis_group,
                condition,
                source_condition,
                replicate,
                sample_id,
                cell_id,
            ) = keys
            meta = self.CONDITION_METADATA.get(
                condition,
                {"actuator": "Unknown", "stimulation": "Other", "ligand": "none"},
            )
            actuator = str(meta["actuator"])
            stimulation = str(meta["stimulation"])
            ligand = str(meta["ligand"])

            # Experiment proxy:
            # - use sample_id when available
            # - otherwise use actuator+replicate for alignment across conditions
            # - fallback to source_condition+replicate
            if condition in self.PRIMARY_CONDITION_SET and actuator in self.ACTUATOR_ORDER:
                experiment = f"{actuator}_rep{int(replicate)}"
            elif pd.notna(sample_id):
                experiment = str(sample_id)
            elif actuator != "Unknown":
                experiment = f"{actuator}_rep{int(replicate)}"
            else:
                experiment = f"{source_condition}_rep{int(replicate)}"

            rows.append(
                {
                    "analysis_group": analysis_group,
                    "condition": condition,
                    "source_condition": source_condition,
                    "replicate": int(replicate),
                    "sample_id": sample_id,
                    "cell_id": cell_id,
                    "experiment": experiment,
                    "n_radius_points": int(len(x)),
                    "radius_min": float(np.nanmin(x)),
                    "radius_max": float(np.nanmax(x)),
                    "auc": auc,
                    "actuator": actuator,
                    "stimulation": stimulation,
                    "ligand": ligand,
                    "is_primary_condition": condition in self.PRIMARY_CONDITION_SET,
                }
            )

        auc_df = pd.DataFrame(rows)
        auc_df = auc_df.sort_values(["condition", "replicate"]).reset_index(drop=True)
        return auc_df

    def build_clean_metadata_table(self, auc_df: pd.DataFrame) -> pd.DataFrame:
        """Build one-row-per-neuron metadata table required for reanalysis."""
        meta_df = auc_df[
            [
                "cell_id",
                "experiment",
                "actuator",
                "stimulation",
                "ligand",
                "condition",
                "source_condition",
                "replicate",
                "sample_id",
            ]
        ].copy()
        meta_df = meta_df.rename(
            columns={
                "cell_id": "neuron_id",
                "experiment": "experiment_id",
                "stimulation": "condition_vehicle_vs_stim",
            }
        )
        meta_df = meta_df.sort_values(["actuator", "condition", "replicate"]).reset_index(
            drop=True
        )
        return meta_df

    def apply_qc_rules(
        self,
        auc_df: pd.DataFrame,
        min_radius_points: int = 20,
        min_radius_max_um: float = 150.0,
        iqr_multiplier: float = 3.0,
    ) -> pd.DataFrame:
        """
        Apply predefined QC rules (non-manual) and return row-level QC flags.

        Rules:
        - Minimum number of radius points.
        - Minimum max radius coverage.
        - AUC inlier within condition using IQR fences.
        """
        qc_df = auc_df.copy()
        qc_df["qc_min_points"] = qc_df["n_radius_points"] >= int(min_radius_points)
        qc_df["qc_radius_coverage"] = qc_df["radius_max"] >= float(min_radius_max_um)
        qc_df["qc_auc_inlier"] = True

        for condition, idx in qc_df.groupby("condition").groups.items():
            sub = qc_df.loc[idx, "auc"]
            if len(sub) < 4:
                continue
            q1 = float(sub.quantile(0.25))
            q3 = float(sub.quantile(0.75))
            iqr = q3 - q1
            low = q1 - iqr_multiplier * iqr
            high = q3 + iqr_multiplier * iqr
            qc_df.loc[idx, "qc_auc_inlier"] = (sub >= low) & (sub <= high)

        qc_df["qc_pass"] = (
            qc_df["qc_min_points"] & qc_df["qc_radius_coverage"] & qc_df["qc_auc_inlier"]
        )
        qc_df["qc_fail_reasons"] = ""
        qc_df.loc[~qc_df["qc_min_points"], "qc_fail_reasons"] += "min_points;"
        qc_df.loc[~qc_df["qc_radius_coverage"], "qc_fail_reasons"] += "radius_coverage;"
        qc_df.loc[~qc_df["qc_auc_inlier"], "qc_fail_reasons"] += "auc_outlier;"
        qc_df["qc_fail_reasons"] = qc_df["qc_fail_reasons"].str.rstrip(";")
        qc_df = qc_df.sort_values(["condition", "replicate"]).reset_index(drop=True)
        return qc_df

    def run_primary_within_actuator(self, auc_df: pd.DataFrame) -> pd.DataFrame:
        """Run within-actuator mixed models on AUC."""
        rows: list[dict[str, Any]] = []
        for actuator in self.ACTUATOR_ORDER:
            sub = auc_df.loc[auc_df["actuator"] == actuator].copy()
            if sub.empty:
                continue
            sub["stimulation"] = pd.Categorical(
                sub["stimulation"], categories=["Vehicle", "Stimulated"]
            )
            sub["stimulation_binary"] = (sub["stimulation"] == "Stimulated").astype(int)

            model_out = self._fit_mixedlm_with_fallback(
                formula="auc ~ stimulation_binary",
                data=sub,
                group_col="experiment",
            )

            coeff = self._extract_term(model_out.result, "stimulation_binary")
            summary_row = {
                "actuator": actuator,
                "model_method": model_out.method,
                "converged": model_out.converged,
                "n_neurons": int(len(sub)),
                "n_experiments": int(sub["experiment"].nunique()),
                "mean_auc_vehicle": float(
                    sub.loc[sub["stimulation"] == "Vehicle", "auc"].mean()
                ),
                "mean_auc_stimulated": float(
                    sub.loc[sub["stimulation"] == "Stimulated", "auc"].mean()
                ),
                "delta_stim_minus_vehicle": float(
                    sub.loc[sub["stimulation"] == "Stimulated", "auc"].mean()
                    - sub.loc[sub["stimulation"] == "Vehicle", "auc"].mean()
                ),
                "coef_stimulation": coeff["coef"],
                "se_stimulation": coeff["se"],
                "pvalue_stimulation": coeff["pvalue"],
                "ci_low_stimulation": coeff["ci_low"],
                "ci_high_stimulation": coeff["ci_high"],
            }
            rows.append(summary_row)

        return pd.DataFrame(rows)

    def run_secondary_across_technologies(
        self, auc_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run across-technology mixed model and return fixed effects + key tests."""
        sub = auc_df.loc[auc_df["actuator"].isin(self.ACTUATOR_ORDER)].copy()
        sub["actuator"] = pd.Categorical(sub["actuator"], categories=self.ACTUATOR_ORDER)
        sub["stimulation_binary"] = (sub["stimulation"] == "Stimulated").astype(int)

        model_out = self._fit_mixedlm_with_fallback(
            formula="auc ~ C(actuator, Treatment(reference='DREADD')) * stimulation_binary",
            data=sub,
            group_col="experiment",
        )

        coef_df = self._all_fixed_effects(model_out.result)
        coef_df.insert(0, "model_method", model_out.method)
        coef_df.insert(1, "converged", model_out.converged)

        key_terms = [
            "stimulation_binary",
            "C(actuator, Treatment(reference='DREADD'))[T.LMO7]",
            "C(actuator, Treatment(reference='DREADD'))[T.PSAM]",
            "C(actuator, Treatment(reference='DREADD'))[T.LMO7]:stimulation_binary",
            "C(actuator, Treatment(reference='DREADD'))[T.PSAM]:stimulation_binary",
        ]
        key_rows = []
        for term in key_terms:
            term_stats = self._extract_term(model_out.result, term)
            key_rows.append({"term": term, **term_stats})

        key_df = pd.DataFrame(key_rows)
        key_df.insert(0, "model_method", model_out.method)
        key_df.insert(1, "converged", model_out.converged)
        key_df.insert(2, "n_neurons", int(len(sub)))
        key_df.insert(3, "n_experiments", int(sub["experiment"].nunique()))
        return coef_df, key_df

    def summarize_experiment_deltas(
        self, auc_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Cross-check with experiment-level deltas: stimulated - vehicle AUC.
        """
        sub = auc_df.loc[auc_df["actuator"].isin(self.ACTUATOR_ORDER)].copy()
        exp_level = (
            sub.groupby(["actuator", "experiment", "stimulation"], as_index=False)["auc"]
            .mean()
            .pivot_table(
                index=["actuator", "experiment"],
                columns="stimulation",
                values="auc",
                aggfunc="first",
            )
            .reset_index()
        )
        exp_level.columns.name = None
        for col in ["Vehicle", "Stimulated"]:
            if col not in exp_level.columns:
                exp_level[col] = np.nan
        exp_level = exp_level.rename(
            columns={"Vehicle": "auc_vehicle", "Stimulated": "auc_stimulated"}
        )
        exp_level["delta_stim_minus_vehicle"] = (
            exp_level["auc_stimulated"] - exp_level["auc_vehicle"]
        )
        exp_level["pair_complete"] = exp_level["auc_vehicle"].notna() & exp_level[
            "auc_stimulated"
        ].notna()
        exp_level = exp_level.sort_values(["actuator", "experiment"]).reset_index(drop=True)

        rows: list[dict[str, Any]] = []
        for actuator in self.ACTUATOR_ORDER:
            a = exp_level.loc[
                (exp_level["actuator"] == actuator) & (exp_level["pair_complete"])
            ].copy()
            deltas = a["delta_stim_minus_vehicle"].dropna().to_numpy(dtype=float)
            n = len(deltas)
            ttest_p = np.nan
            wilcoxon_p = np.nan
            if n >= 2:
                ttest_p = float(stats.ttest_1samp(deltas, popmean=0.0).pvalue)
                try:
                    wilcoxon_p = float(stats.wilcoxon(deltas).pvalue)
                except ValueError:
                    wilcoxon_p = np.nan
            rows.append(
                {
                    "actuator": actuator,
                    "n_paired_experiments": int(n),
                    "mean_delta_stim_minus_vehicle": float(np.mean(deltas)) if n else np.nan,
                    "median_delta_stim_minus_vehicle": float(np.median(deltas))
                    if n
                    else np.nan,
                    "n_negative_delta": int(np.sum(deltas < 0)) if n else 0,
                    "n_positive_delta": int(np.sum(deltas > 0)) if n else 0,
                    "pct_negative_delta": float(np.mean(deltas < 0)) if n else np.nan,
                    "ttest_pvalue_delta_zero": ttest_p,
                    "wilcoxon_pvalue_delta_zero": wilcoxon_p,
                }
            )

        summary_df = pd.DataFrame(rows)
        return exp_level, summary_df

    @staticmethod
    def _fit_mixedlm_with_fallback(
        formula: str, data: pd.DataFrame, group_col: str
    ) -> ModelOutput:
        try:
            model = smf.mixedlm(formula, data=data, groups=data[group_col])
            result = model.fit(reml=False, method="lbfgs", maxiter=300, disp=False)
            converged = bool(getattr(result, "converged", True))
            return ModelOutput(method="mixedlm", result=result, converged=converged)
        except Exception:
            ols = smf.ols(formula, data=data).fit()
            if data[group_col].nunique() > 1:
                ols = ols.get_robustcov_results(
                    cov_type="cluster", groups=data[group_col]
                )
                method = "ols_cluster_fallback"
            else:
                method = "ols_fallback"
            return ModelOutput(method=method, result=ols, converged=True)

    @staticmethod
    def _extract_term(result: Any, term: str) -> dict[str, float]:
        name_to_pos = ShollStatsAnalyzer._name_to_position(result)
        if term not in name_to_pos:
            return {
                "coef": np.nan,
                "se": np.nan,
                "pvalue": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
            }
        pos = name_to_pos[term]

        params = np.asarray(result.params, dtype=float)
        bse = np.asarray(result.bse, dtype=float)
        pvalues = np.asarray(result.pvalues, dtype=float)
        ci = np.asarray(result.conf_int(), dtype=float)
        return {
            "coef": float(params[pos]),
            "se": float(bse[pos]),
            "pvalue": float(pvalues[pos]),
            "ci_low": float(ci[pos, 0]),
            "ci_high": float(ci[pos, 1]),
        }

    @staticmethod
    def _all_fixed_effects(result: Any) -> pd.DataFrame:
        name_to_pos = ShollStatsAnalyzer._name_to_position(result)
        params = np.asarray(result.params, dtype=float)
        bse = np.asarray(result.bse, dtype=float)
        pvalues = np.asarray(result.pvalues, dtype=float)
        ci = np.asarray(result.conf_int(), dtype=float)

        rows = []
        for name, pos in name_to_pos.items():
            rows.append(
                {
                    "term": name,
                    "coef": float(params[pos]),
                    "se": float(bse[pos]),
                    "pvalue": float(pvalues[pos]),
                    "ci_low": float(ci[pos, 0]),
                    "ci_high": float(ci[pos, 1]),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _name_to_position(result: Any) -> dict[str, int]:
        params = result.params
        if hasattr(params, "index"):
            return {name: i for i, name in enumerate(params.index.tolist())}
        names = list(result.model.exog_names)
        return {name: i for i, name in enumerate(names)}
