from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

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

    PRIMARY_CONDITIONS = {
        "DREADD_CNO": ("DREADD", "Stimulated"),
        "DREADD_Vehicle": ("DREADD", "Vehicle"),
        "LMO7_hCTZ": ("LMO7", "Stimulated"),
        "LMO7_Vehicle": ("LMO7", "Vehicle"),
        "PSAM_uPSEM": ("PSAM", "Stimulated"),
        "PSAM_Vehicle": ("PSAM", "Vehicle"),
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
            actuator, stimulation = self.PRIMARY_CONDITIONS.get(condition, (None, None))

            # Experiment proxy:
            # - use sample_id when available
            # - otherwise use actuator+replicate for primary comparisons
            # - fallback to source_condition+replicate
            if pd.notna(sample_id):
                experiment = str(sample_id)
            elif actuator is not None:
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
                    "is_primary_condition": actuator is not None,
                }
            )

        auc_df = pd.DataFrame(rows)
        auc_df = auc_df.sort_values(["condition", "replicate"]).reset_index(drop=True)
        return auc_df

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
