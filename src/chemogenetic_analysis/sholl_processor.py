from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


class ShollDataProcessor:
    """Load and tidy wide-format Sholl analysis CSV files."""

    DISTANCE_COLUMN = "Distance from Soma (µm)"
    DEFAULT_CONDITION_MAP = {
        "DREADD/CNO": "DREADD_CNO",
        "PSAM/uPSEM": "PSAM_uPSEM",
        "LMO7/hCTZ": "LMO7_hCTZ",
        "DREADD": "DREADD_Vehicle",
        "PSAM": "PSAM_Vehicle",
        "LMO7": "LMO7_Vehicle",
        "CONTROL": "EYFP_Vehicle",
        "CONTROL/MEDIA": "EYFP_Vehicle",
        "CNO": "None_CNO",
        "uPSEM": "None_uPSEM",
        "hCTZ": "None_hCTZ",
        "MEDIA": "None_Vehicle",
    }
    GROUP_BY_CONDITION = {
        "DREADD_CNO": "Group I (Activation)",
        "PSAM_uPSEM": "Group I (Activation)",
        "LMO7_hCTZ": "Group I (Activation)",
        "DREADD_Vehicle": "Group II (Expression only)",
        "PSAM_Vehicle": "Group II (Expression only)",
        "LMO7_Vehicle": "Group II (Expression only)",
        "None_CNO": "Group III (Effector only)",
        "None_uPSEM": "Group III (Effector only)",
        "None_hCTZ": "Group III (Effector only)",
        "None_Vehicle": "Group III (Effector only)",
    }
    TECHNOLOGY_CONDITIONS = {
        "DREADD": {
            "Group I (Activation)": "DREADD_CNO",
            "Group II (Expression only)": "DREADD_Vehicle",
            "Group III (Effector only)": "None_CNO",
        },
        "PSAM": {
            "Group I (Activation)": "PSAM_uPSEM",
            "Group II (Expression only)": "PSAM_Vehicle",
            "Group III (Effector only)": "None_uPSEM",
        },
        "LMO7": {
            "Group I (Activation)": "LMO7_hCTZ",
            "Group II (Expression only)": "LMO7_Vehicle",
            "Group III (Effector only)": "None_hCTZ",
        },
        "EYFP": {
            "Group I (Activation)": "EYFP_Vehicle",
            "Group II (Expression only)": "EYFP_Vehicle",
            "Group III (Effector only)": "None_Vehicle",
        },
    }
    GROUP_COLORS = {
        "Group I (Activation)": "#d1495b",
        "Group II (Expression only)": "#2e86ab",
        "Group III (Effector only)": "#3caea3",
    }

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self._raw_df: pd.DataFrame | None = None

    def load_csv(self) -> pd.DataFrame:
        """Load CSV from disk and return a copy of the raw dataframe."""
        if self._raw_df is None:
            self._raw_df = pd.read_csv(self.csv_path)
        return self._raw_df.copy()

    def tidy(self, drop_missing_intersections: bool = True) -> pd.DataFrame:
        """
        Convert wide data into tidy/long format.

        Output columns:
        - radius_um: distance from soma in microns
        - condition: experimental condition derived from the wide column name
        - replicate: 1-based replicate index derived from suffix (.1, .2, ...)
        - sample_id: optional ID extracted from metadata row where radius is empty
        - intersections: Sholl intersections at each radius
        """
        raw_df = self.load_csv()

        if self.DISTANCE_COLUMN not in raw_df.columns:
            raise ValueError(
                f"Expected distance column '{self.DISTANCE_COLUMN}' in {self.csv_path}"
            )

        data_columns = [c for c in raw_df.columns if c != self.DISTANCE_COLUMN]
        metadata_mask = raw_df[self.DISTANCE_COLUMN].isna()

        sample_ids: dict[str, float | str | None] = {}
        if metadata_mask.any():
            metadata_index = raw_df.index[metadata_mask][0]
            sample_ids = raw_df.loc[metadata_index, data_columns].to_dict()

        data_df = raw_df.loc[~metadata_mask, [self.DISTANCE_COLUMN, *data_columns]].copy()
        data_df[self.DISTANCE_COLUMN] = pd.to_numeric(
            data_df[self.DISTANCE_COLUMN], errors="coerce"
        )
        data_df = data_df.dropna(subset=[self.DISTANCE_COLUMN])

        tidy_df = data_df.melt(
            id_vars=[self.DISTANCE_COLUMN],
            value_vars=data_columns,
            var_name="raw_column",
            value_name="intersections",
        )

        if drop_missing_intersections:
            tidy_df = tidy_df.dropna(subset=["intersections"])

        parsed = tidy_df["raw_column"].apply(self._parse_column)
        tidy_df["condition"] = parsed.apply(lambda x: x[0])
        tidy_df["replicate"] = parsed.apply(lambda x: x[1])
        tidy_df["sample_id"] = tidy_df["raw_column"].map(sample_ids)

        tidy_df = tidy_df.rename(columns={self.DISTANCE_COLUMN: "radius_um"})
        tidy_df["radius_um"] = pd.to_numeric(tidy_df["radius_um"], errors="coerce")
        tidy_df["intersections"] = pd.to_numeric(tidy_df["intersections"], errors="coerce")

        tidy_df = tidy_df[
            ["radius_um", "condition", "replicate", "sample_id", "intersections"]
        ]
        tidy_df = tidy_df.dropna(subset=["radius_um"])
        tidy_df = tidy_df.sort_values(["condition", "replicate", "radius_um"]).reset_index(
            drop=True
        )
        return tidy_df

    def write_tidy_csv(
        self, output_path: str | Path, drop_missing_intersections: bool = True
    ) -> Path:
        """Run tidy processing and write tidy CSV to disk."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        tidy_df = self.tidy(drop_missing_intersections=drop_missing_intersections)
        tidy_df.to_csv(output, index=False)
        return output

    def recode_conditions(
        self,
        tidy_df: pd.DataFrame | None = None,
        split_shared_control: bool = True,
    ) -> pd.DataFrame:
        """
        Recode raw condition names into study condition names and analysis groups.

        Assumptions used for current dataset:
        - CONTROL and CONTROL/MEDIA both map to EYFP_Vehicle
        - MEDIA maps to None_Vehicle

        If split_shared_control=True, EYFP_Vehicle rows are duplicated into:
        - Group I (Activation)
        - Group II (Expression only)
        """
        if tidy_df is None:
            tidy_df = self.tidy()

        recoded_df = tidy_df.copy()
        recoded_df["source_condition"] = recoded_df["condition"]
        recoded_df["condition"] = (
            recoded_df["condition"]
            .map(self.DEFAULT_CONDITION_MAP)
            .fillna(recoded_df["condition"])
        )
        recoded_df["analysis_group"] = recoded_df["condition"].map(self.GROUP_BY_CONDITION)
        recoded_df.loc[
            recoded_df["condition"] == "EYFP_Vehicle", "analysis_group"
        ] = "Group I/II (Shared Control)"

        if split_shared_control:
            shared_df = recoded_df.loc[recoded_df["condition"] == "EYFP_Vehicle"].copy()
            non_shared_df = recoded_df.loc[recoded_df["condition"] != "EYFP_Vehicle"]
            shared_g1 = shared_df.copy()
            shared_g1["analysis_group"] = "Group I (Activation)"
            shared_g2 = shared_df.copy()
            shared_g2["analysis_group"] = "Group II (Expression only)"
            recoded_df = pd.concat([non_shared_df, shared_g1, shared_g2], ignore_index=True)

        recoded_df = recoded_df[
            [
                "radius_um",
                "analysis_group",
                "condition",
                "replicate",
                "sample_id",
                "intersections",
                "source_condition",
            ]
        ]
        recoded_df = recoded_df.sort_values(
            ["analysis_group", "condition", "replicate", "radius_um"]
        ).reset_index(drop=True)
        return recoded_df

    def write_recoded_csv(
        self,
        output_path: str | Path,
        drop_missing_intersections: bool = True,
        split_shared_control: bool = True,
    ) -> Path:
        """Write tidy data with recoded condition/group labels."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        tidy_df = self.tidy(drop_missing_intersections=drop_missing_intersections)
        recoded_df = self.recode_conditions(
            tidy_df=tidy_df, split_shared_control=split_shared_control
        )
        recoded_df.to_csv(output, index=False)
        return output

    def summarize_cell_counts(
        self,
        recoded_df: pd.DataFrame | None = None,
        split_shared_control: bool = True,
    ) -> pd.DataFrame:
        """
        Summarize total unique cells per analysis group and condition.

        A cell is identified by source condition + replicate (+ sample_id when present).
        """
        if recoded_df is None:
            recoded_df = self.recode_conditions(split_shared_control=split_shared_control)

        counts_df = recoded_df.copy()
        sample_series = counts_df["sample_id"].where(
            counts_df["sample_id"].notna(), "NA"
        ).astype(str)
        counts_df["cell_id"] = (
            counts_df["source_condition"].astype(str)
            + "__r"
            + counts_df["replicate"].astype(int).astype(str)
            + "__s"
            + sample_series
        )

        summary_df = (
            counts_df.groupby(["analysis_group", "condition"], as_index=False)["cell_id"]
            .nunique()
            .rename(columns={"cell_id": "total_cells"})
            .sort_values(["analysis_group", "condition"])
            .reset_index(drop=True)
        )

        return summary_df

    def summarize_mean_sem_by_technology(
        self,
        recoded_df: pd.DataFrame | None = None,
        split_shared_control: bool = True,
    ) -> pd.DataFrame:
        """Summarize mean and SEM intersections by technology, group, and radius."""
        if recoded_df is None:
            recoded_df = self.recode_conditions(split_shared_control=split_shared_control)

        summary_frames: list[pd.DataFrame] = []
        for technology, group_map in self.TECHNOLOGY_CONDITIONS.items():
            for group_name, condition_name in group_map.items():
                subset = recoded_df[
                    (recoded_df["analysis_group"] == group_name)
                    & (recoded_df["condition"] == condition_name)
                ].copy()
                if subset.empty:
                    continue

                stats = (
                    subset.groupby("radius_um", as_index=False)["intersections"]
                    .agg(
                        mean_intersections="mean",
                        sem_intersections="sem",
                        n_cells="count",
                    )
                    .sort_values("radius_um")
                )
                stats["sem_intersections"] = stats["sem_intersections"].fillna(0.0)
                stats["technology"] = technology
                stats["analysis_group"] = group_name
                stats["condition"] = condition_name
                summary_frames.append(stats)

        if not summary_frames:
            return pd.DataFrame(
                columns=[
                    "radius_um",
                    "mean_intersections",
                    "sem_intersections",
                    "n_cells",
                    "technology",
                    "analysis_group",
                    "condition",
                ]
            )

        summary_df = pd.concat(summary_frames, ignore_index=True)
        summary_df = summary_df[
            [
                "technology",
                "analysis_group",
                "condition",
                "radius_um",
                "mean_intersections",
                "sem_intersections",
                "n_cells",
            ]
        ]
        summary_df = summary_df.sort_values(
            ["technology", "analysis_group", "radius_um"]
        ).reset_index(drop=True)
        return summary_df

    def write_mean_sem_summary(
        self,
        output_path: str | Path,
        split_shared_control: bool = True,
    ) -> Path:
        """Write mean/SEM summary by technology and group to CSV."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary_df = self.summarize_mean_sem_by_technology(
            split_shared_control=split_shared_control
        )
        summary_df.to_csv(output, index=False)
        return output

    def plot_technology_overlays(
        self,
        output_dir: str | Path,
        summary_df: pd.DataFrame | None = None,
        split_shared_control: bool = True,
        dpi: int = 180,
    ) -> list[Path]:
        """
        Save one mean/SEM point plot per technology.

        Mean values are plotted as dots and SEM is shown as vertical error bars.
        A small x-offset is applied per group so overlapping traces (for example,
        EYFP Activation vs Expression) remain visible.
        """
        if summary_df is None:
            summary_df = self.summarize_mean_sem_by_technology(
                split_shared_control=split_shared_control
            )

        import matplotlib.pyplot as plt

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_paths: list[Path] = []
        group_offsets = {
            "Group I (Activation)": -1.0,
            "Group II (Expression only)": 0.0,
            "Group III (Effector only)": 1.0,
        }

        for technology, group_map in self.TECHNOLOGY_CONDITIONS.items():
            tech_df = summary_df[summary_df["technology"] == technology].copy()
            if tech_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(9, 5.5))
            for group_name, condition_name in group_map.items():
                line_df = tech_df[
                    (tech_df["analysis_group"] == group_name)
                    & (tech_df["condition"] == condition_name)
                ].sort_values("radius_um")
                if line_df.empty:
                    continue

                x = line_df["radius_um"] + group_offsets.get(group_name, 0.0)
                y = line_df["mean_intersections"]
                sem = line_df["sem_intersections"]
                color = self.GROUP_COLORS.get(group_name, "#4c4c4c")
                label = f"{group_name}: {condition_name}"

                ax.errorbar(
                    x,
                    y,
                    yerr=sem,
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=2.5,
                    markersize=3.5,
                    alpha=0.9,
                    label=label,
                )

            ax.set_title(f"{technology}: Mean (dots) +/- SEM (bars)")
            ax.set_xlabel("Radius from Soma (um)")
            ax.set_ylabel("Intersections")
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right", fontsize=8, frameon=False)

            file_name = f"{technology.lower()}_group_mean_sem_points.png"
            file_path = output_path / file_name
            fig.tight_layout()
            fig.savefig(file_path, dpi=dpi)
            plt.close(fig)
            plot_paths.append(file_path)

        return plot_paths

    def write_cell_count_summary(
        self,
        output_path: str | Path,
        split_shared_control: bool = True,
    ) -> Path:
        """Write total unique cells per group/condition to CSV."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        recoded_df = self.recode_conditions(split_shared_control=split_shared_control)
        summary_df = self.summarize_cell_counts(
            recoded_df=recoded_df, split_shared_control=split_shared_control
        )
        summary_df.to_csv(output, index=False)
        return output

    @staticmethod
    def _parse_column(column_name: str) -> tuple[str, int]:
        """
        Parse a wide column name into (condition, replicate).

        Examples:
        - "CONTROL" -> ("CONTROL", 1)
        - "CONTROL.3" -> ("CONTROL", 4)
        """
        match = re.match(r"^(?P<condition>.*?)(?:\.(?P<suffix>\d+))?$", column_name)
        if not match:
            return column_name, 1

        condition = match.group("condition")
        suffix = match.group("suffix")
        replicate = 1 if suffix is None else int(suffix) + 1
        return condition, replicate
