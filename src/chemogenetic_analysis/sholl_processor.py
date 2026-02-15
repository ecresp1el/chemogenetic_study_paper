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
