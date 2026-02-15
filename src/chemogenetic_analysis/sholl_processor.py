from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


class ShollDataProcessor:
    """Load and tidy wide-format Sholl analysis CSV files."""

    DISTANCE_COLUMN = "Distance from Soma (µm)"

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
