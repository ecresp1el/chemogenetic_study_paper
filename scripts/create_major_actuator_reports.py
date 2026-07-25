#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chemogenetic_analysis import ConditionAnalysisReport, ReportCondition


CONTROL_GRAY = "#9AA0A6"


@dataclass(frozen=True)
class ActuatorReportSpec:
    slug: str
    title: str
    conditions: tuple[ReportCondition, ...]


REPORTS = (
    ActuatorReportSpec(
        slug="hm3dq_dreadd",
        title="hM3Dq (DREADD) Sholl Analysis Report",
        conditions=(
            ReportCondition("DREADD_CNO", "hM3Dq + CNO", "#6AA84F"),
            ReportCondition("DREADD_Vehicle", "hM3Dq + vehicle", CONTROL_GRAY),
            ReportCondition("None_CNO", "CNO only", "#B7D7A8"),
            ReportCondition("None_Vehicle", "Media only", CONTROL_GRAY),
        ),
    ),
    ActuatorReportSpec(
        slug="psam4_5ht3",
        title="PSAM4-5HT3 Sholl Analysis Report",
        conditions=(
            ReportCondition("PSAM_uPSEM", "PSAM4-5HT3 + uPSEM", "#8E7CC3"),
            ReportCondition("PSAM_Vehicle", "PSAM4-5HT3 + vehicle", CONTROL_GRAY),
            ReportCondition("None_uPSEM", "uPSEM only", "#C9C2E4"),
            ReportCondition("None_Vehicle", "Media only", CONTROL_GRAY),
        ),
    ),
    ActuatorReportSpec(
        slug="lmo7",
        title="LMO7 Sholl Analysis Report",
        conditions=(
            ReportCondition("LMO7_hCTZ", "LMO7 + hCTZ", "#46B3C3"),
            ReportCondition("LMO7_Vehicle", "LMO7 + vehicle", CONTROL_GRAY),
            ReportCondition("None_hCTZ", "hCTZ only", "#A9DCE5"),
            ReportCondition("None_Vehicle", "Media only", CONTROL_GRAY),
        ),
    ),
)


def main() -> None:
    input_path = REPO_ROOT / "input_data" / "Sholl_Analysis_unsorted.csv"
    output_dir = REPO_ROOT / "output" / "plots"
    for spec in REPORTS:
        report = ConditionAnalysisReport.from_raw_csv(
            input_path,
            conditions=spec.conditions,
            title=spec.title,
        )
        figure_path = output_dir / f"{spec.slug}_analysis_report.png"
        metrics_path = output_dir / f"{spec.slug}_analysis_report_metrics.csv"
        report.render(figure_path, metrics_output_path=metrics_path, dpi=300)
        print(f"Report saved to: {figure_path}")
        print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
