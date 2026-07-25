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
class MajorGroupReportSpec:
    slug: str
    title: str
    reference_condition: str
    conditions: tuple[ReportCondition, ...]


REPORTS = (
    MajorGroupReportSpec(
        slug="group_i_treatment",
        title="Group I: Treatment Condition Analysis Report",
        reference_condition="EYFP_Control_Media",
        conditions=(
            ReportCondition("DREADD_CNO", "hM3Dq + CNO", "#6AA84F"),
            ReportCondition("PSAM_uPSEM", "PSAM4-5HT3 + uPSEM", "#8E7CC3"),
            ReportCondition("LMO7_hCTZ", "LMO7 + hCTZ", "#46B3C3"),
            ReportCondition("EYFP_Control_Media", "EYFP + media", CONTROL_GRAY),
        ),
    ),
    MajorGroupReportSpec(
        slug="group_ii_vehicle",
        title="Group II: Vehicle Condition Analysis Report",
        reference_condition="EYFP_Control",
        conditions=(
            ReportCondition("DREADD_Vehicle", "hM3Dq + vehicle", CONTROL_GRAY),
            ReportCondition("PSAM_Vehicle", "PSAM4-5HT3 + vehicle", CONTROL_GRAY),
            ReportCondition("LMO7_Vehicle", "LMO7 + vehicle", CONTROL_GRAY),
            ReportCondition("EYFP_Control", "EYFP control", CONTROL_GRAY),
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
            reference_condition=spec.reference_condition,
            title=spec.title,
        )
        figure_path = output_dir / f"{spec.slug}_analysis_report.png"
        metrics_path = output_dir / f"{spec.slug}_analysis_report_metrics.csv"
        report.render(figure_path, metrics_output_path=metrics_path, dpi=300)
        print(f"Report saved to: {figure_path}")
        print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
