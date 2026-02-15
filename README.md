# chemogenetic_analysis_paper

Python-only workflow for Sholl CSV import and tidying.

## Conda Environment

```bash
conda env create -f environment.yml
conda activate chemogenetic_analysis_paper
```

## Tidy the Input CSV

```bash
python scripts/tidy_sholl_csv.py
```

Default input:
- `input_data/Sholl_Analysis_unsorted.csv`

Default output:
- `output/tidy_sholl_analysis.csv`

## Recode to Analysis Groups

```bash
python scripts/recode_sholl_conditions.py
```

Default output:
- `output/tidy_sholl_analysis_grouped.csv`
- `output/cell_counts_by_group_condition.csv` (total cells by group and condition)

Current recode assumptions:
- `CONTROL` -> `EYFP_Control`
- `CONTROL/MEDIA` -> `EYFP_Control_Media`
- `MEDIA` -> `None_Vehicle`

## Plot Mean +/- SEM By Technology

```bash
python scripts/plot_technology_overlays.py
```

This writes 4 plots (DREADD, PSAM, LMO7, EYFP), each containing:
- Group I (Activation)
- Group II (Expression only)
- Group III (Effector only)

Plot style:
- Mean as dots at each radius
- SEM as vertical error bars
- Small x-offset by group so overlapping groups (for example EYFP Activation vs Expression) are both visible

Outputs:
- `output/plots/mean_sem_by_technology_group.csv`
- `output/plots/dreadd_group_mean_sem_points.png`
- `output/plots/psam_group_mean_sem_points.png`
- `output/plots/lmo7_group_mean_sem_points.png`
- `output/plots/eyfp_group_mean_sem_points.png`

## Radius Coverage and Zero Trend Diagnostics

```bash
python scripts/plot_radius_coverage.py
```

This writes radius-focused diagnostics to show:
- where data are still shared by a majority of cells (default threshold 50%)
- where intersections trend toward zero across radius
- a visual guide at radius `200 um`

Outputs:
- `output/plots/radius_coverage_summary.csv`
- `output/plots/radius_majority_windows.csv`
- `output/plots/dreadd_radius_coverage_zero.png`
- `output/plots/psam_radius_coverage_zero.png`
- `output/plots/lmo7_radius_coverage_zero.png`
- `output/plots/eyfp_radius_coverage_zero.png`

## Activation vs Expression in 1x4 Layout

```bash
python scripts/plot_activation_vs_expression_1x4.py
```

Creates a single `1x4` figure with one panel per technology (`DREADD`, `PSAM`, `LMO7`, `EYFP`) showing:
- Group I (Activation)
- Group II (Expression only)

Outputs:
- `output/plots/activation_vs_expression_1x4.png`
- `output/plots/activation_vs_expression_mean_sem.csv`
