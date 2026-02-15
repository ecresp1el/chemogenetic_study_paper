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
- `CONTROL` and `CONTROL/MEDIA` -> `EYFP_Vehicle`
- `MEDIA` -> `None_Vehicle`

## Plot Mean +/- SEM Overlays By Technology

```bash
python scripts/plot_technology_overlays.py
```

This writes 4 overlay plots (DREADD, PSAM, LMO7, EYFP), each containing:
- Group I (Activation)
- Group II (Expression only)
- Group III (Effector only)

Outputs:
- `output/plots/mean_sem_by_technology_group.csv`
- `output/plots/dreadd_group_overlay_mean_sem.png`
- `output/plots/psam_group_overlay_mean_sem.png`
- `output/plots/lmo7_group_overlay_mean_sem.png`
- `output/plots/eyfp_group_overlay_mean_sem.png`
