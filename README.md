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

Analysis-group definitions preserve the original three study arms without implying
that the EYFP vector control is chemogenetically activated:
- Group I (Treatment condition): actuator plus its matched ligand, or EYFP plus matched media
- Group II (Vehicle condition): actuator-vector or EYFP-vector without the Group-I treatment
- Group III (Ligand/media-only control): ligand-only or media-only control without an actuator vector

Color policy is locked across every figure:
- hM3Dq/DREADD + CNO: green (`#6AA84F`)
- LMO7 + hCTZ: cyan (`#46B3C3`)
- PSAM4-5HT3 + uPSEM: purple (`#8E7CC3`)
- CNO-only, hCTZ-only, and uPSEM-only: lighter green, cyan, and purple versions of their matched treatment
- vehicle, EYFP, and media-only controls: gray (`#9AA0A6`)

The group-comparison figures use a separate palette and never reuse those
technology colors: Group I treatment is gold (`#C9A227`), while Groups II and
III are gray and are distinguished by marker shape.

## Plot Mean +/- SEM By Technology

```bash
python scripts/plot_technology_overlays.py
```

This writes 4 plots (DREADD, PSAM, LMO7, EYFP), each containing:
- Group I (Treatment condition)
- Group II (Vehicle condition)
- Group III (Ligand/media-only control)

Plot style:
- Mean as dots at each radius
- SEM as vertical error bars
- Small x-offset by group so overlapping groups (for example the EYFP treatment and vehicle conditions) are both visible

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

## Treatment vs Vehicle in 1x4 Layout

```bash
python scripts/plot_treatment_vs_vehicle_1x4.py
```

Creates a single `1x4` figure with one panel per technology (`DREADD`, `PSAM`, `LMO7`, `EYFP`) showing:
- Group I (Treatment condition)
- Group II (Vehicle condition)

Outputs:
- `output/plots/treatment_vs_vehicle_1x4.png`
- `output/plots/treatment_vs_vehicle_mean_sem.csv`

## Three Subgroups in 1x4 Layout

```bash
python scripts/plot_three_subgroups_1x4.py
```

Creates a single `1x4` figure (DREADD, PSAM, LMO7, EYFP) with:
- Group I (Treatment condition)
- Group II (Vehicle condition)
- Group III (Ligand/media-only control)

Outputs:
- `output/plots/three_subgroups_1x4.png`
- `output/plots/three_subgroups_mean_sem.csv`

## AUC Mixed-Model Statistics

```bash
python scripts/run_auc_stats.py
```

Primary analysis:
- `DREADD_Vehicle` vs `DREADD_CNO`
- `LMO7_Vehicle` vs `LMO7_hCTZ`
- `PSAM_Vehicle` vs `PSAM_uPSEM`
- Model: `auc ~ stimulation_binary + (1 | experiment)`

Secondary analysis:
- Across technologies: `auc ~ C(actuator) * stimulation_binary + (1 | experiment)`

Outputs:
- `output/stats/auc_per_neuron.csv`
- `output/stats/primary_within_actuator_mixedlm.csv`
- `output/stats/secondary_across_technologies_fixed_effects.csv`
- `output/stats/secondary_key_tests.csv`
- `output/stats/analysis_notes.md`

## Four-Condition Analysis Report

```bash
python scripts/create_condition_analysis_report.py
```

Creates a publication-style four-column report for the no-actuator treatment controls:
`CNO`, `Media`, `hCTZ`, and `uPSEM`. It includes representative cells, individual-cell
traces, treatment-color overlays, mean +/- SEM profiles, fold-change and AUC boxplots,
and a grand mean overlay. The default output colors are CNO red, Media gray, hCTZ cyan,
and uPSEM purple.

Outputs:
- `output/plots/no_actuator_analysis_report.png`
- `output/plots/no_actuator_analysis_report_metrics.csv`

## Major Actuator Analysis Reports

```bash
python scripts/create_major_actuator_reports.py
```

Creates matched four-condition reports for hM3Dq/DREADD, PSAM4-5HT3, and LMO7.
Each report includes actuator + ligand, actuator + vehicle, ligand-only, and media-only
conditions. The active actuator condition uses its locked tool color; every control is gray.

## Major Group Analysis Reports

```bash
python scripts/create_major_group_reports.py
```

Creates the two reports that were missing from the three-arm study summary:
- Group I treatment: hM3Dq + CNO, PSAM4-5HT3 + uPSEM, LMO7 + hCTZ, and EYFP + media
- Group II vehicle: hM3Dq, PSAM4-5HT3, LMO7, and EYFP vector controls without Group-I treatment

Together with `no_actuator_analysis_report.png` (Group III ligand/media-only control),
these provide one report for each major study group.

## AUC Comparison Across Major Groups

```bash
python scripts/plot_auc_major_groups.py
```

Creates a three-panel AUC dot-and-box figure for Groups III, II, and I. Each y-axis
uses the full biological condition label and its cell count, including the correct
vector, ligand-only, and media-only controls for that major group.

## Condition-Level AUC Outlier QC

```bash
python scripts/plot_auc_outlier_qc.py
```

Creates a condition-wise AUC diagnostic using a 95% prediction interval for individual
observations. It reports total, plotted, and candidate-excluded cell counts for every
condition, without automatically changing the analysis dataset.

## AUC Distribution Diagnostics

```bash
python scripts/run_auc_distribution_diagnostics.py
```

Writes per-cell Sholl AUC values, a condition-wise Shapiro-Wilk normality table, and
normal Q-Q plots. AUC is the area under each cell's Sholl curve (intersections versus
radius), and is the response variable used for the downstream statistical models.

## AUC Variance Homogeneity

```bash
python scripts/run_auc_variance_homogeneity.py
```

Runs Brown-Forsythe tests (median-centered Levene tests) for the three primary actuator
comparisons and each four-condition major group. Results include condition membership,
cell counts, test statistic, p-value, and the equal-variance decision.

## Fast Fix Reanalysis (Metadata -> AUC -> QC -> Models -> Delta Check)

```bash
python scripts/run_fast_fix_strategy.py
```

Outputs are saved to `output/stats/fast_fix/`:
- `1_metadata_per_neuron.csv`
- `2_auc_per_neuron_raw.csv`
- `3_qc_flags_per_neuron.csv`
- `3_auc_per_neuron_qc_pass.csv`
- `4_primary_within_actuator_qc.csv`
- `4_secondary_across_technologies_qc_fixed_effects.csv`
- `4_secondary_across_technologies_qc_key_tests.csv`
- `5_experiment_level_deltas.csv`
- `5_experiment_level_delta_summary.csv`
- `fast_fix_notes.md`
