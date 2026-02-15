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

Current recode assumptions:
- `CONTROL` and `CONTROL/MEDIA` -> `EYFP_Vehicle`
- `MEDIA` -> `None_Vehicle`
