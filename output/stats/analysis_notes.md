# AUC Statistical Analysis Notes

## Primary analysis
- Per-neuron AUC computed via trapezoidal integration across radius.
- Within each actuator:
  - `DREADD_Vehicle` vs `DREADD_CNO`
  - `LMO7_Vehicle` vs `LMO7_hCTZ`
  - `PSAM_Vehicle` vs `PSAM_uPSEM`
- Model: `auc ~ stimulation_binary + (1 | experiment)`

## Secondary analysis
- Across technologies (DREADD, LMO7, PSAM):
- Model: `auc ~ C(actuator) * stimulation_binary + (1 | experiment)`
- Interaction terms test whether stimulation effect magnitude differs by actuator.

## Experiment variable assumption
- The source data does not include an explicit `Experiment` column.
- This pipeline uses:
  - For primary actuator conditions, `"{actuator}_rep{replicate}"` to align vehicle/stim pairs.
  - Otherwise, `sample_id` when present.
  - Fallback to `"{source_condition}_rep{replicate}"`.

## Fallback behavior
- MixedLM is used by default.
- If MixedLM fails to converge/fit, OLS with cluster-robust SE by `experiment` is used.
