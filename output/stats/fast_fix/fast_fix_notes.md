# Fast Fix Reanalysis Notes

Pipeline order:
1. Metadata rebuild (`1_metadata_per_neuron.csv`)
2. AUC recomputation from raw curves (`2_auc_per_neuron_raw.csv`)
3. Predefined QC filters (`3_qc_flags_per_neuron.csv`, `3_auc_per_neuron_qc_pass.csv`)
4. Within-actuator comparisons + secondary across-technologies on QC-pass
5. Experiment-level delta cross-check

QC rules:
- min_radius_points >= 20
- radius_max >= 150.0
- AUC inlier by condition using IQR fence (multiplier=3.0)

Experiment id definition:
- for primary actuator conditions: actuator + replicate index
- otherwise sample_id if present
- fallback to source_condition + replicate index

Interpretation check:
- Compare the sign of `coef_stimulation` in step 4 with mean experiment deltas in step 5.
