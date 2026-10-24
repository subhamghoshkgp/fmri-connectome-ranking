Verified 83-region outer-model analysis

Source: dataset1_83_full_01_FINAL_outer_models.zip
All 1,200 model pairs were inspected without refitting.

fold_metrics.csv: 1,200 data rows, 10 columns.
subject_predictions.csv: 12,960 repeated out-of-fold prediction records.
method_summary.csv: all 12 variants, including one-SE variants.
paired_auc_comparisons.csv: 66 model contrasts on identical folds.
resolution_83_vs_129.csv: same-subject, same-split descriptive resolution comparison.
feature_selection_frequency.csv: nonzero selection counts; ROI IDs are MATLAB 1-based.
numerical_diagnostics.csv: recorded residuals plus independently checked objectives, gates and ridge gradients.
chosen_parameters.csv: selected candidates recorded in outer results, not independently verified against absent inner summaries.
audit.json: checks, maximum discrepancies and material limitations.
audit_results.py: script used for independent checks; requires the source ZIP plus the originally delivered code/data package in the paths shown at its top.

Repeated folds reuse subjects. AUC differences, fold SDs and win counts are descriptive.
Win/tie reporting uses a 1e-12 tolerance for numerical equality; tuning tie-breaking is unchanged.
Ridge ranking uses all input edges and does not perform feature selection.
Inner tuning records, full conic dual variables and SVM calibration objects are not in the input archive.
