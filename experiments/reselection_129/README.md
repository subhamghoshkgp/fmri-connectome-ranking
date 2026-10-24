# Dataset-1: outer-only max-inner-AUC re-selection

This package implements the professor's follow-up request. The original one-SE
experiment is retained as a read-only input. Its source files are in `original/`
and their aggregate SHA-256 must match the completed run's manifest.

## Selection rule

For each saved outer split and method, use only valid rows in its existing tuning
summary. Choose the lexicographically smallest key

`(-mean_inner_auc, mean_inner_support, candidate_index)`.

Thus the greatest saved mean inner-CV AUC wins. **Only exact AUC ties** are broken
by smaller saved mean inner support, then smaller candidate index. There is no
one-SE allowance, numerical tie tolerance, rounding, or test-set selection.

The cached summaries/inner records are validated without fitting any inner
tuning fold. Each selected candidate is refitted on the saved full outer-training
indices and evaluated on its saved outer-test indices. All 400 selected outer
fits are run, including folds where the choice is unchanged.

## What stays unchanged

- Original MAT/atlas/label file hashes, subject ordering, diagnosis mapping,
  Fisher transform, edge mapping, incidence matrix, and training-only scaling.
- All 100 saved outer splits and all saved inner splits. No splits are regenerated.
- Hyperparameter grids, penalties, lambda-max calculation, support threshold,
  prediction thresholds, seeds, model definitions, solver order/options/KKT
  tolerances, and RFE-SVM pipeline.
- RFE-SVM's original training-only **3-fold probability calibration** is preserved
  as part of its outer refit. This is separate from the five-fold hyperparameter
  tuning; no tuning fold is rerun.

Python 3.13.15 and every package version recorded in the original manifest are
required. The original manifest did **not** record SCS/joblib/threadpoolctl
versions, so their historical binary versions cannot be independently verified.
Solver algorithms and settings are unchanged. This package pins and records the
full new environment, including SCS 3.3.1, for reproducible refits/resume.

## Run independently in Colab

Open `fmri_dataset1_max_auc_reselection_colab.ipynb` in Colab, choose a CPU
runtime, and run cells in order. Keep the original completed results and dataset
in Google Drive. Defaults use your previous project:

- Original results: `MyDrive/fmri_dataset1_v3/results/final_full_colab_01`.
- Alternative original input: `MyDrive/fmri_dataset1_v3/final_full_colab_01_results.zip`.
- Dataset: `MyDrive/fmri_dataset1_v3/27_SCHZ_CTRL_dataset_downloaded.mat`.
- New outputs: `MyDrive/fmri_dataset1_v3_reselection/results/max_inner_auc_01`.

Edit the input-path cell if the completed files are elsewhere. The notebook
contains the complete code and reference files; the original results and raw
dataset are intentionally not embedded. If needed, it downloads and hash-checks
the same raw dataset into the new project, without replacing the old data file.

The first run cell refits five outer splits as a preview; the next resumes to all
100. You can use limits 10, 20, 30, etc. to run in smaller sessions. Rerunning the
notebook resumes committed refits. Never delete a manifest to bypass a mismatch.
Use a new output name if you intentionally change the execution environment.

## Run locally

Use Python 3.13.15 and install `requirements-reselection.lock.txt` into an isolated
environment. From this package directory:

```bash
python -m pytest -q test_reselect.py
python reselect.py --original /path/to/final_full_colab_01_results.zip \
  --out /path/to/NEW_max_auc_results --audit-only
python reselect.py --original /path/to/final_full_colab_01_results.zip \
  --data /path/to/27_SCHZ_CTRL_dataset.mat \
  --out /path/to/NEW_max_auc_results --jobs 2
```

`--original` also accepts the original results directory. `--max-outer 5` runs
only the first five saved splits and can be resumed by removing that option.
`--compare-only` regenerates comparisons from committed refits without fitting.
The new output must be separate from the original results. A failed selected
candidate is recorded as `METHOD_failed.json`; it is never replaced by a runner-up.
Rerunning retries that same selected candidate.

## Outputs

- `candidate_choices.csv`: all 400 old/new indices, parameters, inner AUC/support,
  and whether the choice changed. Available before any refits.
- `selection_audit.json`: completeness/integrity checks and candidate-change counts.
- `manifest.json`: original config/data/splits, parent run identity, source/input
  hashes, exact new environment, selection rule, and hub-statistic definitions.
- `repeat_XX/fold_YY/METHOD.json`: old full record; old/new selection summaries;
  new parameters, metrics, intercept, penalties, convergence, KKT/duality checks,
  iterations, fit/total runtime, and hashes of original result/tuning files.
- Paired `METHOD.npz`: new arrays under their original names and the complete old
  arrays with `old_` prefixes. Includes predictions/probabilities/scores, labels,
  train/test indices, supports/ROI pairs, standardized/raw coefficients, scalers,
  and V3 capacities. JSON is the atomic commit marker after its NPZ is saved.
- `comparison/SUMMARY.md`, `summary.json`, `metric_comparison.csv`, and
  `paired_fold_comparison.csv`: AUC, accuracy, balanced accuracy, F1, sensitivity,
  specificity, support, old/new means/SDs, paired deltas, and candidate changes.
- `stability_pairs.csv`, `per_repeat_stability_and_oof.csv`: overall/per-repeat
  Jaccard, equal-size Kuncheva, unequal-size adjusted overlap, Nogueira, and
  descriptive pooled out-of-fold metrics. Edge frequencies are also saved.
- V3 hub concentration and ROI hub tables for **both** original and new fits.

## Hub concentration definition

The supplied original code/document did not define a numerical hub statistic.
This follow-up explicitly adds these descriptive measures:

1. `degree_hhi = sum_v [d_v / (2 * |S|)]^2`, where `d_v` is the number of selected
   undirected edges incident on ROI v and S is the saved support.
2. `strength_hhi = sum_v [s_v / sum_u s_u]^2`, where
   `s_v = sum_{e incident on v, e in S} |standardized_beta_e|`.
3. Top-five ROI shares, maximum ROI shares, and effective ROI counts (`1/HHI`).

Every edge contributes to both endpoints. Higher HHI indicates more concentration.
Empty support gives undefined/null concentration. HHI depends on support size;
interpret it alongside sparsity. These are empirical selected-edge measures,
not counts of nonzero V3 capacity variables and not proof of biological hubs.

## Interpretation

This comparison measures sensitivity to the selection rule. Repeated folds reuse
the same 54 subjects. Fold SD is not a confidence interval, and no independent-fold
significance tests or clinical validity claims are made. Review all predictive,
sparsity, stability, and hub outcomes before deciding on algorithm changes.

## Verification performed

Tests cover exact ties, tiny non-ties, invalid candidates, hub formulas,
zero inner-fitting calls, preservation of original files, complete old-array
copies, outer-only refits, resume, input mismatch, and no fallback after failure.
The original cached tuning summaries are independently checked against their
saved inner records; stored outer metrics are checked against saved predictions.

The delivered completed run passed all 23 tests (18 original, 5 re-selection).
All 400 refits were also audited against raw-data scalers, coefficient maps,
test scores, metrics, and saved candidate choices. Use `audit_real_refits.py`
with `--original`, `--data`, and `--out` to reproduce this completed-run audit.
