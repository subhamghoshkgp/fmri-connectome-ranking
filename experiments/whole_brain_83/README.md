# Dataset-1: 83-region full comparison

Use `fmri_dataset1_83_all_algorithms_colab.ipynb` in Google Colab. The notebook
embeds this entire package, including the verified compact data; no separate
dataset upload or original 360 MiB download is necessary.

The experiment includes original V3, Elastic Net, LASSO, RFE-SVM, expanded V3,
ranking V3, ranking V3 with ridge refit, and dense ridge ranking. The four original
baselines are evaluated under both maximum mean inner AUC and one-SE selection:
12 variants in total. Maximum-AUC ties are broken by smaller mean inner support,
then smaller candidate index. Invalid inner fits cannot win selection.

## Data and protocol

- 54 original subjects: 27 controls, then 27 schizophrenia subjects in their
  original within-group order. IDs indicate group/position, not clinical IDs.
- Lausanne2008 `scale33`: all 83 regions, including subcortex and brainstem;
  3,403 distinct upper-triangle connections. This is a genuine supplied scale,
  not a crop of the 129-region matrices.
- `data/dataset1_83.mat` contains exact raw Pearson FC matrices extracted from
  `27_SCHZ_CTRL_dataset.mat`, SHA-256 recorded in the reference protocol.
- Fisher `arctanh` once, matching the saved 129-region preprocessing; every
  scaler is fit on the corresponding training subset only.
- 20 repeats x 5 outer folds, with 5 inner folds; seed 20260929. Full runs use
  the exact saved outer/inner splits recovered from the completed 129-region run.
- Model definitions, grids, thresholds and solver settings are unchanged.
  Source checksums are recorded in `validation/UNCHANGED_MODEL_SOURCES.json`.
- The mathematical preprocessing and original source provenance match the
  129-region run. Its transformed feature byte checksum did not reproduce in
  this execution; this is recorded in `reference/dataset1_129_protocol.json`.
  Byte-identical transformed 129 features are not claimed.

## Run outside Colab, if needed

Install Python 3.13.15 and `requirements.lock.txt` into an isolated environment,
then from this directory use an imported entrypoint (also avoids the prior
joblib `__main__` serialization failure):

```bash
python -u -c 'import run_dataset1_83 as runner; runner.main()' --out ../results/full_01 --jobs 2
```

Re-run that same command to resume. `--max-outer 1` limits an invocation to the
first outer fold; it does not change the full saved study or mean "one more fold".
Changing data/config/source/package versions rejects resume; CPU worker count
may change. Use a new output directory for a deliberately changed experiment.

Each inner fit is committed separately. Outer model arrays are written before
their JSON commit marker. An interrupted fit may be recomputed; committed fits
and completed models are reused. Drive synchronization is required to retain
files through a Colab runtime reset. Do not run two writers on the same output.
Create `STOP_REQUESTED` in the output folder for a graceful pause between fits;
the next run clears this flag after obtaining the run lock.

## Outputs

`analysis/fold_metrics.csv` has 1,200 data rows and 10 columns when complete.
Other outputs contain subject predictions, all pairwise fold comparisons,
Jaccard/Nogueira stability, stability per repeat, hub concentration, runtime and
one-SE/max-AUC selection changes. Outer JSON files record chosen parameters,
metrics, convergence/KKT diagnostics and runtime; NPZ files contain coefficients,
scalers, selected edges, predictions, and train/test indices. Inner checkpoints
and tuning summaries remain available in the run directory.

After all variants have 100 valid outer fits, the runner saves paired 83-vs-129
fold comparisons and a resolution summary. The reference CSV contains the
previously completed 129-region results; these are not reused as 83-region fits.
Repeated folds reuse subjects, so fold SDs, win counts and paired deltas are
descriptive. Dense ridge uses all 3,403 edges; feature-selection Jaccard is NA.
The inherited grids have unequal candidate budgets.

The default export contains analysis, code and provenance. This avoids rereading
158,000 inner records merely to produce a ZIP. Share the full Drive result folder
if model arrays or individual inner fits are requested. A separate optional ZIP
includes the 1,200 outer model pairs but excludes inner checkpoints.

## Validation status

23 automated tests passed. A reduced-grid engineering check on the actual
83-region matrices completed 2 folds x 12 variants (24 model pairs), and resume
left all fitted model files unchanged. These are execution checks, not the final
100-fold scientific results. Keep smoke output separate from the full run.
