# Dataset 1 Version 3 independent rebuild

This package implements the professor's gated-incidence sparse logistic model, logistic LASSO, Elastic Net, and an explicitly configured RFE-SVM reconstruction. It includes nested cross-validation, fit-level checkpoints, diagnostic checks, and descriptive stability reports.

The real Dataset-1 MAT file was retrieved from the public link supplied by the user, verified as complete, and loaded at the 129-ROI scale. `validation/` includes real-data numerical refits and independent exploratory nested-CV runs. Read `REAL_DATA_AUDIT.md` and `AUDIT.md` before using any estimate. The old notebook and result ZIP were supplied; the `szv3_pipeline.zip` imported by that notebook was not supplied, so line-by-line review of that old package is unavailable.

The companion `old_run_audit/` directory holds the safely decoded saved fold arrays, the verification script, corrected atlas names and audit tables. The original 377 MB subject-level MAT file is intentionally absent from the archive. Put that file at a local path and pass it explicitly to the CLI. For example, from the package directory:

```bash
python run.py --config configs/real_v3_pilot.json --data /path/to/27_SCHZ_CTRL_dataset.mat \
  --atlas reference/ParcellationLausanne2008.xls \
  --labels reference/labels_index_CORTICAL_Laus2008_all_scales.mat \
  --representation fisher --out results/real_v3_pilot --jobs 4
python refit_saved_folds.py --method v3 --data /path/to/27_SCHZ_CTRL_dataset.mat \
  --out results/rechecked_v3.jsonl
```

## Start here

Use Python 3.11 or later in Linux, macOS, WSL, or Colab. The run lock uses a POSIX OS lock; native Windows execution is not supported. No GPU is used by this implementation. A notebook is included for Colab, but remote Colab execution was not tested from this workspace.

```bash
cd fmri_v3_rebuild
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python run.py --synthetic --config configs/smoke.json --out results/synthetic_smoke
```

`requirements-tested.txt` records the versions used for validation. Pin those versions for strict replication. Package and Python versions are included in the resume signature: a changed environment requires a new output directory rather than silently mixing fits.

## Make the upload smaller

If the original approximately hundreds-of-megabytes MAT file cannot be attached, run `export_fc129.m` in MATLAB in the same folder as the original file. It creates `Dataset1_FC129_only.mat` containing the control and schizophrenia FC arrays only, without structural data or other scales. The reported original layout is validated before export; it fails rather than guessing if that layout differs.

Upload that smaller file. `inspect_dataset.py path/to/file.mat` prints paths, shapes and types for troubleshooting without dumping the measurements. The loader supports the reported MATLAB v7.3 reference layout and compact v7/v7.3 files. It selects the scale by shape, not an unverified cell position.

## Real data commands

Place the real MAT file in `data/` or use its absolute path. The atlas and cortical-label reference files are already included. The pipeline preserves all 129 regions, yielding 8,256 unique upper-triangle edges in NumPy row-major triangle order. The stored map makes this order explicit. Internal indices are zero based; exported ROI IDs are MATLAB-compatible one based.

The saved Claude fold scores and lambda-max values reconstruct from Fisher-transformed Pearson correlations with training-only scaling. An explicit representation argument remains mandatory. The examples below use `fisher` to match that saved run. Use separate output directories for different representations.

```bash
python benchmark.py --data data/Dataset1_FC129_only.mat \
  --atlas reference/ParcellationLausanne2008.xls \
  --labels reference/labels_index_CORTICAL_Laus2008_all_scales.mat \
  --representation fisher --out results/real_fit_benchmark.json

python run.py --data data/Dataset1_FC129_only.mat \
  --atlas reference/ParcellationLausanne2008.xls \
  --labels reference/labels_index_CORTICAL_Laus2008_all_scales.mat \
  --representation raw --config configs/development_full.json \
  --out results/development_full_raw --jobs 1 --max-outer 1
```

The second command runs one outer fold of the exact development protocol and reports an INCOMPLETE run. Measure its runtime and inspect convergence/tuning records. Resume the remaining folds with the same command and output path after removing `--max-outer 1`. Start with one worker; use two or more only when memory and CPU capacity permit. Each process pins numerical-library threads to one. Parallelism is across outer folds; no hardware speedup is promised.

The complete development run is 5 repeats x 5 outer folds with 5 inner folds and the full 72-setting V3 grid. After numerical validation, use `configs/final_full.json` and a NEW output directory for 20 repeats x 5 outer folds. The first five repeat seeds are deliberately shared; these repeated observations are not independent validation subjects. Freeze all analysis choices before the final run and disclose earlier exploration.

## Presets and model selection

| Config | Outer design | Inner folds | V3 settings | Purpose |
|---|---|---:|---:|---|
| smoke.json | 1 x 3 | 3 | 2 | End-to-end software check only |
| development_coarse.json | 5 x 5 | 5 | 24 | Optional predeclared computational development grid |
| development_full.json | 5 x 5 | 5 | 72 | Full requested parameter grid at development scale |
| final_full.json | 20 x 5 | 5 | 72 | Full requested grid at final resampling scale |

The full V3 grid is alpha in {0.2,0.5,0.8}, ratio in {1,0.7,0.5,0.3,0.2,0.1,0.05,0.02}, ridge in {0,0.001,0.01}. lambda_E = alpha*lambda and lambda_V = (1-alpha)*lambda. Each inner-training split computes its OWN lambda_max using Equation 9; the final outer-training refit recomputes it on that training set.

LASSO uses the same ratio path. Elastic Net uses alpha in {0.2,0.5,0.8}, lambda_1 = alpha*lambda, lambda_2 = (1-alpha)*lambda and lambda_max = max(abs(g))/alpha. These conventions match mean logistic loss. RFE-SVM tunes C in {0.01,0.1,1,10} and selected counts {10,25,50,100,250,500,1000}; RFE removes 20% of original features per step. These baseline grids are reconstruction choices, since the historical settings were not supplied.

The default sparsity tie-break uses the one-standard-error heuristic: retain candidates within one estimated SE of the best mean inner AUC, then choose the smallest mean support. This is NOT a test of statistical indistinguishability. Set `selection_rule` to `max_auc` for strict best-mean-AUC selection with a numerical-tie sparsity rule. Any config change requires a new output folder. No outer test metric participates in selection.

## Solver and numerical checks

CVXPY with CLARABEL solves the explicit convex epigraph formulation. The objective is mean logistic loss plus lambda_E*sum(t), lambda_V*sum(z), and ridge*||beta||^2/2, with t >= beta, t >= -beta, H'z >= beta, H'z >= -beta and z >= 0. The intercept is unpenalized. The same ungated model handles LASSO and Elastic Net.

Every fit must have optimal status AND absolute KKT residual at most 2e-5. The residual is the maximum of primal violation, dual violation, stationarity, and complementary slackness, recomputed from original variables and constraint multipliers. Solver tolerances begin at 1e-9; a failed check triggers a tighter 1e-11 attempt and then a numerically distinct SCS solve. Every accepted solution is checked against the same KKT threshold. Failed inner fits are recorded and their entire candidate is excluded. If no valid candidate remains, or the selected outer refit fails, the run stops. Missing convergence is never converted into a success flag.

Parameterized problems are reused within an inner training split and warm starts are requested along descending paths. Actual warm-start benefit depends on the solver and is not assumed. This certified implementation is intentionally the reference-quality starting point; no unverified custom fast solver is shipped. Profile real folds before considering a faster implementation and compare any replacement on multiple folds/settings.

RFE-SVM checks libsvm completion and an independently computed primal-dual gap for the final SVM fit. Warnings during elimination/calibration are promoted to errors. Calibration uses 3-fold training-only out-of-fold predictions of the entire scaler/RFE/SVM pipeline, with `ensemble=False`, to report probabilities. SVM primary predictions and AUC use the fitted margin, not calibrated-probability thresholding. Calibration is additional to the baseline reconstruction and does not alter its selected support.

## Leakage and validation safeguards

- Scalers are fitted on each inner-training set during tuning and on outer training during the final refit; no full-dataset scaling precedes CV.
- Feature selection, regularization tuning and calibration never see the outer test subjects.
- All methods share stored outer and inner splits.
- No univariate prefilter, imputation, SMOTE, sign removal, or cortical-only restriction is added.
- Nonfinite correlations, asymmetry, unexpected group counts and atlas mismatches stop loading.
- Full source, config, input hashes, versions and splits are saved before fits. Resuming changed inputs or settings is refused.
- Per-inner-fit JSON checkpoints and per-outer-method NPZ/JSON commits avoid a single shared pickle. Completed fits are reused on restart. Only the in-progress solve is lost on interruption.
- A local advisory lock prevents two processes on the same machine from writing the same run. It is NOT a distributed lock across separate machines/Colab VMs: never run two sessions against one output directory.

## Output and interpretation

The output folder includes `manifest.json`, `incidence.npz`, `dataset_metadata.npz`, all inner settings and diagnostics, and per-method fold arrays. Saved arrays include standardized and raw-unit coefficients/intercepts, capacities, training scalers, indices, scores, probabilities, predicted labels, selected edge indices and ROI pairs. Failed fits remain visible in inner JSON logs; no test result is written for a failed final fit.

`analysis/` contains prediction tables, paired descriptive differences, full edge and ROI frequencies, maps, and `SUMMARY.md`/`summary.json`. Scores are recalculated from saved predictions during reporting and checked against fold records. Full repeats require exactly one held-out prediction per subject.

Default numerical support is abs(beta) > 1e-4; each fold also reports counts at 1e-5 and 1e-3. This tolerance is part of the configuration and must be frozen. RFE support is its explicitly selected feature subset, not a logistic zero threshold. Node capacity and observed incident-edge support are reported separately because capacities can be nonunique.

Jaccard, standard equal-size Kuncheva (with valid-pair counts), explicitly named unequal-size adjusted overlap, and Nogueira stability are reported per repeat and overall. Fold SD is descriptive and must not be called a confidence interval. Repeats reuse the same subjects. There are no naive fold-level p-values and no automatic research-success verdict. Full subject-level resampling inference is not implemented in this first rebuild.

The synthetic sample intentionally verifies software behaviour; its AUC is not evidence for or against the scientific model and is not a reproduction of the professor's synthetic pilot.

## What still needs completion

Run the actual Dataset-1 loader and benchmark after a readable upload, confirm the historical representation/baseline settings, then complete the 25-fold development and 100-fold final runs. Actual dataset compatibility and real-data performance remain unverified. Old source and fold records are needed to certify the old run. Versions 1 and 2 and literature-extension baselines are outside this V3-first package, consistent with the professor's stated priority.
