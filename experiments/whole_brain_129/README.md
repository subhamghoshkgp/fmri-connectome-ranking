# Dataset-1: ranking V3 hypothesis test

This is a new experiment, separate from the professor's completed max-inner-AUC
re-selection. It tests whether a **V3 ranking selector followed by ridge ranking
refitting** improves held-out AUC. It does not promise an improvement. The original
data loader, atlas mapping, edge incidence geometry, logistic V3 model, and
saved subject-level splits are reused. The completed baseline results are read
only. Changing a loss or adding a refit requires new inner-CV fits.

## Run independently in Colab

1. Upload `fmri_dataset1_rank_v3_hypothesis_colab.ipynb` into Google Colab.
2. Select a CPU runtime and run cells in order. The notebook embeds this source
   code and the atlas files; uploading the code ZIP is optional.
3. In the first cell, check the paths to your completed `max_inner_auc_01`
   results directory or `max_inner_auc_01_results.zip`, and the original MAT
   dataset. The default paths follow the previous re-selection notebook.
4. The notebook creates a separate `fmri_dataset1_rank_v3` Drive project. It
   matches the recorded Python/scientific-package versions and runs tests.
5. Run one saved outer split first, then resume all 100 splits. Completed inner
   fits and outer methods are committed separately in Drive. After a disconnect,
   reconnect, run setup cells again, and keep **the same RUN_NAME**.
6. Export the new results ZIP after all four new arms have completed all folds.

The initial reference/data validation deliberately fails if the source run,
dataset identity, subject order, labels, atlas, preprocessing, or recorded Python/
scientific-package versions differ. New output inside or enclosing an existing
reference directory is rejected. Original baseline files are never edited.

The full experiment is substantial: for 100 outer folds and five inner folds,
there are 15,000 ordinary V3 selector fits, 15,000 ranking selector fits, 60,000
subset ridge refits, and 2,000 ungated ridge fits, plus final outer fits. Selector
fits and SVD factorizations are reused across the four refit strengths. Checkpoints
avoid repeating committed fits; they cannot recover an individual fit interrupted
before its commit. Runtime depends on solver convergence and the available CPU.
The notebook prints timing for the first outer split to inform your estimate.

## Mathematical definition

For a fitting split only, form all positive-negative pairs of its subjects.
With standardized features and labels 1=SCHZ and 0=CTRL, define

`L_rank(beta) = mean_(i:y_i=1,j:y_j=0) log(1 + exp(-(x_i-x_j) @ beta))`.

The selector minimizes

`L_rank(beta) + alpha*lambda*sum(abs(beta)) + (1-alpha)*lambda*sum(z)
                    + lambda_2*sum(beta**2)/2`,

subject to `z >= 0` and `abs(beta_uv) <= z_u + z_v`.
All pairs are constructed **after subject splitting**. They are dependent
observations; they do not increase the number of independent subjects.

`g0 = -0.5 * (mean(X[y=1]) - mean(X[y=0]))`.
The smallest zero-activation boundary is found by bisection of

`max(H @ maximum(abs(g0)-alpha*lambda,0)) <= (1-alpha)*lambda`.

Here `H` is the ROI-by-edge incidence matrix, and `lambda=ratio*lambda_max`
is recomputed on every fitting split. This is the ranking boundary; the ordinary
logistic control uses the original logistic boundary.

The selector support is `S={e: abs(beta_e)>1e-4}`. On **the same fitting subjects**,
the refit minimizes

`L_rank(w using only X[:,S]) + rho*sum(w**2)/2`.

Its coefficients can change magnitude and sign. The original node capacities
constrain the selector, not the refitted coefficients. Both stages are repeated
inside each inner-training split before scoring its validation subjects. After
tuning, both stages are fit anew on the complete outer-training set.

## Prespecified grids and controls

| Arm | Objective / pipeline | Candidates per inner split |
|---|---|---:|
| `v3_expanded` | Original logistic V3, expanded matched selector grid | 30 |
| `v3_rank` | Ranking V3 selector alone | 30 |
| `v3_rank_refit` | Ranking V3 selector plus ridge ranking refit, jointly tuned | 120 |
| `ridge_rank` | Ridge ranking using all edges, without V3 selection | 4 |

Selector grid: `alpha=[0.2,0.5,0.8]`, `ratio=[0.1,0.05,0.02,0.01,0.005]`,
`lambda_2=[0.01,0.1]`. Refit grid: `rho=[0.01,0.1,1,10]`.

Each candidate must be valid in every inner fold. Choose the highest mean inner
AUC; **exact** ties use smaller mean support, then smaller candidate index. Support
for a refit means its input subset S, as for RFE. Ungated ridge uses all edges.
The separate count of final coefficients above 1e-4 is also saved.

The four completed max-inner-AUC methods (V3, Elastic Net, LASSO, RFE-SVM) and
their original one-SE results are imported as eight reference rows. They are not
trained again. The new logistic control distinguishes ranking-loss changes from
the enlarged selector grid. Pipeline comparisons also involve different tuning
searches; they are not isolated causal estimates. The saved baselines were not
retuned with an equal enlarged search budget.

## Optimization and classification

The ranking selector uses subject-score variables, avoiding a dense pair-by-edge
matrix. It retains the original SCS then CLARABEL fallback settings. Each accepted
solution requires optimal solver status and independently recomputed maximum KKT
residual <=2e-5, including the subject-score equality constraint. Ridge ranking
uses an exact thin-SVD row-space parameterization, **without truncating singular
vectors**. L-BFGS-B output is accepted only if the full-feature objective gradient
is <=2e-5; optimizer success alone is insufficient. Empty selected subsets have
an exact zero-score solution.

Ranking loss has no identifiable intercept and produces ranking scores, not
calibrated probabilities. Binary labels use the midpoint of the two training
class mean scores; no validation or test labels set the threshold. Centering
scores preserves AUC. The logistic control retains its original intercept and
probabilities. Classification metrics are secondary because this threshold
differs from the saved classifiers' rules. An empty ranking support ties all
scores at zero, producing AUC=0.5 and positive labels at the `>=0` cutoff.

## Checkpoints and outputs

- `manifest.json`: configuration, source digest, environment, unchanged saved
  splits, preprocessing provenance, X hash, and reference-content fingerprint.
  Resume rejects changed source/config/data/environment/reference contents.
- `reference_records.json` and `reference_predictions.npz`: compact verified
  saved one-SE/max-AUC metrics, choices, diagnostics, predictions and supports.
- `repeat_XX/fold_YY/inner/`: one atomic JSON per inner fit; ranking selectors
  also save their input masks for refitting. Failed candidates retain their
  identity and error and are invalidated only for their own candidate rows.
- `*_tuning.json`: every candidate's inner means, standard error, support,
  validity, predictions, parameters, convergence diagnostics and runtime.
- `v3_expanded`, `v3_rank`, `v3_rank_refit`, `ridge_rank` `.json/.npz`: selected
  parameters/index, metrics, full standardized/raw coefficients and intercepts,
  scaler statistics, saved train/test indices, predictions, input supports,
  selector coefficients/node capacities and convergence/runtime records.
- `analysis/`: readable `SUMMARY.md`, `summary.json`, fold metrics, descriptive
  paired AUC/accuracy/support comparisons, stability overall/by repeat and hub
  concentration. The hypothesis check compares observed mean AUC only.

Hub concentration uses the previous report's explicit definition: incident
degree shares or absolute standardized-coefficient strength shares, summed over
both edge endpoints; HHI is the sum of squared ROI shares, and top-five share is
the sum of the five largest shares. Empty supports yield null. These are computed
for new gated arms and saved V3 references; larger HHI means greater concentration.
Report alongside support size. Degree concentration for the refit describes S;
strength concentration uses the refitted coefficients on S.

Stability includes Jaccard, equal-size Kuncheva, an explicitly descriptive
unequal-size adjusted overlap, and Nogueira. Larger input supports can inflate
overlap. Ungated ridge stability is omitted because selecting all edges makes it
trivially one. Refit input support and coefficient-threshold counts are distinct.

Stage solver times, SVD time and final-method wall time are recorded separately.
Shared work is reused, so summing per-stage times double-counts some computations;
saved baseline solver times are not comparable to full new tuning wall time.
The reference's core environment is enforced. The historical first run did not
record SCS's dependency version; this new experiment pins SCS=3.3.1 and logs it.

## Failures and interpretation

A failed inner solve invalidates its candidate; a failed selector also invalidates
its associated refits. Invalid checkpoint results remain invalid when resuming.
If every candidate for an arm is invalid, stop. Final outer refit failure stops
the run without choosing a different candidate based on outer-test behavior.
Do not interpret a partial run as final or delete failing records to manufacture
a complete experiment. Investigate errors and use a new RUN_NAME for an amended
configuration or implementation.

This is exploratory development on the same 54 subjects whose earlier outer
results informed the hypothesis. Fold SD is descriptive, not a confidence interval;
the 100 folds and their pairs are not independent replicates. An observed AUC
increase would support further study, not prove superiority. Comparable baseline
tuning and an independent cohort are needed for a strong performance claim.

## Local / CLI use

Create an environment matching the reference manifest (Python 3.13.15 for the
supplied completed Colab run) and install `requirements.lock.txt`:

```bash
python -m pytest -q tests
python run_rank_v3.py --reference /path/max_inner_auc_01_results.zip \
  --data /path/27_SCHZ_CTRL_dataset_downloaded.mat \
  --out /path/new_rank_results --jobs 1 --max-outer 1
# Resume all saved outer folds; keep the same output path and configuration.
python run_rank_v3.py --reference /path/max_inner_auc_01_results.zip \
  --data /path/27_SCHZ_CTRL_dataset_downloaded.mat \
  --out /path/new_rank_results --jobs 1
python run_rank_v3.py --out /path/new_rank_results --analyze-only
```

The mathematical tests use independent dense-pair convex fits, finite-difference
gradients, zero-activation certificates, and full-feature ridge fits. Runner tests
cover exact ties, failed-candidate isolation, saved-split validation, training-only
scaling, prediction reconstruction, source protection, interruption and resume.
`validation/` contains development test results and a **synthetic** 129-ROI dimension
check; it contains no new real Dataset-1 AUC results.
