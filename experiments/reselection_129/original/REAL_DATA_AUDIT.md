# Dataset-1 Version 3: verified results and limits

## Source data and numerical reconstruction

The user-provided public link yielded the complete **377,836,209-byte** MATLAB v7.3 file (SHA-256 `8e6944e1a3d31abce6123eea14c31a6b38cc92f01df90c8d95aefdcdacff2b6f`). It contains 27 CTRL and 27 SCHZ subjects at the 129-ROI scale. Upper-triangle Pearson correlations produce 8,256 edges; Fisher transformation and scaling fitted on training subjects reconstruct every saved old fold score to within 1e-5 (in the first checked fold, below 5e-8). The old lambda-max value matches independently to about 5e-11. No subject-level MAT data is redistributed in this package.

The old ZIP has 100 fold records (25 per method); all scores, labels, selected edge indices and reported prediction metrics pass an independent consistency check. The 25 repeated folds reuse the same 54 subjects and must not be treated as 25 independent test datasets. The original imported `szv3_pipeline.zip` source was not supplied, so the old inner solver code cannot be reviewed line by line.

## Saved run and certified outer refits

| Method | Saved mean fold AUC | Saved converged outer fits | Saved mean selected edges | Certified refit mean fold AUC | Certified mean selected edges |
|---|---:|---:|---:|---:|---:|
| rfe_svm | 0.8045 | 25/25 | 404.0 | — | — |
| enet | 0.7869 | 1/25 | 530.2 | 0.7517 | 155.1 |
| v3 | 0.7011 | 0/25 | 142.2 | 0.7064 | 137.7 |
| lasso | 0.6693 | 25/25 | 24.7 | — | — |

The V3 refits changed held-out AUC in 4 of 25 folds; the mean AUC moved from 0.7011 to 0.7064. The Elastic Net refits changed 18 of 25 fold AUCs; its mean moved from 0.7869 to 0.7517. Maximum independently checked KKT residuals were 8.2e-07 and 1.1e-06, below the 2e-5 acceptance tolerance. Elastic Net used the saved C and l1_ratio with the equivalent mean-loss penalty; the small-data scikit-learn equivalence test is included.

**These are numerical refits at hyperparameters chosen by the old inner search.** The old inner fits did not have saved convergence certificates. This exercise diagnoses the outer solver effect but does not retroactively validate the earlier hyperparameter search.

## Fresh nested cross-validation pilots

| Experiment | Models | Outer × inner folds | V3 mean fold AUC | V3 balanced accuracy | V3 mean selected edges |
|---|---|---|---:|---:|---:|
| Exploratory comparison | V3, LASSO, Elastic Net, RFE-SVM | 3 × 3 | 0.7037 | 0.6481 | 58.7 |
| V3 only | V3 | 5 × 5 | 0.7187 | 0.6100 | 108.6 |

The five V3 held-out fold AUCs are 0.733, 0.767, 0.700, 0.633, 0.760. All 100 inner and five outer fits completed with checked primal feasibility, dual feasibility, stationarity and complementarity. The V3 grid has four (alpha, ratio) choices: alpha 0.2 or 0.8 crossed with ratio 0.2 or 0.05, ridge 0. The one-standard-error selection rule chose models within the inner folds. Mean pairwise edge Jaccard across five outer supports is 0.090; high edge frequency in the accompanying CSV is exploratory, not a validated biomarker.

The three-fold comparison used two candidates per method and only three outer folds; its reported mean fold AUCs are v3 0.704, lasso 0.650, enet 0.700, rfe_svm 0.683. The reconstructed RFE-SVM is not an exact reproduction of the professor’s original baseline without its source and settings.

## Interpretation and outstanding experiment

The old atlas name array mismatches the Lausanne IDs for **72 of 129 ROIs**. The edge index order is correct, so numerical scores were unaffected, but named connections in the old reports must be relabelled by the atlas ID. The corrected mapping and old-run edge tables are in `old_run_audit/`. `selected_edge_frequency.csv` uses corrected names from the new five-fold pilot.

No paired-fold p-value is reported because repeated folds overlap substantially. With only 54 subjects, the results are variable and do not identify a clinical biomarker or a winning model. The complete professor grid (72 V3 configurations, 5 outer folds × 5 repeats for development and 20 repeats for a final run) remains configured but has **not** been executed on real data. These pilots cannot be represented as that final protocol. V1 and V2 are also outside this V3-first rebuild.

The independent implementation, full configs, immutable split manifests, fold predictions, probabilities, selected edges, numerical diagnostics, and 17 passing tests are included. To run the prespecified full grid on suitable compute, use `run.py` with `configs/development_full.json` and the explicit `--representation fisher` argument; keep all fold checkpoints and report failed candidates rather than accepting them.
