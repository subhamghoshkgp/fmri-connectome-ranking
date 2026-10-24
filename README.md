# Sparse-to-Dense Functional Connectome Ranking

Research code and Dataset-1 results for professor review.

**Upload status: PARTIAL.** Code, core analysis tables and the manuscript are available. Large inputs, standalone notebooks, detailed tables and fitted-model archives have not all been committed. This repository is not yet a complete standalone reproduction package. See `UPLOAD_STATUS.md`.

## Study and results

The same 54 participants (27 controls and 27 patients) are evaluated at four representations using saved 20 × 5 outer CV splits, with five inner folds. Inputs use signed Fisher-transformed Pearson correlations and training-only standardization.

| Representation | Regions | Edges | Completed metric rows |
|---|---:|---:|---:|
| Whole brain | 83 | 3,403 | 1,200 |
| Whole brain | 129 | 8,256 | 1,200 |
| Cortical only | 68 | 2,278 | 1,200 |
| Cortical only | 114 | 6,441 | 1,200 |

All **4,800 outer-fold metric rows** are committed under `results/{whole_brain_83,whole_brain_129,cortical_68,cortical_114}/analysis/fold_metrics.csv`. The 32 primary summaries are in `results/primary_results_overview.csv`.

## Contents currently available

- `experiments/`: Python runners, model definitions, configurations and tests for whole-brain experiments, re-selection and cortical-only experiments. Some input/reference files remain pending.
- `results/`: core final fold metrics, summaries and available supporting text analyses; some large detailed tables remain pending.
- `paper/isbi2027_sparse_to_dense.pdf`: four-page manuscript.
- `paper/`: LaTeX, official style, table source and supporting evidence. Some binary figures remain pending.
- `docs/`: experiment/result descriptions.
- `provenance/`: source-package inventories and coverage records. These describe the prepared source package, not proof that every listed file has been committed.

## Method names

IGL-ref is the reference-grid incidence-gated logistic model (`max_auc_v3`); IGL is expanded-grid logistic (`v3_expanded`); IGR is ranking (`v3_rank`); IGR-R adds ridge refitting (`v3_rank_refit`); DRR is dense ridge ranking (`ridge_rank`). Comparisons include LASSO, elastic net and RFE-SVM, plus one-standard-error controls.

Dense ridge ranking has the highest observed AUC across these four representations. The resolutions reuse the same participants; results do not establish independent-cohort or clinical superiority.

## Source-archive limitations

Even the prepared full package has only 65 outer fits per new ranking method at 129 regions, although its analysis summaries contain all 100 evaluations. The supplied cortical export omits per-fold model NPZ files. No missing fits were fabricated or retrained.

Institutional secondary-use ethics, funding/conflicts and final authorship still require author confirmation before submission.
