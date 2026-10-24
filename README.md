# Functional Connectome Classification and Ranking

Research implementation and aggregate results for Dataset-1. No manuscript or conference assets are included.

## Study

The same 54 participants (27 controls and 27 patients) are evaluated using whole-brain representations at 83 and 129 regions and cortical subsets at 68 and 114 regions. Training uses signed Fisher-transformed Pearson correlations, training-only standardization and saved repeated nested cross-validation splits.

## Included

- `experiments/`: Python runners, model modules, frozen hyperparameter grids, tests, solver settings and available protocol references.
- `results/whole_brain_83/analysis/fold_metrics.csv`
- `results/whole_brain_129/analysis/fold_metrics.csv`
- `results/cortical_68/analysis/fold_metrics.csv`
- `results/cortical_114/analysis/fold_metrics.csv`
- `results/primary_results_overview.csv`: 32 primary summaries.
- Supporting aggregate analyses and experiment descriptions.

Each fold-metrics file contains 1,200 rows: 12 variants with 100 outer evaluations each, totaling **4,800 result rows**.

## Algorithms

Incidence-gated logistic learning (reference and expanded grids), incidence-gated ranking, ranking with support-restricted ridge refitting, and dense ridge ranking are compared against logistic LASSO, elastic net and linear RFE-SVM. One-standard-error controls are retained. Internal V3 identifiers are preserved for compatibility with the original runners and results.

## Reproduction and data access

This repository contains the code and aggregate results, but is **not a complete standalone rerun bundle**. Raw/compact participant inputs, participant-level predictions, fitted-model archives containing participant information, and standalone notebooks with embedded dataset payloads were not uploaded. Automatic approval review blocked transfers of participant-level fMRI data.

Use the original dataset and previously supplied notebooks in your authorized local/Drive environment. Preserve the original data, code versions, split definitions, configurations and output directory when resuming an existing experiment. Run the relevant runner with `--help` to inspect its exact interface and use its original pinned environment.

The full source package's model-archive limitations also remain: the older 129-region ranking export contains only 65 fits per new method despite complete 100-fold aggregate summaries, and the supplied cortical export omits raw per-fold model NPZ files. No missing fit was fabricated.

## Interpretation

Dense ridge ranking gives the highest observed AUC across these four representations. All representations reuse the same participants; they are not independent validation datasets. Repeated CV fold variability is descriptive.

See `UPLOAD_STATUS.md` for the final transfer scope.
