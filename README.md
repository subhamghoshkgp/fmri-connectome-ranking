# Functional Connectome Classification and Ranking

Compare sparse and dense learning methods for classifying functional brain connectomes. This repository contains experiment code, frozen configurations, and aggregate results for whole-brain and cortical-only analyses.

## Start here

- **View results:** [Results guide](docs/RESULTS_INDEX.md)
- **Run an experiment:** [Reproduction guide](docs/REPRODUCIBILITY.md)
- **Check the saved tables:** `python scripts/verify_results.py`

## Dataset and evaluation

The study uses the same 54 participants: 27 controls and 27 patients. The four representations are different views of this cohort.

| Representation | Regions | Connections |
|---|---:|---:|
| Whole brain | 83 | 3,403 |
| Whole brain | 129 | 8,256 |
| Cortical only, from 83 regions | 68 | 2,278 |
| Cortical only, from 129 regions | 114 | 6,441 |

Experiments use signed Fisher-transformed correlations, training-only standardization, and repeated nested cross-validation: 20 repeats of five outer folds, with five inner folds. Each representation has 100 evaluations for each of 12 saved variants.

## Methods

| Method | Implementation identifier | Purpose |
|---|---|---|
| Incidence-gated logistic, reference grid | `max_auc_v3` | Structured sparse baseline |
| Incidence-gated logistic, expanded grid | `v3_expanded` | Broader regularization search |
| Incidence-gated ranking | `v3_rank` | Learn positive–negative score ordering |
| Ranking with ridge refit | `v3_rank_refit` | Re-estimate weights on selected connections |
| Dense ridge ranking | `ridge_rank` | Prediction using all connections |
| LASSO, elastic net, RFE-SVM | `max_auc_lasso`, `max_auc_enet`, `max_auc_rfe_svm` | Comparison methods |

Four additional one-standard-error variants are retained as controls. Existing filenames and identifiers are preserved so the code remains compatible with its saved configurations and checkpoints.

## Results at a glance

Mean outer-fold ROC-AUC:

| Method | Whole brain 83 | Whole brain 129 | Cortex 68 | Cortex 114 |
|---|---:|---:|---:|---:|
| LASSO | 0.6519 | 0.6449 | 0.6337 | 0.6519 |
| Elastic net | 0.6867 | 0.7294 | 0.6330 | 0.7217 |
| RFE-SVM | 0.7491 | 0.7929 | 0.6475 | 0.7588 |
| Gated logistic, reference | 0.7069 | 0.7056 | 0.6297 | 0.6878 |
| Gated logistic, expanded | 0.7612 | 0.7977 | 0.6675 | 0.7514 |
| Gated ranking | 0.7554 | 0.7762 | 0.6563 | 0.7402 |
| Ranking with refit | 0.7609 | 0.7936 | 0.6752 | 0.7554 |
| Dense ridge ranking | **0.7940** | **0.8229** | **0.7219** | **0.7851** |

AUC measures ranking discrimination, not accuracy. Repeated folds reuse participants, so these comparisons are descriptive rather than independent validation studies.

## Repository layout

```text
experiments/
  whole_brain_83/     All-method 83-region experiment
  whole_brain_129/    Ranking experiment at 129 regions
  reselection_129/    Re-selection of saved baseline candidates
  cortical/          Cortical-only experiments at both resolutions
results/
  whole_brain_83/analysis/
  whole_brain_129/analysis/
  cortical_68/analysis/
  cortical_114/analysis/
  primary_results_overview.csv
docs/                Setup, run instructions, and results guide
scripts/             Aggregate-results verification
```

## Quick setup

Clone this repository and open a terminal in its root folder. Use Python **3.13.15**, matching the recorded later experiment environments.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the lockfile for the experiment you want to run. For example:

```bash
python -m pip install -r experiments/whole_brain_83/requirements.lock.txt
python scripts/verify_results.py
```

Datasets, participant-level outputs, and large saved-model files are managed separately to keep this repository lightweight. The [reproduction guide](docs/REPRODUCIBILITY.md) lists the input files and commands for each experiment. Start with one outer fold before launching all 100.
