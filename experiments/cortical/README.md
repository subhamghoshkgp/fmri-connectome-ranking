# Dataset-1 cortical-only experiment: both atlas resolutions

Open `fmri_dataset1_cortical_only_colab.ipynb` in Google Colab. Choose a CPU runtime, mount Drive when requested, and run the cells in order. The notebook embeds all code, atlas metadata, cortical data and prior whole-brain reference metrics. No separate data upload or ZIP extraction is required.

## Anatomical restriction

The professor-supplied `ixc` MATLAB lists, cross-checked with `llist` and the XLS atlas labels, define the subset. MATLAB ROI IDs are converted to zero-based indices on **both** matrix axes. Only unique off-diagonal cortical-to-cortical connections are retained.

| Original atlas | Cortical ROIs | Features per participant | Participants |
|---|---:|---:|---:|
| 83 regions (scale33) | 68 | 2,278 | 54 |
| 129 regions (scale60) | 114 | 6,441 | 54 |

Excluded original MATLAB IDs are 35–41 and 76–83 at the 83-region scale, and 58–64 and 122–129 at the 129-region scale. Each atlas removes 14 subcortical ROIs and one brainstem ROI. Cortical–subcortical connections are also removed. A contiguous crop would be wrong.

The signed Pearson FC was extracted from the exact original Drive file, SHA256 `8e6944e1a3d31abce6123eea14c31a6b38cc92f01df90c8d95aefdcdacff2b6f`. Fisher arctanh is applied once, before training-only standardization. The bundled NPZ files contain Fisher features, so they must not be transformed again. The original 54-subject order is unchanged: 27 controls, followed by 27 patients. `DATA_AUDIT.json`, `cortical_regions.csv` and `cortical_edges.csv` document every retained original ROI/edge.

## Algorithms and evaluation

All 12 previous variants are rerun, with fresh cortical inner fits: reference-grid gated logistic, elastic net, LASSO and RFE-SVM under maximum-inner-AUC and one-SE selection (8 variants), plus expanded-grid gated logistic, gated ranking, gated ranking plus ridge refit and dense ridge ranking (4 variants). The training modules and full grids/solver tolerances are unchanged. Internal V3 filenames are retained for code/checkpoint traceability; the notebook provides descriptive method names.

The full protocol uses the exact saved 20 repeats × 5 stratified outer folds, with 5 inner folds: 100 outer evaluations per variant per resolution, or 2,400 outer models overall. All scaling, selector fitting, tuning, ranking pairs, refits and thresholds use training participants only. No previous whole-brain fit is reused to train cortical models. One-SE and max-AUC variants share the *new* cortical inner grid records when appropriate.

## Running, stopping and resuming

Default `JOBS` is at most 2 and numerical library threads are capped at 1 per worker. Increasing jobs beyond the CPUs/RAM available is not helpful. Use the notebook's CPU setting to change workers. GPU acceleration is not implemented.

The main cell runs both resolutions sequentially. Set `ADDITIONAL_FOLDS_PER_RESOLUTION=2` for short batches, or leave it `None` for the full remaining study. `--max-outer` denotes the first N saved outer folds, not N extra folds. All completed inner fits and outer model pairs are kept in Drive in new cortical-specific result folders.

Press the stop button once for a graceful pause between fitted/checkpointed tasks; another interrupt forces termination if necessary. After a runtime disconnect, rerun setup, unpacking, environment/tests and controls, then rerun the main computation cell. Keep the same `PROJECT`, `RUN_NAME`, code, inputs and configuration. Completed fits are reused. An in-progress uncommitted fit may run again. Never start two runtimes writing the same result directory, change/delete committed checkpoints, or reuse the old whole-brain output directory. If a lock cannot be obtained, wait for the original writer to exit.

Code runs locally in the Colab VM; persistent checkpoints live under `MyDrive/fmri_dataset1_cortical_only/results/`. Drive must be mounted before each session. Saving through Drive cannot guarantee preservation of a write that had not finished syncing at an abrupt disconnect; atomic JSON commits and numerical verification protect against incomplete committed model pairs.

The analysis cell reads actual committed arrays to recompute metrics and verify splits, scalers, scores, selected support, inner choices and convergence certificates. Interim metrics are marked incomplete. The full export requires 100 verified folds for **every variant at both resolutions**. The default export contains analysis, code, compact input data, provenance and comparison references; it deliberately excludes the large inner-fit checkpoint trees. Full outer-model export is optional, and may take longer.

## Outputs and interpretation

Each resolution produces `analysis/fold_metrics.csv` (1,200 rows × 10 columns when complete), subject predictions, chosen parameters, convergence certificates, runtime, hub concentration, Jaccard/Nogueira stability, cortical ROI/edge mapping and selection frequencies. Coefficients, scalers, masks and test scores are in each outer model's NPZ file. JSON holds tuning choices, convergence, KKT/gradient checks and timing.

`cortical_vs_fullbrain_summary.csv` and its paired-fold companion compare each new cortical result against the completed original whole-brain result at the matching resolution and saved fold. The combined report also compares the two cortical resolutions. These are the **same 54 participants**, not independent datasets. Repeated folds are dependent; differences are descriptive. Different feature universes make raw feature counts and Jaccard values sensitive to the available number of edges. Dense ridge ranking uses all available cortical edges; feature-selection stability is omitted. Cortical restriction is a hypothesis to test, not a guarantee of higher AUC or biological validity.

## Validation supplied with this package

See `validation/VALIDATION_STATUS.json` for automated tests, source/checksum comparisons, reduced-grid real-data runs and restart verification. Engineering smoke outputs are not final scientific results and are not merged into the final study.

## Optional local use

Extract the code ZIP, install Python 3.13.15 and `requirements.lock.txt`, then run from the extracted folder:

```bash
python -u -c "import run_cortical as r; r.main()" --resolution 83 --out /path/to/NEW/cortex68_results --jobs 2
python -u -c "import run_cortical as r; r.main()" --resolution 129 --out /path/to/NEW/cortex114_results --jobs 2
```

Use the **imported entry point** shown above when using multiple workers. The original script-style entry point previously caused a joblib serialization error in Colab. `MPLBACKEND=Agg` is set before imports to avoid the earlier inline-backend error.
