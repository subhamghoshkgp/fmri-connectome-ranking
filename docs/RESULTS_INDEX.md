# Results guide

Start with `results/primary_results_overview.csv`: 32 rows covering eight primary methods across four representations.

| Representation | Detailed aggregate results | Evaluations |
|---|---|---:|
| Whole brain 83 | `results/whole_brain_83/analysis/fold_metrics.csv` | 1,200 |
| Whole brain 129 | `results/whole_brain_129/analysis/fold_metrics.csv` | 1,200 |
| Cortical 68 | `results/cortical_68/analysis/fold_metrics.csv` | 1,200 |
| Cortical 114 | `results/cortical_114/analysis/fold_metrics.csv` | 1,200 |

Each detailed table has 12 variants × 100 outer evaluations. Analysis folders also contain available summaries, support/stability statistics, paired comparisons and diagnostics. Large participant-level outputs and saved models are kept with the original experiment runs.

## Reading the metrics

- **ROC-AUC:** discrimination from continuous scores; higher is better.
- **Balanced accuracy:** average of sensitivity and specificity.
- **F1:** balance between positive-class precision and recall.
- **Support:** number of connections used by a model.
- **Jaccard / Nogueira:** reproducibility of selected connections across fits; interpret alongside support size.

Dense ridge ranking has the highest observed mean AUC in all four representations. Sparse methods trade discrimination against connection reduction and selection stability. Fold SD describes variability; repeated folds reuse participants and should not be treated as independent samples.

Run `python scripts/verify_results.py` to check the aggregate tables against the primary summaries.
