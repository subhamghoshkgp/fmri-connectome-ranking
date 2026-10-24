# Results index

`results/primary_results_overview.csv` contains the 32 primary method/representation summaries used by the manuscript. Each representation has 12 saved variants x 100 outer folds in its `analysis/fold_metrics.csv`.

| Representation | Final analysis | Available fitted models |
|---|---|---|
| Whole brain 83 | `results/whole_brain_83/analysis/` | all 1,200 outer-model pairs |
| Whole brain 129 | `results/whole_brain_129/analysis/` | new rankers: 65 folds/method in explicitly partial directory; full baseline re-selection under `results/reselection_129/` |
| Cortex 68 | `results/cortical_68/analysis/` | export contains predictions/frequencies/certificates; original per-fold NPZ omitted |
| Cortex 114 | `results/cortical_114/analysis/` | export contains predictions/frequencies/certificates; original per-fold NPZ omitted |

The cortical folders include chosen candidates, subject predictions, selection frequencies, stability, convergence/KKT diagnostics, runtime, hub concentration and matched cortical-versus-whole-brain comparisons. `results/cortical_comparison/` combines cortical resolutions. Full sensitivity, specificity, Jaccard and one-SE controls remain available even though the four-page paper uses a compact main table.

Mean DRR AUC is 0.7940/0.8229 for whole-brain 83/129 and 0.7219/0.7851 for cortex 68/114. The same participants recur across folds and representations; these are descriptive comparisons rather than 100 independent samples. See the paper and original analysis summaries for support/stability and limitations.
