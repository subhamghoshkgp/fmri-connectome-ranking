# Cortical-only Dataset-1: 129 original regions → 114 cortical regions

Complete: **True**. Expected outer folds per variant: **100**.

| Method | Folds | Mean AUC | Fold SD | Balanced accuracy | F1 | Sensitivity | Specificity | Mean edges | Jaccard | Nogueira |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max_auc_v3 | 100 | 0.6878 | 0.1579 | 0.6273 | 0.6160 | 0.6200 | 0.6347 | 201.4100 | 0.1348 | 0.2339 |
| one_se_v3 | 100 | 0.6414 | 0.1555 | 0.5903 | 0.5800 | 0.5940 | 0.5867 | 68.1600 | 0.0876 | 0.1577 |
| max_auc_enet | 100 | 0.7217 | 0.1566 | 0.6417 | 0.6304 | 0.6377 | 0.6457 | 149.9500 | 0.1598 | 0.2895 |
| one_se_enet | 100 | 0.6659 | 0.1534 | 0.6118 | 0.5993 | 0.6120 | 0.6117 | 35.2900 | 0.1186 | 0.1997 |
| max_auc_lasso | 100 | 0.6519 | 0.1517 | 0.6045 | 0.5874 | 0.5993 | 0.6097 | 19.5100 | 0.1019 | 0.1887 |
| one_se_lasso | 100 | 0.6233 | 0.1451 | 0.5972 | 0.5851 | 0.6163 | 0.5780 | 10.7100 | 0.0948 | 0.1756 |
| max_auc_rfe_svm | 100 | 0.7588 | 0.1447 | 0.6758 | 0.6662 | 0.6683 | 0.6833 | 437.3000 | 0.1680 | 0.2641 |
| one_se_rfe_svm | 100 | 0.6846 | 0.1621 | 0.6165 | 0.6041 | 0.6183 | 0.6147 | 98.5000 | 0.1124 | 0.1748 |
| v3_expanded | 100 | 0.7514 | 0.1528 | 0.6738 | 0.6622 | 0.6617 | 0.6860 | 1149.2100 | 0.2278 | 0.2711 |
| v3_rank | 100 | 0.7402 | 0.1427 | 0.6508 | 0.6407 | 0.6490 | 0.6527 | 636.7400 | 0.1977 | 0.2886 |
| v3_rank_refit | 100 | 0.7554 | 0.1402 | 0.6815 | 0.6625 | 0.6587 | 0.7043 | 583.4400 | 0.1821 | 0.2663 |
| ridge_rank | 100 | 0.7851 | 0.1261 | 0.7267 | 0.7204 | 0.7250 | 0.7283 | 6441.0000 | NA | NA |

Primary endpoint: mean outer-fold AUC. Ranking scores are not calibrated probabilities.

Fold variability and paired differences are descriptive; repeated folds reuse subjects. Frozen grids are inherited from Dataset-1, with unequal search budgets.

All 54 subjects are retained: 27 per class. No augmentation or synthetic subjects are introduced.

CORTICAL ONLY: original 54 subjects, atlas-defined cortical-to-cortical edges, unchanged models/grids/solvers, exact saved 20x5 outer and 5 inner CV splits; all inner tuning refitted on cortical features.


Scalers, scores, raw coefficients, support masks, atlas indices, inner candidate choices, convergence/KKT, saved splits, metrics and hubs checked.
Cortical-only performance is a hypothesis to test, not a guaranteed improvement.
