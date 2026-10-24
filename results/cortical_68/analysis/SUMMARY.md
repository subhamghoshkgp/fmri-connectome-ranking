# Cortical-only Dataset-1: 83 original regions → 68 cortical regions

Complete: **True**. Expected outer folds per variant: **100**.

| Method | Folds | Mean AUC | Fold SD | Balanced accuracy | F1 | Sensitivity | Specificity | Mean edges | Jaccard | Nogueira |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| max_auc_v3 | 100 | 0.6297 | 0.1502 | 0.5858 | 0.5829 | 0.5997 | 0.5720 | 99.9300 | 0.1241 | 0.2013 |
| one_se_v3 | 100 | 0.6349 | 0.1566 | 0.5770 | 0.5665 | 0.5917 | 0.5623 | 33.1800 | 0.1271 | 0.1821 |
| max_auc_enet | 100 | 0.6330 | 0.1491 | 0.5860 | 0.5633 | 0.5763 | 0.5957 | 91.4900 | 0.1371 | 0.2342 |
| one_se_enet | 100 | 0.6308 | 0.1547 | 0.5757 | 0.5739 | 0.6160 | 0.5353 | 23.9700 | 0.1312 | 0.1838 |
| max_auc_lasso | 100 | 0.6337 | 0.1654 | 0.5962 | 0.5809 | 0.5943 | 0.5980 | 17.9600 | 0.1346 | 0.2285 |
| one_se_lasso | 100 | 0.6089 | 0.1508 | 0.5618 | 0.5554 | 0.6183 | 0.5053 | 8.2900 | 0.1281 | 0.2019 |
| max_auc_rfe_svm | 100 | 0.6475 | 0.1585 | 0.5940 | 0.5840 | 0.5927 | 0.5953 | 370.3500 | 0.1486 | 0.1634 |
| one_se_rfe_svm | 100 | 0.6189 | 0.1692 | 0.5872 | 0.5612 | 0.5650 | 0.6093 | 89.2500 | 0.1079 | 0.1189 |
| v3_expanded | 100 | 0.6675 | 0.1515 | 0.6155 | 0.6047 | 0.6097 | 0.6213 | 705.8400 | 0.2528 | 0.1857 |
| v3_rank | 100 | 0.6563 | 0.1503 | 0.6043 | 0.6013 | 0.6180 | 0.5907 | 449.4100 | 0.2267 | 0.2596 |
| v3_rank_refit | 100 | 0.6752 | 0.1484 | 0.6180 | 0.6129 | 0.6270 | 0.6090 | 414.1500 | 0.2308 | 0.2678 |
| ridge_rank | 100 | 0.7219 | 0.1320 | 0.6667 | 0.6644 | 0.6800 | 0.6533 | 2278.0000 | NA | NA |

Primary endpoint: mean outer-fold AUC. Ranking scores are not calibrated probabilities.

Fold variability and paired differences are descriptive; repeated folds reuse subjects. Frozen grids are inherited from Dataset-1, with unequal search budgets.

All 54 subjects are retained: 27 per class. No augmentation or synthetic subjects are introduced.

CORTICAL ONLY: original 54 subjects, atlas-defined cortical-to-cortical edges, unchanged models/grids/solvers, exact saved 20x5 outer and 5 inner CV splits; all inner tuning refitted on cortical features.


Scalers, scores, raw coefficients, support masks, atlas indices, inner candidate choices, convergence/KKT, saved splits, metrics and hubs checked.
Cortical-only performance is a hypothesis to test, not a guaranteed improvement.
