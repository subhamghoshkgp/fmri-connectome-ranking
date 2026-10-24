# Ranking V3 hypothesis test

Complete: **True**. Expected outer folds per new method: **100**.

| Method | Folds | AUC mean | AUC fold SD | Balanced accuracy | F1 | Mean features | Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|
| saved_max_auc_v3 | 100 | 0.7056 | 0.151338865797209 | 0.6463 | 0.6470 | 255.25 | 0.1349281823362149 |
| saved_one_se_v3 | 100 | 0.6489 | 0.14135802782801327 | 0.6107 | 0.6141 | 79.04 | 0.09469376262345094 |
| saved_max_auc_enet | 100 | 0.7294 | 0.15378529496427895 | 0.6450 | 0.6486 | 176.52 | 0.15906058322479272 |
| saved_one_se_enet | 100 | 0.6855 | 0.15077615877888892 | 0.6250 | 0.6180 | 52.02 | 0.11660120366897395 |
| saved_max_auc_lasso | 100 | 0.6449 | 0.14998876126995378 | 0.6015 | 0.5902 | 22.10 | 0.09804982000791033 |
| saved_one_se_lasso | 100 | 0.6328 | 0.15451798263785876 | 0.5910 | 0.5710 | 12.64 | 0.08565737053477288 |
| saved_max_auc_rfe_svm | 100 | 0.7929 | 0.14073024238710902 | 0.7090 | 0.7028 | 614.15 | 0.19645971870465243 |
| saved_one_se_rfe_svm | 100 | 0.7297 | 0.15174074115165156 | 0.6555 | 0.6478 | 181.15 | 0.12567687986387754 |
| v3_expanded | 100 | 0.7977 | 0.15327394546008907 | 0.7098 | 0.7066 | 1690.66 | 0.28607064658120945 |
| v3_rank | 100 | 0.7762 | 0.15854063692883358 | 0.6880 | 0.6875 | 970.99 | 0.2541884444313466 |
| v3_rank_refit | 100 | 0.7936 | 0.15186064850927172 | 0.7048 | 0.7074 | 839.83 | 0.22766758974673054 |
| ridge_rank | 100 | 0.8229 | 0.12444920625810196 | 0.7423 | 0.7470 | 8256.00 | None |

New arms: expanded ordinary logistic V3; ranking V3; ranking V3 with ridge refit; ungated ridge ranking. Saved one-SE and max-AUC baselines are reference rows, not newly trained models.

The ranking-vs-expanded-logistic contrast matches selector grids. The refit-vs-ranking contrast adds the jointly tuned refit strength. The ungated ranker checks whether gating adds value. These comparisons assess pipelines and tuning procedures, not isolated causal effects.

Only label a run complete when each new method has all expected folds. New inner fits are essential; old tuning summaries cannot validate a changed loss or a new refit.

AUC is primary. Ranking scores are not calibrated probabilities; labels use the training class-score midpoint. Larger input supports can inflate overlap. V3 selector capacities describe the selector, not the refitted coefficients.

Fold SD and paired fold deltas are descriptive. The same subjects and outer results have already informed development. No independent-fold significance test is provided. Retune baselines with comparable search effort and validate on a new cohort before making a superiority claim.
