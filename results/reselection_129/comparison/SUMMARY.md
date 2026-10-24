# Dataset-1: one-SE versus max-inner-AUC

Complete: **True**. Outer refits: **400/400**. Inner tuning fits rerun: **0**.

Fold means are descriptive. Paired deltas use the same saved outer splits. Repetitions share the same subjects; SD is not a confidence interval.

| Method | Folds | Candidate changes | AUC old/new | Balanced accuracy old/new | F1 old/new | Sensitivity old/new | Specificity old/new | Support old/new |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 | 100/100 | 85 | 0.6489 / 0.7056 | 0.6107 / 0.6463 | 0.6141 / 0.6470 | 0.6453 / 0.6773 | 0.5760 / 0.6153 | 79.0400 / 255.2600 |
| enet | 100/100 | 90 | 0.6855 / 0.7294 | 0.6250 / 0.6450 | 0.6180 / 0.6486 | 0.6473 / 0.6780 | 0.6027 / 0.6120 | 52.0200 / 176.5200 |
| lasso | 100/100 | 75 | 0.6328 / 0.6449 | 0.5910 / 0.6015 | 0.5710 / 0.5902 | 0.6043 / 0.6070 | 0.5777 / 0.5960 | 12.6400 / 22.1000 |
| rfe_svm | 100/100 | 85 | 0.7297 / 0.7929 | 0.6555 / 0.7090 | 0.6478 / 0.7028 | 0.6690 / 0.7173 | 0.6420 / 0.7007 | 181.1500 / 614.1500 |

| Method | Jaccard old/new | Nogueira old/new | Kuncheva old/new (equal sizes only) |
|---|---:|---:|---:|
| v3 | 0.0947 / 0.1349 | 0.1653 / 0.2430 | 0.1810 / 0.2229 |
| enet | 0.1166 / 0.1591 | 0.2061 / 0.2909 | 0.2741 / 0.4063 |
| lasso | 0.0857 / 0.0980 | 0.1668 / 0.1833 | 0.2351 / 0.2131 |
| rfe_svm | 0.1257 / 0.1965 | 0.2157 / 0.3137 | 0.3240 / 0.4273 |

## V3 hub concentration

No numerical hub statistic was defined in the supplied original implementation; the following definitions are explicit additions to reporting.

`degree_HHI = sum_v (d_v / (2 * selected_edges))^2`, where d_v is selected-edge incidence. The weighted analogue uses sums of absolute standardized coefficients on selected incident edges. Larger values mean more concentration. Support size affects HHI. Top-5 shares count both endpoints; empty support is undefined.

- degree_hhi: old 0.05997461706889184, new 0.03815918705421376; paired delta -0.021815430014678082.
- strength_hhi: old 0.08756641385399438, new 0.05631232426351279; paired delta -0.03125408959048159.
- degree_top5_share: old 0.3780349825997444, new 0.25067129005322863; paired delta -0.12736369254651572.
- strength_top5_share: old 0.4781517747901261, new 0.3468037900349733; paired delta -0.1313479847551529.
- degree_effective_rois: old 27.26441314964989, new 48.09325502337329; paired delta 20.828841873723405.
- strength_effective_rois: old 18.676876872120506, new 32.32879641990224; paired delta 13.651919547781729.

Selection used only saved inner-CV summaries, without test scores. Every selected candidate was refitted, including unchanged choices. RFE-SVM retains its original training-only 3-fold probability calibration, which is distinct from hyperparameter tuning.

Convergence/KKT, parameters, runtime, and old/new predictions/coefficients/support are saved per fold. A failed outer refit cannot trigger a different candidate choice.

This comparison assesses sensitivity to the selection rule. Algorithm changes should be considered after reviewing these results; validation on a new cohort would still be needed.
