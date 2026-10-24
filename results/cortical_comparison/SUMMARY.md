# Cortical-only Dataset-1: both resolutions

Full study complete: **True**.

Each final resolution requires 100 outer evaluations for each of 12 variants; 2,400 outer models overall.

| Original atlas | Cortex | Method | Folds | AUC mean | Fold SD | BA | F1 | Mean edges | Jaccard | Nogueira |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 83 | 68 | Incidence-gated logistic (reference grid) | 100 | 0.6297 | 0.1502 | 0.5858 | 0.5829 | 99.9300 | 0.1241 | 0.2013 |
| 83 | 68 | Reference-grid gated logistic (one-SE) | 100 | 0.6349 | 0.1566 | 0.5770 | 0.5665 | 33.1800 | 0.1271 | 0.1821 |
| 83 | 68 | Elastic net | 100 | 0.6330 | 0.1491 | 0.5860 | 0.5633 | 91.4900 | 0.1371 | 0.2342 |
| 83 | 68 | Elastic net (one-SE) | 100 | 0.6308 | 0.1547 | 0.5757 | 0.5739 | 23.9700 | 0.1312 | 0.1838 |
| 83 | 68 | LASSO | 100 | 0.6337 | 0.1654 | 0.5962 | 0.5809 | 17.9600 | 0.1346 | 0.2285 |
| 83 | 68 | LASSO (one-SE) | 100 | 0.6089 | 0.1508 | 0.5618 | 0.5554 | 8.2900 | 0.1281 | 0.2019 |
| 83 | 68 | RFE-SVM | 100 | 0.6475 | 0.1585 | 0.5940 | 0.5840 | 370.3500 | 0.1486 | 0.1634 |
| 83 | 68 | RFE-SVM (one-SE) | 100 | 0.6189 | 0.1692 | 0.5872 | 0.5612 | 89.2500 | 0.1079 | 0.1189 |
| 83 | 68 | Incidence-gated logistic (expanded grid) | 100 | 0.6675 | 0.1515 | 0.6155 | 0.6047 | 705.8400 | 0.2528 | 0.1857 |
| 83 | 68 | Incidence-gated ranking | 100 | 0.6563 | 0.1503 | 0.6043 | 0.6013 | 449.4100 | 0.2267 | 0.2596 |
| 83 | 68 | Incidence-gated ranking + ridge refit | 100 | 0.6752 | 0.1484 | 0.6180 | 0.6129 | 414.1500 | 0.2308 | 0.2678 |
| 83 | 68 | Dense ridge ranking | 100 | 0.7219 | 0.1320 | 0.6667 | 0.6644 | 2278.0000 | — | — |
| 129 | 114 | Incidence-gated logistic (reference grid) | 100 | 0.6878 | 0.1579 | 0.6273 | 0.6160 | 201.4100 | 0.1348 | 0.2339 |
| 129 | 114 | Reference-grid gated logistic (one-SE) | 100 | 0.6414 | 0.1555 | 0.5903 | 0.5800 | 68.1600 | 0.0876 | 0.1577 |
| 129 | 114 | Elastic net | 100 | 0.7217 | 0.1566 | 0.6417 | 0.6304 | 149.9500 | 0.1598 | 0.2895 |
| 129 | 114 | Elastic net (one-SE) | 100 | 0.6659 | 0.1534 | 0.6118 | 0.5993 | 35.2900 | 0.1186 | 0.1997 |
| 129 | 114 | LASSO | 100 | 0.6519 | 0.1517 | 0.6045 | 0.5874 | 19.5100 | 0.1019 | 0.1887 |
| 129 | 114 | LASSO (one-SE) | 100 | 0.6233 | 0.1451 | 0.5972 | 0.5851 | 10.7100 | 0.0948 | 0.1756 |
| 129 | 114 | RFE-SVM | 100 | 0.7588 | 0.1447 | 0.6758 | 0.6662 | 437.3000 | 0.1680 | 0.2641 |
| 129 | 114 | RFE-SVM (one-SE) | 100 | 0.6846 | 0.1621 | 0.6165 | 0.6041 | 98.5000 | 0.1124 | 0.1748 |
| 129 | 114 | Incidence-gated logistic (expanded grid) | 100 | 0.7514 | 0.1528 | 0.6738 | 0.6622 | 1149.2100 | 0.2278 | 0.2711 |
| 129 | 114 | Incidence-gated ranking | 100 | 0.7402 | 0.1427 | 0.6508 | 0.6407 | 636.7400 | 0.1977 | 0.2886 |
| 129 | 114 | Incidence-gated ranking + ridge refit | 100 | 0.7554 | 0.1402 | 0.6815 | 0.6625 | 583.4400 | 0.1821 | 0.2663 |
| 129 | 114 | Dense ridge ranking | 100 | 0.7851 | 0.1261 | 0.7267 | 0.7204 | 6441.0000 | — | — |

Same 54 participants, dependent repeated CV folds. Descriptive comparisons; no independent-cohort or significance claim. Feature universes differ between resolutions and full/cortical analyses. No cross-atlas Jaccard is computed.

The anatomical restriction removes cortical–subcortical and subcortical–subcortical edges, not just columns selected by a classifier. All model tuning is performed afresh on cortical features.

Compare cortical_vs_fullbrain_summary.csv within each resolution for matched-fold changes. Dense ridge ranking uses every available cortical connection; its feature-selection stability is not informative. Raw support counts/Jaccard depend on the available feature universe.
