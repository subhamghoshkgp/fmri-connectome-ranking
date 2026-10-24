# Numerical and methodological verification

## Input FC: signed Fisher-transformed correlation

The original `27_SCHZ_CTRL_dataset.mat` has SHA256:

`8e6944e1a3d31abce6123eea14c31a6b38cc92f01df90c8d95aefdcdacff2b6f`

The analysis selects one unique off-diagonal upper-triangular edge per ROI pair. All 54 participants remain, in their original control/patient ordering.

| Scale | Raw signed Pearson range | Fisher feature range | Negative edge values / all participant-edge values |
|---|---|---|---|
| 83 ROIs | −0.7532017 to 0.9788603 | −0.9803138 to 2.2695608 | 42,655 / 183,762 |
| 129 ROIs | −0.7773523 to 0.9761280 | −1.0386446 to 2.2080938 | 116,353 / 445,824 |

Recomputing `arctanh(raw)` matches the full-resolution loaders with maximum absolute difference zero in the verification execution. In contrast, `arctanh(abs(raw))` differs by as much as 1.9606276 and 2.0772892. Signed Fisher input is therefore directly established from the arrays and implementations, in agreement with the saved `representation="fisher"` configuration. Floating-point byte checksums of independently recomputed Fisher features are not assumed portable across CPUs. Prior audit discrepancies and their scope are preserved in `evidence/protocol_129.json` and `evidence/revision_results_review.json`.

The dataset source describes Pearson connectivity: https://doi.org/10.5281/zenodo.3758534 . The source repository's Pearson definition alone does not establish the subsequent Fisher transform; that transform was verified in the experiment code and saved configurations.

## Primary quantitative interpretation

| Resolution | DRR mean AUC | RFE–SVM mean AUC | DRR minus RFE–SVM | DRR minus IGL |
|---|---:|---:|---:|---:|
| 83 ROIs | 0.7940 | 0.7491 | 0.0449 | 0.0328 |
| 129 ROIs | 0.8229 | 0.7929 | 0.0300 | 0.0252 |

IGR-R retains 643.85 and 839.83 mean edges, representing 81.1% and 89.8% fewer than DRR's full inputs. Its mean-AUC gaps to DRR are 0.0331 and 0.0293. Against IGL it reduces mean support by 37.5% and 50.3%, with AUC decreases of 0.0003 and 0.0041. IGR has the highest Nogueira stability among the eight primary pipelines. These are descriptive comparisons, not significance claims.

## Mathematical checks

The proposition is stated for supports of equal size and coefficient magnitude, where both graph configurations exist. It applies to the edge-plus-capacity penalty; adding the identical ridge term to both supports preserves the ordering. The proof does not claim test-AUC superiority.

The ridge lemma uses an **untruncated thin SVD**, including every thin coordinate even if the design is rank deficient. The loss depends only on scores and the positive ridge penalty removes components orthogonal to the thin right-singular subspace. Numerical checks independently compare star/matching capacity linear programs and full/reduced ridge optimization on a centered rank-deficient design; see `evidence/MATHEMATICAL_CHECKS.json`. These checks support the displayed derivations and are not additional clinical experiments.

## Sources and claims

The manuscript keeps a focused set of 13 numbered references. Newly explicit foundational references are Hoerl and Kennard (1970), DOI 10.1080/00401706.1970.10488634, and Belloni and Chernozhukov (2013), DOI 10.3150/11-BEJ410. The latter is cited as prior post-selection estimation, without importing its OLS theory as a guarantee for our ranking refit.

Primary sources consulted include the official ISBI instructions, source dataset, latent overlapping-group paper, IJCAI AUC-consistency paper, JMLR stability paper, connectome RFE/structured-classification papers, ridge paper, and post-selection paper. Useful direct links:

- https://biomedicalimaging.org/2027/papers/
- https://zenodo.org/records/3758534
- https://arxiv.org/abs/1110.0413
- https://www.ijcai.org/Proceedings/15/Papers/137.pdf
- https://www.jmlr.org/beta/papers/v18/17-514.html
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7334612/
- https://arxiv.org/abs/1701.08140
- https://arxiv.org/abs/1310.5415
- https://arxiv.org/abs/1001.0188
- https://www.stat.cmu.edu/technometrics/70-79/VOL-12-01/v1201055.pdf
- https://arxiv.org/abs/2511.05531
- https://arxiv.org/abs/1706.07581

No new training or new cortical-only performance result is claimed in the paper.
