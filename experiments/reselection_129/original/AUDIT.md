# Audit of the prior fMRI work

**Update after receipt of the real dataset and saved results:** The source MAT file, old result archive, and old Colab notebook became available after this initial audit was written. The old pipeline source ZIP is still missing. `REAL_DATA_AUDIT.md` supersedes the availability statements and quantifies the independent numerical verification. The saved scores reconstruct from Fisher-transformed source FC data and the saved coefficients; all 100 fold metrics match. The 25 V3 fits and 25 Elastic Net fits have since been re-solved with explicit KKT checks. Historical claims in the sections below should be read alongside that new evidence.

## Conclusion

The professor's Version 3 formulation can be implemented as a convex constrained logistic model. The attached conversation is useful development history, but it does not establish that the old Dataset-1 results are numerically converged or that the quoted significance tests are valid. Treat those results as provisional until the original fits are examined or the experiment is rerun with explicit numerical checks.

This package is an independent rebuild. The source MAT was subsequently retrieved from the user-provided public link, and real-data numerical audits and pilots are now included. The available files reviewed were SZ_Idea.pdf, README.rtf, the cortical-label MAT file, the Lausanne spreadsheet, and the Claude conversation export. The result ZIP and notebook were subsequently supplied. The old source ZIP imported by that notebook was not.

## Findings that change the experiment

### Solver convergence was not established for the old run

The transcript reports `converged: 0/25` for the completed development run and repeatedly argues that 80 iterations are enough because one full-data problem matched a reference support. This does not establish convergence on different training subsets, penalties, or inner-validation fits. Matching the selected support alone does not imply matching coefficients, objective, or predicted probabilities. Small constraint infeasibility establishes approximate feasibility, not optimality. A non-converged fit is not automatically wrong, but it needs additional evidence before being accepted.

The rebuild checks primal feasibility, dual feasibility, stationarity, and complementary slackness for each constrained logistic fit. An optimal solver status alone is insufficient: residuals must also meet the declared tolerance. A failed candidate is recorded and excluded; failure of every candidate or the final refit stops the experiment. The fit cannot be relabelled as converged by changing a display flag.

### The regularization direction was reversed

In the discussion dated September 20, ratio 0.05 is called the sparsest tested point, and 0.02 or 0.01 is proposed as an even sparser direction. Here lambda equals ratio times lambda_max. Lower ratios weaken both the edge and node penalties, generally allowing more activity. Individual support sizes need not be monotone in correlated problems, but the penalty direction is unambiguous. The proposed interpretation was reversed.

### The quoted significance claims are unsupported

The conversation reports p = 0.0001 for V3 versus Elastic Net and RFE-SVM from a paired sign-flip procedure over repeated folds, then describes the gaps as proven real. Overlapping training sets and repeated use of the same 54 subjects invalidate a naive independent-fold interpretation. The exact old statistical code is absent, so its calculation cannot be independently reconstructed here. Neither those p-values nor a new automatic GOOD/BAD verdict should be used as confirmatory evidence.

The rebuild reports fold metrics, per-repeat summaries, and paired descriptive differences without p-values. A defensible inferential analysis needs a separately specified method and assumptions. A subject-label permutation experiment must rerun the full preprocessing/tuning/evaluation pipeline under each permissible permutation; it tests a different null from a generic pairwise algorithm superiority claim. If actual matching, family, or site constraints exist, exchangeability must respect them.

### Elastic Net convergence warnings were dismissed

The transcript calls these warnings harmless and lowers the iteration cap to reduce runtime while acknowledging non-convergence. This does not demonstrate that the predictive or sparsity estimates are unaffected. The rebuild uses the same certified conic logistic machinery for LASSO and Elastic Net. A separate small-data comparison against scikit-learn's independently implemented SAGA solver checks the loss and penalty normalization.

### The old ROI naming bug affects interpretation

The old implementation reportedly appended 15 missing labels after 114 cortical labels. At the 129-region scale, the noncortical IDs are 58-64 and 122-129 in MATLAB's one-based indexing. The left cortical block starts at ID 65. Appending placeholders shifts many names and can therefore mislabel cortical edges too.

Independently verified from the supplied files: the spreadsheet's LABELS sheet, scale60 columns, covers IDs 1-129 exactly; all 114 cortical ID/name pairs match the cortical-index MAT file. The rebuild maps by explicit ID, cross-checks the two sources, and labels the midline brainstem without a left-hemisphere prefix. Atlas spelling is preserved for traceability. A naming-only fix would not change predictions if the old numerical feature order was correct, but the saved scores now reconstruct from the source MAT, confirming the old numerical edge order. High structural degree is anatomical plausibility evidence, not proof of a region's identity.

### Preprocessing and baseline reproduction are still unresolved

The proposal requires the same functional-connectivity representation as the existing Dataset-1 implementation. The transcript used Fisher-transformed correlations. The supplied README establishes Pearson correlation connectivity but does not specify whether the prior classifier transformed it. The rebuild therefore requires an explicit `raw` or `fisher` argument and records the choice. It never takes absolute values or drops negative connections. The saved scores independently reconstruct under Fisher transformation, confirming the old convention.

RFE-SVM in this rebuild is a documented reconstruction with inner tuning of C and selected feature count. It is not labelled an exact reproduction of the published or professor's prior baseline without that code and its settings. Calibration is additional probability reporting; primary SVM AUC and labels use decision margins and the zero threshold.

### Stability measures need defined conventions

Standard Kuncheva requires equal, nonempty, non-full support sizes. The rebuild reports it only for valid pairs and reports the pair count. Unequal-size pairs use a separately named descriptive adjusted-overlap quantity: (intersection - k*l/p)/(min(k,l) - k*l/p). This is not silently presented as standard Kuncheva. Nogueira's variable-size stability is included as an additional descriptive summary. Empty/empty Jaccard is undefined in this implementation, so two empty discoveries do not become evidence of perfect biomarker stability.

No arbitrary truncation by feature index is used. Raw Jaccard can depend on support size; neither a larger Jaccard nor an adjusted overlap alone establishes a better biomarker method.

### Development choices must not be justified by test performance

The transcript changes grids after examining outer results. Such work can be useful exploration, but repeated outer-test feedback compromises confirmatory interpretation. Freezing a grid and using 20 new resampling seeds on the same subjects does not create an independent replication. The included full grids are taken directly from the professor's specification rather than selected to improve a previously observed outer result.

### Runtime claims were too certain

The transcript attributes large slowdowns to thread oversubscription, then retracts that diagnosis; it also extrapolates runtime from fixed iteration counts. Per-iteration work, line searches, inner-prox effort, convergence difficulty, hardware, and parallel overhead can change. This rebuild provides a one-fit benchmark and a one-outer-fold execution option. Measure actual runtime before committing to the full run; no completion-time guarantee is made.

## What the earlier numerical table means

The export reports these development means: RFE-SVM AUC 0.805, Elastic Net 0.787, V3 0.701, LASSO 0.669, with approximately 404, 530, 142, and 25 selected edges respectively. These means are independently recomputed from the old saved fold scores, whose model fits still require convergence checks. They do not establish superiority or inferiority after the convergence and inference issues above. A proposed scattered-signal explanation for V3's performance is a hypothesis, not an identified biological mechanism.

## Mathematical review

Equation 5 is convex: logistic loss is convex; the edge and ridge penalties are convex; and the gates can be written as two linear inequalities per edge plus z >= 0. Eliminating z yields the stated fractional-cover norm. The triangle example gives cost 1.5 for three unit edges; that is the intended relaxation, not a coding error. Integer-cover or cycle-tightening extensions are not included in the current estimator.

The intercept-only gradient and Equation 9 define lambda_max separately on each training subset after training-only scaling. Bisection is valid because the left-hand side is nonincreasing and the right-hand side increases. The ridge derivative vanishes at zero, so lambda_max does not depend on ridge strength. The implementation uses mean logistic loss consistently; changing to summed loss without rescaling penalties would change the fitted model.

The nonnegative capacities z can be nonunique and are not binary affected-region indicators. ROI interpretation must include the actual incident nonzero edges. Strong claims of a new penalty also require care: the proposal itself gives its equivalence to latent star-group LASSO with within-group infinity norm.

## Remaining evidence needed

2. The professor's original preprocessing/RFE-SVM implementation to establish exact reproduction.
3. The old szv3_pipeline.zip if a line-by-line code audit is desired; the result archive has been supplied and numerically audited.
4. Demographic/motion/site information and any true subject-pair structure if confound checks or restricted resampling are required. The README calls the sample age-balanced but reports different group mean ages; this deserves examination, not an assumption of perfect matching.

## Primary technical references used

- Supplied SZ_Idea.pdf, Sections II-D, III, VI and VII, with mathematical appendices checked selectively.
- scikit-learn: nested versus non-nested cross-validation, https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- scikit-learn: pipelines and preprocessing leakage, https://scikit-learn.org/stable/modules/compose.html
- CVXPY solver capabilities and tolerances, https://www.cvxpy.org/tutorial/solvers/index.html
- scikit-learn probability calibration, https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
- Nogueira, Sechidis and Brown, On the Stability of Feature Selection Algorithms, JMLR 18(174), 2018, https://jmlr.org/papers/v18/17-514.html
