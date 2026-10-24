# ISBI 2027: whole-brain and cortical-only revision

Title: Sparse-to-Dense Connectome Ranking: Incidence-Gated Models and Dense Ridge Prediction

## Compile in Overleaf

Upload this ZIP as a project, or extract it and upload the files into your existing project. Choose `isbi2027_sparse_to_dense.tex` as the main document and **pdfLaTeX** as the compiler. Compile twice.

Compilation requires the main TEX, `spconf.sty`, `results_table.tex`, and the two PDFs in `figures/`. The inline bibliography needs no BibTeX run. Python and shell escape are not required to compile.

The existing title, author block (Subham Ghosh, Electrical Engineering, IIT Kharagpur), theory, proposition, lemma, bibliography, both figures and all whole-brain numerical entries are preserved. The updates are limited to the cortical comparison, its protocol/audit description, related abstract/conclusion wording and necessary layout changes.

## Strict four-page layout

The manuscript is **exactly four pages total**, including both figures, the merged results table, declarations and all references. The supported nine-point text size, official style, margins and column geometry are unchanged. All eight professor-requested changes are retained; see `evidence/EIGHT_REQUIREMENTS_CHECKLIST.json`.

The two former results tables are merged into Table 1, with AUC/SD, mean edges, BA, F1 and adjusted stability for all eight primary methods in all four representations. Full sensitivity, specificity, Jaccard and one-SE results remain in the archived supporting CSVs. Repeated prose and captions have been condensed while preserving both formal results and their proofs.

## Results included

- Whole brain: 83 regions / 3,403 edges and 129 regions / 8,256 edges.
- Cortical only: 68 regions / 2,278 edges and 114 regions / 6,441 edges.
- The same 54 participants, original subject order, exact saved 20 x 5 outer splits and five inner folds are used throughout.
- Atlas-defined restriction removes 14 subcortical regions and the brainstem from both matrix axes. Every retained edge is cortical-to-cortical.
- Cortical models were retuned with unchanged grids, solvers, model definitions and preprocessing rules. This manuscript update did not rerun training.
- All 12 variants have 100 outer records per scope and resolution: 4,800 records overall. The manuscript presents eight maximum-inner-AUC pipelines; all one-SE controls remain in supporting CSVs.

Table 1 combines whole-brain and cortical AUC/SD, selected edges, BA, F1 and adjusted stability. Matching-fold AUC differences are discussed in the text. Full cortical BA, F1, sensitivity, specificity, Jaccard and secondary variants are in `evidence/all_cortical_results.csv` and the exported source summaries under `evidence/cortical_83/` and `evidence/cortical_129/`.

## Verification and interpretation

The revision independently recalculated 14,400 classification metrics from 25,920 exported held-out prediction rows. All agreed with the fold CSVs to floating-point precision. It also checked all 2,400 cortical fold records/certificates, exact split identity, configuration identity, full-brain reference consistency, support-frequency totals and chance-adjusted Nogueira stability. See `evidence/CORTICAL_RESULTS_VERIFICATION.json`.

The uploaded analysis export omits the original outer-model NPZ files and inner-fit caches. Their fitted coefficients/scalers, individual support masks, Jaccard overlaps and inner candidate rankings were not independently reconstructed in this revision. The notebook's prior array verification statement and this revision's export-level checks should not be conflated.

Whole-brain mean AUC exceeds cortical-only mean AUC in 15 of the 16 primary comparisons; the exception is LASSO at the 129-region resolution. DRR leads AUC, BA and F1 across all four representations. IGR-R leads sparse cortical AUC at 68 regions; RFE-SVM leads at 114. These are descriptive results from one small reused cohort. They do not isolate subcortical disease biology or establish independent-cohort superiority.

## Regenerate assets

With Python, Matplotlib and NumPy installed:

```sh
python generate_assets.py
pdflatex -interaction=nonstopmode -halt-on-error isbi2027_sparse_to_dense.tex
pdflatex -interaction=nonstopmode -halt-on-error isbi2027_sparse_to_dense.tex
```

The generator recreates the merged table (through `generate_results_table.py`) and the two original figures. Data and verification summaries are included; no model training is performed.

## Author confirmations

The existing marked fields for the institutional secondary-use ethics determination and funding/conflicts remain for author completion. Confirm authorship and the AI assistance disclosure before submission. The source-study ethics approvals are retained without inventing a secondary-use exemption.
