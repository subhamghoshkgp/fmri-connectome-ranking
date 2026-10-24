# Reproducing the experiments

The archived experiment folders preserve the original model code and checkpoint-sensitive filenames. Do not merge or rename implementation files inside those folders before resuming old checkpoints.

## Environments and entry points

Use the notebook's exact Python version and per-experiment `requirements.lock.txt`; cortical and later ranking runs record Python 3.13.15. Different archived stages may have different recorded environments, so do not replace all lockfiles with one guessed environment. Numerical-library threads should be capped at one per outer worker. The original runners support CPU workers; no GPU implementation is supplied.

For multiple workers, use imported entry points to avoid the earlier joblib `expit` serialization issue:

```sh
# Run from experiments/whole_brain_83 after installing its lockfile.
python -u -c "import run_dataset1_83 as r; r.main()" --help

# Run from experiments/cortical.
python -u -c "import run_cortical as r; r.main()" --help

# Run from experiments/whole_brain_129.
python -u -c "import run_rank_v3 as r; r.main()" --help
```

The original notebooks supply full arguments, Drive mounting, frozen input checks, environment setup, tests, progress, graceful pause/resume and export. Read their cells before changing paths. A single interrupted uncommitted fit may need recomputation; completed atomic checkpoints are reused.

## Inputs and protocol

The exact original dataset source was provided at https://drive.google.com/file/d/1q7o9qA9M5il2SwFdiMLpBi-7RNr2q0y8/view . Its SHA256 is `8e6944e1a3d31abce6123eea14c31a6b38cc92f01df90c8d95aefdcdacff2b6f`. The manuscript also cites the public Lausanne dataset DOI 10.5281/zenodo.3758534.

The bundled 83-region MAT fixture and cortical NPZ files are compact derived inputs. Cortical NPZ data are already Fisher-transformed; do not apply arctanh again. Scalers are learned separately in every training subset. Subject order is controls 01-27 followed by patients 01-27. Full input audits and original ROI/edge mappings are archived.

Use the original saved subject-level splits; do not make new splits for comparing anatomical scopes. Select valid candidates by highest mean inner-CV AUC, breaking exact ties by smaller inner support and then candidate index. One-SE rows are secondary controls. Cortical models are freshly retuned on cortical features rather than reusing whole-brain selected candidates.

The shared original ranking project is https://drive.google.com/drive/folders/162wqRcyLQa_StXgf4ckz9yfkUOSLPui3 . It is the source for the historical 129-region code, reference files, final analyses and partial model archive. The folder has not been modified by this packaging task.

## Tests versus scientific runs

Each experiment retains its original tests and validation records. Run them in that stage's pinned environment. Packaging verification checks artifact coverage and syntax, not numerical solver equivalence in a substituted environment. Full 100-fold training is computationally expensive and is not rerun by repository verification.

## Compile the manuscript

From `paper/`, run pdfLaTeX twice on `isbi2027_sparse_to_dense.tex`, or upload the contents of that folder into Overleaf. The manuscript is four pages including figures, table, declarations and references. `generate_assets.py` and `generate_results_table.py` recreate the table/figures from frozen supporting results; they do not train models.
