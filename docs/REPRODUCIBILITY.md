# Reproducing the experiments

## 1. Create an isolated environment

Use Python 3.13.15 for the later experiment runners. Activate a virtual environment and install the selected experiment's own lockfile. Keep earlier baseline/re-selection environments separate when their recorded dependencies differ.

```bash
python -m pip install -r experiments/whole_brain_83/requirements.lock.txt
```

Use one numerical-library thread per worker. Runners set these limits where implemented. Begin with `--jobs 1`; use `--jobs 2` only when CPU and RAM allow it. Imported entry points below avoid the earlier multiprocessing serialization issue.

## 2. Prepare the external inputs

Use the original Dataset-1 files and reference assets from your existing experiment downloads. Large data and model outputs stay outside Git.

Original dataset: [27_SCHZ_CTRL_dataset.mat](https://drive.google.com/file/d/1q7o9qA9M5il2SwFdiMLpBi-7RNr2q0y8/view).

Existing ranking project: [data/code and reference results](https://drive.google.com/drive/folders/162wqRcyLQa_StXgf4ckz9yfkUOSLPui3).

Required assets depend on the runner:

| Experiment | Inputs |
|---|---|
| Whole brain 83 | Original MAT (or verified compact 83-region MAT), atlas XLS and cortical-label MAT (saved protocol JSON is included) |
| Cortical only | Original MAT (or verified cortical NPZ for the matching scale), atlas XLS and cortical-label MAT (saved protocol JSON is included) |
| Whole brain 129 ranking | Original MAT, atlas/label assets, completed max-inner-AUC baseline re-selection directory or ZIP |
| Baseline re-selection | Original completed baseline results, including tuning summaries and saved splits |

For the 83-region and cortical runners, place the atlas assets in that experiment's `reference/` folder with these exact names; the saved protocol JSON is already included:

```text
ParcellationLausanne2008.xls
labels_index_CORTICAL_Laus2008_all_scales.mat
dataset1_129_protocol.json
```

The original MAT SHA-256 is `8e6944e1a3d31abce6123eea14c31a6b38cc92f01df90c8d95aefdcdacff2b6f`. Preserve source files and checksums: loaders deliberately reject unexpected inputs. Do not apply Fisher transformation twice to cached cortical features.

## 3. Run one outer fold

Replace `/path/to/...` with your local paths. Commands below assume the corresponding environment is active.

### Whole brain, 83 regions

From `experiments/whole_brain_83`:

```bash
python -u -c "import run_dataset1_83 as r; r.main()" --data /path/to/27_SCHZ_CTRL_dataset.mat --out /path/to/new_whole_brain_83 --config configs/full.json --jobs 1 --max-outer 1
```

### Cortical-only, both resolutions

Install `experiments/cortical/requirements.lock.txt` in its environment. From `experiments/cortical`:

```bash
python -u -c "import run_cortical as r; r.main()" --resolution 83 --data /path/to/27_SCHZ_CTRL_dataset.mat --out /path/to/new_cortical_68 --config configs/full.json --jobs 1 --max-outer 1
python -u -c "import run_cortical as r; r.main()" --resolution 129 --data /path/to/27_SCHZ_CTRL_dataset.mat --out /path/to/new_cortical_114 --config configs/full.json --jobs 1 --max-outer 1
```

`--resolution` names the original atlas; 83 and 129 become 68 and 114 cortical regions respectively.

### Whole brain, 129-region ranking

Install `experiments/whole_brain_129/requirements.lock.txt`. From that folder:

```bash
python -u -c "import run_rank_v3 as r; r.main()" --reference /path/to/completed_max_inner_auc_results --data /path/to/27_SCHZ_CTRL_dataset.mat --out /path/to/new_rank_129 --config configs/full.json --jobs 1 --max-outer 1
```

Keep the reference results read-only and use a separate output directory. Supply `--atlas` and `--labels` if their locations differ from the runner's defaults. This runner trains the four ranking/expanded arms and compares them with saved baseline rows.

For baseline re-selection, follow `experiments/reselection_129/README.md` and its `requirements-reselection.lock.txt`. Re-selection uses saved tuning summaries rather than rerunning inner folds.

## 4. Run all folds and resume

Once the first fold passes, rerun the same command with `--max-outer 100`. Preserve the same output path, input files, configuration and numerical code to resume. Completed checkpoints are reused; an interrupted fit that was not committed may be recomputed. Changing these inputs requires a new output directory.

Use the runner's `--help` for inspection and analysis options. Run experiment tests in the appropriate pinned environment; tests involving source data require those external inputs.

## 5. Verify aggregate results

From the repository root:

```bash
python scripts/verify_results.py
```

This standard-library check validates fold counts, unique evaluation keys, and agreement between the detailed fold tables and primary summaries. It does not train models or re-audit saved participant predictions.

All reported comparisons preserve the saved splits, training-only scaling, inner-AUC tuning and original model definitions. The anatomical representations reuse the same participants.
