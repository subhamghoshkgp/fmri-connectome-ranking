# Fitted-model storage

Fitted-model trees are archived by repeat to reduce the number of GitHub upload operations. Each ZIP contains original repository-relative paths and unchanged source bytes. No training is repeated. Use `scripts/extract_model_archives.py` to restore them.

The available-artifact limitations in `provenance/ARTIFACT_COVERAGE.json` remain unchanged: 1,200 whole-brain 83 model pairs, 400 baseline re-selection pairs at 129 regions, and 260 ranking pairs at 129 regions (65 per new method). Cortical exports contain final analyses and predictions but omit raw model NPZ files.

`provenance/SOURCE_FILE_SHA256SUMS.csv` records the unarchived source inventory. `provenance/FILE_SHA256SUMS.csv` records the actual published files.
