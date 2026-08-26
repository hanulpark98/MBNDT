# MBNDT: Multi-Branch Neural Decision Tree

Implementation and experiment artifacts for MBNDT from the paper **Adaptive
Multi-Branching for Shallow Decision Tree Induction** (IEEE ICDM 2026).

MBNDT is an axis-aligned decision tree learning algorithm trained end-to-end with
differentiable multi-way splits, designed for shallow decision tree induction. Each node learns ordered thresholds and an
optional branch mask. Inference follows one deterministic root-to-leaf path.

<img width="4273" height="1530" alt="figure1_v7" src="https://github.com/user-attachments/assets/41c93fd5-9483-4033-b3a7-64aec1cff7da" />

## Repository status

This is a basic camera-ready release version. The core model, paper HPO scripts,
21-dataset manifest, compact aggregate results, and a smoke test are present. We are planned to update the final release as soon as possible.

## Installation

Create a fresh Python 3.10+ environment, then install the project:

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e '.[dev]'
pytest
```

PyTorch installation can be platform-specific. If CUDA is required, install
the appropriate PyTorch build for the host before installing this project.

## Minimal example

```python
import torch
from mbndt import MBNDTClassifier

torch.manual_seed(0)
X = torch.randn(160, 8)
y = (X[:, 0] - 0.5 * X[:, 1] > 0).long()

classifier = MBNDTClassifier(
    depth=3,
    branching_factor=3,
    epochs=20,
    patience=5,
    random_state=0,
).fit(
    X[:128],
    y[:128],
    X_val=X[128:],
    y_val=y[128:],
)

probabilities = classifier.predict_proba(X[128:])
predictions = classifier.predict(X[128:])
```

The lightweight constructor uses a single fit (`n_restarts=0`). To enable the
paper's restart-selection procedure directly, set `n_restarts` and its stage
controls:

```python
classifier = MBNDTClassifier(
    n_restarts=5,
    restart_epochs=40,
    restart_patience=8,
    stage1_select_metric="val_loss",
    stage2_es_metric="val_bacc",
)
```

For a compute-intensive no-HPO configuration summarized from the 105 selected
hyperparameter sets in the paper's 21-dataset benchmark:

```python
classifier = MBNDTClassifier.paper_recommended()
```

This preset is a cross-dataset heuristic, not a substitute for the paper's
per-split HPO.

Run the complete example with:

```bash
python examples/train_and_predict.py
```

The classifier accepts numeric, already-preprocessed arrays or tensors. See
[the interface guide](docs/INTERFACE.md) for MBNDT-PP, save/load, multiclass,
and full-configuration examples.

## Paper-equivalent path

`MBNDTClassifier` is a thin facade: it delegates model construction and
training to the unchanged research implementation. Deterministic tests require
its learned parameters, logits, and metrics to exactly match the low-level path
for the same seed, configuration, batches, and data.

The paper experiments additionally use random restarts, leaf-budget
regularization, longer early-stopped training, nested evaluation, and HPO.
Their authoritative entry point remains `experiments/run_mbndt_hpo.py`; the
facade does not replace that reproduction runner.

## Reproducing experiments

The dataset list is in `configs/paper_datasets.txt`. Datasets are fetched by
OpenML ID; downloaded data are not committed.

A small functional run can be launched by lowering trials and timeout:

```bash
python experiments/run_mbndt_hpo.py \
  --datasets openml:55 \
  --n_hpo_trials 1 \
  --hpo_timeout_sec 120 \
  --artifact_base artifacts/smoke
```

For paper-scale experiments, use the defaults and the full dataset list. The
paper used five outer splits, five inner folds, and a one-hour HPO budget per
outer split. Full reproduction requires substantial compute.

Baseline instructions and third-party requirements are documented in
`experiments/README.md`.

## Results and figures

Compact aggregate results are under `results/tables/`. Regenerate the
21-dataset leaf-budget figure with:

```bash
python scripts/make_leaf_budget_plot.py
```

The repository excludes downloaded datasets, checkpoints, raw trial histories,
notebook copies, and archived exploratory experiments.

## Project layout

```text
src/mbndt/       Core model, training configuration, and preprocessing
examples/        Small end-to-end training and inference example
docs/            Clean-interface and paper-parity guidance
experiments/     MBNDT and baseline HPO entry points
configs/         Paper dataset manifest
results/         Compact aggregate tables and derived figures
scripts/         Reproducible analysis/plot scripts
tests/           Fast smoke tests
paper/           Citation and paper-posting notes
```

## Citation

Citation metadata will be provided after publication.

## License

The MBNDT code in this repository is released under the [MIT License](LICENSE).
Third-party projects and datasets remain subject to their own licenses and
terms.
