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

`MBNDT.forward` returns logits. The example below trains a small binary model
through the same configuration-based training path used by the experiment
runner, then converts its logits to probabilities and class predictions.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from mbndt import (
    LeafBudgetHP,
    LoadBalanceHP,
    MBNDTConfig,
    ModelHP,
    OptimHP,
    RegularizersHP,
    TrainingHP,
    train_from_cfg,
)
from mbndt import model as mbndt_module

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace these tensors with preprocessed, numeric features and binary labels.
x = torch.randn(160, 8)
y = (x[:, 0] - 0.5 * x[:, 1] > 0).float()
train_loader = DataLoader(
    TensorDataset(x[:128], y[:128]), batch_size=32, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(x[128:], y[128:]), batch_size=32
)

config = MBNDTConfig(
    model=ModelHP(n_features=8, D=3, B=3, use_masks=True),
    optim=OptimHP(),
    training=TrainingHP(
        random_restart=False,
        stage2_epoch=20,
        stage2_patience=5,
        stage2_es_metric="val_loss",
        verbose=False,
    ),
    regularizers=RegularizersHP(
        load_balance=LoadBalanceHP(on=False),
        leaf_budget=LeafBudgetHP(on=False),
    ),
)

model, history = train_from_cfg(
    mbndt_module, config, train_loader, val_loader, device
)

model.eval()
with torch.no_grad():
    logits = model(x[128:].to(device))
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).long()
```

Run the complete example with:

```bash
python examples/train_and_predict.py
```

The paper experiments additionally enable random restarts and leaf-budget
regularization, and use longer early-stopped training through these same
configuration objects. Their complete settings are in
`experiments/run_mbndt_hpo.py`.

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
