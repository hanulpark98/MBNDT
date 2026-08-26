# MBNDT: Multi-Branch Neural Decision Tree

Implementation and experiment artifacts for MBNDT from the paper **Adaptive
Multi-Branching for Shallow Decision Tree Induction** (ICDM 2026).

MBNDT is a shallow, axis-aligned decision tree trained end-to-end with
differentiable multi-way splits. Each node learns ordered thresholds and an
optional branch mask. Inference follows one deterministic root-to-leaf path.

<img width="4273" height="1530" alt="figure1_v7" src="https://github.com/user-attachments/assets/41c93fd5-9483-4033-b3a7-64aec1cff7da" />

## Repository status

This is a camera-ready release candidate. The core model, paper HPO scripts,
21-dataset manifest, compact aggregate results, and a smoke test are present.
Before public release, the authors should record the exact upstream revisions
used for GradTree and SPLIT.

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
from mbndt import MBNDT

model = MBNDT(n_features=8, D=3, B=3, use_masks=True)
x = torch.randn(32, 8)
logits = model(x)
probabilities = torch.sigmoid(logits)
```

The low-level class exposes the exact research architecture. For the complete
training policy, including random restarts, early stopping, and leaf-budget
regularization, use the configuration objects and paper experiment runner.

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
experiments/     MBNDT and baseline HPO entry points
configs/         Paper dataset manifest
results/         Compact aggregate tables and derived figures
scripts/         Reproducible analysis/plot scripts
tests/           Fast smoke tests
paper/           Citation and paper-posting notes
```

## Citation

Citation metadata are provided in `CITATION.cff`. Add the DOI and final IEEE
Xplore bibliographic fields after publication.

## License

The MBNDT code in this repository is released under the [MIT License](LICENSE).
Third-party projects and datasets remain subject to their own licenses and
terms.
