# MBNDT classifier interface

`MBNDTClassifier` provides a compact interface while preserving the paper
implementation as the computational backend. It accepts numeric,
already-preprocessed feature matrices. Use the split-aware utilities in
`mbndt.preprocessing` before fitting when a dataset contains missing values or
categorical columns.

## Fit and predict

```python
from mbndt import MBNDTClassifier

classifier = MBNDTClassifier(
    depth=3,
    branching_factor=3,
    use_masks=True,
    epochs=100,
    patience=20,
    n_restarts=0,
    leaf_budget=8,
    random_state=0,
)
classifier.fit(X_train, y_train, X_val=X_valid, y_val=y_valid)

logits = classifier.decision_function(X_test)
probabilities = classifier.predict_proba(X_test)
predictions = classifier.predict(X_test)
accuracy = classifier.score(X_test, y_test)
```

For binary classification, `predict_proba` returns two columns ordered by
`classifier.classes_`. For multiclass classification it applies softmax and
returns one column per class. `predict` maps indices back to the labels
originally supplied to `fit`.

## Performance controls and defaults

The convenience constructor is intentionally lightweight. Its defaults use one
direct fit (`n_restarts=0`), a shared learning-rate fallback of `0.01`, 100
maximum epochs, and no leaf-budget penalty. Random restarts are therefore
available but are not silently enabled.

The main performance controls are exposed directly:

```python
classifier = MBNDTClassifier(
    lr_feature=0.01,
    lr_thresh=0.01,
    lr_leaf=0.02,
    lr_mask=0.01,
    n_restarts=5,
    restart_epochs=40,
    restart_patience=8,
    stage1_es_metric="val_loss",
    stage1_select_metric="val_loss",
    stage2_es_metric="val_bacc",
    epochs=500,
    patience=25,
    loss_type="balanced_bce",
    leaf_budget=36,
    batch_size="auto",
)
```

When a component learning rate is omitted, `learning_rate` supplies its value.
Setting `n_restarts=0` uses the direct training path; a positive value enables
the paper's two-stage restart-selection path with that many candidates.
`batch_size="auto"` reproduces the paper's deterministic rule based on the
training-set size.

## Paper-derived no-HPO preset

```python
classifier = MBNDTClassifier.paper_recommended(
    random_state=0,
    device="cuda",
)
```

This preset was summarized from the 105 selected hyperparameter sets (21
datasets × 5 outer splits) used in the main benchmark:

| Parameter | Preset value | Summary rule |
|---|---:|---|
| Branching factor, depth, budget | `B=3, D=4, K=36` | Most frequent valid joint tuple |
| `tau_cdf` | `0.30716` | Geometric mean |
| Feature learning rate | `0.010145` | Geometric mean |
| Threshold learning rate | `0.009401` | Geometric mean |
| Leaf learning rate | `0.020192` | Geometric mean |
| Mask learning rate | `0.011648` | Geometric mean |
| Loss | `balanced_bce` | Selected in all 105 runs |
| Restarts | `5` | Final paper training policy |
| Restart stage | `40` epochs, patience `8` | Final paper policy |
| Continuation stage | `500` epochs, patience `25` | Final paper policy |
| Batch size | `"auto"` | Paper's dataset-size rule |

The continuous HPO parameters were sampled logarithmically, so geometric means
are more appropriate than arithmetic means. Architecture and leaf budget were
tuned jointly; averaging them independently could create an unrepresentative
or invalid combination.

The preset is deliberately opt-in because it is substantially more expensive
than the lightweight defaults. It is a reasonable fixed starting point when
HPO is unavailable, but it does not reproduce the paper's dataset- and
split-specific HPO selections.

## MBNDT-PP

The paper constructs MBNDT-PP from the combined training and early-stopping
validation features:

```python
import numpy as np

X_trainval = np.concatenate((X_train, X_valid), axis=0)
classifier.prune(X_trainval, min_branch_hit=1)
pp_predictions = classifier.predict(X_test, post_pruned=True)
```

The raw and post-pruned PyTorch modules remain available as `model_` and
`posthoc_model_`.

## Save and load

```python
classifier.save("mbndt.pt")

restored = MBNDTClassifier.load("mbndt.pt", device="cpu")
predictions = restored.predict(X_test)
```

The serialized checkpoint contains the raw MBNDT model, its complete
configuration, feature count, label ordering, and interface metadata.
Post-pruning can be reconstructed by calling `prune` with the same reference
features.

## Full configuration

Use `from_config` when every research setting must be controlled explicitly:

```python
from mbndt import MBNDTClassifier, MBNDTConfig, ModelHP, TrainingHP

config = MBNDTConfig(
    model=ModelHP(n_features=X_train.shape[1], D=3, B=3),
    training=TrainingHP(
        random_restart=True,
        n_restarts=5,
        base_seed=0,
    ),
)
classifier = MBNDTClassifier.from_config(
    config,
    batch_size=256,
    device="cuda",
)
classifier.fit(X_train, y_train, X_val=X_valid, y_val=y_valid)
```

For exact paper reproduction, use `experiments/run_mbndt_hpo.py`. It controls
the nested splits, preprocessing, conditional architecture search,
random-restart policy, HPO budget, reporting, and MBNDT-PP construction.

## Parity guarantee

The test suite trains both interfaces with identical configurations, seeds,
batches, and data, then requires exact equality of:

- every learned state-dictionary tensor;
- raw logits;
- training and validation metrics;
- post-pruned logits.

The facade does not duplicate or replace the tree, optimizer, regularizers,
early stopping, random restart, or post-pruning algorithms.
