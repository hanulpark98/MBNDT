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
