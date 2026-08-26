from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mbndt import (
    LeafBudgetHP,
    LoadBalanceHP,
    MBNDTClassifier,
    MBNDTConfig,
    ModelHP,
    OptimHP,
    RegularizersHP,
    TrainingHP,
    train_from_cfg,
)
from mbndt import model as paper_model


def _binary_data():
    generator = torch.Generator().manual_seed(41)
    features = torch.randn(64, 4, generator=generator)
    targets = (features[:, 0] - 0.4 * features[:, 1] > 0).float()
    return features, targets


def _parity_config():
    return MBNDTConfig(
        model=ModelHP(
            n_features=4,
            D=2,
            B=3,
            num_classes=1,
            use_masks=True,
            tau_cdf=0.6,
        ),
        optim=OptimHP(
            lr_feature=0.01,
            lr_thresh=0.01,
            lr_leaf=0.02,
            lr_mask=0.01,
        ),
        training=TrainingHP(
            random_restart=False,
            base_seed=17,
            stage2_epoch=2,
            stage2_patience=2,
            stage2_es_metric="val_loss",
            loss_type="bce",
            verbose=False,
        ),
        regularizers=RegularizersHP(
            load_balance=LoadBalanceHP(on=False),
            leaf_budget=LeafBudgetHP(on=False),
        ),
    )


def test_facade_is_exactly_equivalent_to_paper_training_path(tmp_path):
    features, targets = _binary_data()
    config = _parity_config()
    train_x, train_y = features[:48], targets[:48]
    val_x, val_y = features[48:], targets[48:]

    classifier = MBNDTClassifier.from_config(
        config,
        batch_size=16,
        random_state=17,
        device="cpu",
    ).fit(train_x, train_y, X_val=val_x, y_val=val_y)

    paper_model.set_global_seed(17)
    train_generator = torch.Generator().manual_seed(17)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=16,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        pin_memory=False,
    )
    validation_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    reference_model, reference_history = train_from_cfg(
        paper_model,
        deepcopy(config),
        train_loader,
        validation_loader,
        torch.device("cpu"),
    )

    for name, expected in reference_model.state_dict().items():
        torch.testing.assert_close(
            classifier.model_.state_dict()[name],
            expected,
            rtol=0,
            atol=0,
        )

    with torch.no_grad():
        reference_logits = reference_model(val_x).numpy()
    np.testing.assert_array_equal(
        classifier.decision_function(val_x),
        reference_logits,
    )
    for key in ("train_loss", "val_loss", "val_auc", "val_acc", "val_bacc"):
        np.testing.assert_array_equal(
            classifier.history_[key],
            reference_history[key],
        )

    probabilities = classifier.predict_proba(val_x)
    assert probabilities.shape == (16, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    checkpoint = tmp_path / "classifier.pt"
    classifier.save(checkpoint)
    restored = MBNDTClassifier.load(checkpoint, device="cpu")
    np.testing.assert_array_equal(
        classifier.decision_function(val_x),
        restored.decision_function(val_x),
    )
    np.testing.assert_array_equal(
        classifier.predict(val_x),
        restored.predict(val_x),
    )


def test_post_pruned_facade_matches_paper_predictor():
    features, targets = _binary_data()
    classifier = MBNDTClassifier.from_config(
        _parity_config(),
        batch_size=16,
        random_state=17,
        device="cpu",
    ).fit(
        features[:48],
        targets[:48],
        X_val=features[48:56],
        y_val=targets[48:56],
    )
    reference_x = features[:56]
    classifier.prune(reference_x)

    reference_loader = DataLoader(
        TensorDataset(reference_x, torch.zeros(len(reference_x))),
        batch_size=16,
        shuffle=False,
    )
    reference_posthoc, _ = paper_model.build_posthoc_merged_mbndt(
        classifier.model_,
        reference_loader,
        device="cpu",
        min_branch_hit=1,
        freeze_base=True,
    )
    with torch.no_grad():
        expected = reference_posthoc(features[56:]).numpy()
    np.testing.assert_array_equal(
        classifier.decision_function(features[56:], post_pruned=True),
        expected,
    )


def test_random_restart_and_leaf_budget_match_paper_path():
    features, targets = _binary_data()
    train_x, train_y = features[:48], targets[:48]
    val_x, val_y = features[48:], targets[48:]
    config = _parity_config()
    config.training.random_restart = True
    config.training.n_restarts = 2
    config.training.base_seed = 23
    config.training.stage1_epoch = 1
    config.training.stage1_patience = 1
    config.training.stage2_epoch = 1
    config.training.stage2_patience = 1
    config.training.stage1_es_metric = "val_loss"
    config.training.stage1_select_metric = "val_loss"
    config.training.stage2_es_metric = "val_loss"
    config.regularizers.leaf_budget = LeafBudgetHP(
        on=True,
        K=4,
        mode="mass_log",
    )

    classifier = MBNDTClassifier.from_config(
        config,
        batch_size=16,
        random_state=23,
        device="cpu",
    ).fit(train_x, train_y, X_val=val_x, y_val=val_y)

    paper_model.set_global_seed(23)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=16,
        shuffle=True,
        generator=torch.Generator().manual_seed(23),
        num_workers=0,
        pin_memory=False,
    )
    validation_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    reference_model, _ = train_from_cfg(
        paper_model,
        deepcopy(config),
        train_loader,
        validation_loader,
        torch.device("cpu"),
    )

    for name, expected in reference_model.state_dict().items():
        torch.testing.assert_close(
            classifier.model_.state_dict()[name],
            expected,
            rtol=0,
            atol=0,
        )
    with torch.no_grad():
        expected_logits = reference_model(val_x).numpy()
    np.testing.assert_array_equal(
        classifier.decision_function(val_x),
        expected_logits,
    )


def test_multiclass_labels_and_probability_shape():
    generator = torch.Generator().manual_seed(8)
    features = torch.randn(60, 4, generator=generator)
    encoded = torch.argmax(features[:, :3], dim=1).numpy()
    labels = np.asarray(["alpha", "beta", "gamma"])[encoded]

    classifier = MBNDTClassifier(
        depth=2,
        branching_factor=3,
        epochs=1,
        patience=1,
        batch_size=15,
        random_state=9,
        device="cpu",
    ).fit(
        features[:45],
        labels[:45],
        X_val=features[45:],
        y_val=labels[45:],
    )

    probabilities = classifier.predict_proba(features[45:])
    predictions = classifier.predict(features[45:])
    assert probabilities.shape == (15, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert set(predictions).issubset(set(classifier.classes_))


def test_interface_rejects_unfitted_and_invalid_inputs():
    classifier = MBNDTClassifier(epochs=1, patience=1, device="cpu")
    with pytest.raises(RuntimeError, match="not fitted"):
        classifier.predict(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="provided together"):
        classifier.fit(
            np.zeros((4, 3), dtype=np.float32),
            [0, 1, 0, 1],
            X_val=np.zeros((2, 3), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="at least two classes"):
        classifier.fit(
            np.zeros((4, 3), dtype=np.float32),
            [1, 1, 1, 1],
        )
