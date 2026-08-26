import torch
from torch.utils.data import DataLoader, TensorDataset

from mbndt import (
    LeafBudgetHP,
    LoadBalanceHP,
    MBNDT,
    MBNDTConfig,
    ModelHP,
    RegularizersHP,
    TrainingHP,
    train_from_cfg,
)
from mbndt import model as mbndt_module


def test_binary_forward_is_finite_and_single_output_per_row():
    torch.manual_seed(0)
    model = MBNDT(n_features=4, D=2, B=3, use_masks=True)
    inputs = torch.randn(8, 4)

    logits, auxiliary = model(inputs, return_aux=True)

    assert logits.shape == (8,)
    assert torch.isfinite(logits).all()
    assert auxiliary["g_soft"].shape == (8, model.num_internal_nodes, 3)
    torch.testing.assert_close(
        auxiliary["g_soft"].sum(dim=-1),
        torch.ones(8, model.num_internal_nodes),
    )


def test_configuration_path_trains_and_predicts():
    torch.manual_seed(0)
    device = torch.device("cpu")
    features = torch.randn(48, 4)
    labels = (features[:, 0] > 0).float()
    train_loader = DataLoader(
        TensorDataset(features[:32], labels[:32]),
        batch_size=16,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(features[32:], labels[32:]),
        batch_size=16,
    )
    config = MBNDTConfig(
        model=ModelHP(n_features=4, D=2, B=3, use_masks=True),
        training=TrainingHP(
            random_restart=False,
            stage2_epoch=2,
            stage2_patience=2,
            stage2_es_metric="val_loss",
            verbose=False,
        ),
        regularizers=RegularizersHP(
            load_balance=LoadBalanceHP(on=False),
            leaf_budget=LeafBudgetHP(on=False),
        ),
    )

    model, history = train_from_cfg(
        mbndt_module,
        config,
        train_loader,
        validation_loader,
        device,
    )

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(features[32:]))
        predictions = (probabilities >= 0.5).long()

    assert len(history["train_loss"]) == 2
    assert probabilities.shape == (16,)
    assert torch.isfinite(probabilities).all()
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert predictions.shape == (16,)
