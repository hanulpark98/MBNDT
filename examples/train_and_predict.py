"""Train MBNDT on a small synthetic binary-classification dataset."""

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


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Replace these tensors with preprocessed, numeric features and binary labels.
    features = torch.randn(160, 8)
    labels = (features[:, 0] - 0.5 * features[:, 1] > 0).float()

    train_loader = DataLoader(
        TensorDataset(features[:128], labels[:128]),
        batch_size=32,
        shuffle=True,
    )
    validation_features = features[128:]
    validation_labels = labels[128:]
    validation_loader = DataLoader(
        TensorDataset(validation_features, validation_labels),
        batch_size=32,
    )

    config = MBNDTConfig(
        model=ModelHP(n_features=features.shape[1], D=3, B=3, use_masks=True),
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
        mbndt_module,
        config,
        train_loader,
        validation_loader,
        device,
    )

    model.eval()
    with torch.no_grad():
        logits = model(validation_features.to(device))
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).long().cpu()

    accuracy = (predictions == validation_labels.long()).float().mean()
    print(f"epochs run: {len(history.get('train_loss', []))}")
    print(f"validation accuracy: {accuracy.item():.3f}")
    print(f"first five probabilities: {probabilities[:5].cpu().tolist()}")


if __name__ == "__main__":
    main()
