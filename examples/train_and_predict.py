"""Train and evaluate MBNDT through the user-facing classifier interface."""

import torch

from mbndt import MBNDTClassifier


def main() -> None:
    torch.manual_seed(0)
    features = torch.randn(160, 8)
    labels = (features[:, 0] - 0.5 * features[:, 1] > 0).long()

    train_features = features[:128]
    train_labels = labels[:128]
    validation_features = features[128:]
    validation_labels = labels[128:]

    classifier = MBNDTClassifier(
        depth=3,
        branching_factor=3,
        use_masks=True,
        epochs=20,
        patience=5,
        batch_size=32,
        random_state=0,
    ).fit(
        train_features,
        train_labels,
        X_val=validation_features,
        y_val=validation_labels,
    )

    probabilities = classifier.predict_proba(validation_features)
    predictions = classifier.predict(validation_features)
    accuracy = classifier.score(validation_features, validation_labels)

    print(f"epochs run: {len(classifier.history_.get('train_loss', []))}")
    print(f"validation accuracy: {accuracy:.3f}")
    print(f"first five probabilities: {probabilities[:5].tolist()}")
    print(f"first five predictions: {predictions[:5].tolist()}")

    # MBNDT-PP uses the combined train + early-stopping validation set in the
    # paper. The raw model remains available through classifier.model_.
    classifier.prune(torch.cat((train_features, validation_features)))
    post_pruned_accuracy = classifier.score(
        validation_features,
        validation_labels,
        post_pruned=True,
    )
    print(f"post-pruned validation accuracy: {post_pruned_accuracy:.3f}")


if __name__ == "__main__":
    main()
