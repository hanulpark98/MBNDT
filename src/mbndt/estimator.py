"""User-facing classifier facade for the unchanged paper implementation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import model as _paper
from .config import (
    LeafBudgetHP,
    LoadBalanceHP,
    MBNDTConfig,
    ModelHP,
    OptimHP,
    RegularizersHP,
    TrainingHP,
    build_model_from_cfg,
    train_from_cfg,
)


class MBNDTClassifier:
    """Convenience interface over the unchanged MBNDT research code.

    The constructor covers the common single-fit case. Use from_config when
    reproducing a complete paper configuration, including random restarts and
    separate optimizer learning rates.

    X must already be numeric and preprocessed. The paper's split-aware
    preprocessing functions remain available from mbndt.preprocessing.
    """

    _SERIALIZATION_VERSION = 1

    def __init__(
        self,
        *,
        depth: int = 3,
        branching_factor: int = 3,
        use_masks: bool = True,
        tau_cdf: float = 0.6,
        learning_rate: float = 1e-2,
        epochs: int = 100,
        patience: int = 20,
        loss_type: str = "bce",
        leaf_budget: float | None = None,
        load_balance_weight: float | None = None,
        batch_size: int = 256,
        random_state: int = 0,
        device: str | torch.device | None = None,
    ) -> None:
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        if branching_factor < 2:
            raise ValueError(
                f"branching_factor must be at least 2, got {branching_factor}"
            )
        if epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {epochs}")
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if leaf_budget is not None and leaf_budget <= 0:
            raise ValueError(f"leaf_budget must be positive, got {leaf_budget}")

        self.depth = int(depth)
        self.branching_factor = int(branching_factor)
        self.use_masks = bool(use_masks)
        self.tau_cdf = float(tau_cdf)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.loss_type = str(loss_type)
        self.leaf_budget = None if leaf_budget is None else float(leaf_budget)
        self.load_balance_weight = (
            None if load_balance_weight is None else float(load_balance_weight)
        )
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.device = device
        self._config_template: MBNDTConfig | None = None

    @classmethod
    def from_config(
        cls,
        config: MBNDTConfig,
        *,
        batch_size: int = 256,
        random_state: int | None = None,
        device: str | torch.device | None = None,
    ) -> MBNDTClassifier:
        """Create a classifier from a full paper-style configuration."""
        if not isinstance(config, MBNDTConfig):
            raise TypeError("config must be an MBNDTConfig instance")
        seed = (
            int(config.training.base_seed)
            if random_state is None
            else int(random_state)
        )
        estimator = cls(
            depth=config.model.D,
            branching_factor=config.model.B,
            use_masks=config.model.use_masks,
            tau_cdf=config.model.tau_cdf,
            learning_rate=config.optim.lr_feature,
            epochs=config.training.stage2_epoch,
            patience=config.training.stage2_patience,
            loss_type=config.training.loss_type,
            leaf_budget=(
                config.regularizers.leaf_budget.K
                if config.regularizers.leaf_budget.on
                else None
            ),
            load_balance_weight=(
                config.regularizers.load_balance.weight
                if config.regularizers.load_balance.on
                else None
            ),
            batch_size=batch_size,
            random_state=seed,
            device=device,
        )
        estimator._config_template = deepcopy(config)
        return estimator

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        X_val: Any | None = None,
        y_val: Any | None = None,
    ) -> MBNDTClassifier:
        """Fit an MBNDT model and return self."""
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be provided together")

        features = self._as_features(X, name="X")
        labels, classes = self._encode_training_targets(y, len(features))
        validation_features = (
            None if X_val is None else self._as_features(X_val, name="X_val")
        )
        validation_labels = (
            None
            if y_val is None
            else self._encode_known_targets(y_val, classes, len(validation_features))
        )
        if (
            validation_features is not None
            and validation_features.shape[1] != features.shape[1]
        ):
            raise ValueError(
                "X_val has a different number of features: "
                f"{validation_features.shape[1]} != {features.shape[1]}"
            )

        self.device_ = self._resolve_device(self.device)
        _paper.set_global_seed(self.random_state)
        train_loader = self._make_loader(
            features, labels, shuffle=True, seed=self.random_state
        )
        validation_loader = (
            None
            if validation_features is None
            else self._make_loader(
                validation_features,
                validation_labels,
                shuffle=False,
                seed=self.random_state + 1,
            )
        )
        config = self._make_config(
            n_features=features.shape[1],
            num_classes=(1 if len(classes) == 2 else len(classes)),
        )
        model, history = train_from_cfg(
            _paper, config, train_loader, validation_loader, self.device_
        )

        self.model_ = model
        self.history_ = history
        self.config_ = config
        self.classes_ = classes
        self.n_features_in_ = int(features.shape[1])
        self.posthoc_model_ = None
        self.prune_spec_ = None
        return self

    def decision_function(
        self, X: Any, *, post_pruned: bool = False
    ) -> np.ndarray:
        """Return raw binary or multiclass logits."""
        model = self._prediction_model(post_pruned)
        features = self._checked_prediction_features(X)
        loader = DataLoader(
            TensorDataset(features),
            batch_size=min(self.batch_size, max(1, len(features))),
            shuffle=False,
        )
        outputs: list[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                outputs.append(model(batch.to(self.device_).float()).detach().cpu())
        return torch.cat(outputs, dim=0).numpy()

    def predict_proba(
        self, X: Any, *, post_pruned: bool = False
    ) -> np.ndarray:
        """Return probabilities in standard (n_samples, n_classes) form."""
        logits = torch.from_numpy(self.decision_function(X, post_pruned=post_pruned))
        if len(self.classes_) == 2:
            positive = torch.sigmoid(logits)
            probabilities = torch.stack((1.0 - positive, positive), dim=1)
        else:
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.numpy()

    def predict(self, X: Any, *, post_pruned: bool = False) -> np.ndarray:
        """Return predictions mapped back to the labels supplied to fit."""
        indices = self.predict_proba(X, post_pruned=post_pruned).argmax(axis=1)
        return self.classes_[indices]

    def score(self, X: Any, y: Any, *, post_pruned: bool = False) -> float:
        """Return mean classification accuracy."""
        expected = np.asarray(y).reshape(-1)
        predicted = self.predict(X, post_pruned=post_pruned)
        if len(expected) != len(predicted):
            raise ValueError(
                f"y contains {len(expected)} samples; expected {len(predicted)}"
            )
        return float(np.mean(predicted == expected))

    def prune(
        self, X_reference: Any, *, min_branch_hit: int = 1
    ) -> MBNDTClassifier:
        """Build the paper's post-hoc MBNDT-PP predictor.

        For paper-equivalent usage, X_reference must be the combined
        preprocessed training and early-stopping validation data.
        """
        self._require_fitted()
        if min_branch_hit < 1:
            raise ValueError("min_branch_hit must be at least 1")
        features = self._checked_prediction_features(X_reference)
        dummy_targets = torch.zeros(len(features), dtype=torch.float32)
        loader = self._make_loader(
            features,
            dummy_targets,
            shuffle=False,
            seed=self.random_state + 2,
        )
        posthoc_model, prune_spec = _paper.build_posthoc_merged_mbndt(
            model=self.model_,
            train_loader=loader,
            device=str(self.device_),
            min_branch_hit=int(min_branch_hit),
            freeze_base=True,
        )
        self.posthoc_model_ = posthoc_model
        self.prune_spec_ = prune_spec
        return self

    def save(self, path: str | Path) -> None:
        """Save a fitted raw MBNDT model and facade metadata."""
        self._require_fitted()
        payload = {
            "format_version": self._SERIALIZATION_VERSION,
            "config": asdict(self.config_),
            "state_dict": {
                key: value.detach().cpu()
                for key, value in self.model_.state_dict().items()
            },
            "classes": self.classes_.tolist(),
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "n_features_in": self.n_features_in_,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> MBNDTClassifier:
        """Load a model written by save."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        version = payload.get("format_version")
        if version != cls._SERIALIZATION_VERSION:
            raise ValueError(
                f"Unsupported MBNDTClassifier format version: {version}"
            )
        config = _config_from_dict(payload["config"])
        estimator = cls.from_config(
            config,
            batch_size=int(payload["batch_size"]),
            random_state=int(payload["random_state"]),
            device=device,
        )
        estimator.device_ = estimator._resolve_device(device)
        estimator.model_ = build_model_from_cfg(
            _paper, config, estimator.device_
        )
        estimator.model_.load_state_dict(payload["state_dict"])
        estimator.model_.eval()
        estimator.config_ = config
        estimator.classes_ = np.asarray(payload["classes"])
        estimator.n_features_in_ = int(payload["n_features_in"])
        estimator.history_ = {}
        estimator.posthoc_model_ = None
        estimator.prune_spec_ = None
        return estimator

    def _make_config(self, *, n_features: int, num_classes: int) -> MBNDTConfig:
        if self._config_template is not None:
            config = deepcopy(self._config_template)
            config.model.n_features = int(n_features)
            config.model.num_classes = int(num_classes)
            return config

        load_balance_on = self.load_balance_weight is not None
        leaf_budget_on = self.leaf_budget is not None
        return MBNDTConfig(
            model=ModelHP(
                n_features=int(n_features),
                D=self.depth,
                B=self.branching_factor,
                num_classes=int(num_classes),
                use_masks=self.use_masks,
                tau_cdf=self.tau_cdf,
            ),
            optim=OptimHP(
                lr_feature=self.learning_rate,
                lr_thresh=self.learning_rate,
                lr_leaf=self.learning_rate,
                lr_mask=self.learning_rate,
            ),
            training=TrainingHP(
                random_restart=False,
                base_seed=self.random_state,
                stage2_epoch=self.epochs,
                stage2_patience=self.patience,
                stage2_es_metric="val_loss",
                loss_type=self.loss_type,
                verbose=False,
            ),
            regularizers=RegularizersHP(
                load_balance=LoadBalanceHP(
                    on=load_balance_on,
                    weight=(
                        1e-3
                        if self.load_balance_weight is None
                        else self.load_balance_weight
                    ),
                ),
                leaf_budget=LeafBudgetHP(
                    on=leaf_budget_on,
                    K=(8.0 if self.leaf_budget is None else self.leaf_budget),
                ),
            ),
        )

    def _make_loader(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        shuffle: bool,
        seed: int,
    ) -> DataLoader:
        generator = torch.Generator().manual_seed(int(seed))
        return DataLoader(
            TensorDataset(features, labels),
            batch_size=min(self.batch_size, max(1, len(features))),
            shuffle=shuffle,
            generator=(generator if shuffle else None),
            num_workers=0,
            pin_memory=False,
        )

    @staticmethod
    def _as_features(X: Any, *, name: str) -> torch.Tensor:
        try:
            features = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{name} must be a numeric two-dimensional array. "
                "Use mbndt.preprocessing for categorical data."
            ) from exc
        if features.ndim != 2:
            raise ValueError(
                f"{name} must be two-dimensional, got shape {tuple(features.shape)}"
            )
        if len(features) == 0:
            raise ValueError(f"{name} must contain at least one sample")
        if not torch.isfinite(features).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return features

    @staticmethod
    def _encode_training_targets(
        y: Any, expected_samples: int
    ) -> tuple[torch.Tensor, np.ndarray]:
        targets = np.asarray(y).reshape(-1)
        if len(targets) != expected_samples:
            raise ValueError(
                f"y contains {len(targets)} samples; expected {expected_samples}"
            )
        classes, encoded = np.unique(targets, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("y must contain at least two classes")
        dtype = torch.float32 if len(classes) == 2 else torch.long
        return torch.as_tensor(encoded, dtype=dtype), classes

    @staticmethod
    def _encode_known_targets(
        y: Any, classes: np.ndarray, expected_samples: int
    ) -> torch.Tensor:
        targets = np.asarray(y).reshape(-1)
        if len(targets) != expected_samples:
            raise ValueError(
                f"y_val contains {len(targets)} samples; expected {expected_samples}"
            )
        class_to_index = {label: index for index, label in enumerate(classes)}
        unseen = [label for label in np.unique(targets) if label not in class_to_index]
        if unseen:
            raise ValueError(f"y_val contains unseen classes: {unseen}")
        encoded = np.asarray([class_to_index[label] for label in targets])
        dtype = torch.float32 if len(classes) == 2 else torch.long
        return torch.as_tensor(encoded, dtype=dtype)

    @staticmethod
    def _resolve_device(
        device: str | torch.device | None,
    ) -> torch.device:
        resolved = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return resolved

    def _checked_prediction_features(self, X: Any) -> torch.Tensor:
        self._require_fitted()
        features = self._as_features(X, name="X")
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features: "
                f"{features.shape[1]} != {self.n_features_in_}"
            )
        return features

    def _prediction_model(self, post_pruned: bool) -> torch.nn.Module:
        self._require_fitted()
        if not post_pruned:
            return self.model_
        if self.posthoc_model_ is None:
            raise RuntimeError("Call prune(X_reference) before post-pruned prediction")
        return self.posthoc_model_

    def _require_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("This MBNDTClassifier instance is not fitted")


def _config_from_dict(data: dict[str, Any]) -> MBNDTConfig:
    """Reconstruct nested configuration dataclasses from serialized metadata."""
    regularizers = data["regularizers"]
    return MBNDTConfig(
        model=ModelHP(**data["model"]),
        optim=OptimHP(**data["optim"]),
        training=TrainingHP(**data["training"]),
        regularizers=RegularizersHP(
            load_balance=LoadBalanceHP(**regularizers["load_balance"]),
            leaf_budget=LeafBudgetHP(**regularizers["leaf_budget"]),
        ),
    )
