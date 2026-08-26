#!/usr/bin/env python3
"""
CART HPO command-line runner.

Examples
--------
# UCI repo datasets 27 and 468, HPO objective AUROC, max_depth searched from 1 to 4
python cart_hpo_cli.py \
  --source ucirepo \
  --dataset_ids 27 468 \
  --hpo_metric auroc \
  --max_depth 4 \
  --root_dir ./HPO_reports/cart/ucirepo

# OpenML datasets, using each dataset's default target
python cart_hpo_cli.py \
  --source openml \
  --dataset_ids 31 1461 \
  --hpo_metric bal_acc \
  --max_depth 6 \
  --root_dir ./HPO_reports/cart/openml

# Mixed sources in one command
python cart_hpo_cli.py \
  --datasets ucirepo:27 ucirepo:468 openml:31 \
  --hpo_metric f1_macro \
  --max_depth 4 \
  --root_dir ./HPO_reports/cart/mixed

# OpenML with explicit target for one dataset
python cart_hpo_cli.py \
  --datasets openml:1590:target \
  --hpo_metric auprc \
  --max_depth 5
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import optuna
import pandas as pd
from ucimlrepo import fetch_ucirepo

try:
    import openml
except ImportError:  # only required when --source openml is used
    openml = None

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

try:
    import category_encoders as ce
except ImportError as e:
    raise ImportError(
        "This script requires category_encoders. Install it with: pip install category-encoders"
    ) from e


# ------------------------------
# Defaults
# ------------------------------
DEFAULT_K_OUTER_SPLITS = 5
DEFAULT_TEST_RATIO = 0.2
DEFAULT_INNER_REPEATS = 5
DEFAULT_USE_STRATIFY = True
DEFAULT_SEED_BASE = 0
DEFAULT_HPO_TIMEOUT_SEC = 60 * 60
DEFAULT_MAX_HPO_TRIALS = 100000
DEFAULT_CAT_CARD_THRESHOLD = 10
DEFAULT_STD_PENALTY_ALPHA = 0.5

MAXIMIZE_METRICS = {
    "accuracy",
    "bal_acc",
    "mcc",
    "auroc",
    "auprc",
    "f1_macro",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
}
MINIMIZE_METRICS = {"brier", "log_loss"}
SUPPORTED_HPO_METRICS = sorted(MAXIMIZE_METRICS | MINIMIZE_METRICS)


@dataclass(frozen=True)
class DatasetSpec:
    source: str
    dataset_id: int
    target: Optional[str] = None

    @property
    def short_name(self) -> str:
        return f"{self.source}_{self.dataset_id}"


# ------------------------------
# Reproducibility
# ------------------------------
def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ------------------------------
# Metrics
# ------------------------------
def _safe_metric(fn, default=np.nan):
    try:
        out = fn()
        if out is None:
            return default
        return float(out)
    except Exception:
        return default


def eval_binary_metrics(y_true, y_pred, y_prob) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob, dtype=float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    # Clip only for loss metrics that require valid probabilities.
    y_prob_clip = np.clip(y_prob, 1e-15, 1 - 1e-15)

    return {
        "accuracy": _safe_metric(lambda: accuracy_score(y_true, y_pred)),
        "bal_acc": _safe_metric(lambda: balanced_accuracy_score(y_true, y_pred)),
        "mcc": _safe_metric(lambda: matthews_corrcoef(y_true, y_pred)),
        "auroc": _safe_metric(lambda: roc_auc_score(y_true, y_prob)),
        "auprc": _safe_metric(lambda: average_precision_score(y_true, y_prob)),
        "brier": _safe_metric(lambda: brier_score_loss(y_true, y_prob_clip)),
        "log_loss": _safe_metric(lambda: log_loss(y_true, np.vstack([1 - y_prob_clip, y_prob_clip]).T, labels=[0, 1])),
        "f1_macro": _safe_metric(lambda: f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": _safe_metric(lambda: precision_score(y_true, y_pred, zero_division=0)),
        "recall": _safe_metric(lambda: recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def metric_direction(metric_name: str) -> str:
    if metric_name in MAXIMIZE_METRICS:
        return "maximize"
    if metric_name in MINIMIZE_METRICS:
        return "minimize"
    raise ValueError(f"Unsupported hpo_metric={metric_name}. Choose one of {SUPPORTED_HPO_METRICS}")


def robust_objective(scores: list[float], metric_name: str, std_penalty_alpha: float) -> float:
    vals = np.asarray(scores, dtype=float)
    vals = vals[np.isfinite(vals)]

    direction = metric_direction(metric_name)
    if len(vals) == 0:
        return -1e9 if direction == "maximize" else 1e9

    mu = float(np.mean(vals))
    sigma = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    # For a score: maximize mean - alpha*std.
    # For a loss:  minimize mean + alpha*std.
    if direction == "maximize":
        return mu - std_penalty_alpha * sigma
    return mu + std_penalty_alpha * sigma


# ------------------------------
# CART search space
# ------------------------------
def sample_cart_params(trial: optuna.Trial, max_depth: int) -> dict:
    return {
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
        # If --max_depth 4, this searches max_depth = 1, 2, 3, 4.
        "max_depth": trial.suggest_int("max_depth", 1, int(max_depth)),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.001, 0.20, log=True),
        "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.001, 0.20, log=True),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-8, 1e-2, log=True),
    }


# ------------------------------
# Complexity extraction for CART
# ------------------------------
def _quantile_higher(x, q):
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    try:
        return float(np.quantile(x, q, method="higher"))
    except TypeError:
        return float(np.quantile(x, q, interpolation="higher"))


def _decision_steps_stats(clf, X, q=0.90):
    """
    Steps per sample = (#nodes visited in decision_path) - 1.
    This excludes the final leaf node.
    """
    path = clf.decision_path(X)
    n_nodes_on_path = np.diff(path.indptr)
    steps = n_nodes_on_path - 1

    return {
        "pathlen_decisions_mean": float(np.mean(steps)) if len(steps) else 0.0,
        "pathlen_decisions_median": float(np.median(steps)) if len(steps) else 0.0,
        "pathlen_decisions_p90": _quantile_higher(steps, q),
        "pathlen_decisions_max": float(np.max(steps)) if len(steps) else 0.0,
    }


def _leaf_usage_stats(clf, X, topk=10):
    leaf_ids = clf.apply(X)
    unique, counts = np.unique(leaf_ids, return_counts=True)

    n = int(counts.sum()) if len(counts) else 0
    out = {
        "hit_leaves": int(len(unique)),
        "n_samples": n,
    }

    if n > 0 and len(counts) > 0:
        p = counts / counts.sum()
        ent = -np.sum(p * np.log(p + 1e-12))
        out["leaf_support_entropy"] = float(ent)
        out["leaf_support_eff_leaves"] = float(np.exp(ent))
        k = min(topk, len(counts))
        out[f"leaf_support_top{topk}_coverage"] = float(np.sort(counts)[-k:].sum() / n)

    return out


def extract_cart_complexity_e1(clf, X_train=None, X_test=None):
    cx = {
        "tree_max_depth": int(clf.get_depth()),
        "n_leaves_struct": int(clf.get_n_leaves()),
        "n_nodes_struct": int(clf.tree_.node_count),
    }
    cx["n_internal_nodes_struct"] = int(cx["n_nodes_struct"] - cx["n_leaves_struct"])

    feats = getattr(clf.tree_, "feature", None)
    if feats is not None:
        cx["n_features_used"] = int(len(set(feats[feats >= 0].tolist())))

    if X_train is not None:
        cx.update({f"train_{k}": v for k, v in _decision_steps_stats(clf, X_train).items()})
        cx.update({f"train_{k}": v for k, v in _leaf_usage_stats(clf, X_train).items()})

    if X_test is not None:
        cx.update({f"test_{k}": v for k, v in _decision_steps_stats(clf, X_test).items()})
        cx.update({f"test_{k}": v for k, v in _leaf_usage_stats(clf, X_test).items()})

    return cx


# ------------------------------
# Preprocessor: numeric median + low-cardinality OHE + high-cardinality LOO
# ------------------------------
def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _to_df(cols):
    return FunctionTransformer(lambda X: pd.DataFrame(X, columns=cols), validate=False)


def make_preprocessor_ohe_loo(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    feature_names,
    cat_indicator,
    card_threshold: int,
):
    cat_cols = [c for c, is_cat in zip(feature_names, cat_indicator) if is_cat]
    num_cols = [c for c, is_cat in zip(feature_names, cat_indicator) if not is_cat]

    low_card_cols, high_card_cols = [], []
    for c in cat_cols:
        card = X_fit[c].nunique(dropna=True)
        (low_card_cols if card < card_threshold else high_card_cols).append(c)

    transformers = []

    if num_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols))

    if low_card_cols:
        low_cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", _make_ohe()),
            ]
        )
        transformers.append(("cat_ohe", low_cat_pipe, low_card_cols))

    if high_card_cols:
        high_cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("to_df", _to_df(high_card_cols)),
                (
                    "loo",
                    ce.LeaveOneOutEncoder(
                        cols=high_card_cols,
                        sigma=0.0,
                        handle_unknown="value",
                        handle_missing="value",
                    ),
                ),
            ]
        )
        transformers.append(("cat_loo", high_cat_pipe, high_card_cols))

    return ColumnTransformer(transformers)


# ------------------------------
# Dataset loading
# ------------------------------
def build_cat_indicator_from_ucirepo(dataset, X: pd.DataFrame):
    feature_names = X.columns.tolist()

    cat_from_meta = None
    try:
        meta = dataset.variables[["name", "type"]].copy()
        meta["type_norm"] = meta["type"].astype(str).str.lower()
        categorical_type_labels = {"categorical", "nominal", "binary"}
        name_to_is_cat = dict(zip(meta["name"], meta["type_norm"].isin(categorical_type_labels)))
        cat_from_meta = [bool(name_to_is_cat.get(col, False)) for col in feature_names]
    except Exception:
        pass

    cat_from_dtype = [(str(X[col].dtype) == "object" or str(X[col].dtype).startswith("category")) for col in feature_names]

    if cat_from_meta is None:
        return cat_from_dtype

    return [a or b for a, b in zip(cat_from_meta, cat_from_dtype)]


def build_cat_indicator_from_openml(cat_indicator, X: pd.DataFrame):
    if cat_indicator is not None:
        return [bool(v) for v in cat_indicator]
    return [(str(X[col].dtype) == "object" or str(X[col].dtype).startswith("category")) for col in X.columns]


def load_dataset(spec: DatasetSpec):
    source = spec.source.lower()

    if source in {"uci", "ucirepo"}:
        dataset = fetch_ucirepo(id=int(spec.dataset_id))
        X = dataset.data.features.copy()
        y = dataset.data.targets.iloc[:, 0].copy()
        cat_indicator = build_cat_indicator_from_ucirepo(dataset, X)
        dataset_name = getattr(dataset.metadata, "name", None) if hasattr(dataset, "metadata") else None
        dataset_name = dataset_name or f"ucirepo_{spec.dataset_id}"

    elif source == "openml":
        if openml is None:
            raise ImportError("openml is not installed. Install it with: pip install openml")

        ds = openml.datasets.get_dataset(int(spec.dataset_id))
        target = spec.target or ds.default_target_attribute
        if target is None:
            raise ValueError(
                f"[OpenML {spec.dataset_id}] No default target found. Use --datasets openml:{spec.dataset_id}:TARGET_NAME"
            )

        X, y, cat_ind, _ = ds.get_data(dataset_format="dataframe", target=target)
        X = X.copy()
        y = y.copy()
        cat_indicator = build_cat_indicator_from_openml(cat_ind, X)
        dataset_name = ds.name

    else:
        raise ValueError(f"Unsupported source={spec.source}. Use 'ucirepo' or 'openml'.")

    # Drop rows with missing target.
    valid_y = pd.notna(y)
    if not np.all(valid_y):
        X = X.loc[valid_y].reset_index(drop=True)
        y = pd.Series(y).loc[valid_y].reset_index(drop=True)

    # Make sure column names are strings and unique enough for ColumnTransformer.
    X = pd.DataFrame(X).reset_index(drop=True)
    X.columns = [str(c) for c in X.columns]

    return X, y, cat_indicator, dataset_name


def encode_binary_target(y, spec: DatasetSpec):
    le = LabelEncoder()
    y_enc = le.fit_transform(np.asarray(y))

    if len(le.classes_) != 2:
        raise ValueError(
            f"[{spec.source} {spec.dataset_id}] Binary-only runner; got classes={list(le.classes_)}"
        )

    return y_enc, [str(c) for c in le.classes_]


# ------------------------------
# Run CART HPO for one dataset
# ------------------------------
def run_cart_hpo_one(
    spec: DatasetSpec,
    *,
    out_dir: Path,
    k_outer_splits: int,
    test_ratio: float,
    inner_repeats: int,
    use_stratify: bool,
    seed_base: int,
    hpo_timeout_sec: int,
    max_hpo_trials: int,
    cat_card_threshold: int,
    hpo_metric: str,
    max_depth: int,
    std_penalty_alpha: float,
    show_progress_bar: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, cat_indicator, dataset_name = load_dataset(spec)
    feature_names = X.columns.tolist()
    y_enc, classes = encode_binary_target(y, spec)

    if hpo_metric not in SUPPORTED_HPO_METRICS:
        raise ValueError(f"Unsupported --hpo_metric={hpo_metric}; choose from {SUPPORTED_HPO_METRICS}")

    direction = metric_direction(hpo_metric)

    config = {
        "dataset": asdict(spec),
        "dataset_name": dataset_name,
        "classes_after_label_encoding": classes,
        "k_outer_splits": k_outer_splits,
        "test_ratio": test_ratio,
        "inner_repeats": inner_repeats,
        "use_stratify": use_stratify,
        "seed_base": seed_base,
        "hpo_timeout_sec": hpo_timeout_sec,
        "max_hpo_trials": max_hpo_trials,
        "cat_card_threshold": cat_card_threshold,
        "hpo_metric": hpo_metric,
        "hpo_direction": direction,
        "max_depth_arg": max_depth,
        "searched_max_depth_values": list(range(1, int(max_depth) + 1)),
        "std_penalty_alpha": std_penalty_alpha,
        "n_samples": int(len(y_enc)),
        "n_features_raw": int(X.shape[1]),
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    outer_splitter = (
        StratifiedShuffleSplit(n_splits=k_outer_splits, test_size=test_ratio, random_state=seed_base)
        if use_stratify
        else ShuffleSplit(n_splits=k_outer_splits, test_size=test_ratio, random_state=seed_base)
    )
    outer_splits = list(outer_splitter.split(np.zeros((len(y_enc), 1)), y_enc))

    all_results = []

    for split_idx, (tv_idx, te_idx) in enumerate(outer_splits):
        outer_seed = seed_base + split_idx
        set_all_seeds(outer_seed)

        X_tv = X.iloc[tv_idx].reset_index(drop=True)
        y_tv = y_enc[tv_idx]
        X_test = X.iloc[te_idx].reset_index(drop=True)
        y_test = y_enc[te_idx]

        inner_seed = outer_seed + 111
        y_tv_arr = np.asarray(y_tv).astype(int)
        n_tv = len(y_tv_arr)

        inner_splitter = (
            StratifiedKFold(n_splits=inner_repeats, shuffle=True, random_state=inner_seed)
            if use_stratify
            else KFold(n_splits=inner_repeats, shuffle=True, random_state=inner_seed)
        )
        inner_splits = list(inner_splitter.split(np.zeros((n_tv, 1)), y_tv_arr))

        # Cache transformed splits once per outer split.
        cache_t0 = time.perf_counter()
        cached = []
        for fold_id, (tr_idx, va_idx) in enumerate(inner_splits):
            X_tr, y_tr = X_tv.iloc[tr_idx], y_tv_arr[tr_idx]
            X_va, y_va = X_tv.iloc[va_idx], y_tv_arr[va_idx]

            pre = make_preprocessor_ohe_loo(
                X_fit=X_tr,
                y_fit=y_tr,
                feature_names=feature_names,
                cat_indicator=cat_indicator,
                card_threshold=cat_card_threshold,
            )
            Xtr = pre.fit_transform(X_tr, y_tr)
            Xva = pre.transform(X_va)
            cached.append((Xtr, y_tr, Xva, y_va))

        cache_preprocess_time_sec = float(time.perf_counter() - cache_t0)

        def objective(trial: optuna.Trial):
            params = sample_cart_params(trial, max_depth=max_depth)
            scores = []

            for rep_id, (Xtr, ytr, Xva, yva) in enumerate(cached):
                clf = DecisionTreeClassifier(
                    random_state=outer_seed * 1_000_000 + trial.number + rep_id,
                    **params,
                )
                clf.fit(Xtr, ytr)
                pred_va = clf.predict(Xva)
                prob_va = clf.predict_proba(Xva)[:, 1]

                metrics = eval_binary_metrics(yva, pred_va, prob_va)
                scores.append(float(metrics[hpo_metric]))

            value = robust_objective(scores, hpo_metric, std_penalty_alpha)

            vals = np.asarray(scores, dtype=float)
            vals = vals[np.isfinite(vals)]
            trial.set_user_attr("inner_metric", hpo_metric)
            trial.set_user_attr("inner_mean", float(np.mean(vals)) if len(vals) else None)
            trial.set_user_attr("inner_std", float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            trial.set_user_attr("robust_value", float(value))

            return value

        hpo_t0 = time.perf_counter()
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=outer_seed),
        )
        study.optimize(
            objective,
            timeout=hpo_timeout_sec,
            n_trials=max_hpo_trials,
            n_jobs=1,
            show_progress_bar=show_progress_bar,
        )

        hpo_time_sec = float(time.perf_counter() - hpo_t0)
        best_params = dict(study.best_trial.params)

        final_seed = outer_seed * 10_000_000 + 999
        set_all_seeds(final_seed)

        pp_t0 = time.perf_counter()
        pre = make_preprocessor_ohe_loo(
            X_fit=X_tv,
            y_fit=y_tv,
            feature_names=feature_names,
            cat_indicator=cat_indicator,
            card_threshold=cat_card_threshold,
        )
        Xtv_tr = pre.fit_transform(X_tv, y_tv)
        XTe = pre.transform(X_test)
        final_preprocess_time_sec = float(time.perf_counter() - pp_t0)

        clf = DecisionTreeClassifier(random_state=final_seed, **best_params)

        fit_t0 = time.perf_counter()
        clf.fit(Xtv_tr, y_tv)
        final_fit_time_sec = float(time.perf_counter() - fit_t0)

        inf_t0 = time.perf_counter()
        y_pred = clf.predict(XTe)
        y_prob = clf.predict_proba(XTe)[:, 1]
        final_infer_time_sec = float(time.perf_counter() - inf_t0)

        train_pred = clf.predict(Xtv_tr)
        train_prob = clf.predict_proba(Xtv_tr)[:, 1]
        train_metrics = eval_binary_metrics(y_tv, train_pred, train_prob)
        test_metrics = eval_binary_metrics(y_test, y_pred, y_prob)

        complexity = extract_cart_complexity_e1(clf, X_train=Xtv_tr, X_test=XTe)
        complexity["n_features_after_pp"] = int(Xtv_tr.shape[1])

        report = {
            "dataset": asdict(spec),
            "dataset_name": dataset_name,
            "split": int(split_idx),
            "outer_seed": int(outer_seed),
            "hpo_metric": hpo_metric,
            "hpo_direction": direction,
            "best_value_robust": float(study.best_value),
            "best_inner_mean": study.best_trial.user_attrs.get("inner_mean"),
            "best_inner_std": study.best_trial.user_attrs.get("inner_std"),
            "best_params": best_params,
            "time_sec": {
                "cache_preprocess_time": cache_preprocess_time_sec,
                "hpo_time": hpo_time_sec,
                "final_preprocess_time": final_preprocess_time_sec,
                "final_fit_time": final_fit_time_sec,
                "final_infer_time": final_infer_time_sec,
            },
            "complexity": complexity,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        all_results.append(report)

        print(
            f"\n[{spec.source.upper()} {spec.dataset_id} | Outer {split_idx}] "
            f"best_robust_{hpo_metric}={report['best_value_robust']:.4f} "
            f"test_{hpo_metric}={test_metrics.get(hpo_metric, float('nan')):.4f} "
            f"test_auc={test_metrics.get('auroc', float('nan')):.4f} "
            f"test_bal_acc={test_metrics.get('bal_acc', float('nan')):.4f} "
            f"p90steps={complexity.get('test_pathlen_decisions_p90', None)} "
            f"leaves={complexity.get('n_leaves_struct', None)} "
            f"fit={final_fit_time_sec:.2f}s"
        )

    with open(out_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    rows = []
    for r in all_results:
        row = {
            "source": r["dataset"]["source"],
            "dataset_id": r["dataset"]["dataset_id"],
            "dataset_name": r["dataset_name"],
            "split": r["split"],
            "outer_seed": r["outer_seed"],
            "hpo_metric": r["hpo_metric"],
            "best_value_robust": r["best_value_robust"],
            "best_inner_mean": r["best_inner_mean"],
            "best_inner_std": r["best_inner_std"],
            **{f"hp/{k}": v for k, v in r["best_params"].items()},
            **{f"time/{k}": v for k, v in r["time_sec"].items()},
            **{f"cx/{k}": v for k, v in r["complexity"].items()},
            **{f"train/{k}": v for k, v in r["train_metrics"].items()},
            **{f"test/{k}": v for k, v in r["test_metrics"].items()},
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "outer_reports_per_split.csv", index=False)

    def mean_std(series: pd.Series):
        vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        return {
            "mean": float(np.mean(vals)) if len(vals) else 0.0,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
        }

    summary = {}
    for prefix in ("test/", "train/", "cx/", "time/"):
        cols = [c for c in df.columns if c.startswith(prefix)]
        for c in cols:
            summary[c] = mean_std(df[c])

    with open(out_dir / "outer_reports_mean_std.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return df, summary


def run_cart_hpo_many(specs: Iterable[DatasetSpec], *, root_dir: str, **kwargs):
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for spec in specs:
        out_dir = root / spec.short_name
        try:
            _, summary = run_cart_hpo_one(spec, out_dir=out_dir, **kwargs)
            all_summaries.append({"dataset": asdict(spec), "summary": summary})
        except Exception as e:
            print(f"\n[{spec.source.upper()} {spec.dataset_id}] FAILED: {repr(e)}")
            all_summaries.append({"dataset": asdict(spec), "error": repr(e)})

    with open(root / "summaries_all_datasets.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nSaved: {root / 'summaries_all_datasets.json'}")
    return all_summaries


# ------------------------------
# CLI
# ------------------------------
def parse_dataset_spec(text: str) -> DatasetSpec:
    """
    Accepts:
      ucirepo:27
      uci:27
      openml:31
      openml:1590:target_name
    """
    parts = text.split(":")
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError(
            f"Bad dataset spec '{text}'. Use SOURCE:ID or openml:ID:TARGET."
        )

    source = parts[0].lower()
    if source == "uci":
        source = "ucirepo"
    if source not in {"ucirepo", "openml"}:
        raise argparse.ArgumentTypeError(
            f"Bad source '{source}'. Use ucirepo or openml."
        )

    try:
        dataset_id = int(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Bad dataset id in '{text}'.") from e

    target = parts[2] if len(parts) == 3 else None
    if source != "openml" and target is not None:
        raise argparse.ArgumentTypeError("Explicit target is only supported for OpenML specs.")

    return DatasetSpec(source=source, dataset_id=dataset_id, target=target)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run CART HPO on UCIrepo/OpenML binary classification datasets."
    )

    dataset_group = parser.add_argument_group("dataset selection")
    dataset_group.add_argument(
        "--source",
        choices=["ucirepo", "uci", "openml"],
        default="ucirepo",
        help="Dataset source used with --dataset_ids. Ignored for --datasets specs.",
    )
    dataset_group.add_argument(
        "--dataset_ids",
        nargs="+",
        type=int,
        default=None,
        help="Dataset IDs from --source, e.g. --source ucirepo --dataset_ids 27 468.",
    )
    dataset_group.add_argument(
        "--datasets",
        nargs="+",
        type=parse_dataset_spec,
        default=None,
        help="Mixed dataset specs, e.g. --datasets ucirepo:27 openml:31 openml:1590:target.",
    )

    hpo_group = parser.add_argument_group("HPO settings")
    hpo_group.add_argument("--hpo_metric", choices=SUPPORTED_HPO_METRICS, default="auroc")
    hpo_group.add_argument(
        "--max_depth",
        type=int,
        default=4,
        help="Upper bound for CART max_depth search. If 4, HPO searches 1,2,3,4.",
    )
    hpo_group.add_argument("--max_hpo_trials", type=int, default=DEFAULT_MAX_HPO_TRIALS)
    hpo_group.add_argument("--hpo_timeout_sec", type=int, default=DEFAULT_HPO_TIMEOUT_SEC)
    hpo_group.add_argument("--std_penalty_alpha", type=float, default=DEFAULT_STD_PENALTY_ALPHA)
    hpo_group.add_argument("--no_progress_bar", action="store_true")

    cv_group = parser.add_argument_group("split/CV settings")
    cv_group.add_argument("--k_outer_splits", type=int, default=DEFAULT_K_OUTER_SPLITS)
    cv_group.add_argument("--test_ratio", type=float, default=DEFAULT_TEST_RATIO)
    cv_group.add_argument(
        "--inner_repeats",
        type=int,
        default=DEFAULT_INNER_REPEATS,
        help="Number of inner K-fold splits.",
    )
    cv_group.add_argument("--seed_base", type=int, default=DEFAULT_SEED_BASE)
    cv_group.add_argument("--no_stratify", action="store_true")

    misc_group = parser.add_argument_group("misc")
    misc_group.add_argument("--cat_card_threshold", type=int, default=DEFAULT_CAT_CARD_THRESHOLD)
    misc_group.add_argument("--root_dir", type=str, default="./cart_hpo_runs")
    misc_group.add_argument("--optuna_verbosity", choices=["debug", "info", "warning", "error"], default="warning")

    return parser


def specs_from_args(args) -> list[DatasetSpec]:
    specs = []

    if args.datasets:
        specs.extend(args.datasets)

    if args.dataset_ids:
        source = "ucirepo" if args.source == "uci" else args.source
        specs.extend(DatasetSpec(source=source, dataset_id=int(did)) for did in args.dataset_ids)

    if not specs:
        raise ValueError(
            "No datasets provided. Use either --dataset_ids with --source, or --datasets SOURCE:ID."
        )

    return specs


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    verbosity = {
        "debug": optuna.logging.DEBUG,
        "info": optuna.logging.INFO,
        "warning": optuna.logging.WARNING,
        "error": optuna.logging.ERROR,
    }[args.optuna_verbosity]
    optuna.logging.set_verbosity(verbosity)

    if args.max_depth < 1:
        raise ValueError("--max_depth must be >= 1")
    if args.inner_repeats < 2:
        raise ValueError("--inner_repeats must be >= 2")
    if args.k_outer_splits < 1:
        raise ValueError("--k_outer_splits must be >= 1")

    specs = specs_from_args(args)

    print("Running CART HPO with:")
    print(f"  datasets        = {[asdict(s) for s in specs]}")
    print(f"  hpo_metric      = {args.hpo_metric} ({metric_direction(args.hpo_metric)})")
    print(f"  max_depth grid  = {list(range(1, args.max_depth + 1))}")
    print(f"  root_dir        = {args.root_dir}")

    run_cart_hpo_many(
        specs,
        root_dir=args.root_dir,
        k_outer_splits=args.k_outer_splits,
        test_ratio=args.test_ratio,
        inner_repeats=args.inner_repeats,
        use_stratify=not args.no_stratify,
        seed_base=args.seed_base,
        hpo_timeout_sec=args.hpo_timeout_sec,
        max_hpo_trials=args.max_hpo_trials,
        cat_card_threshold=args.cat_card_threshold,
        hpo_metric=args.hpo_metric,
        max_depth=args.max_depth,
        std_penalty_alpha=args.std_penalty_alpha,
        show_progress_bar=not args.no_progress_bar,
    )


if __name__ == "__main__":
    main()
