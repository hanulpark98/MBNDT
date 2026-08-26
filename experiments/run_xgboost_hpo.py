import argparse
import json
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import optuna
import pandas as pd
from ucimlrepo import fetch_ucirepo

try:
    import openml
except ImportError:
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
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder

try:
    import category_encoders as ce
except ImportError as e:
    raise ImportError("Install category_encoders with: pip install category-encoders") from e

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError("Install xgboost with: pip install xgboost") from e


# ------------------------------
# Defaults aligned with cart_hpo_cli.py / split_hpo_cli.py
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

# XGBoost defaults/search bounds
DEFAULT_N_ESTIMATORS_MIN = 50
DEFAULT_N_ESTIMATORS_MAX = 500
DEFAULT_LEARNING_RATE_LOW = 1e-2
DEFAULT_LEARNING_RATE_HIGH = 3e-1
DEFAULT_MAX_DEPTH = 4
DEFAULT_N_JOBS = 4
DEFAULT_TREE_METHOD = "hist"

MAXIMIZE_METRICS = {
    "accuracy", "bal_acc", "mcc", "auroc", "auprc", "f1_macro",
    "precision", "recall", "sensitivity", "specificity", "ppv", "npv",
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


def seed32(x: int) -> int:
    return int(x % (2**32 - 1))


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

    if direction == "maximize":
        return mu - std_penalty_alpha * sigma
    return mu + std_penalty_alpha * sigma


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

    cat_from_dtype = [
        (str(X[col].dtype) == "object" or str(X[col].dtype).startswith("category"))
        for col in feature_names
    ]

    if cat_from_meta is None:
        return cat_from_dtype
    return [a or b for a, b in zip(cat_from_meta, cat_from_dtype)]


def build_cat_indicator_from_openml(cat_indicator, X: pd.DataFrame):
    if cat_indicator is not None:
        return [bool(v) for v in cat_indicator]
    return [
        (str(X[col].dtype) == "object" or str(X[col].dtype).startswith("category"))
        for col in X.columns
    ]


def load_dataset(spec: DatasetSpec):
    source = spec.source.lower()

    if source in {"uci", "ucirepo"}:
        dataset = fetch_ucirepo(id=int(spec.dataset_id))
        X = dataset.data.features.copy()
        y = dataset.data.targets.iloc[:, 0].copy()
        cat_indicator = build_cat_indicator_from_ucirepo(dataset, X)

        metadata = getattr(dataset, "metadata", None)
        if isinstance(metadata, dict):
            dataset_name = metadata.get("name")
        else:
            dataset_name = getattr(metadata, "name", None)
        dataset_name = dataset_name or f"ucirepo_{spec.dataset_id}"

    elif source == "openml":
        if openml is None:
            raise ImportError("openml is not installed. Install it with: pip install openml")

        ds = openml.datasets.get_dataset(int(spec.dataset_id))
        target = spec.target or ds.default_target_attribute
        if target is None:
            raise ValueError(f"[OpenML {spec.dataset_id}] No default target found. Use --datasets openml:{spec.dataset_id}:TARGET_NAME")

        X, y, cat_ind, _ = ds.get_data(dataset_format="dataframe", target=target)
        X = X.copy()
        y = y.copy()
        cat_indicator = build_cat_indicator_from_openml(cat_ind, X)
        dataset_name = ds.name

    else:
        raise ValueError(f"Unsupported source={spec.source}. Use 'ucirepo' or 'openml'.")

    valid_y = pd.notna(y)
    if not np.all(valid_y):
        X = X.loc[valid_y].reset_index(drop=True)
        y = pd.Series(y).loc[valid_y].reset_index(drop=True)

    X = pd.DataFrame(X).reset_index(drop=True)
    X.columns = [str(c) for c in X.columns]

    return X, y, cat_indicator, dataset_name


def encode_binary_target(y, spec: DatasetSpec):
    le = LabelEncoder()
    y_enc = le.fit_transform(np.asarray(y))

    if len(le.classes_) != 2:
        raise ValueError(f"[{spec.source} {spec.dataset_id}] Binary-only runner; got classes={list(le.classes_)}")

    return y_enc, [str(c) for c in le.classes_]


# ------------------------------
# Preprocessor: numeric median + low-card OHE + high-card LOO
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
        low_cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", _make_ohe()),
        ])
        transformers.append(("cat_ohe", low_cat_pipe, low_card_cols))

    if high_card_cols:
        high_cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_df", _to_df(high_card_cols)),
            ("loo", ce.LeaveOneOutEncoder(
                cols=high_card_cols,
                sigma=0.0,
                handle_unknown="value",
                handle_missing="value",
            )),
        ])
        transformers.append(("cat_loo", high_cat_pipe, high_card_cols))

    return ColumnTransformer(transformers)


# ------------------------------
# Split generation / verification
# ------------------------------
def make_outer_splits_like_cart(
    y_enc: np.ndarray,
    *,
    k_outer_splits: int,
    test_ratio: float,
    use_stratify: bool,
    seed_base: int,
):
    outer_splitter = (
        StratifiedShuffleSplit(n_splits=k_outer_splits, test_size=test_ratio, random_state=seed_base)
        if use_stratify
        else ShuffleSplit(n_splits=k_outer_splits, test_size=test_ratio, random_state=seed_base)
    )
    outer_splits = list(outer_splitter.split(np.zeros((len(y_enc), 1)), y_enc))
    return [(np.asarray(tv_idx, dtype=int), np.asarray(te_idx, dtype=int)) for tv_idx, te_idx in outer_splits]


def save_outer_splits(path_json: Path, splits, *, seed_base, k_outer_splits, test_ratio, use_stratify) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed_base": int(seed_base),
        "k_outer_splits": int(k_outer_splits),
        "test_ratio": float(test_ratio),
        "use_stratify": bool(use_stratify),
        "outer_splits": [{"tv_idx": tv.tolist(), "te_idx": te.tolist()} for (tv, te) in splits],
    }
    path_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_outer_splits(path_json: Path):
    obj = json.loads(path_json.read_text(encoding="utf-8"))
    return [(np.asarray(s["tv_idx"], dtype=int), np.asarray(s["te_idx"], dtype=int)) for s in obj["outer_splits"]]


def load_or_make_outer_splits(
    path_json: Optional[Path],
    y_enc: np.ndarray,
    *,
    k_outer_splits: int,
    test_ratio: float,
    use_stratify: bool,
    seed_base: int,
):
    if path_json is not None and path_json.exists():
        return load_outer_splits(path_json)

    splits = make_outer_splits_like_cart(
        y_enc,
        k_outer_splits=k_outer_splits,
        test_ratio=test_ratio,
        use_stratify=use_stratify,
        seed_base=seed_base,
    )

    if path_json is not None:
        save_outer_splits(
            path_json,
            splits,
            seed_base=seed_base,
            k_outer_splits=k_outer_splits,
            test_ratio=test_ratio,
            use_stratify=use_stratify,
        )

    return splits


# ------------------------------
# XGBoost search space
# ------------------------------
def sample_xgb_params(
    trial: optuna.Trial,
    *,
    max_depth: int,
    n_estimators_min: int,
    n_estimators_max: int,
    learning_rate_low: float,
    learning_rate_high: float,
):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", int(n_estimators_min), int(n_estimators_max)),
        "learning_rate": trial.suggest_float("learning_rate", learning_rate_low, learning_rate_high, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, int(max_depth)),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "scale_pos_weight_mode": trial.suggest_categorical("scale_pos_weight_mode", ["none", "balanced"]),
    }

    return params

def _scale_pos_weight_from_params(params: dict, y_train: np.ndarray) -> float:
    mode = params.get("scale_pos_weight_mode", "none")

    if mode == "none":
        return 1.0

    if mode != "balanced":
        raise ValueError(
            f"Unsupported scale_pos_weight_mode={mode!r}; use 'none' or 'balanced'."
        )

    y_train = np.asarray(y_train).astype(int)
    n_pos = max(1, int(np.sum(y_train == 1)))
    n_neg = max(1, int(np.sum(y_train == 0)))
    return float(n_neg / n_pos)


def materialize_xgb_params(
    sampled_params: dict,
    y_train: np.ndarray,
    *,
    random_state: int,
    n_jobs: int,
    tree_method: str,
    device: Optional[str],
):
    # Remove meta params used only to construct scale_pos_weight.
    params = {k: v for k, v in sampled_params.items() if k not in {"scale_pos_weight_mode", "scale_pos_weight_mult"}}
    params["scale_pos_weight"] = _scale_pos_weight_from_params(sampled_params, y_train)

    base = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
        "tree_method": tree_method,
        "verbosity": 0,
        **params,
    }
    # XGBoost >=2 supports `device`; older versions may reject it, so keep optional.
    if device:
        base["device"] = device
    return base


# ------------------------------
# XGBoost complexity / ensemble path length
# ------------------------------
def _quantile_higher(x, q):
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    try:
        return float(np.quantile(x, q, method="higher"))
    except TypeError:
        return float(np.quantile(x, q, interpolation="higher"))


_NODE_RE = re.compile(r"^\t*(\d+):")
_SPLIT_RE = re.compile(r"\[(f\d+)<")


def _parse_xgb_dump(clf):
    booster = clf.get_booster()
    dump = booster.get_dump(with_stats=False, dump_format="text")

    tree_stats = []
    leaf_depth_maps = []
    features_used = set()

    for tree_str in dump:
        lines = tree_str.strip().split("\n") if tree_str.strip() else []
        n_leaves = 0
        n_internal = 0
        depths = []
        leaf_depth = {}

        for line in lines:
            depth = len(line) - len(line.lstrip("\t"))
            depths.append(depth)

            m_node = _NODE_RE.match(line)
            node_id = int(m_node.group(1)) if m_node else None

            if "leaf=" in line:
                n_leaves += 1
                if node_id is not None:
                    leaf_depth[node_id] = depth
            elif ":[" in line:
                n_internal += 1
                m_feat = _SPLIT_RE.search(line)
                if m_feat:
                    features_used.add(m_feat.group(1))

        tree_stats.append({
            "n_leaves": int(n_leaves),
            "n_internal": int(n_internal),
            "n_nodes": int(n_leaves + n_internal),
            "max_depth": int(max(depths) if depths else 0),
        })
        leaf_depth_maps.append(leaf_depth)

    return tree_stats, leaf_depth_maps, features_used


def _leaf_usage_stats_xgb(clf, X, topk=10):
    leaf_matrix = clf.apply(X)
    leaf_matrix = np.asarray(leaf_matrix)
    if leaf_matrix.ndim == 1:
        leaf_matrix = leaf_matrix.reshape(-1, 1)

    n_samples, n_trees = leaf_matrix.shape
    all_leaf_ids = []
    for tree_idx in range(n_trees):
        # Composite id; safe enough because node ids are small.
        all_leaf_ids.extend((tree_idx, int(v)) for v in leaf_matrix[:, tree_idx])

    counts = np.asarray(list(Counter(all_leaf_ids).values()), dtype=float)
    n_total_hits = int(counts.sum()) if len(counts) else 0
    out = {
        "hit_unique_leaves": int(len(counts)),
        "n_leaf_hits_total": n_total_hits,
    }

    if n_total_hits > 0 and len(counts) > 0:
        p = counts / counts.sum()
        ent = -np.sum(p * np.log(p + 1e-12))
        out["leaf_support_entropy"] = float(ent)
        out["leaf_support_eff_leaves"] = float(np.exp(ent))
        k = min(topk, len(counts))
        out[f"leaf_support_top{topk}_coverage"] = float(np.sort(counts)[-k:].sum() / n_total_hits)

    return out


def _ensemble_path_stats_xgb(clf, X, leaf_depth_maps, q=0.90):
    leaf_matrix = np.asarray(clf.apply(X))
    if leaf_matrix.ndim == 1:
        leaf_matrix = leaf_matrix.reshape(-1, 1)

    n_samples, n_trees = leaf_matrix.shape
    total_steps = np.zeros(n_samples, dtype=float)

    for t in range(n_trees):
        depth_map = leaf_depth_maps[t] if t < len(leaf_depth_maps) else {}
        # Fallback to 0 when a leaf id is not found.
        total_steps += np.asarray([depth_map.get(int(v), 0) for v in leaf_matrix[:, t]], dtype=float)

    per_tree_steps = total_steps / max(1, n_trees)

    return {
        "ensemble_pathlen_total_decisions_mean": float(np.mean(total_steps)) if len(total_steps) else 0.0,
        "ensemble_pathlen_total_decisions_median": float(np.median(total_steps)) if len(total_steps) else 0.0,
        "ensemble_pathlen_total_decisions_p90": _quantile_higher(total_steps, q) if len(total_steps) else 0.0,
        "ensemble_pathlen_total_decisions_max": float(np.max(total_steps)) if len(total_steps) else 0.0,
        "ensemble_pathlen_per_tree_mean": float(np.mean(per_tree_steps)) if len(per_tree_steps) else 0.0,
        "ensemble_pathlen_per_tree_median": float(np.median(per_tree_steps)) if len(per_tree_steps) else 0.0,
        "ensemble_pathlen_per_tree_p90": _quantile_higher(per_tree_steps, q) if len(per_tree_steps) else 0.0,
    }


def extract_xgb_complexity(clf, X_train=None, X_test=None):
    cx = {}
    booster = clf.get_booster()

    cx["n_trees"] = int(booster.num_boosted_rounds())

    try:
        tree_stats, leaf_depth_maps, features_used = _parse_xgb_dump(clf)
        if tree_stats:
            cx["total_leaves_struct"] = int(sum(t["n_leaves"] for t in tree_stats))
            cx["total_internal_nodes_struct"] = int(sum(t["n_internal"] for t in tree_stats))
            cx["total_nodes_struct"] = int(sum(t["n_nodes"] for t in tree_stats))
            cx["avg_leaves_per_tree"] = float(np.mean([t["n_leaves"] for t in tree_stats]))
            cx["avg_tree_depth"] = float(np.mean([t["max_depth"] for t in tree_stats]))
            cx["max_tree_depth"] = int(max(t["max_depth"] for t in tree_stats))
            cx["min_tree_depth"] = int(min(t["max_depth"] for t in tree_stats))
            cx["n_features_used"] = int(len(features_used))
    except Exception:
        tree_stats, leaf_depth_maps = [], []

    try:
        gain_imp = booster.get_score(importance_type="gain")
        if gain_imp:
            vals = np.array(list(gain_imp.values()), dtype=float)
            vals = vals / (vals.sum() + 1e-12)
            sorted_vals = np.sort(vals)
            n = len(sorted_vals)
            gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_vals) - (n + 1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals) + 1e-12)
            cx["importance_gain_gini"] = float(gini)

        weight_imp = booster.get_score(importance_type="weight")
        if weight_imp:
            vals = np.array(list(weight_imp.values()), dtype=float)
            vals = vals / (vals.sum() + 1e-12)
            sorted_vals = np.sort(vals)
            n = len(sorted_vals)
            gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_vals) - (n + 1) * np.sum(sorted_vals)) / (n * np.sum(sorted_vals) + 1e-12)
            cx["importance_weight_gini"] = float(gini)
            cx["total_splits"] = int(sum(weight_imp.values()))
    except Exception:
        pass

    if X_train is not None:
        try:
            cx.update({f"train_{k}": v for k, v in _leaf_usage_stats_xgb(clf, X_train).items()})
            cx.update({f"train_{k}": v for k, v in _ensemble_path_stats_xgb(clf, X_train, leaf_depth_maps).items()})
        except Exception:
            pass

    if X_test is not None:
        try:
            cx.update({f"test_{k}": v for k, v in _leaf_usage_stats_xgb(clf, X_test).items()})
            cx.update({f"test_{k}": v for k, v in _ensemble_path_stats_xgb(clf, X_test, leaf_depth_maps).items()})
        except Exception:
            pass

    return cx


# ------------------------------
# Run XGBoost HPO for one dataset
# ------------------------------
def run_xgb_hpo_one(
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
    n_estimators_min: int,
    n_estimators_max: int,
    learning_rate_low: float,
    learning_rate_high: float,
    n_jobs: int,
    tree_method: str,
    device: Optional[str],
    show_progress_bar: bool,
    outer_splits_json: Optional[Path] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, cat_indicator, dataset_name = load_dataset(spec)
    feature_names = X.columns.tolist()
    y_enc, classes = encode_binary_target(y, spec)

    if hpo_metric not in SUPPORTED_HPO_METRICS:
        raise ValueError(f"Unsupported --hpo_metric={hpo_metric}; choose from {SUPPORTED_HPO_METRICS}")

    direction = metric_direction(hpo_metric)

    if outer_splits_json is None:
        outer_splits_json = out_dir / f"outer_splits_seed{seed_base}.json"

    outer_splits = load_or_make_outer_splits(
        outer_splits_json,
        y_enc,
        k_outer_splits=k_outer_splits,
        test_ratio=test_ratio,
        use_stratify=use_stratify,
        seed_base=seed_base,
    )

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
        "n_estimators_min": n_estimators_min,
        "n_estimators_max": n_estimators_max,
        "learning_rate_low": learning_rate_low,
        "learning_rate_high": learning_rate_high,
        "tree_method": tree_method,
        "device": device,
        "outer_splits_json": str(outer_splits_json),
        "n_samples": int(len(y_enc)),
        "n_features_raw": int(X.shape[1]),
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

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
            sampled_params = sample_xgb_params(
                trial,
                max_depth=max_depth,
                n_estimators_min=n_estimators_min,
                n_estimators_max=n_estimators_max,
                learning_rate_low=learning_rate_low,
                learning_rate_high=learning_rate_high,
            )
            scores = []

            for rep_id, (Xtr, ytr, Xva, yva) in enumerate(cached):
                params = materialize_xgb_params(
                    sampled_params,
                    ytr,
                    random_state=seed32(outer_seed * 1_000_000 + trial.number + rep_id),
                    n_jobs=n_jobs,
                    tree_method=tree_method,
                    device=device,
                )
                clf = XGBClassifier(**params)
                clf.fit(Xtr, ytr)

                pred_va = clf.predict(Xva)
                prob_va = clf.predict_proba(Xva)[:, 1]

                metrics = eval_binary_metrics(yva, pred_va, prob_va)
                scores.append(float(metrics[hpo_metric]))

                partial_value = robust_objective(scores, hpo_metric, std_penalty_alpha)
                trial.report(float(partial_value), step=rep_id)
                if trial.should_prune():
                    raise optuna.TrialPruned()

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
            sampler=optuna.samplers.TPESampler(seed=seed32(outer_seed)),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=max(10, inner_repeats), n_warmup_steps=1),
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

        final_params = materialize_xgb_params(
            best_params,
            y_tv,
            random_state=final_seed,
            n_jobs=n_jobs,
            tree_method=tree_method,
            device=device,
        )
        clf = XGBClassifier(**final_params)

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

        complexity = extract_xgb_complexity(clf, X_train=Xtv_tr, X_test=XTe)
        complexity["n_features_after_pp"] = int(Xtv_tr.shape[1])
        complexity["hp_scale_pos_weight_eff"] = float(final_params.get("scale_pos_weight", 1.0))

        trial_states = Counter(t.state.name for t in study.trials)
        hpo_stats = {
            "n_trials_total": int(len(study.trials)),
            "n_trials_complete": int(trial_states.get("COMPLETE", 0)),
            "n_trials_pruned": int(trial_states.get("PRUNED", 0)),
            "n_trials_failed": int(trial_states.get("FAIL", 0)),
            "best_trial_number": int(study.best_trial.number),
        }

        report = {
            "dataset": asdict(spec),
            "dataset_name": dataset_name,
            "split": int(split_idx),
            "outer_seed": int(outer_seed),
            "outer_splits_json": str(outer_splits_json),
            "hpo_metric": hpo_metric,
            "hpo_direction": direction,
            "best_value_robust": float(study.best_value),
            "best_inner_mean": study.best_trial.user_attrs.get("inner_mean"),
            "best_inner_std": study.best_trial.user_attrs.get("inner_std"),
            "best_params": best_params,
            "hpo_stats": hpo_stats,
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
            f"trees={complexity.get('n_trees', None)} "
            f"leaves_total={complexity.get('total_leaves_struct', None)} "
            f"ens_p90steps={complexity.get('test_ensemble_pathlen_total_decisions_p90', None)} "
            f"fit={final_fit_time_sec:.2f}s"
        )

        split_dir = out_dir / f"outer_{split_idx:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)
        with open(split_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

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
            "outer_splits_json": r["outer_splits_json"],
            "hpo_metric": r["hpo_metric"],
            "best_value_robust": r["best_value_robust"],
            "best_inner_mean": r["best_inner_mean"],
            "best_inner_std": r["best_inner_std"],
            **{f"hp/{k}": v for k, v in r["best_params"].items()},
            **{f"hpo/{k}": v for k, v in r["hpo_stats"].items()},
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
    for prefix in ("test/", "train/", "cx/", "time/", "hpo/"):
        cols = [c for c in df.columns if c.startswith(prefix)]
        for c in cols:
            summary[c] = mean_std(df[c])

    with open(out_dir / "outer_reports_mean_std.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return df, summary


def run_xgb_hpo_many(specs: Iterable[DatasetSpec], *, root_dir: str, outer_splits_dir: Optional[str] = None, **kwargs):
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    split_root = Path(outer_splits_dir) if outer_splits_dir else None
    if split_root is not None:
        split_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for spec in specs:
        out_dir = root / spec.short_name
        try:
            shared_split_json = None
            if split_root is not None:
                shared_split_json = split_root / f"{spec.short_name}_outer_splits_seed{kwargs.get('seed_base', DEFAULT_SEED_BASE)}.json"

            _, summary = run_xgb_hpo_one(
                spec,
                out_dir=out_dir,
                outer_splits_json=shared_split_json,
                **kwargs,
            )
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
    parts = text.split(":")
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError(f"Bad dataset spec '{text}'. Use SOURCE:ID or openml:ID:TARGET.")

    source = parts[0].lower()
    if source == "uci":
        source = "ucirepo"
    if source not in {"ucirepo", "openml"}:
        raise argparse.ArgumentTypeError(f"Bad source '{source}'. Use ucirepo or openml.")

    try:
        dataset_id = int(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Bad dataset id in '{text}'.") from e

    target = parts[2] if len(parts) == 3 else None
    if source != "openml" and target is not None:
        raise argparse.ArgumentTypeError("Explicit target is only supported for OpenML specs.")

    return DatasetSpec(source=source, dataset_id=dataset_id, target=target)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run XGBoost HPO on UCIrepo/OpenML binary classification datasets.")

    dataset_group = parser.add_argument_group("dataset selection")
    dataset_group.add_argument("--source", choices=["ucirepo", "uci", "openml"], default="ucirepo")
    dataset_group.add_argument("--dataset_ids", nargs="+", type=int, default=None)
    dataset_group.add_argument("--datasets", nargs="+", type=parse_dataset_spec, default=None)

    hpo_group = parser.add_argument_group("HPO settings")
    hpo_group.add_argument("--hpo_metric", choices=SUPPORTED_HPO_METRICS, default="bal_acc")
    hpo_group.add_argument("--max_depth", type=int, default=DEFAULT_MAX_DEPTH, help="Upper bound for XGBoost per-tree max_depth search. If 4, HPO searches 1..4.")
    hpo_group.add_argument("--max_hpo_trials", type=int, default=DEFAULT_MAX_HPO_TRIALS)
    hpo_group.add_argument("--hpo_timeout_sec", type=int, default=DEFAULT_HPO_TIMEOUT_SEC)
    hpo_group.add_argument("--std_penalty_alpha", type=float, default=DEFAULT_STD_PENALTY_ALPHA)
    hpo_group.add_argument("--no_progress_bar", action="store_true")

    xgb_group = parser.add_argument_group("XGBoost settings")
    xgb_group.add_argument("--n_estimators_min", type=int, default=DEFAULT_N_ESTIMATORS_MIN)
    xgb_group.add_argument("--n_estimators_max", type=int, default=DEFAULT_N_ESTIMATORS_MAX)
    xgb_group.add_argument("--learning_rate_low", type=float, default=DEFAULT_LEARNING_RATE_LOW)
    xgb_group.add_argument("--learning_rate_high", type=float, default=DEFAULT_LEARNING_RATE_HIGH)
    xgb_group.add_argument("--n_jobs", type=int, default=DEFAULT_N_JOBS)
    xgb_group.add_argument("--tree_method", choices=["auto", "exact", "approx", "hist"], default=DEFAULT_TREE_METHOD)
    xgb_group.add_argument("--device", type=str, default=None, help="Optional XGBoost device, e.g. cuda. Leave unset for older XGBoost versions.")

    cv_group = parser.add_argument_group("split/CV settings")
    cv_group.add_argument("--k_outer_splits", type=int, default=DEFAULT_K_OUTER_SPLITS)
    cv_group.add_argument("--test_ratio", type=float, default=DEFAULT_TEST_RATIO)
    cv_group.add_argument("--inner_repeats", type=int, default=DEFAULT_INNER_REPEATS)
    cv_group.add_argument("--seed_base", type=int, default=DEFAULT_SEED_BASE)
    cv_group.add_argument("--no_stratify", action="store_true")
    cv_group.add_argument("--outer_splits_dir", type=str, default=None)

    misc_group = parser.add_argument_group("misc")
    misc_group.add_argument("--cat_card_threshold", type=int, default=DEFAULT_CAT_CARD_THRESHOLD)
    misc_group.add_argument("--root_dir", type=str, default="./xgb_hpo_runs")
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
        raise ValueError("No datasets provided. Use either --dataset_ids with --source, or --datasets SOURCE:ID.")
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
    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError("--test_ratio must be in (0, 1)")
    if args.n_estimators_min < 1 or args.n_estimators_max < args.n_estimators_min:
        raise ValueError("Require 1 <= --n_estimators_min <= --n_estimators_max")
    if args.learning_rate_low <= 0 or args.learning_rate_high <= 0 or args.learning_rate_low >= args.learning_rate_high:
        raise ValueError("Require 0 < --learning_rate_low < --learning_rate_high")

    specs = specs_from_args(args)

    print("Running XGBoost HPO with:")
    print(f"  datasets        = {[asdict(s) for s in specs]}")
    print(f"  hpo_metric      = {args.hpo_metric} ({metric_direction(args.hpo_metric)})")
    print(f"  max_depth grid  = {list(range(1, args.max_depth + 1))}")
    print(f"  n_estimators    = [{args.n_estimators_min}, {args.n_estimators_max}]")
    print(f"  tree_method     = {args.tree_method}")
    print(f"  root_dir        = {args.root_dir}")

    run_xgb_hpo_many(
        specs,
        root_dir=args.root_dir,
        outer_splits_dir=args.outer_splits_dir,
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
        n_estimators_min=args.n_estimators_min,
        n_estimators_max=args.n_estimators_max,
        learning_rate_low=args.learning_rate_low,
        learning_rate_high=args.learning_rate_high,
        n_jobs=args.n_jobs,
        tree_method=args.tree_method,
        device=args.device,
        show_progress_bar=not args.no_progress_bar,
    )


if __name__ == "__main__":
    main()
