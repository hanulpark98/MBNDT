import argparse
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import optuna
import pandas as pd
from ucimlrepo import fetch_ucirepo

try:
    import openml
except ImportError:  # only required when --source openml is used
    openml = None

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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

try:
    import category_encoders as ce
except ImportError as e:
    raise ImportError(
        "This script requires category_encoders. Install it with: pip install category-encoders"
    ) from e

try:
    from split import SPLIT
    from split._tree import Leaf as SplitLeaf
except ImportError as e:
    raise ImportError(
        "This script requires the SPLIT package/repo to be importable. "
        "Run it from the SPLIT repo root or install the package first."
    ) from e

import warnings
import contextlib


# ------------------------------
# Optional warning suppression for noisy SPLIT/GOSDT internals
# ------------------------------
@contextlib.contextmanager
def split_warning_context(
    *,
    suppress_split_warnings: bool = False,
    suppress_small_reg_warning: bool = False,
):
    with warnings.catch_warnings():
        if suppress_split_warnings:
            warnings.filterwarnings(
                "ignore",
                message=r".*Mean of empty slice.*",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*invalid value encountered in scalar divide.*",
                category=RuntimeWarning,
            )

        if suppress_small_reg_warning:
            warnings.filterwarnings(
                "ignore",
                message=r".*regularization was chosen that is less than 1 /.*",
            )

        if suppress_small_reg_warning:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    yield
        else:
            yield


# ------------------------------
# Defaults aligned with cart_hpo_cli.py
# ------------------------------
DEFAULT_K_OUTER_SPLITS = 5
DEFAULT_TEST_RATIO = 0.2
DEFAULT_INNER_REPEATS = 5
DEFAULT_USE_STRATIFY = True
DEFAULT_SEED_BASE = 0
DEFAULT_HPO_TIMEOUT_SEC = 60 * 60
DEFAULT_MAX_HPO_TRIALS = 100000
DEFAULT_CAT_CARD_THRESHOLD = 10
DEFAULT_STD_PENALTY_ALPHA = 0

# SPLIT-specific defaults from the notebook
DEFAULT_PER_FIT_LIMIT_SEC = 5 * 60
DEFAULT_FINAL_FIT_LIMIT_SEC = 20 * 60
DEFAULT_REG_LOW = 1e-6
DEFAULT_REG_HIGH = 1e-1
DEFAULT_GBDT_N_EST = 50
DEFAULT_GBDT_MAX_DEPTH = 1
DEFAULT_GBDT_N_EST_CHOICES = [50, 100, 200]
DEFAULT_GBDT_MAX_DEPTH_CHOICES = [1, 2, 3]

DEFAULT_LOOKAHEAD_POLICY = "auto"   # {"fixed", "auto", "tune_capped"}
DEFAULT_LOOKAHEAD_FIXED = 2
DEFAULT_LOOKAHEAD_MIN = 2
DEFAULT_LOOKAHEAD_SMALL_N_THRESHOLD = 1000
DEFAULT_LOOKAHEAD_SMALL_VALUE = 3

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


def choose_lookahead_depth(
    trial: optuna.Trial,
    *,
    full_depth_budget: int,
    n_train: int,
    lookahead_policy: str,
    lookahead_fixed: int,
    lookahead_min: int,
    lookahead_small_n_threshold: int,
    lookahead_small_value: int,
) -> int:
    """
    Choose SPLIT lookahead depth.

    Policy:
      fixed:
        always use lookahead_fixed, clipped to full_depth_budget.

      auto:
        use lookahead_small_value for small datasets, otherwise lookahead_fixed.

      tune_capped:
        tune lookahead_depth_budget in a small capped range.
        For small datasets: lookahead_min..lookahead_small_value.
        Otherwise: lookahead_min..lookahead_fixed.

    Note:
      lookahead cannot exceed full_depth_budget.
      If full_depth_budget == 1, lookahead must be 1.
    """
    full = int(full_depth_budget)
    n_train = int(n_train)

    if full <= 1:
        return 1

    lo = max(1, int(lookahead_min))
    lo = min(lo, full)

    policy = str(lookahead_policy).lower()

    if policy == "fixed":
        return min(max(int(lookahead_fixed), lo), full)

    if policy == "auto":
        target = int(lookahead_small_value) if n_train < int(lookahead_small_n_threshold) else int(lookahead_fixed)
        return min(max(target, lo), full)

    if policy == "tune_capped":
        cap = int(lookahead_small_value) if n_train < int(lookahead_small_n_threshold) else int(lookahead_fixed)
        hi = min(max(cap, lo), full)

        if hi <= lo:
            return int(lo)

        return trial.suggest_int(
            "lookahead_depth_budget",
            int(lo),
            int(hi),
        )

    raise ValueError(
        f"Unknown lookahead_policy={lookahead_policy}. "
        "Use one of: fixed, auto, tune_capped."
    )
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
        raise ValueError(
            f"[{spec.source} {spec.dataset_id}] Binary-only runner; got classes={list(le.classes_)}"
        )

    return y_enc, [str(c) for c in le.classes_]


# ------------------------------
# Preprocessor: DataFrame output for SPLIT
# ------------------------------
class UnifiedOheLooOrNumericPreprocessor:
    """
    Same preprocessing idea as CART:
      numeric: median imputation
      low-cardinality categorical: OHE
      high-cardinality categorical: Leave-One-Out encoding

    Difference: SPLIT expects tabular/DataFrame-like input, so this class returns a dense DataFrame.
    """

    def __init__(self, feature_names, cat_indicator, card_threshold=10):
        self.feature_names = list(feature_names)
        self.cat_indicator = list(cat_indicator)
        self.card_threshold = int(card_threshold)

        self.num_cols = []
        self.cat_cols = []
        self.low_cat_cols = []
        self.high_cat_cols = []

        self.num_medians_ = None
        self.cat_modes_ = None
        self.ohe_ = None
        self.loo_ = None
        self.ohe_names_ = []

    def _make_dense_ohe(self):
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X = X.copy()
        y = np.asarray(y).ravel()

        self.cat_cols = [c for c, is_cat in zip(self.feature_names, self.cat_indicator) if is_cat]
        self.num_cols = [c for c, is_cat in zip(self.feature_names, self.cat_indicator) if not is_cat]

        if self.num_cols:
            for c in self.num_cols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
            self.num_medians_ = X[self.num_cols].median(numeric_only=True)
        else:
            self.num_medians_ = None

        self.cat_modes_ = {}
        for c in self.cat_cols:
            m = X[c].mode(dropna=True)
            self.cat_modes_[c] = m.iloc[0] if len(m) else "__MISSING__"

        low, high = [], []
        for c in self.cat_cols:
            card = X[c].nunique(dropna=True)
            (low if card < self.card_threshold else high).append(c)
        self.low_cat_cols, self.high_cat_cols = low, high

        if self.low_cat_cols:
            X_low = X[self.low_cat_cols].copy()
            for c in self.low_cat_cols:
                X_low[c] = X_low[c].fillna(self.cat_modes_[c]).astype(str)
            self.ohe_ = self._make_dense_ohe()
            self.ohe_.fit(X_low)
            if hasattr(self.ohe_, "get_feature_names_out"):
                self.ohe_names_ = list(self.ohe_.get_feature_names_out(self.low_cat_cols))
            else:
                self.ohe_names_ = []
        else:
            self.ohe_ = None
            self.ohe_names_ = []

        if self.high_cat_cols:
            X_high = X[self.high_cat_cols].copy()
            for c in self.high_cat_cols:
                X_high[c] = X_high[c].fillna(self.cat_modes_[c]).astype(str)
            self.loo_ = ce.LeaveOneOutEncoder(
                cols=self.high_cat_cols,
                sigma=0.0,
                handle_unknown="value",
                handle_missing="value",
            )
            self.loo_.fit(X_high, y)
        else:
            self.loo_ = None

        return self

    def fit_transform_df(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform_df(X, y=y)

    def transform_df(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        parts = []
        cols = []

        if self.num_cols:
            X_num = X[self.num_cols].copy()
            for c in self.num_cols:
                X_num[c] = pd.to_numeric(X_num[c], errors="coerce")
            if self.num_medians_ is not None:
                X_num = X_num.fillna(self.num_medians_)
            parts.append(X_num.to_numpy(dtype=np.float32))
            cols += self.num_cols

        if self.ohe_ is not None:
            X_low = X[self.low_cat_cols].copy()
            for c in self.low_cat_cols:
                X_low[c] = X_low[c].fillna(self.cat_modes_[c]).astype(str)
            Z = np.asarray(self.ohe_.transform(X_low), dtype=np.float32)
            parts.append(Z)
            cols += self.ohe_names_

        if self.loo_ is not None:
            X_high = X[self.high_cat_cols].copy()
            for c in self.high_cat_cols:
                X_high[c] = X_high[c].fillna(self.cat_modes_[c]).astype(str)
            if y is not None:
                Z = self.loo_.transform(X_high, y=np.asarray(y).ravel()).to_numpy(dtype=np.float32)
            else:
                Z = self.loo_.transform(X_high).to_numpy(dtype=np.float32)
            parts.append(Z)
            cols += self.high_cat_cols

        if not parts:
            return pd.DataFrame(index=X.index)

        out = np.hstack(parts)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return pd.DataFrame(out, columns=cols, index=X.index)


# ------------------------------
# SPLIT helpers
# ------------------------------
def split_predict_proba_pos(model: SPLIT, X_df: pd.DataFrame) -> np.ndarray:
    """
    SPLIT wrapper may not expose predict_proba, so use its internal encoder/classifier.
    """
    Xb = model.enc.transform(X_df)
    prob = model.clf.predict_proba(np.asarray(Xb, dtype=bool))
    prob = np.asarray(prob)
    if prob.ndim == 2:
        return prob[:, 1]
    return prob.astype(float)


def _children(node):
    return [node.left_child, node.right_child]


def _is_leaf(node) -> bool:
    return isinstance(node, SplitLeaf)


def summarize_split_tree(root) -> Dict[str, int]:
    feats = set()

    def dfs(node, depth):
        if node is None:
            return 0, 0, depth - 1
        if _is_leaf(node):
            return 1, 1, depth
        feats.add(int(node.feature))
        n_nodes = 1
        n_leaves = 0
        max_d = depth
        for c in _children(node):
            cn, cl, cd = dfs(c, depth + 1)
            n_nodes += cn
            n_leaves += cl
            max_d = max(max_d, cd)
        return n_nodes, n_leaves, max_d

    n_nodes, n_leaves, max_depth = dfs(root, 0)
    return {
        "tree_max_depth": int(max_depth),
        "n_nodes_struct": int(n_nodes),
        "n_leaves_struct": int(n_leaves),
        "n_internal_nodes_struct": int(n_nodes - n_leaves),
        "n_features_used": int(len(feats)),
    }


def route_one_binary(root, x_row_bool: np.ndarray):
    """
    SPLIT.py predict logic:
      if x_i[node.feature] is True -> left_child
      else                         -> right_child
    """
    node = root
    steps = 0
    path_bits = []
    while node is not None and (not _is_leaf(node)):
        f = int(node.feature)
        go_left = bool(x_row_bool[f])
        steps += 1
        path_bits.append(1 if go_left else 0)
        node = node.left_child if go_left else node.right_child
    return tuple(path_bits), steps


def _quantile_higher(x, q):
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    try:
        return float(np.quantile(x, q, method="higher"))
    except TypeError:
        return float(np.quantile(x, q, interpolation="higher"))


def decision_steps_stats_split(root, Xb_bool: np.ndarray, q=0.90) -> Dict[str, float]:
    steps = np.zeros(len(Xb_bool), dtype=int)
    for i in range(len(Xb_bool)):
        _, s = route_one_binary(root, Xb_bool[i])
        steps[i] = s

    return {
        "pathlen_decisions_mean": float(np.mean(steps)) if len(steps) else 0.0,
        "pathlen_decisions_median": float(np.median(steps)) if len(steps) else 0.0,
        "pathlen_decisions_p90": _quantile_higher(steps, q) if len(steps) else 0.0,
        "pathlen_decisions_max": float(np.max(steps)) if len(steps) else 0.0,
    }


def leaf_usage_stats_split(root, Xb_bool: np.ndarray, topk=10) -> Dict[str, float]:
    leaf_ids = []
    for i in range(len(Xb_bool)):
        lid, _ = route_one_binary(root, Xb_bool[i])
        leaf_ids.append(lid)

    counts = np.asarray(list(Counter(leaf_ids).values()), dtype=float)
    n = int(counts.sum()) if len(counts) else 0
    out = {
        "hit_leaves": int(len(counts)),
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


def extract_split_complexity_e1(root, X_train_bool=None, X_test_bool=None) -> Dict[str, Any]:
    cx = {}
    cx.update(summarize_split_tree(root))

    if X_train_bool is not None:
        cx.update({f"train_{k}": v for k, v in decision_steps_stats_split(root, X_train_bool).items()})
        cx.update({f"train_{k}": v for k, v in leaf_usage_stats_split(root, X_train_bool).items()})

    if X_test_bool is not None:
        cx.update({f"test_{k}": v for k, v in decision_steps_stats_split(root, X_test_bool).items()})
        cx.update({f"test_{k}": v for k, v in leaf_usage_stats_split(root, X_test_bool).items()})

    return cx


def get_split_root(model: SPLIT):
    if getattr(model, "tree", None) is not None:
        return model.tree
    if hasattr(model, "clf") and hasattr(model.clf, "trees_") and len(model.clf.trees_) > 0:
        return model.clf.trees_[0].tree
    raise RuntimeError("Could not find SPLIT tree root on model.tree or model.clf.trees_[0].tree")


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
    splits = []
    for s in obj["outer_splits"]:
        splits.append((np.asarray(s["tv_idx"], dtype=int), np.asarray(s["te_idx"], dtype=int)))
    return splits


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
# SPLIT search space and HPO
# ------------------------------
def sample_split_params(
    trial: optuna.Trial,
    *,
    max_depth: int,
    tune_full_depth: bool,
    reg_low: float,
    reg_high: float,
    tune_binarizer: bool,
    tune_greedy_postprocess: bool,
    gbdt_n_est: int,
    gbdt_max_depth: int,
    gbdt_n_est_choices: list[int],
    gbdt_max_depth_choices: list[int],
    greedy_postprocess: bool,
    n_train: int,
    lookahead_policy: str,
    lookahead_fixed: int,
    lookahead_min: int,
    lookahead_small_n_threshold: int,
    lookahead_small_value: int,
):
    """Sample SPLIT hyperparameters.

    Depth sampling is deliberately constrained to valid pairs:
    lookahead_depth_budget <= full_depth_budget <= max_depth.
    """
    if tune_full_depth:
        full_depth_budget = trial.suggest_int("full_depth_budget", 1, int(max_depth))
    else:
        full_depth_budget = int(max_depth)

    reg = float(trial.suggest_float("reg", reg_low, reg_high, log=True))



    lookahead_depth_budget = choose_lookahead_depth(
        trial,
        full_depth_budget=full_depth_budget,
        n_train=n_train,
        lookahead_policy=lookahead_policy,
        lookahead_fixed=lookahead_fixed,
        lookahead_min=lookahead_min,
        lookahead_small_n_threshold=lookahead_small_n_threshold,
        lookahead_small_value=lookahead_small_value,
    )


    if tune_binarizer:
        sampled_gbdt_n_est = trial.suggest_categorical(
            "gbdt_n_est", [int(v) for v in gbdt_n_est_choices]
        )
        sampled_gbdt_max_depth = trial.suggest_categorical(
            "gbdt_max_depth", [int(v) for v in gbdt_max_depth_choices]
        )
    else:
        sampled_gbdt_n_est = int(gbdt_n_est)
        sampled_gbdt_max_depth = int(gbdt_max_depth)

    if tune_greedy_postprocess:
        sampled_greedy_postprocess = trial.suggest_categorical(
            "greedy_postprocess", [False, True]
        )
    else:
        sampled_greedy_postprocess = bool(greedy_postprocess)

    return {
        "reg": reg,
        "full_depth_budget": int(full_depth_budget),
        "lookahead_depth_budget": int(lookahead_depth_budget),
        "gbdt_n_est": int(sampled_gbdt_n_est),
        "gbdt_max_depth": int(sampled_gbdt_max_depth),
        "greedy_postprocess": bool(sampled_greedy_postprocess),
    }


def tune_split_optuna(
    X_tv: pd.DataFrame,
    y_tv: np.ndarray,
    *,
    outer_seed: int,
    inner_repeats: int,
    use_stratify: bool,
    feature_names,
    cat_indicator,
    card_threshold: int,
    hpo_timeout_sec: int,
    max_hpo_trials: int,
    hpo_metric: str,
    max_depth: int,
    tune_full_depth: bool,
    std_penalty_alpha: float,
    reg_low: float,
    reg_high: float,
    per_fit_limit_sec: int,
    similar_support: bool,
    greedy_postprocess: bool,
    gbdt_n_est: int,
    gbdt_max_depth: int,
    tune_binarizer: bool,
    tune_greedy_postprocess: bool,
    clamp_reg_to_inv_n: bool,
    gbdt_n_est_choices: list[int],
    gbdt_max_depth_choices: list[int],
    show_progress_bar: bool= True,
    suppress_split_warnings: bool = False,
    suppress_small_reg_warning: bool = False,
    lookahead_policy: str,
    lookahead_fixed: int,
    lookahead_min: int,
    lookahead_small_n_threshold: int,
    lookahead_small_value: int,
):
    inner_seed = outer_seed + 111
    y_arr = np.asarray(y_tv).astype(int)
    n = len(y_arr)

    inner_splitter = (
        StratifiedKFold(n_splits=inner_repeats, shuffle=True, random_state=inner_seed)
        if use_stratify
        else KFold(n_splits=inner_repeats, shuffle=True, random_state=inner_seed)
    )
    inner_splits = list(inner_splitter.split(np.zeros((n, 1)), y_arr))

    cache_t0 = time.perf_counter()
    cached = []
    for fold_id, (tr_idx, va_idx) in enumerate(inner_splits):
        X_tr, y_tr = X_tv.iloc[tr_idx], y_arr[tr_idx]
        X_va, y_va = X_tv.iloc[va_idx], y_arr[va_idx]

        prep = UnifiedOheLooOrNumericPreprocessor(
            feature_names=feature_names,
            cat_indicator=cat_indicator,
            card_threshold=card_threshold,
        )
        Xtr = prep.fit_transform_df(X_tr, y_tr)
        Xva = prep.transform_df(X_va)
        cached.append((Xtr, y_tr, Xva, y_va))

    cache_preprocess_time_sec = float(time.perf_counter() - cache_t0)

    direction = metric_direction(hpo_metric)

    def objective(trial: optuna.Trial) -> float:
        params = sample_split_params(
            trial,
            max_depth=max_depth,
            tune_full_depth=tune_full_depth,
            reg_low=reg_low,
            reg_high=reg_high,
            tune_binarizer=tune_binarizer,
            tune_greedy_postprocess=tune_greedy_postprocess,
            gbdt_n_est=gbdt_n_est,
            gbdt_max_depth=gbdt_max_depth,
            gbdt_n_est_choices=gbdt_n_est_choices,
            gbdt_max_depth_choices=gbdt_max_depth_choices,
            greedy_postprocess=greedy_postprocess,
            n_train=n,
            lookahead_policy=lookahead_policy,
            lookahead_fixed=lookahead_fixed,
            lookahead_min=lookahead_min,
            lookahead_small_n_threshold=lookahead_small_n_threshold,
            lookahead_small_value=lookahead_small_value,
        )
        scores = []

        for rep_id, (Xtr, ytr, Xva, yva) in enumerate(cached):
            set_all_seeds(seed32(outer_seed * 1_000_000 + trial.number + rep_id))

            reg_raw = float(params["reg"])
            reg_eff = (
                max(reg_raw, (1.0 / max(1, len(ytr))) + 1e-12)
                if clamp_reg_to_inv_n
                else reg_raw
            )

            model = SPLIT(
                time_limit=int(per_fit_limit_sec),
                verbose=False,
                reg=float(reg_eff),
                lookahead_depth_budget=int(params["lookahead_depth_budget"]),
                full_depth_budget=int(params["full_depth_budget"]),
                similar_support=bool(similar_support),
                allow_small_reg=True,
                greedy_postprocess=bool(params["greedy_postprocess"]),
                binarize=True,
                gbdt_n_est=int(params["gbdt_n_est"]),
                gbdt_max_depth=int(params["gbdt_max_depth"]),
            )

            with split_warning_context(
                suppress_split_warnings=suppress_split_warnings,
                suppress_small_reg_warning=suppress_small_reg_warning,
            ):
                model.fit(Xtr, ytr)

            pred_va = model.predict(Xva)
            prob_va = split_predict_proba_pos(model, Xva)

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
    # Add fixed values that are not present in Optuna params when corresponding tuning flags are off.
    if not tune_full_depth:
        best_params["full_depth_budget"] = int(max_depth)

    # Reconstruct fixed/dynamic lookahead if it was not sampled by Optuna.
    #
    # For lookahead_policy in {"fixed", "auto"}, lookahead_depth_budget is not an
    # Optuna parameter, so study.best_trial.params will not contain it. Also, for
    # tune_capped, if the valid range collapses to a single value, the helper
    # returns that value without calling trial.suggest_int(...), so it will not be
    # present in best_trial.params either.
    #
    # Important: use the exact same helper as sample_split_params(...) to avoid
    # stale variables/rules from an older reconstruction branch.
    if "lookahead_depth_budget" not in best_params:
        full_depth = int(best_params["full_depth_budget"])
        best_params["lookahead_depth_budget"] = int(
            choose_lookahead_depth(
                study.best_trial,
                full_depth_budget=full_depth,
                n_train=n,
                lookahead_policy=lookahead_policy,
                lookahead_fixed=lookahead_fixed,
                lookahead_min=lookahead_min,
                lookahead_small_n_threshold=lookahead_small_n_threshold,
                lookahead_small_value=lookahead_small_value,
            )
        )





    if not tune_binarizer:
        best_params["gbdt_n_est"] = int(gbdt_n_est)
        best_params["gbdt_max_depth"] = int(gbdt_max_depth)
    if not tune_greedy_postprocess:
        best_params["greedy_postprocess"] = bool(greedy_postprocess)

    trial_states = Counter(t.state.name for t in study.trials)
    hpo_stats = {
        "n_trials_total": int(len(study.trials)),
        "n_trials_complete": int(trial_states.get("COMPLETE", 0)),
        "n_trials_pruned": int(trial_states.get("PRUNED", 0)),
        "n_trials_failed": int(trial_states.get("FAIL", 0)),
        "best_trial_number": int(study.best_trial.number),
    }

    return {
        "study": study,
        "best_params": best_params,
        "hpo_stats": hpo_stats,
        "hpo_time_sec": hpo_time_sec,
        "cache_preprocess_time_sec": cache_preprocess_time_sec,
    }


# ------------------------------
# Run SPLIT HPO for one dataset
# ------------------------------
def run_split_hpo_one(
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
    tune_full_depth: bool,
    std_penalty_alpha: float,
    reg_low: float,
    reg_high: float,
    per_fit_limit_sec: int,
    final_fit_limit_sec: int,
    similar_support: bool,
    greedy_postprocess: bool,
    gbdt_n_est: int,
    gbdt_max_depth: int,
    tune_binarizer: bool,
    tune_greedy_postprocess: bool,
    clamp_reg_to_inv_n: bool,
    gbdt_n_est_choices: list[int],
    gbdt_max_depth_choices: list[int],
    show_progress_bar: bool = True,
    outer_splits_json: Optional[Path] = None,
    suppress_split_warnings: bool = False,
    suppress_small_reg_warning: bool = False,
    lookahead_policy: str,
    lookahead_fixed: int,
    lookahead_min: int,
    lookahead_small_n_threshold: int,
    lookahead_small_value: int,
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
        "tune_full_depth": bool(tune_full_depth),
        "searched_full_depth_values": list(range(1, int(max_depth) + 1)) if tune_full_depth else [int(max_depth)],
        "searched_lookahead_depth_values": {
            "policy": str(lookahead_policy),
            "min": int(lookahead_min),
            "fixed_or_cap": int(lookahead_fixed),
            "small_n_threshold": int(lookahead_small_n_threshold),
            "small_n_value": int(lookahead_small_value),
        },
        "std_penalty_alpha": std_penalty_alpha,
        "reg_low": reg_low,
        "reg_high": reg_high,
        "per_fit_limit_sec": per_fit_limit_sec,
        "final_fit_limit_sec": final_fit_limit_sec,
        "similar_support": bool(similar_support),
        "greedy_postprocess": bool(greedy_postprocess),
        "tune_greedy_postprocess": bool(tune_greedy_postprocess),
        "gbdt_n_est": int(gbdt_n_est),
        "gbdt_max_depth": int(gbdt_max_depth),
        "tune_binarizer": bool(tune_binarizer),
        "gbdt_n_est_choices": [int(v) for v in gbdt_n_est_choices],
        "gbdt_max_depth_choices": [int(v) for v in gbdt_max_depth_choices],
        "clamp_reg_to_inv_n": bool(clamp_reg_to_inv_n),
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

        hpo_out = tune_split_optuna(
            X_tv,
            y_tv,
            outer_seed=outer_seed,
            inner_repeats=inner_repeats,
            use_stratify=use_stratify,
            feature_names=feature_names,
            cat_indicator=cat_indicator,
            card_threshold=cat_card_threshold,
            hpo_timeout_sec=hpo_timeout_sec,
            max_hpo_trials=max_hpo_trials,
            hpo_metric=hpo_metric,
            max_depth=max_depth,
            tune_full_depth=tune_full_depth,
            std_penalty_alpha=std_penalty_alpha,
            reg_low=reg_low,
            reg_high=reg_high,
            per_fit_limit_sec=per_fit_limit_sec,
            similar_support=similar_support,
            greedy_postprocess=greedy_postprocess,
            gbdt_n_est=gbdt_n_est,
            gbdt_max_depth=gbdt_max_depth,
            tune_binarizer=tune_binarizer,
            tune_greedy_postprocess=tune_greedy_postprocess,
            clamp_reg_to_inv_n=clamp_reg_to_inv_n,
            gbdt_n_est_choices=gbdt_n_est_choices,
            gbdt_max_depth_choices=gbdt_max_depth_choices,
            show_progress_bar=show_progress_bar,
            suppress_split_warnings=suppress_split_warnings,
            suppress_small_reg_warning=suppress_small_reg_warning,
            lookahead_policy=lookahead_policy,
            lookahead_fixed=lookahead_fixed,
            lookahead_min=lookahead_min,
            lookahead_small_n_threshold=lookahead_small_n_threshold,
            lookahead_small_value=lookahead_small_value,
        )
        study = hpo_out["study"]
        best_params = dict(hpo_out["best_params"])

        final_seed = outer_seed * 10_000_000 + 999
        set_all_seeds(final_seed)

        pp_t0 = time.perf_counter()
        prep = UnifiedOheLooOrNumericPreprocessor(
            feature_names=feature_names,
            cat_indicator=cat_indicator,
            card_threshold=cat_card_threshold,
        )
        Xtv_tr = prep.fit_transform_df(X_tv, y_tv)
        XTe = prep.transform_df(X_test)
        final_preprocess_time_sec = float(time.perf_counter() - pp_t0)

        reg_raw = float(best_params["reg"])
        reg_eff = (
            max(reg_raw, (1.0 / max(1, len(y_tv))) + 1e-12)
            if clamp_reg_to_inv_n
            else reg_raw
        )

        model = SPLIT(
            time_limit=int(final_fit_limit_sec),
            verbose=False,
            reg=float(reg_eff),
            lookahead_depth_budget=int(best_params["lookahead_depth_budget"]),
            full_depth_budget=int(best_params["full_depth_budget"]),
            similar_support=bool(similar_support),
            allow_small_reg=True,
            greedy_postprocess=bool(best_params["greedy_postprocess"]),
            binarize=True,
            gbdt_n_est=int(best_params["gbdt_n_est"]),
            gbdt_max_depth=int(best_params["gbdt_max_depth"]),
        )

        fit_t0 = time.perf_counter()
        with split_warning_context(
            suppress_split_warnings=suppress_split_warnings,
            suppress_small_reg_warning=suppress_small_reg_warning,
        ):
            model.fit(Xtv_tr, y_tv)
        final_fit_time_sec = float(time.perf_counter() - fit_t0)

        inf_t0 = time.perf_counter()
        y_pred = model.predict(XTe)
        y_prob = split_predict_proba_pos(model, XTe)
        final_infer_time_sec = float(time.perf_counter() - inf_t0)

        train_pred = model.predict(Xtv_tr)
        train_prob = split_predict_proba_pos(model, Xtv_tr)
        train_metrics = eval_binary_metrics(y_tv, train_pred, train_prob)
        test_metrics = eval_binary_metrics(y_test, y_pred, y_prob)

        Xtv_bin = model.enc.transform(Xtv_tr)
        XTe_bin = model.enc.transform(XTe)
        root = get_split_root(model)

        complexity = extract_split_complexity_e1(
            root,
            X_train_bool=np.asarray(Xtv_bin, dtype=bool),
            X_test_bool=np.asarray(XTe_bin, dtype=bool),
        )
        complexity["n_features_after_pp"] = int(Xtv_tr.shape[1])
        complexity["hp_full_depth_budget"] = int(best_params["full_depth_budget"])
        complexity["hp_lookahead_depth_budget"] = int(best_params["lookahead_depth_budget"])
        complexity["hp_reg_eff"] = float(reg_eff)
        complexity["hp_reg_raw"] = float(reg_raw)
        complexity["hp_gbdt_n_est"] = int(best_params["gbdt_n_est"])
        complexity["hp_gbdt_max_depth"] = int(best_params["gbdt_max_depth"])
        complexity["hp_greedy_postprocess"] = bool(best_params["greedy_postprocess"])

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
            "hpo_stats": hpo_out["hpo_stats"],
            "time_sec": {
                "cache_preprocess_time": float(hpo_out["cache_preprocess_time_sec"]),
                "hpo_time": float(hpo_out["hpo_time_sec"]),
                "final_preprocess_time": float(final_preprocess_time_sec),
                "final_fit_time": float(final_fit_time_sec),
                "final_infer_time": float(final_infer_time_sec),
            },
            "complexity": complexity,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "lookahead_policy": str(lookahead_policy),
            "lookahead_fixed": int(lookahead_fixed),
            "lookahead_min": int(lookahead_min),
            "lookahead_small_n_threshold": int(lookahead_small_n_threshold),
            "lookahead_small_value": int(lookahead_small_value),
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


def run_split_hpo_many(specs: Iterable[DatasetSpec], *, root_dir: str, outer_splits_dir: Optional[str] = None, **kwargs):
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

            _, summary = run_split_hpo_one(
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
        description="Run SPLIT HPO on UCIrepo/OpenML binary classification datasets."
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
    hpo_group.add_argument("--hpo_metric", choices=SUPPORTED_HPO_METRICS, default="bal_acc")
    hpo_group.add_argument(
        "--max_depth",
        type=int,
        default=4,
        help="Upper bound/full budget for SPLIT. Default fixes full_depth_budget=4.",
    )
    hpo_group.add_argument(
        "--tune_full_depth",
        action="store_true",
        help="Tune full_depth_budget from 1..max_depth. Without this, full_depth_budget is fixed to max_depth.",
    )
    hpo_group.add_argument("--max_hpo_trials", type=int, default=DEFAULT_MAX_HPO_TRIALS)
    hpo_group.add_argument("--hpo_timeout_sec", type=int, default=DEFAULT_HPO_TIMEOUT_SEC)
    hpo_group.add_argument("--std_penalty_alpha", type=float, default=DEFAULT_STD_PENALTY_ALPHA)
    hpo_group.add_argument("--reg_low", type=float, default=DEFAULT_REG_LOW)
    hpo_group.add_argument("--reg_high", type=float, default=DEFAULT_REG_HIGH)
    hpo_group.add_argument("--no_progress_bar", action="store_true")

    split_group = parser.add_argument_group("SPLIT settings")
    split_group.add_argument("--per_fit_limit_sec", type=int, default=DEFAULT_PER_FIT_LIMIT_SEC)
    split_group.add_argument("--final_fit_limit_sec", type=int, default=DEFAULT_FINAL_FIT_LIMIT_SEC)
    split_group.add_argument("--similar_support", action="store_true")
    split_group.add_argument("--greedy_postprocess", action="store_true", help="Use greedy postprocess when --tune_greedy_postprocess is not set.")
    split_group.add_argument("--tune_greedy_postprocess", action="store_true", help="Tune greedy_postprocess over {False, True}.")
    split_group.add_argument("--gbdt_n_est", type=int, default=DEFAULT_GBDT_N_EST, help="Fixed binarizer n_estimators when --tune_binarizer is not set.")
    split_group.add_argument("--gbdt_max_depth", type=int, default=DEFAULT_GBDT_MAX_DEPTH, help="Fixed binarizer max_depth when --tune_binarizer is not set.")
    split_group.add_argument("--tune_binarizer", action="store_true", help="Tune SPLIT threshold-guessing binarizer parameters.")
    split_group.add_argument("--gbdt_n_est_choices", nargs="+", type=int, default=DEFAULT_GBDT_N_EST_CHOICES)
    split_group.add_argument("--gbdt_max_depth_choices", nargs="+", type=int, default=DEFAULT_GBDT_MAX_DEPTH_CHOICES)
    split_group.add_argument("--clamp_reg_to_inv_n", action="store_true", help="Clamp reg to at least 1/n_train. Off by default for a stronger SPLIT search.")

    split_group.add_argument(
        "--suppress_split_warnings",
        action="store_true",
        help="Suppress noisy NumPy RuntimeWarnings emitted inside SPLIT/GOSDT internals.",
    )
    split_group.add_argument(
        "--suppress_small_reg_warning",
        action="store_true",
        help="Suppress SPLIT/GOSDT small-regularization warnings. Cosmetic only; does not clamp reg.",
    )

    split_group.add_argument(
        "--lookahead_policy",
        choices=["fixed", "auto", "tune_capped"],
        default=DEFAULT_LOOKAHEAD_POLICY,
        help=(
            "How to set SPLIT lookahead_depth_budget. "
            "'fixed' uses --lookahead_fixed; "
            "'auto' uses --lookahead_small_value for small datasets and --lookahead_fixed otherwise; "
            "'tune_capped' tunes in a capped range."
        ),
    )

    split_group.add_argument(
        "--lookahead_fixed",
        type=int,
        default=DEFAULT_LOOKAHEAD_FIXED,
        help="Fixed/default lookahead depth. Recommended: 2.",
    )

    split_group.add_argument(
        "--lookahead_min",
        type=int,
        default=DEFAULT_LOOKAHEAD_MIN,
        help="Minimum lookahead depth when full_depth_budget allows it. Recommended: 2.",
    )

    split_group.add_argument(
        "--lookahead_small_n_threshold",
        type=int,
        default=DEFAULT_LOOKAHEAD_SMALL_N_THRESHOLD,
        help="Datasets with n_train below this threshold use --lookahead_small_value under auto/tune_capped.",
    )

    split_group.add_argument(
        "--lookahead_small_value",
        type=int,
        default=DEFAULT_LOOKAHEAD_SMALL_VALUE,
        help="Lookahead value/cap for small datasets. Recommended: 3.",
    )

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
    cv_group.add_argument(
        "--outer_splits_dir",
        type=str,
        default=None,
        help="Optional directory for shared outer split JSONs. Useful to verify/reuse exact splits across models.",
    )

    misc_group = parser.add_argument_group("misc")
    misc_group.add_argument("--cat_card_threshold", type=int, default=DEFAULT_CAT_CARD_THRESHOLD)
    misc_group.add_argument("--root_dir", type=str, default="./split_hpo_runs")
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
    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError("--test_ratio must be in (0, 1)")
    if args.reg_low <= 0 or args.reg_high <= 0 or args.reg_low >= args.reg_high:
        raise ValueError("Require 0 < --reg_low < --reg_high")
    if args.lookahead_fixed < 1:
        raise ValueError("--lookahead_fixed must be >= 1")
    if args.lookahead_min < 1:
        raise ValueError("--lookahead_min must be >= 1")
    if args.lookahead_small_value < 1:
        raise ValueError("--lookahead_small_value must be >= 1")
    if args.lookahead_small_n_threshold < 0:
        raise ValueError("--lookahead_small_n_threshold must be >= 0")

    specs = specs_from_args(args)

    print("Running SPLIT HPO with:")
    print(f"  datasets        = {[asdict(s) for s in specs]}")
    print(f"  hpo_metric      = {args.hpo_metric} ({metric_direction(args.hpo_metric)})")
    print(f"  max_depth       = {args.max_depth}")
    print(f"  tune_full_depth = {args.tune_full_depth}")
    print(f"  tune_binarizer  = {args.tune_binarizer}")
    print(f"  tune_greedy_pp  = {args.tune_greedy_postprocess}")
    print(f"  root_dir        = {args.root_dir}")

    run_split_hpo_many(
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
        tune_full_depth=args.tune_full_depth,
        std_penalty_alpha=args.std_penalty_alpha,
        reg_low=args.reg_low,
        reg_high=args.reg_high,
        per_fit_limit_sec=args.per_fit_limit_sec,
        final_fit_limit_sec=args.final_fit_limit_sec,
        similar_support=args.similar_support,
        greedy_postprocess=args.greedy_postprocess,
        gbdt_n_est=args.gbdt_n_est,
        gbdt_max_depth=args.gbdt_max_depth,
        tune_binarizer=args.tune_binarizer,
        tune_greedy_postprocess=args.tune_greedy_postprocess,
        clamp_reg_to_inv_n=args.clamp_reg_to_inv_n,
        gbdt_n_est_choices=args.gbdt_n_est_choices,
        gbdt_max_depth_choices=args.gbdt_max_depth_choices,
        show_progress_bar=not args.no_progress_bar,
        suppress_split_warnings=args.suppress_split_warnings,
        suppress_small_reg_warning=args.suppress_small_reg_warning,
        lookahead_policy=args.lookahead_policy,
        lookahead_fixed=args.lookahead_fixed,
        lookahead_min=args.lookahead_min,
        lookahead_small_n_threshold=args.lookahead_small_n_threshold,
        lookahead_small_value=args.lookahead_small_value,
    )


if __name__ == "__main__":
    main()
