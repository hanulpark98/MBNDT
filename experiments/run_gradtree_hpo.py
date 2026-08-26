"""
GradTree HPO Script (REVISED for Fair Comparison with MBNDT)

Changes from original:
1. Inner CV: StratifiedShuffleSplit -> StratifiedKFold (match MBNDT)
2. Objective: mean -> mean - 0.5*std (robust objective, match MBNDT)
3. Batch size: removed Optuna batch-fraction tuning; uses strict constant GradTree batch size
4. Optuna reporting: stores fold/partial objective diagnostics; uses NopPruner by default to match the current MBNDT runner
5. Reporting: saves consolidated per-split report.json plus outer_split_summary.csv / mean_std.json
6. CLI: supports MBNDT-style comma-separated --datasets

Usage:
    # Single dataset
    python GradTree_HPO_MBNDT_scheme.py --dataset_type uci --dataset_id 176

    # Multiple datasets
    python GradTree_HPO_MBNDT_scheme.py --datasets openml:55,openml:27 --output_root ./HPO_results_gradtree
"""

import os, sys, gc, time, json, random, warnings

# -----------------------------------------------------------------------------
# Conservative TensorFlow runtime defaults. These must be set before importing
# TensorFlow/GradTree. CLI args and shell env can still override them.
# -----------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_force_compilation_parallelism=1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")
os.environ.setdefault("AUTOGRAPH_VERBOSITY", "0")

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import argparse


# ==============================================================================
# EARLY CUDA / TENSORFLOW CONFIGURATION
# ------------------------------------------------------------------------------
# TensorFlow reads CUDA_VISIBLE_DEVICES during import, so these arguments must be
# parsed before importing TensorFlow or GradTree. The main argparse parser below
# also defines the same options so they appear in --help.
# ==============================================================================
def _preparse_cuda_config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cuda_device",
        type=str,
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        help="CUDA device id(s) visible to TensorFlow, e.g. 0, 1, or 0,1. Use 'all' to leave unchanged.",
    )
    parser.add_argument(
        "--cuda_num",
        type=str,
        default=None,
        help="MBNDT-style alias for --cuda_device. Parsed before TensorFlow import.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Disable CUDA by setting CUDA_VISIBLE_DEVICES=-1 before TensorFlow import.",
    )
    parser.add_argument(
        "--no_tf_allow_growth",
        action="store_true",
        help="Disable TensorFlow GPU memory growth. Default: allow growth.",
    )
    parser.add_argument(
        "--tf_xla_jit",
        type=int,
        choices=[0, 1],
        default=0,
        help="Set TF_XLA_FLAGS=--tf_xla_auto_jit={0,1}. Default: 0.",
    )
    args, _ = parser.parse_known_args()

    selected_cuda = args.cuda_num if args.cuda_num is not None else args.cuda_device

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif str(selected_cuda).lower() not in {"all", "unchanged"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_cuda)

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false" if args.no_tf_allow_growth else "true"
    os.environ["TF_XLA_FLAGS"] = f"--tf_xla_auto_jit={int(args.tf_xla_jit)}"

    return {
        "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu": bool(args.cpu),
        "tf_allow_growth": not bool(args.no_tf_allow_growth),
        "tf_xla_jit": int(args.tf_xla_jit),
    }


CUDA_CONFIG = _preparse_cuda_config()

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from tqdm.auto import tqdm

# CHANGED: Added StratifiedKFold import
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, roc_auc_score,
    recall_score, log_loss,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from mbndt.preprocessing import PreprocessConfig, preprocess_splits
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo
import tensorflow as tf
from tensorflow.keras import backend as K
import logging

tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.config.optimizer.set_jit(False)
# Stronger XLA/JIT hardening. TF_XLA_FLAGS disables auto-clustering, but some
# libraries may still create @tf.function(jit_compile=True). GradTree can be
# sensitive to XLA on some CUDA/TF stacks, where failures abort the process from
# C++ and cannot be caught in Python. This wrapper forces jit_compile=False for
# functions defined after this point, including GradTree internals imported below.
try:
    _original_tf_function = tf.function
    def _tf_function_no_xla(*args, **kwargs):
        if kwargs.get("jit_compile", False):
            kwargs = dict(kwargs)
            kwargs["jit_compile"] = False
        return _original_tf_function(*args, **kwargs)
    tf.function = _tf_function_no_xla
except Exception as e:
    print(f"[WARN] Could not monkeypatch tf.function jit_compile: {e}", flush=True)

try:
    tf.config.optimizer.set_experimental_options({
        "layout_optimizer": False,
        "constant_folding": False,
        "shape_optimization": False,
        "remapping": False,
        "arithmetic_optimization": False,
        "dependency_optimization": False,
        "loop_optimization": False,
        "function_optimization": False,
        "debug_stripper": False,
        "scoped_allocator_optimization": False,
        "pin_to_host_optimization": False,
        "implementation_selector": False,
        "auto_mixed_precision": False,
    })
except Exception as e:
    print(f"[WARN] Could not set TF optimizer experimental options: {e}", flush=True)

gpus = tf.config.list_physical_devices('GPU')
if CUDA_CONFIG.get("tf_allow_growth", True):
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("Could not set memory growth:", e)
print(
    f"[CUDA CONFIG] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
    f"| physical_gpus={len(gpus)} | allow_growth={CUDA_CONFIG.get('tf_allow_growth')} "
    f"| xla_jit={CUDA_CONFIG.get('tf_xla_jit')} "
    f"| TF_XLA_FLAGS={os.environ.get('TF_XLA_FLAGS')} "
    f"| TF_CPP_MIN_LOG_LEVEL={os.environ.get('TF_CPP_MIN_LOG_LEVEL')} "
    f"| XLA_FLAGS={os.environ.get('XLA_FLAGS')}"
)

# Import GradTree only after TensorFlow CUDA visibility and memory-growth settings
# have been applied.
from GradTree import GradTree

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=".*n_quantiles.*greater than the total number of samples.*",
    category=UserWarning,
)


# ==============================================================================
# DEFAULT CONFIGURATION
# ==============================================================================
DEFAULT_CONFIG = {
    "K_OUTER_SPLITS": 5,
    "TEST_RATIO": 0.2,
    "HPO_TIMEOUT_SEC": 60 * 60,  # 1 hour per outer split
    "N_HPO_TRIALS": 10000,
    "INNER_VAL_RATIO": 0.2,
    "INNER_REPEATS": 5,
    "INNER_ES_RATIO": 0.2,

    # MBNDT-style public metric names are accepted. They are mapped to
    # GradTree's metric keys internally by normalize_metric_key().
    "MONITOR_METRIC": "balanced_acc",
    "STD_PENALTY_ALPHA": 0.5,
    "BASE_SEED": 0,
    "USE_STRATIFY": True,
    "USE_SMOTE": False,
    "MAX_DEPTH": 4,

    # Strict constant GradTree batch size. Batch size is NOT tuned by Optuna.
    # BATCH_MIN/BATCH_MAX/MIN_BATCHES_PER_EPOCH are accepted only for CLI
    # compatibility with older commands; they do not affect GradTree training.
    "BATCH_SIZE": 128,
    "BATCH_MIN": 16,
    "BATCH_MAX": 4096,
    "MIN_BATCHES_PER_EPOCH": 12,

    # Match the current MBNDT runner by default: report partial values,
    # but do not let Optuna prune unless explicitly enabled.
    "USE_OPTUNA_PRUNER": False,
    "SHOW_PROGRESS_BAR": False,
    "LOG_EVERY_N_TRIALS": 10,

    # Optional offline/local dataset store. If provided, OpenML/UCI datasets are
    # loaded from exported pickle payloads instead of remote APIs.
    "LOCAL_DATASET_DIR": None,
}

EVAL_METRIC_CHOICES = ["balanced_acc", "aucroc", "acc", "f1_macro", "bal_acc", "auroc"]

GRADTREE_METRIC_MAP = {
    "balanced_acc": "bal_acc",
    "aucroc": "auroc",
    "bal_acc": "bal_acc",
    "auroc": "auroc",
    "acc": "acc",
    "f1_macro": "f1_macro",
}


# ==============================================================================
# HELPERS
# ==============================================================================
def _infer_cat_num_cols(X: pd.DataFrame, cat_indicator):
    cols = X.columns.tolist()
    if cat_indicator is None:
        return [], cols
    cat_indicator = np.asarray(cat_indicator).astype(bool)
    if len(cat_indicator) != len(cols):
        raise ValueError(f"cat_indicator length mismatch: {len(cat_indicator)} vs n_cols={len(cols)}")
    cat_cols = [c for c, is_cat in zip(cols, cat_indicator) if is_cat]
    num_cols = [c for c, is_cat in zip(cols, cat_indicator) if not is_cat]
    return cat_cols, num_cols


def _to_numpy_y(y):
    return np.asarray(y)


def seed_everything_gradtree(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def flatten_for_row(prefix, d):
    """Flatten dict -> {'prefix/key': value} for CSV columns"""
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        kk = f"{prefix}{k}"
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[kk] = float(v)
        elif isinstance(v, bool):
            out[kk] = int(v)
        elif v is None:
            out[kk] = np.nan
        else:
            out[kk] = str(v)
    return out


def _to_jsonable(x):
    """Convert torch/numpy objects to JSON-serializable format."""
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, set):
        return sorted([_to_jsonable(v) for v in x])
    return str(x)


def save_json(path: Path, obj, indent=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=indent)


def _to_dense_float32(Xm):
    """Handle numpy / sparse outputs from sklearn preprocessors."""
    if hasattr(Xm, "toarray"):
        Xm = Xm.toarray()
    return np.asarray(Xm, dtype=np.float32)


def normalize_metric_key(metric: str) -> str:
    """Map MBNDT-style public metric names to GradTree metric keys."""
    metric = str(metric)
    if metric not in GRADTREE_METRIC_MAP:
        raise KeyError(
            f"Unknown monitor metric {metric!r}. "
            f"Allowed metrics: {sorted(GRADTREE_METRIC_MAP.keys())}"
        )
    return GRADTREE_METRIC_MAP[metric]


def compute_train_batch_size(
    n_train: int,
    *,
    batch_size: int,
    batch_min: int,
    batch_max: int,
    min_batches_per_epoch: int,
) -> int:
    """Return a strict constant GradTree train batch size.

    For GradTree baselines, the user requested --batch_size to mean exactly
    that value for every HPO fold and final refit. Therefore n_train,
    batch_min, batch_max, and min_batches_per_epoch are intentionally ignored
    except that we validate the requested batch is positive. Keras/TensorFlow
    can accept batch_size > n_train; it will simply run one partial/full batch.
    """
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return batch_size


# ==============================================================================
# MBNDT-STYLE REPORT FLATTENING / EXPORT
# ==============================================================================
def _flatten_gradtree_report(r: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one consolidated per-split report into one CSV row."""
    row = {
        "outer_seed": r.get("outer_seed", None),
        "best_value_inner_objective": r.get("best_value_inner_objective", np.nan),
        "best_inner_mean": r.get("best_inner_mean", np.nan),
        "best_inner_std": r.get("best_inner_std", np.nan),
    }

    for k, v in (r.get("metric_config", {}) or {}).items():
        row[f"metric/{k}"] = v

    for k, v in (r.get("best_hps", {}) or {}).items():
        row[f"hp/{k}"] = v

    for k, v in (r.get("best_args", {}) or {}).items():
        row[f"arg/{k}"] = v

    for k, v in (r.get("timing", {}) or {}).items():
        row[f"timing/{k}"] = v

    sub = r.get("gradtree", {}) or {}
    perf = sub.get("perf", {}) or {}
    for split_name in ("train", "val", "test"):
        for k, v in (perf.get(split_name, {}) or {}).items():
            row[f"gradtree/{split_name}/{k}"] = v

    for k, v in (sub.get("structure_and_paths", {}) or {}).items():
        row[f"gradtree/cx/{k}"] = v

    return row


def export_outer_split_summary(all_results: List[Dict[str, Any]], artifact_root: Path):
    """Save MBNDT-style per-split CSV and mean/std JSON."""
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    rows = [_flatten_gradtree_report(r) for r in all_results]
    df = pd.DataFrame(rows)

    out_csv = artifact_root / "outer_split_summary.csv"
    df.to_csv(out_csv, index=False)

    summary = {}
    for col in df.columns:
        if col == "outer_seed":
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue
        summary[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
        }

    depth_counts = {}
    if "hp/depth" in df.columns:
        depth_counts = {str(k): int(v) for k, v in df["hp/depth"].value_counts(dropna=False).to_dict().items()}

    save_json(artifact_root / "outer_split_summary_mean_std.json", summary)
    save_json(artifact_root / "outer_split_depth_counts.json", depth_counts)

    # Backward-compatible names for older aggregation scripts.
    df.to_csv(artifact_root / "outer_reports_per_split.csv", index=False)
    save_json(artifact_root / "outer_reports_mean_std.json", summary)

    print(f"[Saved] {out_csv}")
    print(f"[Saved] {artifact_root / 'outer_split_summary_mean_std.json'}")
    return df, summary


# ==============================================================================
# METRICS AND MODEL HELPERS
# ==============================================================================
def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def gradtree_predict_proba_pos(model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Returns P(y=1) shape [N,].

    Conservative version: avoid wrapping X in tf.data.Dataset first. Repeated
    Dataset graph construction across Optuna folds can trigger more TensorFlow
    graph/XLA compilation work on some servers. GradTree normally accepts numpy
    arrays directly, so we try that first.
    """
    X = np.asarray(X, dtype=np.float32)

    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X, batch_size=batch_size)
        except TypeError:
            p = model.predict_proba(X)
        except Exception:
            p = None
        if p is not None:
            p = np.asarray(p)
            if p.ndim == 2 and p.shape[1] >= 2:
                return p[:, 1].astype(np.float64)
            if p.ndim == 1:
                if (p.min() < 0.0) or (p.max() > 1.0):
                    return _sigmoid(p).astype(np.float64)
                return p.astype(np.float64)

    if hasattr(model, "predict"):
        try:
            out = model.predict(X, batch_size=batch_size)
        except TypeError:
            out = model.predict(X)
        except Exception:
            out = None
        if out is not None:
            out = np.asarray(out)
            if out.ndim == 2 and out.shape[1] >= 2:
                return out[:, 1].astype(np.float64)
            if out.ndim == 1:
                if (out.min() < 0.0) or (out.max() > 1.0):
                    return _sigmoid(out).astype(np.float64)
                return out.astype(np.float64)

    # Last-resort fallback to tf.data.Dataset only if numpy prediction failed.
    try:
        ds = tf.data.Dataset.from_tensor_slices(X).batch(batch_size, drop_remainder=False)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(ds)
        else:
            p = model.predict(ds)
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1].astype(np.float64)
        if p.ndim == 1:
            if (p.min() < 0.0) or (p.max() > 1.0):
                return _sigmoid(p).astype(np.float64)
            return p.astype(np.float64)
    except Exception:
        pass

    return np.zeros((X.shape[0],), dtype=np.float64)


def compute_binary_metrics(y_true: np.ndarray, p_pos: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    p_pos = np.asarray(p_pos).astype(np.float64).reshape(-1)
    y_hat = (p_pos >= thr).astype(int)

    f1m = float(f1_score(y_true, y_hat, average="macro"))
    acc = float(accuracy_score(y_true, y_hat))

    try:
        auroc = float(roc_auc_score(y_true, p_pos))
    except Exception:
        auroc = float("nan")

    try:
        logloss = float(log_loss(y_true, p_pos))
    except Exception:
        logloss = float("nan")

    try:
        mcc = float(matthews_corrcoef(y_true, y_hat))
    except Exception:
        mcc = float("nan")

    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")

    return {
        "f1_macro": f1m,
        "acc": acc,
        "bal_acc": float(balanced_accuracy_score(y_true, y_hat)),
        "auroc": auroc,
        "logloss": logloss,
        "mcc": mcc,
        "precision": prec,
        "sensitivity": sens,
        "specificity": spec,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def gradtree_complexity(depth: int) -> Dict[str, int]:
    """GradTree complexity based on binary tree depth"""
    L = int(2 ** depth)
    I = int(L - 1)
    return {"depth": int(depth), "leaves": L, "internal": I, "nodes": int(L + I)}


# ==============================================================================
# HPO PARAMETER SUGGESTION
# CHANGED: Learning rates from step to log scale
# ==============================================================================
def suggest_params_gradtree(
    trial: optuna.Trial,
    *,
    max_depth: int = 4,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Suggest only GradTree-native hyperparameters.

    Batch size is intentionally excluded from Optuna and is set by
    compute_train_batch_size(), matching the MBNDT runner's scheme format.
    """
    params = {
        "depth": trial.suggest_int("depth", 1, int(max_depth), step=1),

        # CHANGED: from step=0.005 to log=True
        "learning_rate_index": trial.suggest_float("learning_rate_index", 1e-3, 1e-1, log=True),
        "learning_rate_values": trial.suggest_float("learning_rate_values", 1e-3, 1e-1, log=True),
        "learning_rate_leaf": trial.suggest_float("learning_rate_leaf", 1e-3, 1e-1, log=True),

        "optimizer": "SWA",
        "cosine_decay_steps": 0,
        "initializer": "RandomNormal",

        "loss": "crossentropy",
        "focal_loss": trial.suggest_categorical("focal_loss", [True, False]),

        "temperature": 0.0,
        "from_logits": True,
        "apply_class_balancing": True,

        "polyLoss": trial.suggest_categorical("polyLoss", [True, False]),
        "polyLossEpsilon": trial.suggest_int("polyLossEpsilon", 0, 5, step=1),
    }

    args = {
        "epochs": 250,
        "early_stopping_epochs": 25,
        # batch_size is filled per-fold/final-fit after preprocessing, because it
        # depends on the effective training-set size.
        "cat_idx": [],
        "objective": "binary",
        "random_seed": 42,
        "verbose": 0,
    }
    return params, args


# ==============================================================================
# PREPROCESSING CONFIG
# ==============================================================================
def make_pp_cfg(seed: int, *, to_torch: bool, use_stratify: bool, use_smote: bool):
    return PreprocessConfig(
        random_state=seed,
        stratify=use_stratify,
        use_smote=use_smote,
        impute="median",
        encode_categoricals=True,
        cardinality_threshold=10,
        te_sigma=0.0,
        numerical_stability=False,
        scale=True,
        to_torch=to_torch,
    )


# ==============================================================================
# SINGLE OUTER SPLIT RUNNER
# REVISED: StratifiedKFold + Robust Objective + Pruner
# ==============================================================================
def run_one_outer_split_gradtree(
    *,
    outer_seed: int,
    tv_idx,
    te_idx,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names,
    cat_indicator,
    ARTIFACT_ROOT: Path,
    INNER_VAL_RATIO: float,
    INNER_REPEATS: int,
    INNER_ES_RATIO: float,
    USE_STRATIFY: bool,
    USE_SMOTE: bool,
    HPO_TIMEOUT_SEC: int,
    N_HPO_TRIALS: int,
    MONITOR_METRIC: str,
    MAX_DEPTH: int = 4,
    BATCH_SIZE: int = 128,
    BATCH_MIN: int = 16,
    BATCH_MAX: int = 4096,
    MIN_BATCHES_PER_EPOCH: int = 12,
    STD_PENALTY_ALPHA: float = 0.5,
    USE_OPTUNA_PRUNER: bool = False,
    SHOW_PROGRESS_BAR: bool = False,
    LOG_EVERY_N_TRIALS: int = 10,
):
    y = np.asarray(y).reshape(-1)
    cat_cols, num_cols = _infer_cat_num_cols(X, cat_indicator)

    X_tv = X.iloc[tv_idx].reset_index(drop=True)
    y_tv = y[tv_idx]
    X_test = X.iloc[te_idx].reset_index(drop=True)
    y_test = y[te_idx]

    # -------------------------
    # CHANGED: Inner CV using StratifiedKFold (match MBNDT)
    # -------------------------
    inner_seed = outer_seed + 111
    splitter = (
        StratifiedKFold(n_splits=INNER_REPEATS, shuffle=True, random_state=inner_seed)
        if USE_STRATIFY else
        KFold(n_splits=INNER_REPEATS, shuffle=True, random_state=inner_seed)
    )
    inner_splits = list(splitter.split(np.zeros((len(y_tv), 1)), y_tv))

    es_seeds = [inner_seed + 1000 + rep_id for rep_id in range(INNER_REPEATS)]

    rep_raw = []
    for rep_id, (idx_tr, idx_va) in enumerate(inner_splits):
        X_tr_raw = X_tv.iloc[idx_tr]
        y_tr_raw = y_tv[idx_tr]
        X_va_raw = X_tv.iloc[idx_va]
        y_va_raw = y_tv[idx_va]

        X_main_raw, X_es_raw, y_main_raw, y_es_raw = train_test_split(
            X_tr_raw, y_tr_raw,
            test_size=INNER_ES_RATIO,
            random_state=es_seeds[rep_id],
            stratify=y_tr_raw if USE_STRATIFY else None
        )
        rep_raw.append((X_main_raw, y_main_raw, X_es_raw, y_es_raw, X_va_raw, y_va_raw))

    gradtree_metric = normalize_metric_key(MONITOR_METRIC)

    # Strict constant GradTree batch size: same value for every inner fold.
    n_train_ref = min(len(item[1]) for item in rep_raw)
    hpo_batch_size = compute_train_batch_size(
        n_train_ref,
        batch_size=BATCH_SIZE,
        batch_min=BATCH_MIN,
        batch_max=BATCH_MAX,
        min_batches_per_epoch=MIN_BATCHES_PER_EPOCH,
    )
    print(f"[Batch] strict constant (not tuned): n_train_ref={n_train_ref}, batch_size={hpo_batch_size}")

    # -------------------------
    # Objective with robust scoring (match MBNDT)
    # -------------------------
    def objective(trial: optuna.Trial) -> float:
        params, base_args = suggest_params_gradtree(
            trial,
            max_depth=MAX_DEPTH,
        )

        trial_seed = int(outer_seed * 1_000_000 + trial.number + 7)
        seed_everything_gradtree(trial_seed)
        base_args = dict(base_args)
        base_args["random_seed"] = trial_seed
        trial.set_user_attr("batch_size", int(hpo_batch_size))
        trial.set_user_attr("monitor_metric", str(MONITOR_METRIC))
        trial.set_user_attr("gradtree_metric", str(gradtree_metric))

        scores = []

        for rep_id, (X_main_raw, y_main_raw, X_es_raw, y_es_raw, X_va_raw, y_va_raw) in enumerate(rep_raw):
            pp_seed = inner_seed + rep_id
            cfg_pp = make_pp_cfg(pp_seed, to_torch=False, use_stratify=USE_STRATIFY, use_smote=USE_SMOTE)

            res = preprocess_splits(
                X_train=X_main_raw, y_train=y_main_raw,
                X_valid=X_es_raw, y_valid=y_es_raw,
                X_test=X_va_raw, y_test=y_va_raw,
                cfg=cfg_pp,
                cat_cols=cat_cols, num_cols=num_cols
            )

            X_tr = _to_dense_float32(res.X_train)
            y_tr = np.asarray(res.y_train).astype(int).reshape(-1)
            X_es = _to_dense_float32(res.X_valid)
            y_es = np.asarray(res.y_valid).astype(int).reshape(-1)
            X_va = _to_dense_float32(res.X_test)
            y_va = np.asarray(res.y_test).astype(int).reshape(-1)

            args_fold = dict(base_args)
            args_fold["batch_size"] = int(hpo_batch_size)

            # Conservative cleanup before every GradTree fit. This reduces graph
            # accumulation across Optuna folds/trials in TensorFlow.
            tf.keras.backend.clear_session()
            gc.collect()

            model = GradTree(params=params, args=args_fold)
            model.fit(X_train=X_tr, y_train=y_tr, X_val=X_es, y_val=y_es)

            p_va = gradtree_predict_proba_pos(model, X_va)
            m = compute_binary_metrics(y_va, p_va, thr=0.5)
            score = float(m.get(gradtree_metric, m.get("bal_acc", 0.0)))
            scores.append(score)

            # Optuna reporting: store rich fold diagnostics, but report only after
            # at least two folds so the std-penalized objective is meaningful.
            mu = float(np.mean(scores))
            sigma = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            robust_partial = mu - STD_PENALTY_ALPHA * sigma
            trial.set_user_attr(f"fold_{rep_id}_{gradtree_metric}", score)
            trial.set_user_attr(f"partial_mean_after_fold_{rep_id}", mu)
            trial.set_user_attr(f"partial_std_after_fold_{rep_id}", sigma)
            trial.set_user_attr(f"partial_objective_after_fold_{rep_id}", robust_partial)

            if rep_id >= 1:
                trial.report(robust_partial, step=rep_id)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            try:
                del model
            except Exception:
                pass
            tf.keras.backend.clear_session()
            gc.collect()

        # CHANGED: Return robust objective (match MBNDT)
        mu = float(np.mean(scores))
        sigma = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        robust = mu - STD_PENALTY_ALPHA * sigma

        # Store diagnostics
        trial.set_user_attr("inner_mean", mu)
        trial.set_user_attr("inner_std", sigma)
        trial.set_user_attr("inner_objective", robust)
        trial.set_user_attr("depth", int(params["depth"]))
        trial.set_user_attr("leaves", int(2 ** int(params["depth"])))
        trial.set_user_attr("batch_size", int(hpo_batch_size))

        return robust

    # -------------------------
    # Match MBNDT scheme by default: deterministic TPE seed and no pruning.
    # Enable MedianPruner only if explicitly requested.
    # -------------------------
    sampler = optuna.samplers.TPESampler(seed=outer_seed + 1000)
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1, interval_steps=1)
        if USE_OPTUNA_PRUNER else
        optuna.pruners.NopPruner()
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    # Notebook-friendly progress:
    # - If SHOW_PROGRESS_BAR=True, use ONE manual tqdm bar updated by callback.
    #   Do not use Optuna's built-in progress bar because it can duplicate output
    #   when stdout is streamed through notebooks.
    # - If SHOW_PROGRESS_BAR=False, print one compact line every LOG_EVERY_N_TRIALS.
    pbar = None

    def _format_trial_value(v):
        try:
            if v is None or not np.isfinite(float(v)):
                return "nan"
            return f"{float(v):.6f}"
        except Exception:
            return "nan"

    def _clean_trial_callback(study_: optuna.Study, trial_: optuna.trial.FrozenTrial):
        n_done = len(study_.trials)
        best = study_.best_trial
        inner_mean = trial_.user_attrs.get("inner_mean", float("nan"))
        inner_std = trial_.user_attrs.get("inner_std", float("nan"))
        depth = trial_.user_attrs.get("depth", trial_.params.get("depth", "?"))

        if pbar is not None:
            # Move the single bar to the current completed-trial count.
            delta = n_done - pbar.n
            if delta > 0:
                pbar.update(delta)
            elapsed = time.perf_counter() - hpo_t0
            pbar.set_postfix_str(
                f"best=#{best.number}:{best.value:.6f} "
                f"last=#{trial_.number}:{_format_trial_value(trial_.value)} "
                f"mean={inner_mean:.4f} std={inner_std:.4f} "
                f"depth={depth} elapsed={elapsed:.0f}/{HPO_TIMEOUT_SEC}s",
                refresh=True,
            )
            return

        # Fallback compact text logging.
        log_every = max(1, int(LOG_EVERY_N_TRIALS))
        should_log = (n_done == 1) or (n_done % log_every == 0) or (trial_.state.name != "COMPLETE")
        if not should_log:
            return
        print(
            f"[Trial {trial_.number:04d}] state={trial_.state.name} "
            f"value={_format_trial_value(trial_.value)} "
            f"mean={inner_mean:.6f} std={inner_std:.6f} depth={depth} | "
            f"best=#{best.number} {best.value:.6f}",
            flush=True,
        )

    hpo_t0 = time.perf_counter()
    if bool(SHOW_PROGRESS_BAR):
        pbar = tqdm(
            total=int(N_HPO_TRIALS),
            desc=f"HPO outer={outer_seed}",
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout,
            mininterval=1.0,
        )
    try:
        study.optimize(
            objective,
            n_trials=N_HPO_TRIALS,
            timeout=HPO_TIMEOUT_SEC,
            n_jobs=1,
            show_progress_bar=False,
            callbacks=[_clean_trial_callback],
        )
    finally:
        if pbar is not None:
            pbar.close()
    hpo_time_sec = float(time.perf_counter() - hpo_t0)

    best_params_sampled = study.best_params
    best_params_full, best_args_full = suggest_params_gradtree(
        study.best_trial,
        max_depth=MAX_DEPTH,
    )

    # Final refit
    final_seed = outer_seed * 10_000_000 + 999  # CHANGED: match MBNDT seed pattern
    seed_everything_gradtree(final_seed)

    X_main_raw, X_es_raw, y_main_raw, y_es_raw = train_test_split(
        X_tv, y_tv,
        test_size=INNER_ES_RATIO,
        random_state=final_seed + 777,  # CHANGED: match MBNDT
        stratify=y_tv if USE_STRATIFY else None
    )

    cfg_pp = make_pp_cfg(final_seed, to_torch=False, use_stratify=USE_STRATIFY, use_smote=USE_SMOTE)

    res = preprocess_splits(
        X_train=X_main_raw, y_train=y_main_raw,
        X_valid=X_es_raw, y_valid=y_es_raw,
        X_test=X_test, y_test=y_test,
        cfg=cfg_pp,
        cat_cols=cat_cols, num_cols=num_cols
    )

    X_tr = _to_dense_float32(res.X_train)
    y_tr = np.asarray(res.y_train).astype(int).reshape(-1)
    X_es = _to_dense_float32(res.X_valid)
    y_es = np.asarray(res.y_valid).astype(int).reshape(-1)
    X_te = _to_dense_float32(res.X_test)
    y_te = np.asarray(res.y_test).astype(int).reshape(-1)

    final_batch_size = compute_train_batch_size(
        len(y_tr),
        batch_size=BATCH_SIZE,
        batch_min=BATCH_MIN,
        batch_max=BATCH_MAX,
        min_batches_per_epoch=MIN_BATCHES_PER_EPOCH,
    )
    print(f"[Batch] final fit strict constant: n_train={len(y_tr)}, batch_size={final_batch_size}")

    best_args_full = dict(best_args_full)
    best_args_full["random_seed"] = final_seed
    best_args_full["batch_size"] = int(final_batch_size)

    best_params_report = dict(best_params_full)
    best_params_report["hpo_batch_size"] = int(hpo_batch_size)
    best_params_report["final_batch_size"] = int(final_batch_size)
    best_params_report["nominal_leaves"] = int(2 ** int(best_params_full["depth"]))

    tf.keras.backend.clear_session()
    gc.collect()
    model = GradTree(params=best_params_full, args=best_args_full)

    t0 = time.perf_counter()
    model.fit(X_train=X_tr, y_train=y_tr, X_val=X_es, y_val=y_es)
    final_fit_time_sec = float(time.perf_counter() - t0)

    # Train / ES-val / test metrics
    p_tr = gradtree_predict_proba_pos(model, X_tr)
    train_metrics = compute_binary_metrics(y_tr, p_tr, thr=0.5)

    p_es = gradtree_predict_proba_pos(model, X_es)
    val_metrics = compute_binary_metrics(y_es, p_es, thr=0.5)

    p_te = gradtree_predict_proba_pos(model, X_te)
    test_metrics = compute_binary_metrics(y_te, p_te, thr=0.5)

    # Complexity: GradTree is a native binary tree, so this is structural capacity.
    cx = gradtree_complexity(int(best_params_full["depth"]))

    # Save per-outer artifacts
    outer_dir = Path(ARTIFACT_ROOT) / f"outer_{outer_seed:02d}"
    outer_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "outer_seed": int(outer_seed),
        "best_hps": best_params_report,
        "best_args": best_args_full,
        "metric_config": {
            "monitor_metric": str(MONITOR_METRIC),
            "gradtree_metric": str(gradtree_metric),
            "std_penalty_alpha": float(STD_PENALTY_ALPHA),
            "hpo_batch_size": int(hpo_batch_size),
            "final_batch_size": int(final_batch_size),
            "batch_size": int(BATCH_SIZE),
            "batch_min": int(BATCH_MIN),
            "batch_max": int(BATCH_MAX),
            "min_batches_per_epoch_ignored_for_gradtree_constant_batch": int(MIN_BATCHES_PER_EPOCH),
            "batch_policy": "strict_constant",
            "use_optuna_pruner": bool(USE_OPTUNA_PRUNER),
        },
        "best_value_inner_objective": float(study.best_value),
        "best_inner_mean": float(study.best_trial.user_attrs.get("inner_mean", np.nan)),
        "best_inner_std": float(study.best_trial.user_attrs.get("inner_std", np.nan)),
        "timing": {
            "hpo_time_sec": float(hpo_time_sec),
            "final_fit_time_sec": float(final_fit_time_sec),
        },
        "gradtree": {
            "perf": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            },
            "structure_and_paths": cx,
        },
    }

    save_json(outer_dir / "report.json", report)
    save_json(outer_dir / "best_params_sampled.json", best_params_sampled)
    save_json(outer_dir / "best_params.json", best_params_report)
    save_json(outer_dir / "best_args.json", best_args_full)
    save_json(outer_dir / "train_metrics.json", train_metrics)
    save_json(outer_dir / "val_metrics.json", val_metrics)
    save_json(outer_dir / "test_metrics.json", test_metrics)
    save_json(outer_dir / "complexity.json", cx)

    tf.keras.backend.clear_session()
    gc.collect()

    return report


# ==============================================================================
# MAIN RUNNER FOR SINGLE DATASET
# ==============================================================================
def run_gradtree_full(
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names,
    cat_indicator,
    ARTIFACT_ROOT: Path,
    K_OUTER_SPLITS: int,
    TEST_RATIO: float,
    HPO_TIMEOUT_SEC: int,
    N_HPO_TRIALS: int,
    INNER_VAL_RATIO: float,
    INNER_REPEATS: int,
    INNER_ES_RATIO: float,
    MONITOR_METRIC: str,
    BASE_SEED: int,
    USE_STRATIFY: bool,
    USE_SMOTE: bool = False,
    MAX_DEPTH: int = 4,
    BATCH_SIZE: int = 128,
    BATCH_MIN: int = 16,
    BATCH_MAX: int = 4096,
    MIN_BATCHES_PER_EPOCH: int = 12,
    STD_PENALTY_ALPHA: float = 0.5,
    USE_OPTUNA_PRUNER: bool = False,
    SHOW_PROGRESS_BAR: bool = False,
    LOG_EVERY_N_TRIALS: int = 10,
):
    ARTIFACT_ROOT = Path(ARTIFACT_ROOT)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    run_config = {
        "model": "GradTree",
        "cuda_config": CUDA_CONFIG,
        "k_outer_splits": int(K_OUTER_SPLITS),
        "test_ratio": float(TEST_RATIO),
        "hpo_timeout_sec": int(HPO_TIMEOUT_SEC),
        "n_hpo_trials": int(N_HPO_TRIALS),
        "inner_val_ratio": float(INNER_VAL_RATIO),
        "inner_repeats": int(INNER_REPEATS),
        "inner_es_ratio": float(INNER_ES_RATIO),
        "hpo_metric": str(MONITOR_METRIC),
        "gradtree_metric": normalize_metric_key(MONITOR_METRIC),
        "base_seed": int(BASE_SEED),
        "use_stratify": bool(USE_STRATIFY),
        "use_smote": bool(USE_SMOTE),
        "max_depth": int(MAX_DEPTH),
        "batch_size": int(BATCH_SIZE),
        "batch_min": int(BATCH_MIN),
        "batch_max": int(BATCH_MAX),
        "min_batches_per_epoch_ignored_for_gradtree_constant_batch": int(MIN_BATCHES_PER_EPOCH),
        "batch_policy": "strict_constant",
        "std_penalty_alpha": float(STD_PENALTY_ALPHA),
        "use_optuna_pruner": bool(USE_OPTUNA_PRUNER),
        "n_samples": int(len(y)),
        "n_features_raw": int(X.shape[1]),
    }
    save_json(ARTIFACT_ROOT / "run_config.json", run_config)

    y = np.asarray(y).reshape(-1)

    outer_splitter = (
        StratifiedShuffleSplit(n_splits=K_OUTER_SPLITS, test_size=TEST_RATIO, random_state=BASE_SEED)
        if USE_STRATIFY else
        ShuffleSplit(n_splits=K_OUTER_SPLITS, test_size=TEST_RATIO, random_state=BASE_SEED)
    )
    outer_splits = list(outer_splitter.split(np.zeros((len(y), 1)), y))

    all_results = []
    for split_idx, (tv_idx, te_idx) in enumerate(outer_splits):
        outer_seed = BASE_SEED + split_idx

        out = run_one_outer_split_gradtree(
            outer_seed=outer_seed,
            tv_idx=tv_idx,
            te_idx=te_idx,
            X=X, y=y,
            feature_names=feature_names,
            cat_indicator=cat_indicator,
            ARTIFACT_ROOT=ARTIFACT_ROOT,
            INNER_VAL_RATIO=INNER_VAL_RATIO,
            INNER_REPEATS=INNER_REPEATS,
            INNER_ES_RATIO=INNER_ES_RATIO,
            USE_STRATIFY=USE_STRATIFY,
            USE_SMOTE=USE_SMOTE,
            HPO_TIMEOUT_SEC=HPO_TIMEOUT_SEC,
            N_HPO_TRIALS=N_HPO_TRIALS,
            MONITOR_METRIC=MONITOR_METRIC,
            MAX_DEPTH=MAX_DEPTH,
            BATCH_SIZE=BATCH_SIZE,
            BATCH_MIN=BATCH_MIN,
            BATCH_MAX=BATCH_MAX,
            MIN_BATCHES_PER_EPOCH=MIN_BATCHES_PER_EPOCH,
            STD_PENALTY_ALPHA=STD_PENALTY_ALPHA,
            USE_OPTUNA_PRUNER=USE_OPTUNA_PRUNER,
            SHOW_PROGRESS_BAR=SHOW_PROGRESS_BAR,
            LOG_EVERY_N_TRIALS=LOG_EVERY_N_TRIALS,
        )
        all_results.append(out)

        print(f"\n[GradTree Outer split {split_idx}] seed={outer_seed}")
        print("  Best inner objective:", out["best_value_inner_objective"])
        print("  Best params:", out["best_hps"])
        print("  Test metrics:", out["gradtree"]["perf"]["test"])
        print("  Complexity:", out["gradtree"]["structure_and_paths"])

    df, summary = export_outer_split_summary(all_results, ARTIFACT_ROOT)
    print("\nSaved:", ARTIFACT_ROOT / "outer_split_summary.csv")

    return all_results, df, summary


# ==============================================================================

# DATASET LOADING HELPERS
# ==============================================================================
def _dataset_cache_key(source: str, dataset_id) -> str:
    safe_id = str(dataset_id).replace("/", "_").replace(" ", "_").replace(":", "_")
    return f"{source}_{safe_id}"


def _local_dataset_prefix(source: str, dataset_id, local_dataset_dir: Optional[str]) -> Optional[Path]:
    if local_dataset_dir is None or str(local_dataset_dir).strip() == "":
        return None
    return Path(local_dataset_dir) / _dataset_cache_key(source, dataset_id)


def _load_local_dataset_payload(source: str, dataset_id, local_dataset_dir: Optional[str]):
    """Load a frozen dataset exported by export_datasets_for_gradtree_mbndt_csv_v6.py.

    v6 uses CSV.GZ + JSON instead of pickle because pickle is not portable across
    NumPy/Pandas major versions. We intentionally fail on old .pkl files with a
    clear message instead of trying to unpickle incompatible payloads such as
    files created with NumPy 2.x and read under NumPy 1.x.
    """
    prefix = _local_dataset_prefix(source, dataset_id, local_dataset_dir)
    if prefix is None:
        return None

    meta_path = prefix.with_suffix(".meta.json")
    x_path = prefix.with_suffix(".X.csv.gz")
    y_path = prefix.with_suffix(".y.csv.gz")
    old_pickle_path = prefix.with_suffix(".pkl")

    if not meta_path.exists() or not x_path.exists() or not y_path.exists():
        if old_pickle_path.exists():
            raise RuntimeError(
                f"Found old pickle dataset file {old_pickle_path}, but this runner expects "
                "portable CSV/JSON frozen datasets. Re-export on the working server using "
                "export_datasets_for_gradtree_mbndt_csv_v6.py. The pickle error you saw "
                "comes from NumPy/Pandas version incompatibility."
            )
        raise FileNotFoundError(
            f"Local frozen dataset files not found for key={prefix.name} under {Path(local_dataset_dir)}. "
            f"Expected: {meta_path.name}, {x_path.name}, {y_path.name}. "
            "Export them on the working server first, or remove --local_dataset_dir."
        )

    print(f"  [Local] Loading frozen dataset: {prefix.name} (.csv.gz + .json)", flush=True)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = list(meta["feature_names"])
    cat_indicator = [bool(v) for v in meta["cat_indicator"]]

    # Read feature matrix. Low-memory=False avoids mixed-type chunk inference.
    X = pd.read_csv(x_path, compression="gzip", low_memory=False)

    # Restore original column order and names exactly.
    missing_cols = [c for c in feature_names if c not in X.columns]
    if missing_cols:
        raise ValueError(f"Frozen X file {x_path} is missing feature columns: {missing_cols[:10]}")
    X = X[feature_names]

    # Force categorical columns to object/string-like columns for the shared
    # preprocessor; numeric columns are allowed to remain numeric as parsed.
    for col, is_cat in zip(feature_names, cat_indicator):
        if is_cat:
            X[col] = X[col].astype("object")

    y_df = pd.read_csv(y_path, compression="gzip")
    if "target" not in y_df.columns:
        raise ValueError(f"Frozen y file {y_path} must contain a 'target' column")
    y = y_df["target"]

    return (
        X,
        y,
        feature_names,
        cat_indicator,
        str(meta.get("name", f"{source}_{dataset_id}")),
    )


# DATASET LOADING HELPERS
# ==============================================================================
def load_uci_dataset(dataset_id: int, local_dataset_dir: Optional[str] = None):
    """Load UCI dataset with the same feature typing policy as the MBNDT runner."""
    local = _load_local_dataset_payload("uci", dataset_id, local_dataset_dir)
    if local is not None:
        return local
    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0]
    feature_names = X.columns.tolist()

    meta = dataset.variables[["name", "type"]].copy()
    meta["type_norm"] = meta["type"].astype(str).str.lower()

    # Match MBNDT: do NOT treat binary variables as categorical by default.
    categorical_type_labels = {"categorical", "nominal"}
    name_to_is_cat = dict(zip(meta["name"], meta["type_norm"].isin(categorical_type_labels)))

    # Match MBNDT: string-valued binary columns become a single numeric 0/1 column.
    binary_cols = meta.loc[meta["type_norm"] == "binary", "name"].tolist()
    for col in binary_cols:
        if col not in X.columns:
            continue
        if pd.api.types.is_numeric_dtype(X[col]):
            continue
        uniq = sorted(X[col].dropna().astype(str).unique())
        if len(uniq) == 2:
            X[col] = X[col].astype(str).map({uniq[0]: 0, uniq[1]: 1}).astype("float64")
        else:
            name_to_is_cat[col] = True

    cat_indicator = [bool(name_to_is_cat.get(col, False)) for col in feature_names]
    name = getattr(dataset.metadata, "name", f"UCI_{dataset_id}")

    return X, y, feature_names, cat_indicator, name


def load_openml_dataset(dataset_id, local_dataset_dir: Optional[str] = None):
    """Load OpenML dataset using the exact same sklearn.fetch_openml path as MBNDT.

    This intentionally does not import or call openml-python's
    openml.datasets.get_dataset(...).get_data(...). That keeps the API/cache/backend
    aligned with the MBNDT runner.
    """
    local = _load_local_dataset_payload("openml", dataset_id, local_dataset_dir)
    if local is not None:
        return local

    bunch = fetch_openml(
        data_id=dataset_id if isinstance(dataset_id, int) else None,
        name=dataset_id if isinstance(dataset_id, str) else None,
        as_frame=True,
        parser="auto",
    )
    X = bunch.data.copy()
    y = bunch.target
    feature_names = X.columns.tolist()

    # Match MBNDT OpenML preprocessing before the shared preprocessor:
    # 1) bool / nullable boolean -> float.
    for col in feature_names:
        if X[col].dtype.name in ("bool", "boolean"):
            X[col] = X[col].astype("float64")

    # 2) object/category columns with exactly 2 non-null values -> one numeric 0/1 column.
    for col in feature_names:
        if X[col].dtype.name in ("object", "category"):
            uniq = X[col].dropna().unique().tolist()
            if len(uniq) == 2:
                uniq_sorted = sorted(uniq, key=lambda v: str(v))
                mapping = {uniq_sorted[0]: 0.0, uniq_sorted[1]: 1.0}
                X[col] = X[col].astype("object").map(mapping).astype("float64")

    # 3) Remaining object/category columns: force string representation.
    for col in feature_names:
        if X[col].dtype.name in ("object", "category"):
            X[col] = X[col].astype(str)

    # 4) Recompute cat_indicator from final dtypes.
    cat_indicator = [
        X[col].dtype.name in ("category", "object")
        for col in feature_names
    ]

    name = getattr(bunch, "details", {}).get("name", None) or f"OpenML_{dataset_id}"
    return X, y, feature_names, cat_indicator, name


# ==============================================================================
def parse_dataset_spec(text: str) -> Dict[str, Any]:
    """Accept ucirepo:27, uci:27, or openml:31. Kept for legacy space-separated args."""
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Bad dataset spec {text!r}. Use SOURCE:ID."
        )

    source = parts[0].lower()
    if source in {"uci", "ucirepo"}:
        ds_type = "uci"
    elif source == "openml":
        ds_type = "openml"
    else:
        raise argparse.ArgumentTypeError(f"Bad source {source!r}; use ucirepo/uci or openml.")

    if ds_type == "uci":
        try:
            ds_id = int(parts[1])
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Bad UCI dataset id in {text!r}.") from e
    else:
        try:
            ds_id = int(parts[1])
        except ValueError:
            ds_id = str(parts[1])

    return {"type": ds_type, "id": ds_id}


def parse_dataset_specs_arg(s: str) -> List[Dict[str, Any]]:
    """Parse MBNDT-style comma-separated dataset specs.

    Accepted examples:
      --datasets openml:55,openml:27,uci:176
      --datasets uci_46,uci_519
      --datasets 46,519,47     # defaults to UCI
    """
    if s is None or str(s).strip() == "":
        return []

    specs: List[Dict[str, Any]] = []
    for raw_tok in str(s).split(","):
        tok = raw_tok.strip()
        if not tok:
            continue

        if ":" in tok:
            source, did = tok.split(":", 1)
        elif "_" in tok:
            source, did = tok.split("_", 1)
        else:
            source, did = "uci", tok

        source = source.strip().lower()
        did = did.strip()

        if source in {"uci", "ucirepo"}:
            ds_type = "uci"
            try:
                did_val = int(did)
            except ValueError as e:
                raise argparse.ArgumentTypeError(f"UCI dataset id must be int, got {did!r}") from e
        elif source == "openml":
            ds_type = "openml"
            # Match MBNDT: OpenML accepts either integer data_id or string name.
            try:
                did_val = int(did)
            except ValueError:
                did_val = str(did)
        else:
            raise argparse.ArgumentTypeError(f"Bad source {source!r}; use uci/ucirepo/openml.")

        specs.append({"type": ds_type, "id": did_val})

    return specs


def specs_from_cli_args(args) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    if getattr(args, "datasets", None):
        specs.extend(parse_dataset_specs_arg(args.datasets))

    if getattr(args, "dataset_ids", None):
        source = getattr(args, "source", "uci")
        ds_type = "uci" if source in {"uci", "ucirepo"} else "openml"
        specs.extend({"type": ds_type, "id": int(did)} for did in args.dataset_ids)

    return specs


# ==============================================================================
# MULTI-DATASET WRAPPER
# ==============================================================================
def _safe_dataset_name(name: str) -> str:
    return str(name).replace(" ", "_").replace("/", "_")[:80]


def _load_and_prepare_dataset(ds_spec: Dict[str, Any], *, local_dataset_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load one dataset and return encoded binary-classification data, or None."""
    ds_type = ds_spec.get("type", ds_spec.get("source", "uci"))
    ds_type = "uci" if ds_type in {"uci", "ucirepo"} else ds_type
    ds_id = ds_spec["id"]
    print(f"\n[Loading] {ds_type.upper()} dataset ID={ds_id}...", flush=True)

    if ds_type == "uci":
        X, y, feature_names, cat_indicator, auto_name = load_uci_dataset(ds_id, local_dataset_dir=local_dataset_dir)
    elif ds_type == "openml":
        X, y, feature_names, cat_indicator, auto_name = load_openml_dataset(ds_id, local_dataset_dir=local_dataset_dir)
    else:
        print(f"  [SKIP] Unknown dataset type: {ds_type}")
        return None

    ds_name = ds_spec.get("name", auto_name)

    le = LabelEncoder()
    y_enc = le.fit_transform(np.asarray(y))
    n_classes = len(le.classes_)
    if n_classes != 2:
        print(f"  [SKIP] {ds_name} has {n_classes} classes (not binary)")
        return None

    print(f"  [OK] {ds_name}: {len(y_enc)} samples, {X.shape[1]} features", flush=True)
    return {
        "type": ds_type,
        "id": ds_id,
        "name": ds_name,
        "X": X,
        "y": y_enc,
        "feature_names": feature_names,
        "cat_indicator": cat_indicator,
        "n_samples": int(len(y_enc)),
        "n_features": int(X.shape[1]),
        "n_classes": int(n_classes),
    }


def run_multiple_datasets(
    datasets_config: List[Dict],
    output_root: str = "./results",
    config: Optional[Dict] = None,
):
    """Run GradTree HPO on multiple datasets using an MBNDT-style shell."""
    if config is None:
        config = DEFAULT_CONFIG.copy()

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GradTree HPO Runner (MBNDT-style experiment shell)")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Datasets to process: {len(datasets_config)}")
    print(f"  Outer CV splits: {config['K_OUTER_SPLITS']}")
    print(f"  HPO timeout per split: {config['HPO_TIMEOUT_SEC'] / 60:.1f} minutes")
    print(f"  Max trials per split: {config['N_HPO_TRIALS']}")
    print(f"  Monitor metric: {config['MONITOR_METRIC']} -> {normalize_metric_key(config['MONITOR_METRIC'])}")
    print(f"  Max GradTree depth: {config.get('MAX_DEPTH', 4)}")
    print(f"  Batch: strict constant, batch_size={config.get('BATCH_SIZE', 128)}")
    print(f"  Optuna pruner: {'MedianPruner' if config.get('USE_OPTUNA_PRUNER', False) else 'NopPruner'}")
    print(f"  Artifact base: {output_root}")
    print(f"  Local dataset dir: {config.get('LOCAL_DATASET_DIR') or '<remote fetch>'}")

    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    datasets = []
    for ds_spec in datasets_config:
        try:
            loaded = _load_and_prepare_dataset(ds_spec, local_dataset_dir=config.get("LOCAL_DATASET_DIR"))
            if loaded is not None:
                datasets.append(loaded)
        except Exception as e:
            ds_type = ds_spec.get("type", ds_spec.get("source", "uci"))
            ds_id = ds_spec.get("id", "?")
            print(f"  [ERROR] Failed to load {ds_type} dataset ID={ds_id}: {e}")
            import traceback
            traceback.print_exc()

    if not datasets:
        print("[ERROR] No valid binary classification datasets found!")
        return {}

    print(f"\n[INFO] Found {len(datasets)} valid binary classification datasets")

    all_dataset_results = {}
    summary_rows = []

    for i, ds in enumerate(datasets):
        print(f"\n\n{'#' * 70}")
        print(f"# DATASET {i + 1}/{len(datasets)}")
        print(f"{'#' * 70}")

        ds_name = ds["name"]
        ds_type = ds["type"]
        ds_id = ds["id"]
        prefix = "UCIrepo" if ds_type == "uci" else "OpenML"
        artifact_path = output_root / f"{prefix}_{ds_id}_{_safe_dataset_name(ds_name)}"

        print(f"\n{'=' * 70}")
        print(f"RUNNING: {ds_name} ({ds_type.upper()} ID={ds_id})")
        print(f"  Samples: {ds['n_samples']}, Features: {ds['n_features']}")
        print(f"  Monitor metric: {config['MONITOR_METRIC']} -> {normalize_metric_key(config['MONITOR_METRIC'])}")
        print(f"  Batch: strict constant, batch_size={config.get('BATCH_SIZE', 128)}")
        print(f"{'=' * 70}\n")

        try:
            all_results, df, summary = run_gradtree_full(
                X=ds["X"],
                y=ds["y"],
                feature_names=ds["feature_names"],
                cat_indicator=ds["cat_indicator"],
                ARTIFACT_ROOT=artifact_path,
                K_OUTER_SPLITS=config["K_OUTER_SPLITS"],
                TEST_RATIO=config["TEST_RATIO"],
                HPO_TIMEOUT_SEC=config["HPO_TIMEOUT_SEC"],
                N_HPO_TRIALS=config["N_HPO_TRIALS"],
                INNER_VAL_RATIO=config["INNER_VAL_RATIO"],
                INNER_REPEATS=config["INNER_REPEATS"],
                INNER_ES_RATIO=config["INNER_ES_RATIO"],
                MONITOR_METRIC=config["MONITOR_METRIC"],
                BASE_SEED=config["BASE_SEED"],
                USE_STRATIFY=config["USE_STRATIFY"],
                USE_SMOTE=config.get("USE_SMOTE", False),
                MAX_DEPTH=config.get("MAX_DEPTH", 4),
                BATCH_SIZE=config.get("BATCH_SIZE", 128),
                BATCH_MIN=config.get("BATCH_MIN", 16),
                BATCH_MAX=config.get("BATCH_MAX", 4096),
                MIN_BATCHES_PER_EPOCH=config.get("MIN_BATCHES_PER_EPOCH", 12),
                STD_PENALTY_ALPHA=config.get("STD_PENALTY_ALPHA", 0.5),
                USE_OPTUNA_PRUNER=config.get("USE_OPTUNA_PRUNER", False),
                SHOW_PROGRESS_BAR=config.get("SHOW_PROGRESS_BAR", False),
                LOG_EVERY_N_TRIALS=config.get("LOG_EVERY_N_TRIALS", 10),
            )

            all_dataset_results[ds_name] = (all_results, df, summary)

            if summary:
                row = {"dataset": ds_name, "n_samples": ds["n_samples"], "n_features": ds["n_features"]}
                for key, val in summary.items():
                    if isinstance(val, dict) and "mean" in val:
                        row[f"{key}_mean"] = val["mean"]
                        row[f"{key}_std"] = val["std"]
                summary_rows.append(row)

            print(f"\nCOMPLETED: {ds_name}")
            print(f"  Results saved under: {artifact_path}")

        except Exception as e:
            print(f"[ERROR] Failed on dataset {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_root / "all_datasets_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        save_json(output_root / "all_datasets_summary.json", summary_rows)
        print(f"\n[Saved] Overall summary: {summary_path}")

    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for ds_name, (_all_results, _df, _summary) in all_dataset_results.items():
        print(f"  {ds_name}: results saved under {output_root}")

    print(f"\n[DONE] Results saved to: {output_root}")
    return all_dataset_results


# ==============================================================================
# CLI INTERFACE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="GradTree HPO runner with MBNDT-style experiment shell")

    # Mode selection
    parser.add_argument("--multi", action="store_true", help="Run multiple datasets")

    # Single dataset options (legacy)
    parser.add_argument("--dataset_type", type=str, default="uci", choices=["uci", "openml"])
    parser.add_argument("--dataset_id", type=int, default=176)
    parser.add_argument("--dataset_name", type=str, default=None)

    # CART/SPLIT/XGB-style dataset selection
    parser.add_argument("--source", choices=["ucirepo", "uci", "openml"], default="ucirepo",
                        help="Dataset source used with --dataset_ids.")
    parser.add_argument("--dataset_ids", nargs="+", type=int, default=None,
                        help="Dataset IDs from --source, e.g. --source ucirepo --dataset_ids 27 468.")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated specs, e.g. --datasets openml:55,openml:27,uci:176. For backward compatibility, --source/--dataset_ids also works.")

    # Multi-dataset options
    parser.add_argument("--config", type=str, default=None, help="Path to datasets config JSON")
    parser.add_argument("--local_dataset_dir", type=str, default=None,
                        help="Directory containing exported dataset payloads like openml_24.pkl or uci_46.pkl. If set, remote fetching is skipped for those datasets.")
    parser.add_argument("--local_openml_dir", type=str, default=None,
                        help="Backward-compatible alias for --local_dataset_dir.")

    # Output
    parser.add_argument("--output_root", type=str, default="./gradtree_results")
    parser.add_argument("--artifact_base", type=str, default=None,
                        help="MBNDT-style alias for --output_root.")

    # CUDA / TensorFlow configuration. These are pre-parsed before TensorFlow import,
    # so changing them requires passing them on the original command line.
    parser.add_argument("--cuda_device", type=str, default=CUDA_CONFIG.get("cuda_device", "0"),
                        help="CUDA device id(s), e.g. 0, 1, 0,1. Use 'all' to leave unchanged.")
    parser.add_argument("--cuda_num", type=str, default=None,
                        help="MBNDT-style alias for --cuda_device. Must be passed on the original command line.")
    parser.add_argument("--cpu", action="store_true", default=CUDA_CONFIG.get("cpu", False),
                        help="Disable CUDA before TensorFlow import.")
    parser.add_argument("--no_tf_allow_growth", action="store_true",
                        default=not CUDA_CONFIG.get("tf_allow_growth", True),
                        help="Disable TensorFlow GPU memory growth.")
    parser.add_argument("--tf_xla_jit", type=int, choices=[0, 1],
                        default=int(CUDA_CONFIG.get("tf_xla_jit", 0)),
                        help="Set TF_XLA_FLAGS=--tf_xla_auto_jit={0,1}.")

    # HPO config overrides
    parser.add_argument("--k_outer_splits", type=int, default=None)
    parser.add_argument("--hpo_timeout_sec", type=int, default=None)
    parser.add_argument("--n_hpo_trials", type=int, default=None)
    parser.add_argument("--hpo_metric", type=str, default=None, choices=EVAL_METRIC_CHOICES,
                        help="Metric optimized by HPO, e.g. balanced_acc, aucroc, acc, f1_macro.")
    parser.add_argument("--monitor_metric", type=str, default=None, choices=EVAL_METRIC_CHOICES,
                        help="Alias for --hpo_metric, matching the MBNDT runner CLI.")
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--inner_repeats", type=int, default=None)
    parser.add_argument("--inner_es_ratio", type=float, default=None)
    parser.add_argument("--no_stratify", action="store_true")
    parser.add_argument("--use_smote", action="store_true")
    parser.add_argument("--max_depth", type=int, default=None, help="Upper bound for GradTree depth search. Default: 4.")
    parser.add_argument("--batch_size", type=int, default=None, help="Strict constant train batch size for every HPO fold and final refit. Not tuned by Optuna. Default: 128.")
    parser.add_argument("--batch_min", type=int, default=None, help="Minimum batch size clamp. Default: 16.")
    parser.add_argument("--batch_max", type=int, default=None, help="Maximum batch size clamp. Default: 4096.")
    parser.add_argument("--min_batches_per_epoch", type=int, default=None, help="Small-data guardrail. Default: 12.")
    parser.add_argument("--use_optuna_pruner", action="store_true", help="Enable MedianPruner. Default is NopPruner to match current MBNDT runner.")
    parser.add_argument("--show_progress_bar", action="store_true", help="Show Optuna/tqdm progress bar. Default: off for clean notebook output.")
    parser.add_argument("--log_every_n_trials", type=int, default=None, help="Print one compact progress line every N completed trials. Default: 10.")
    parser.add_argument("--base_seed", type=int, default=None)
    parser.add_argument("--std_penalty_alpha", type=float, default=None)

    args = parser.parse_args()
    if args.artifact_base is not None:
        args.output_root = args.artifact_base
    if args.local_dataset_dir is None and args.local_openml_dir is not None:
        args.local_dataset_dir = args.local_openml_dir

    # Build config
    config = DEFAULT_CONFIG.copy()
    if args.local_dataset_dir is not None:
        config["LOCAL_DATASET_DIR"] = args.local_dataset_dir
    if args.k_outer_splits is not None:
        config["K_OUTER_SPLITS"] = args.k_outer_splits
    if args.hpo_timeout_sec is not None:
        config["HPO_TIMEOUT_SEC"] = args.hpo_timeout_sec
    if args.n_hpo_trials is not None:
        config["N_HPO_TRIALS"] = args.n_hpo_trials
    metric_arg = args.monitor_metric if args.monitor_metric is not None else args.hpo_metric
    if metric_arg is not None:
        config["MONITOR_METRIC"] = metric_arg
    if args.test_ratio is not None:
        config["TEST_RATIO"] = args.test_ratio
    if args.inner_repeats is not None:
        config["INNER_REPEATS"] = args.inner_repeats
    if args.inner_es_ratio is not None:
        config["INNER_ES_RATIO"] = args.inner_es_ratio
    if args.no_stratify:
        config["USE_STRATIFY"] = False
    if args.use_smote:
        config["USE_SMOTE"] = True
    if args.max_depth is not None:
        config["MAX_DEPTH"] = args.max_depth
    if args.batch_size is not None:
        config["BATCH_SIZE"] = args.batch_size
    if args.batch_min is not None:
        config["BATCH_MIN"] = args.batch_min
    if args.batch_max is not None:
        config["BATCH_MAX"] = args.batch_max
    if args.min_batches_per_epoch is not None:
        config["MIN_BATCHES_PER_EPOCH"] = args.min_batches_per_epoch
    if args.use_optuna_pruner:
        config["USE_OPTUNA_PRUNER"] = True
    if args.show_progress_bar:
        config["SHOW_PROGRESS_BAR"] = True
    if args.log_every_n_trials is not None:
        config["LOG_EVERY_N_TRIALS"] = args.log_every_n_trials
    if args.base_seed is not None:
        config["BASE_SEED"] = args.base_seed
    if args.std_penalty_alpha is not None:
        config["STD_PENALTY_ALPHA"] = args.std_penalty_alpha

    if int(config.get("MAX_DEPTH", 4)) < 1:
        raise ValueError("MAX_DEPTH must be >= 1")
    if normalize_metric_key(config.get("MONITOR_METRIC", "balanced_acc")) not in {"bal_acc", "auroc", "acc", "f1_macro"}:
        raise ValueError(f"Unsupported MONITOR_METRIC: {config.get('MONITOR_METRIC')}")
    if int(config.get("BATCH_SIZE", 128)) < 1:
        raise ValueError("BATCH_SIZE must be >= 1")
    if int(config.get("BATCH_MIN", 16)) < 1:
        raise ValueError("BATCH_MIN must be >= 1")
    if int(config.get("BATCH_MAX", 4096)) < int(config.get("BATCH_MIN", 16)):
        raise ValueError("BATCH_MAX must be >= BATCH_MIN")
    if int(config.get("MIN_BATCHES_PER_EPOCH", 12)) < 1:
        raise ValueError("MIN_BATCHES_PER_EPOCH must be >= 1")
    if int(config.get("INNER_REPEATS", 5)) < 2:
        raise ValueError("INNER_REPEATS must be at least 2 for KFold/StratifiedKFold")
    if int(config.get("LOG_EVERY_N_TRIALS", 10)) < 1:
        raise ValueError("LOG_EVERY_N_TRIALS must be >= 1")

    cli_specs = specs_from_cli_args(args)

    if args.multi or args.config or cli_specs:
        # Multi-dataset mode. Supports either a JSON config or CART/SPLIT/XGB-style
        # --source/--dataset_ids/--datasets arguments.
        datasets_config = []
        if args.config:
            with open(args.config, "r") as f:
                loaded = json.load(f)
            datasets_config.extend(loaded)
        if cli_specs:
            datasets_config.extend(cli_specs)
        if not datasets_config:
            datasets_config = [
                {"type": "uci", "id": 176},
            ]
            print("No config or dataset ids provided. Using default single dataset.")

        run_multiple_datasets(
            datasets_config=datasets_config,
            output_root=args.output_root,
            config=config,
        )
    else:
        # Single dataset mode, legacy interface.
        if args.dataset_type == "uci":
            X, y, feature_names, cat_indicator, ds_name = load_uci_dataset(args.dataset_id, local_dataset_dir=config.get("LOCAL_DATASET_DIR"))
        else:
            X, y, feature_names, cat_indicator, ds_name = load_openml_dataset(args.dataset_id, local_dataset_dir=config.get("LOCAL_DATASET_DIR"))

        if args.dataset_name:
            ds_name = args.dataset_name

        le = LabelEncoder()
        y_enc = le.fit_transform(np.asarray(y))

        if len(le.classes_) != 2:
            raise ValueError(f"Binary classification only (found {len(le.classes_)} classes)")

        artifact_path = Path(args.output_root) / ds_name

        run_gradtree_full(
            X=X,
            y=y_enc,
            feature_names=feature_names,
            cat_indicator=cat_indicator,
            ARTIFACT_ROOT=artifact_path,
            K_OUTER_SPLITS=config["K_OUTER_SPLITS"],
            TEST_RATIO=config["TEST_RATIO"],
            HPO_TIMEOUT_SEC=config["HPO_TIMEOUT_SEC"],
            N_HPO_TRIALS=config["N_HPO_TRIALS"],
            INNER_VAL_RATIO=config["INNER_VAL_RATIO"],
            INNER_REPEATS=config["INNER_REPEATS"],
            INNER_ES_RATIO=config["INNER_ES_RATIO"],
            MONITOR_METRIC=config["MONITOR_METRIC"],
            BASE_SEED=config["BASE_SEED"],
            USE_STRATIFY=config["USE_STRATIFY"],
            USE_SMOTE=config.get("USE_SMOTE", False),
            MAX_DEPTH=config.get("MAX_DEPTH", 4),
            BATCH_SIZE=config.get("BATCH_SIZE", 128),
            BATCH_MIN=config.get("BATCH_MIN", 16),
            BATCH_MAX=config.get("BATCH_MAX", 4096),
            MIN_BATCHES_PER_EPOCH=config.get("MIN_BATCHES_PER_EPOCH", 12),
            STD_PENALTY_ALPHA=config.get("STD_PENALTY_ALPHA", 0.5),
            USE_OPTUNA_PRUNER=config.get("USE_OPTUNA_PRUNER", False),
            SHOW_PROGRESS_BAR=config.get("SHOW_PROGRESS_BAR", False),
            LOG_EVERY_N_TRIALS=config.get("LOG_EVERY_N_TRIALS", 10),
        )


if __name__ == "__main__":
    main()
