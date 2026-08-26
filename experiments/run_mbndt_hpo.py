# ------------------------------
# Standard imports
# ------------------------------
import os, sys, gc, warnings
import argparse
import time, random
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import math

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

DATASETS = [

    ("uci", 46),
    ("uci", 519),
    ("uci", 47),
    # ("openml", 1169),
    # ("openml", 23512),
    # ("openml", 41159),
]

K_OUTER_SPLITS    = 5
BASE_SEED         = 0
TEST_RATIO        = 0.2

HPO_TIMEOUT_SEC   = 60 * 60
N_HPO_TRIALS      = 10000

INNER_VAL_RATIO   = 0.2
INNER_REPEATS     = 5
INNER_ES_RATIO    = 0.2

# HPO objective / final reported evaluation metric.
# evaluate_model() metric keys: {"balanced_acc", "aucroc", "acc", "f1_macro"}
MONITOR_METRIC = "balanced_acc"

# Explicit training monitors for random-restart training.
# train_model() monitor keys: {"val_loss", "val_auc", "val_f1_macro", "val_bacc", "val_acc"}
# Recommended default for balanced-accuracy HPO: val_loss / val_loss / val_bacc.
STAGE1_ES_METRIC = "val_loss"
STAGE1_SELECT_METRIC = "val_loss"
STAGE2_ES_METRIC = "val_bacc"

# Kept only for old configs / backward compatibility.
# The real stage monitors above are the authoritative controls.
RESTART_METRIC = STAGE2_ES_METRIC

# Whether HPO trials use random restart.
HPO_USE_RANDOM_RESTART = True

TRAIN_MONITOR_CHOICES = ["val_loss", "val_auc", "val_f1_macro", "val_bacc", "val_acc"]
EVAL_METRIC_CHOICES = ["balanced_acc", "aucroc", "acc", "f1_macro"]

LOSS_TYPE_CHOICES = ["balanced_bce"]

STD_PENALTY_ALPHA = 0.5

USE_STRATIFY      = True
USE_SMOTE         = False
USE_MASK          = True

# ----------------------------------------------------------------------------
# Batch size is NOT tuned. It is set by a deterministic rule on the
# training-set size (see compute_train_batch_size below).
#
# BATCH_SIZE doubles as:
#   (a) the absolute *target* batch for the train loader, and
#   (b) the default batch for eval / ES loaders, where it only affects
#       forward-pass throughput and never changes results.
#
# Rationale for an absolute (not proportional) batch:
#   - Minibatch-gradient variance ~ sigma^2 / B with a finite-population
#     correction, so optimization behavior is governed by the ABSOLUTE batch
#     B, not by B/n. A fixed proportion p drives large datasets toward
#     near-full-batch (vanishing gradient noise) and is unprincipled across
#     dataset sizes.
#   - The leaf-mass surrogate's per-batch estimate variance also depends on
#     the absolute number of routed samples, so stabilizing it wants an
#     absolute floor, not a proportion.
#
# The only place the dataset size legitimately enters is a small-data
# guardrail (MIN_BATCHES_PER_EPOCH) that keeps you off full-batch on tiny sets.
# ----------------------------------------------------------------------------
BATCH_SIZE            = 256   # absolute target for the train loader
BATCH_MIN             = 16
BATCH_MAX             = 4096
MIN_BATCHES_PER_EPOCH = 12    # small-data guardrail: keeps you off full-batch


# Artifact root for the single-study experiment.
ARTIFACT_BASE     = Path("./HPO_reports_bacc/maxdepth_4_masslog_newlagconstants")
CUDA_NUM          = 0

# Post-hoc MBNDT-PP reporting
POSTHOC_MIN_BRANCH_HIT = 1
POSTHOC_FREEZE_BASE = True

# Tree template search space.
# B is now tuned jointly with depth D and leaf budget K.
B_CHOICES = [3,4]
D_CHOICES = list(range(1, 5))

# Joint (B, D)-dependent leaf-budget grid.
#
# Rationale:
#   - B=4 has much larger nominal capacity than B=3 at the same depth
#     because nominal leaves = B**D.
#   - Therefore K should not depend only on D.
#   - This grid gives B=4 a fair chance without allowing a fully capacity-scaled
#     explosion that would make the comparison mainly a larger-tree comparison.
LEAF_BUDGET_CHOICES_BY_BD = {
    (3, 1): [2, 3],
    (4, 1): [2, 3, 4],

    (3, 2): [3, 6, 9],
    (4, 2): [4, 8, 12, 16],

    (3, 3): [6, 12, 18, 24],
    (4, 3): [8, 12, 16, 24, 32],

    (3, 4): [12, 24, 36, 48],
    (4, 4): [16, 24, 32, 48, 64],
}

# ------------------------------
# ML imports
# ------------------------------
import numpy as np
import pandas as pd
import matplotlib

import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml

# ------------------------------
# MBNDT modules (adjust path as needed)
# ------------------------------
from mbndt import model as NT5_BASE
from mbndt.model import (
    build_posthoc_merged_mbndt,
    save_mbndt_pp_outer_artifacts,
    compute_perf_gaps,
)
from mbndt.preprocessing import PreprocessConfig, preprocess_splits
from mbndt.config import (
    MBNDTConfig, ModelHP, OptimHP, TrainingHP,
    RegularizersHP, LoadBalanceHP, LeafBudgetHP,
    train_from_cfg,
)

# ------------------------------
# System configuration
# ------------------------------
matplotlib.use("Agg")
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=".*n_quantiles.*greater than the total number of samples.*",
    category=UserWarning,
)
optuna.logging.set_verbosity(optuna.logging.WARNING)



def device_to_str(device):
    if isinstance(device, torch.device):
        return str(device)   # preserves "cuda:0", "cuda:1", etc.
    return str(device)


# ============================================================================
# DATASET LOADING UTILITIES
# ============================================================================

@dataclass
class DatasetInfo:
    dataset_id: Any
    source: str
    name: str
    X: pd.DataFrame
    y: np.ndarray
    feature_names: List[str]
    cat_indicator: List[bool]
    cat_cols: List[str]
    num_cols: List[str]
    n_samples: int
    n_features: int
    n_classes: int


def fetch_uci_dataset(dataset_id: int) -> Optional[DatasetInfo]:
    try:
        dataset = fetch_ucirepo(id=dataset_id)
        X = dataset.data.features
        y_raw = dataset.data.targets.iloc[:, 0]
        feature_names = X.columns.tolist()

        meta = dataset.variables[["name", "type"]].copy()
        meta["type_norm"] = meta["type"].astype(str).str.lower()

        # 'binary' removed — a binary feature is already a perfect split point for
        # tree methods. One-hot encoding it produces two perfectly anti-correlated
        # columns, doubling the feature count for that column and forcing the model
        # to learn the complementarity.
        categorical_type_labels = {"categorical", "nominal"}
        name_to_is_cat = dict(zip(meta["name"], meta["type_norm"].isin(categorical_type_labels)))

        # Handle UCI 'binary'-typed columns: if numeric (0/1), pass through as
        # numeric. If string-valued (e.g. 'yes'/'no'), map to a single 0/1 column
        # rather than letting downstream one-hot it into two.
        binary_cols = meta.loc[meta["type_norm"] == "binary", "name"].tolist()
        for col in binary_cols:
            if col not in X.columns:
                continue
            if pd.api.types.is_numeric_dtype(X[col]):
                # Already numeric — nothing to do, will go through as a num_col.
                continue
            uniq = sorted(X[col].dropna().astype(str).unique())
            if len(uniq) == 2:
                # Two non-null categories → map to 0/1 deterministically.
                X[col] = X[col].astype(str).map({uniq[0]: 0, uniq[1]: 1}).astype("float64")
            else:
                # Defensive fallback: a column marked 'binary' that has ≠2 unique
                # values is suspicious — treat as categorical so the preprocessor
                # encodes it correctly rather than crashing on the map above.
                name_to_is_cat[col] = True

        cat_indicator = [bool(name_to_is_cat.get(col, False)) for col in feature_names]
        cat_cols = [c for c, is_cat in zip(feature_names, cat_indicator) if is_cat]
        num_cols = [c for c, is_cat in zip(feature_names, cat_indicator) if not is_cat]

        le = LabelEncoder()
        y = le.fit_transform(np.asarray(y_raw))
        n_classes = len(le.classes_)

        name = getattr(dataset.metadata, "name", f"UCI_{dataset_id}")

        return DatasetInfo(
            dataset_id=dataset_id, source="uci", name=name,
            X=X, y=y, feature_names=feature_names,
            cat_indicator=cat_indicator, cat_cols=cat_cols, num_cols=num_cols,
            n_samples=len(y), n_features=X.shape[1], n_classes=n_classes,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load dataset {dataset_id}: {e}")
        return None


def fetch_openml_dataset(dataset_id) -> Optional[DatasetInfo]:
    try:
        bunch = fetch_openml(
            data_id=dataset_id if isinstance(dataset_id, int) else None,
            name=dataset_id if isinstance(dataset_id, str) else None,
            as_frame=True, parser="auto",
        )
        X = bunch.data.copy()
        y_raw = bunch.target
        feature_names = X.columns.tolist()

        # ---------------------------------------------------------------
        # Treat binary features as numeric, not categorical.
        # OpenML hands us a mix of dtypes: numpy bool, pandas nullable
        # 'boolean', object strings, and pandas 'category'. Any of these
        # can be effectively binary.
        # ---------------------------------------------------------------

        # 1) bool / nullable boolean -> float (0.0 / 1.0; NA -> NaN).
        for col in feature_names:
            if X[col].dtype.name in ("bool", "boolean"):
                X[col] = X[col].astype("float64")

        # 2) object/category columns with exactly 2 unique non-null
        #    values -> map to 0.0 / 1.0 with deterministic ordering.
        #    Sort by string representation so the encoding is reproducible
        #    across runs and pandas versions.
        for col in feature_names:
            if X[col].dtype.name in ("object", "category"):
                uniq = X[col].dropna().unique().tolist()
                if len(uniq) == 2:
                    uniq_sorted = sorted(uniq, key=lambda v: str(v))
                    mapping = {uniq_sorted[0]: 0.0, uniq_sorted[1]: 1.0}
                    # .astype("object") first so .map() works uniformly on
                    # both Categorical and object dtypes; NaN passes through.
                    X[col] = X[col].astype("object").map(mapping).astype("float64")

        # 3) Remaining object/category columns: force string repr (the
        #    downstream encoder expects strings, not pandas categoricals).
        for col in feature_names:
            if X[col].dtype.name in ("object", "category"):
                X[col] = X[col].astype(str)

        # 4) Re-compute cat_indicator from the FINAL dtypes. After steps
        #    1-2, anything that should be numeric is numeric, so a simple
        #    object/category check is the right rule.
        cat_indicator = [
            X[col].dtype.name in ("category", "object")
            for col in feature_names
        ]

        cat_cols = [c for c, is_cat in zip(feature_names, cat_indicator) if is_cat]
        num_cols = [c for c, is_cat in zip(feature_names, cat_indicator) if not is_cat]

        le = LabelEncoder()
        y = le.fit_transform(np.asarray(y_raw))
        n_classes = len(le.classes_)

        name = getattr(bunch, "details", {}).get("name", None) or f"OpenML_{dataset_id}"

        return DatasetInfo(
            dataset_id=dataset_id, source="openml", name=name,
            X=X, y=y, feature_names=feature_names,
            cat_indicator=cat_indicator, cat_cols=cat_cols, num_cols=num_cols,
            n_samples=len(y), n_features=X.shape[1], n_classes=n_classes,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load OpenML dataset {dataset_id}: {e}")
        return None


def load_and_filter_datasets(dataset_specs, *, binary_only=True) -> List[DatasetInfo]:
    datasets = []
    for source, did in dataset_specs:
        print(f"\n[Loading] {source.upper()} dataset ID={did}...")
        if source == "uci":
            info = fetch_uci_dataset(did)
        elif source == "openml":
            info = fetch_openml_dataset(did)
        else:
            print(f"  [SKIP] Unknown source: {source}")
            continue
        if info is None:
            continue
        if binary_only and info.n_classes != 2:
            print(f"  [SKIP] {info.name} has {info.n_classes} classes (not binary)")
            continue
        print(f"  [OK] {info.name}: {info.n_samples} samples, {info.n_features} features")
        datasets.append(info)
    return datasets


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)


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


def make_pp_cfg(seed: int, *, to_torch: bool):
    return PreprocessConfig(
        random_state=seed, stratify=USE_STRATIFY, use_smote=USE_SMOTE,
        impute="median", encode_categoricals=True, cardinality_threshold=10,
        te_sigma=0.0, numerical_stability=False, scale=True, to_torch=to_torch,
    )


def _to_jsonable(x):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist() if x.ndim > 0 else float(x.detach().cpu().item())
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


def get_leaf_budget_choices_for_BD(B: int, depth: int) -> List[int]:
    vals = LEAF_BUDGET_CHOICES_BY_BD.get((int(B), int(depth)), None)
    if vals is None:
        raise ValueError(f"No leaf-budget choices configured for B={B}, D={depth}")

    # Never allow a leaf budget larger than the nominal full-tree capacity.
    nominal_leaves = int(B) ** int(depth)
    vals = sorted({int(v) for v in vals if int(v) <= nominal_leaves})

    if not vals:
        raise ValueError(f"Empty leaf-budget grid for B={B}, D={depth}")
    return vals


def build_B_depth_leaf_budget_triples() -> List[str]:
    triples = []
    for B in B_CHOICES:
        for d in D_CHOICES:
            for k in get_leaf_budget_choices_for_BD(B, d):
                triples.append(f"B{int(B)}_D{int(d)}_K{int(k)}")
    return triples


BDK_CHOICES = build_B_depth_leaf_budget_triples()


def parse_B_depth_leaf_budget_triple(triple: str):
    b_part, depth_part, budget_part = triple.split("_")
    B = int(b_part[1:])
    depth = int(depth_part[1:])
    leaf_budget_k = int(budget_part[1:])
    return B, depth, leaf_budget_k


def compute_train_batch_size(n_train: int) -> int:
    n_train = int(n_train)
    if   n_train < 10_000:  target = 256
    elif n_train < 100_000: target = 512
    else:                   target = 1024
    b = min(target, max(BATCH_MIN, n_train // MIN_BATCHES_PER_EPOCH))
    b = max(BATCH_MIN, min(BATCH_MAX, b))
    return int(min(b, n_train))


def _make_loader(Xt, yt, *, shuffle, seed, device, batch_size=None):
    gen = torch.Generator()
    gen.manual_seed(seed)

    Xt = Xt.to(device)
    yt = yt.to(device)
    n = Xt.size(0)

    if batch_size is None:
        batch_size = BATCH_SIZE

    batch_size = int(batch_size)
    batch_size = max(BATCH_MIN, min(BATCH_MAX, batch_size))
    batch_size = min(batch_size, n)  # allow full-batch on tiny datasets

    return DataLoader(
        TensorDataset(Xt, yt),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen if shuffle else None,
        num_workers=0, pin_memory=False,
    )


def _eval_metric_from_loader(model, loader, metric_key: str):
    mets = NT5_BASE.evaluate_model(model, loader)
    if metric_key in mets:
        return float(mets[metric_key]), mets
    raise KeyError(f"Metric {metric_key} not found in evaluation output keys={list(mets.keys())}")


def train_from_cfg_timed(NT5_v7, cfg, train_loader, es_loader, device):
    t0 = time.perf_counter()
    model, history = train_from_cfg(NT5_v7, cfg, train_loader, es_loader, device)
    elapsed_sec = float(time.perf_counter() - t0)
    epochs_ran = None
    if isinstance(history, dict):
        for k in ["epoch", "epochs", "n_epochs", "best_epoch"]:
            if k in history:
                epochs_ran = history[k]
                break
    return model, history, {"final_fit_time_sec": elapsed_sec, "epochs_ran": epochs_ran}


def sample_hparams(trial):
    hps = {}

    # Joint architecture choice prevents invalid/dynamically changing K spaces
    # and makes the searched leaf budget explicitly conditional on both B and D.
    triple = trial.suggest_categorical("B_depth_leaf_budget_triple", BDK_CHOICES)
    B, depth, leaf_budget_k = parse_B_depth_leaf_budget_triple(triple)

    hps["B"] = int(B)
    hps["D"] = int(depth)
    hps["leaf_budget_K"] = int(leaf_budget_k)
    hps["nominal_leaves"] = int(B) ** int(depth)
    hps["leaf_budget_ratio"] = float(leaf_budget_k) / float(hps["nominal_leaves"])

    hps["tau_cdf"] = trial.suggest_float("tau_cdf", 0.08, 1.2, log=True)

    hps["lr_feature"] = trial.suggest_float("lr_feature", 1e-3, 1e-1, log=True)
    hps["lr_thresh"]  = trial.suggest_float("lr_thresh",  1e-3, 1e-1, log=True)
    hps["lr_leaf"]    = trial.suggest_float("lr_leaf",    1e-3, 1e-1, log=True)
    hps["lr_mask"]    = trial.suggest_float("lr_mask",    1e-3, 1e-1, log=True)

    # Loss-function HPO. If LOSS_TYPE_CHOICES has length 1, this is effectively fixed.
    hps["loss_type"] = trial.suggest_categorical("loss_type", LOSS_TYPE_CHOICES)

    # NOTE: batch_size is intentionally NOT part of the search space. It is set
    # deterministically by compute_train_batch_size() on the training-set size.
    return hps


def make_cfg(hps, n_features, seed, *, for_hpo: bool):
    if for_hpo:
        stage1_epoch, stage1_patience = 25, 5
        stage2_epoch, stage2_patience = 300, 25
        verbose = False
        random_restart = HPO_USE_RANDOM_RESTART
        random_restart_n = 3
    else:
        stage1_epoch, stage1_patience = 40, 8
        stage2_epoch, stage2_patience = 500, 25
        verbose = True
        random_restart = True
        random_restart_n = 5

    cfg = MBNDTConfig(
        model=ModelHP(
            n_features=n_features,
            D=hps["D"],
            B=hps["B"],
            selector_mode="entmax_st",
            branch_mode="st",
            tau_cdf=hps["tau_cdf"],
            #tau_feat=hps["tau_feat"],
            use_masks=USE_MASK,
        ),
        optim=OptimHP(
            lr_feature=hps["lr_feature"],
            lr_thresh=hps["lr_thresh"],
            lr_leaf=hps["lr_leaf"],
            lr_mask=hps["lr_mask"],
        ),
        training=TrainingHP(
            random_restart=random_restart,
            n_restarts=random_restart_n,
            base_seed=seed,

            restart_metric=STAGE2_ES_METRIC,  # backward compatibility only

            stage1_es_metric=STAGE1_ES_METRIC,
            stage1_select_metric=STAGE1_SELECT_METRIC,
            stage2_es_metric=STAGE2_ES_METRIC,

            stage1_epoch=stage1_epoch,
            stage1_patience=stage1_patience,
            stage2_epoch=stage2_epoch,
            stage2_patience=stage2_patience,

            loss_type=hps.get("loss_type", "balanced_bce"),
            verbose=verbose,
        ),
        regularizers=RegularizersHP(
            load_balance=LoadBalanceHP(on=False, warmup_frac=0.05),
            leaf_budget=LeafBudgetHP(on=True, K=float(hps["leaf_budget_K"])),
        )
    )
    return cfg

# ============================================================================
# MAIN RUNNER FUNCTIONS
# ============================================================================

def run_one_outer_split(
    outer_seed: int,
    tv_idx,
    te_idx,
    X,
    y,
    feature_names,
    cat_indicator,
    device,
    artifact_root: Path,
):
    """
    One outer split:
      - HPO over depth/leaf-budget pair + tau + per-group LRs
      - Final fit on train+ES, evaluate on train / val / test for both MBNDT and MBNDT-PP
      - PP construction uses train+val combined
      - Slim consolidated report.json saved per split

    Batch size is fixed by compute_train_batch_size() on the actual training
    set size (computed once for HPO inner folds, recomputed for the final fit).
    """
    y = _to_numpy_y(y)
    cat_cols, num_cols = _infer_cat_num_cols(X, cat_indicator)

    X_tv   = X.iloc[tv_idx].reset_index(drop=True)
    y_tv   = y[tv_idx]
    X_test = X.iloc[te_idx].reset_index(drop=True)
    y_test = y[te_idx]

    inner_seed = outer_seed + 111
    y_tv_arr = np.asarray(y_tv).astype(int)
    n_tv = len(y_tv_arr)


    splitter = StratifiedKFold(n_splits=INNER_REPEATS, shuffle=True, random_state=inner_seed)
    inner_splits = list(splitter.split(np.zeros((n_tv, 1)), y_tv_arr))
    es_seeds = [inner_seed + 1000 + fold_id for fold_id in range(len(inner_splits))]

    rep_raw = []
    for fold_id, (idx_tr, idx_va) in enumerate(inner_splits):
        X_tr_raw = X_tv.iloc[idx_tr]; y_tr_raw = y_tv_arr[idx_tr]
        X_va_raw = X_tv.iloc[idx_va]; y_va_raw = y_tv_arr[idx_va]
        X_main_raw, X_es_raw, y_main_raw, y_es_raw = train_test_split(
            X_tr_raw, y_tr_raw,
            test_size=INNER_ES_RATIO,
            random_state=es_seeds[fold_id],
            stratify=(y_tr_raw if USE_STRATIFY else None),
        )
        rep_raw.append((X_main_raw, y_main_raw, X_es_raw, y_es_raw, X_va_raw, y_va_raw))

    # Use the smallest inner-training split as the reference so the single fixed
    # batch is valid (<= n) for every inner fold.
    n_train_ref = min(len(item[1]) for item in rep_raw)
    hpo_batch_size = compute_train_batch_size(n_train_ref)
    print(f"[Batch] fixed (not tuned): n_train_ref={n_train_ref}, batch_size={hpo_batch_size}")

    def objective(trial):
        hps = sample_hparams(trial)
        trial_seed = BASE_SEED + outer_seed * 1_000_000 + trial.number
        seed_everything(trial_seed)


        scores = []
        for rep_id, (X_main_raw, y_main_raw, X_es_raw, y_es_raw, X_va_raw, y_va_raw) in enumerate(rep_raw):



            pp_seed = inner_seed + rep_id
            cfg_pp = make_pp_cfg(pp_seed, to_torch=True)

            res = preprocess_splits(
                X_train=X_main_raw, y_train=y_main_raw,
                X_valid=X_es_raw,  y_valid=y_es_raw,
                X_test=X_va_raw,   y_test=y_va_raw,
                cfg=cfg_pp, cat_cols=cat_cols, num_cols=num_cols,
            )

            train_loader = _make_loader(
                res.X_train, res.y_train, shuffle=True,
                seed=trial_seed + 10 + rep_id, device=device,
                batch_size=hpo_batch_size,
            )
            es_loader    = _make_loader(res.X_valid, res.y_valid, shuffle=False, seed=trial_seed + 20 + rep_id, device=device)
            va_loader    = _make_loader(res.X_test,  res.y_test,  shuffle=False, seed=trial_seed + 30 + rep_id, device=device)

            n_features_local = int(res.X_train.shape[1])
            cfg = make_cfg(hps, n_features=n_features_local, seed=trial_seed + rep_id, for_hpo=True)
            cfg.training.verbose = False
            cfg.model.num_classes = res.meta["num_classes"]

            model, history = train_from_cfg(NT5_BASE, cfg, train_loader, es_loader, device)

            # Build MBNDT-PP from train + ES-val (mirrors the final-fit construction).
            trainval_X = torch.cat([res.X_train, res.X_valid], dim=0)
            trainval_y = torch.cat([res.y_train, res.y_valid], dim=0)
            trainval_loader = _make_loader(
                trainval_X, trainval_y, shuffle=False,
                seed=trial_seed + 40 + rep_id, device=device,
            )

            posthoc_model, prune_spec = build_posthoc_merged_mbndt(
                model=model,
                train_loader=trainval_loader,
                device=device_to_str(device),
                min_branch_hit=POSTHOC_MIN_BRANCH_HIT,
                freeze_base=POSTHOC_FREEZE_BASE,
            )

            # Score the headline model — MBNDT-PP — on the inner validation fold.
            s, _ = _eval_metric_from_loader(posthoc_model, va_loader, MONITOR_METRIC)

            #s, _ = _eval_metric_from_loader(model, va_loader, MONITOR_METRIC)
            scores.append(float(s))

            # Optional: log per-fold PP full-depth train-hit diagnostics.
            # Do NOT call this routable_leaves; it is not compressed structural complexity.
            pp_sum = prune_spec.get("summary", {})

            trial.set_user_attr(
                f"pp_train_allocated_leaves_full_depth_fold_{rep_id}",
                int(pp_sum.get("train_allocated_leaves_full_depth", 0)),
            )
            trial.set_user_attr(
                f"pp_train_active_branches_full_depth_fold_{rep_id}",
                int(pp_sum.get("train_active_branches_full_depth", 0)),
            )
            trial.set_user_attr(
                f"pp_train_reached_internal_nodes_full_depth_fold_{rep_id}",
                int(pp_sum.get("train_reached_internal_nodes_full_depth", 0)),
            )

            # mu = float(np.mean(scores))
            # sigma = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            # robust_partial = mu - STD_PENALTY_ALPHA * sigma

            # trial.report(robust_partial, step=rep_id)
            # if rep_id >= 1 and trial.should_prune():
            #     raise optuna.TrialPruned()

            del model, history, posthoc_model, prune_spec
            del res, train_loader, es_loader, va_loader, trainval_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        mu = float(np.mean(scores))
        sigma = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        robust = mu - STD_PENALTY_ALPHA * sigma

        trial.set_user_attr("inner_mean", mu)
        trial.set_user_attr("inner_std", sigma)
        trial.set_user_attr("B", int(hps["B"]))
        trial.set_user_attr("depth", int(hps["D"]))
        trial.set_user_attr("leaf_budget_K", int(hps["leaf_budget_K"]))
        trial.set_user_attr("nominal_leaves", int(hps["nominal_leaves"]))
        trial.set_user_attr("leaf_budget_ratio", float(hps["leaf_budget_ratio"]))
        trial.set_user_attr("loss_type", str(hps["loss_type"]))
        trial.set_user_attr("batch_size", int(hpo_batch_size))
        return robust

    # ---- Run HPO ------------------------------------------------------------
    hpo_t0 = time.perf_counter()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=outer_seed + 1000),
        #pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1, interval_steps=1),
        pruner=optuna.pruners.NopPruner(),
    )
    study.optimize(objective, n_trials=N_HPO_TRIALS, timeout=HPO_TIMEOUT_SEC, n_jobs=1, show_progress_bar=True)
    hpo_time_sec = float(time.perf_counter() - hpo_t0)

    best_hps = dict(study.best_trial.params)
    best_B, best_depth, best_leaf_budget = parse_B_depth_leaf_budget_triple(
        best_hps["B_depth_leaf_budget_triple"]
    )
    best_hps["B"]             = int(best_B)
    best_hps["D"]             = int(best_depth)
    best_hps["leaf_budget_K"] = int(best_leaf_budget)
    best_hps["nominal_leaves"] = int(best_B) ** int(best_depth)
    best_hps["leaf_budget_ratio"] = float(best_leaf_budget) / float(best_hps["nominal_leaves"])
    # Batch size is not a tuned param; record the value the HPO inner folds used.
    best_hps["hpo_batch_size"] = int(hpo_batch_size)

    # ---- Final fit on train+ES split ----------------------------------------
    final_seed = outer_seed * 10_000_000 + 999
    seed_everything(final_seed)

    X_main_raw, X_es_raw, y_main_raw, y_es_raw = train_test_split(
        X_tv, y_tv_arr,
        test_size=INNER_ES_RATIO,
        random_state=final_seed + 777,
        stratify=(y_tv_arr if USE_STRATIFY else None),
    )

    cfg_pp = make_pp_cfg(final_seed, to_torch=True)
    res = preprocess_splits(
        X_train=X_main_raw, y_train=y_main_raw,
        X_valid=X_es_raw,  y_valid=y_es_raw,
        X_test=X_test,     y_test=y_test,
        cfg=cfg_pp, cat_cols=cat_cols, num_cols=num_cols,
    )

    # Recompute on the actual final training-set size (larger than the inner
    # folds, so this is more correct than reusing the HPO value).
    final_batch_size = compute_train_batch_size(int(res.X_train.shape[0]))
    best_hps["final_batch_size"] = int(final_batch_size)
    print(f"[Batch] final fit: n_train={int(res.X_train.shape[0])}, batch_size={final_batch_size}")

    train_loader = _make_loader(
        res.X_train, res.y_train, shuffle=True,
        seed=final_seed + 1, device=device,
        batch_size=final_batch_size,
    )
    es_loader    = _make_loader(res.X_valid, res.y_valid, shuffle=False, seed=final_seed + 2, device=device)
    test_loader  = _make_loader(res.X_test,  res.y_test,  shuffle=False, seed=final_seed + 3, device=device)

    n_features_final = int(res.X_train.shape[1])
    cfg = make_cfg(best_hps, n_features=n_features_final, seed=final_seed, for_hpo=False)

    cfg.model.num_classes = res.meta["num_classes"]

    model, history, time_info = train_from_cfg_timed(NT5_BASE, cfg, train_loader, es_loader, device)

    # ---- Evaluation loaders -------------------------------------------------
    train_eval_loader = _make_loader(res.X_train, res.y_train, shuffle=False, seed=final_seed + 101, device=device)
    val_eval_loader   = _make_loader(res.X_valid, res.y_valid, shuffle=False, seed=final_seed + 102, device=device)

    # Combined (train + ES-val) loader for path-length stats and PP construction
    Xt_trainval = torch.cat([res.X_train, res.X_valid], dim=0)
    yt_trainval = torch.cat([res.y_train, res.y_valid], dim=0)
    trainval_loader = _make_loader(Xt_trainval, yt_trainval, shuffle=False, seed=final_seed + 103, device=device)

    # ---- MBNDT performance on train / val / test ----------------------------
    _, train_metrics = _eval_metric_from_loader(model, train_eval_loader, MONITOR_METRIC)
    _, val_metrics   = _eval_metric_from_loader(model, val_eval_loader,   MONITOR_METRIC)
    _, test_metrics  = _eval_metric_from_loader(model, test_loader,       MONITOR_METRIC)

    # ---- Build MBNDT-PP from train+val combined -----------------------------
    posthoc_model, posthoc_prune_spec = build_posthoc_merged_mbndt(
        model=model,
        train_loader=trainval_loader,
        device=device_to_str(device),
        min_branch_hit=POSTHOC_MIN_BRANCH_HIT,
        freeze_base=POSTHOC_FREEZE_BASE,
    )

    # ---- MBNDT-PP performance on train / val / test -------------------------
    _, train_metrics_pp = _eval_metric_from_loader(posthoc_model, train_eval_loader, MONITOR_METRIC)
    _, val_metrics_pp   = _eval_metric_from_loader(posthoc_model, val_eval_loader,   MONITOR_METRIC)
    _, test_metrics_pp  = _eval_metric_from_loader(posthoc_model, test_loader,       MONITOR_METRIC)

    # ---- Gaps (train/val/test) ---------------------------------------------
    gaps_mbndt = compute_perf_gaps(train_metrics,    val_metrics,    test_metrics)
    gaps_pp    = compute_perf_gaps(train_metrics_pp, val_metrics_pp, test_metrics_pp)

    # ---- Slim structure + path-length scalars ------------------------------
    outer_dir = artifact_root / f"outer_{outer_seed:02d}"
    outer_dir.mkdir(parents=True, exist_ok=True)

    mbndt_scalars = NT5_BASE.save_mbndt_outer_artifacts(
        NT5_v7=NT5_BASE,
        model=model,
        trainval_loader=trainval_loader,
        test_loader=test_loader,
        out_dir=outer_dir,
        device=device_to_str(device),
        z_min=-6, z_max=6, n_grid=8001,
        apply_masks=True, mask_eps=0.0, tol=1e-12, min_edge_hits=1,
    )

    pp_scalars = save_mbndt_pp_outer_artifacts(
        posthoc_model=posthoc_model,
        prune_spec=posthoc_prune_spec,
        trainval_loader=trainval_loader,
        test_loader=test_loader,
        out_dir=outer_dir,
        device=device_to_str(device),

        # New argument from the patched MBNDT-PP code.
        # This catches impossible cases where PP compressed structure exceeds raw MBNDT.
        raw_struct_for_assert=mbndt_scalars,
    )

    # Extra defensive sanity check at runner level.
    # Under the corrected compressed/function-hard convention,
    # MBNDT-PP should not be structurally larger than raw MBNDT.
    for k in ["routable_leaves", "routable_branches_effective", "decision_nodes"]:
        raw_v = int(mbndt_scalars.get(k, 0))
        pp_v = int(pp_scalars.get(k, 0))
        if pp_v > raw_v:
            raise RuntimeError(
                f"[BUG] MBNDT-PP structural metric increased for {k}: "
                f"raw={raw_v}, pp={pp_v}. "
                "This means PP is still not using the same compressed/function-hard "
                "structural convention as raw MBNDT."
            )

    # ---- Consolidated per-split report -------------------------------------
    report = {
        "outer_seed": int(outer_seed),
        "best_hps": best_hps,
        "metric_config": {
            "monitor_metric": MONITOR_METRIC,
            "stage1_es_metric": STAGE1_ES_METRIC,
            "stage1_select_metric": STAGE1_SELECT_METRIC,
            "stage2_es_metric": STAGE2_ES_METRIC,
            "hpo_use_random_restart": HPO_USE_RANDOM_RESTART,
            "hpo_batch_size": int(hpo_batch_size),
            "final_batch_size": int(final_batch_size),
        },
        "best_value_inner_objective": float(study.best_value),
        "best_inner_mean": float(study.best_trial.user_attrs.get("inner_mean", np.nan)),
        "best_inner_std":  float(study.best_trial.user_attrs.get("inner_std",  np.nan)),
        "timing": {
            "hpo_time_sec":       float(hpo_time_sec),
            "final_fit_time_sec": float(time_info["final_fit_time_sec"]),
            "epochs_ran":         time_info.get("epochs_ran", None),
        },
        "mbndt": {
            "perf": {
                "train": train_metrics,
                "val":   val_metrics,
                "test":  test_metrics,
                "gaps":  gaps_mbndt,
            },
            "structure_and_paths": mbndt_scalars,
        },
        "mbndt_pp": {
            "perf": {
                "train": train_metrics_pp,
                "val":   val_metrics_pp,
                "test":  test_metrics_pp,
                "gaps":  gaps_pp,
            },
            "structure_and_paths": pp_scalars,
        },
    }
    save_json(outer_dir / "report.json", report)
    save_json(outer_dir / "best_params.json", best_hps)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return report


# ============================================================================
# EXPORT / AGGREGATION
# ============================================================================

def _flatten_split_report(r: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a per-split report dict into a single-row dict for CSV export."""
    row = {
        "outer_seed": r.get("outer_seed", None),
        "best_value_inner_objective": r.get("best_value_inner_objective", np.nan),
        "best_inner_mean": r.get("best_inner_mean", np.nan),
        "best_inner_std":  r.get("best_inner_std",  np.nan),
    }
    for k, v in (r.get("metric_config", {}) or {}).items():
        row[f"metric/{k}"] = v
    for k, v in (r.get("best_hps", {}) or {}).items():
        row[f"hp/{k}"] = v
    for k, v in (r.get("timing", {}) or {}).items():
        row[f"timing/{k}"] = v

    for variant in ("mbndt", "mbndt_pp"):
        sub = r.get(variant, {}) or {}
        perf = sub.get("perf", {}) or {}
        for split_name in ("train", "val", "test"):
            for k, v in (perf.get(split_name, {}) or {}).items():
                row[f"{variant}/{split_name}/{k}"] = v
        for k, v in (perf.get("gaps", {}) or {}).items():
            row[f"{variant}/{k}"] = v
        for k, v in (sub.get("structure_and_paths", {}) or {}).items():
            row[f"{variant}/cx/{k}"] = v

    return row


def export_outer_split_summary(all_results: List[Dict[str, Any]], artifact_root: Path):
    rows = [_flatten_split_report(r) for r in all_results]
    df = pd.DataFrame(rows)
    out_csv = artifact_root / "outer_split_summary.csv"
    df.to_csv(out_csv, index=False)

    # Mean/std across outer splits for every numeric column
    summary = {}
    for col in df.columns:
        if col == "outer_seed":
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue
        summary[col] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n":    int(len(vals)),
        }

    pair_counts = {}
    pair_col = "hp/B_depth_leaf_budget_triple"
    if pair_col in df.columns:
        counts = df[pair_col].value_counts(dropna=False).to_dict()
        pair_counts = {str(k): int(v) for k, v in counts.items()}

    save_json(artifact_root / "outer_split_summary_mean_std.json", summary)
    save_json(artifact_root / "outer_split_architecture_counts.json", pair_counts)
    return df, summary, pair_counts


def run_single_dataset(dataset_info: DatasetInfo, device: torch.device) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"RUNNING: {dataset_info.name} ({dataset_info.source.upper()} ID={dataset_info.dataset_id})")
    print(f"  Samples: {dataset_info.n_samples}, Features: {dataset_info.n_features}")
    print(f"  B choices: {B_CHOICES}")
    print(f"  D choices: {D_CHOICES}")
    print(f"  Loss choices: {LOSS_TYPE_CHOICES}")
    print(f"  Batch: fixed rule, target={BATCH_SIZE}, min_batches/epoch={MIN_BATCHES_PER_EPOCH}")
    print(f"{'='*70}\n")

    safe_name = dataset_info.name.replace(" ", "_").replace("/", "_")[:50]
    prefix = "UCIrepo" if dataset_info.source == "uci" else "OpenML"
    artifact_root = ARTIFACT_BASE / f"{prefix}_{dataset_info.dataset_id}_{safe_name}"
    artifact_root.mkdir(parents=True, exist_ok=True)

    save_json(artifact_root / "dataset_info.json", {
        "dataset_id": dataset_info.dataset_id,
        "name": dataset_info.name,
        "n_samples": dataset_info.n_samples,
        "n_features": dataset_info.n_features,
        "n_classes": dataset_info.n_classes,
        "feature_names": dataset_info.feature_names,
        "cat_cols": dataset_info.cat_cols,
        "num_cols": dataset_info.num_cols,
        "B_choices": B_CHOICES,
        "D_choices": D_CHOICES,
        "leaf_budget_choices_by_BD": {f"B{b}_D{d}": ks for (b, d), ks in LEAF_BUDGET_CHOICES_BY_BD.items()},
        "B_depth_leaf_budget_triple_choices": BDK_CHOICES,
        "loss_type_choices": LOSS_TYPE_CHOICES,
        "batch_size_target": BATCH_SIZE,
        "batch_rule": f"min({BATCH_SIZE}, max({BATCH_MIN}, n_train//{MIN_BATCHES_PER_EPOCH}))",
        "batch_min": BATCH_MIN,
        "batch_max": BATCH_MAX,
        "min_batches_per_epoch": MIN_BATCHES_PER_EPOCH,
        "inner_repeats": INNER_REPEATS,
        "monitor_metric": MONITOR_METRIC,
        "stage1_es_metric": STAGE1_ES_METRIC,
        "stage1_select_metric": STAGE1_SELECT_METRIC,
        "stage2_es_metric": STAGE2_ES_METRIC,
        "hpo_use_random_restart": HPO_USE_RANDOM_RESTART,
    })

    seed_everything(BASE_SEED)
    outer_splitter = (
        StratifiedShuffleSplit(n_splits=K_OUTER_SPLITS, test_size=TEST_RATIO, random_state=BASE_SEED)
        if USE_STRATIFY else
        ShuffleSplit(n_splits=K_OUTER_SPLITS, test_size=TEST_RATIO, random_state=BASE_SEED)
    )
    outer_splits = list(outer_splitter.split(np.zeros((len(dataset_info.y), 1)), dataset_info.y))

    all_results = []
    for split_idx, (tv_idx, te_idx) in enumerate(outer_splits):
        outer_seed = BASE_SEED + split_idx
        out = run_one_outer_split(
            outer_seed=outer_seed,
            tv_idx=tv_idx, te_idx=te_idx,
            X=dataset_info.X, y=dataset_info.y,
            feature_names=dataset_info.feature_names,
            cat_indicator=dataset_info.cat_indicator,
            device=device,
            artifact_root=artifact_root,
        )
        all_results.append(out)

        print(f"\n[Outer split {split_idx}] seed={outer_seed}")
        print(f"  Best architecture: {out['best_hps'].get('B_depth_leaf_budget_triple', 'N/A')}")
        print(f"  Best inner objective: {out['best_value_inner_objective']:.4f}")
        m_test = out['mbndt']['perf']['test'].get(MONITOR_METRIC, None)
        pp_test = out['mbndt_pp']['perf']['test'].get(MONITOR_METRIC, None)
        print(f"  Test {MONITOR_METRIC} (MBNDT):    {m_test}")
        print(f"  Test {MONITOR_METRIC} (MBNDT-PP): {pp_test}")

    # Aggregated mean/std + pair frequencies
    outer_df, outer_summary, pair_counts = export_outer_split_summary(all_results, artifact_root)
    save_json(artifact_root / "all_results_raw.json", all_results)

    print(f"\n{'='*70}")
    print(f"COMPLETED: {dataset_info.name}")
    print(f"{'='*70}")
    primary_metric_cols = [
        f"mbndt/test/{MONITOR_METRIC}",
        f"mbndt_pp/test/{MONITOR_METRIC}",
    ]
    # Also show balanced accuracy in AUROC runs, because it is useful for sanity checks.
    if MONITOR_METRIC != "balanced_acc":
        primary_metric_cols += ["mbndt/test/balanced_acc", "mbndt_pp/test/balanced_acc"]

    cols_to_show = [c for c in [
        "outer_seed", "hp/B", "hp/D", "hp/leaf_budget_K", "hp/leaf_budget_ratio",
        *primary_metric_cols,

        # Correct compressed/function-hard structural metrics
        "mbndt/cx/routable_leaves",
        "mbndt_pp/cx/routable_leaves",
        "mbndt/cx/routable_branches_effective",
        "mbndt_pp/cx/routable_branches_effective",
        "mbndt/cx/decision_nodes",
        "mbndt_pp/cx/decision_nodes",

        # PP full-depth diagnostics
        "mbndt_pp/cx/train_allocated_leaves_full_depth",
        "mbndt_pp/cx/pp_trainval_redirected_allocated_leaves_full_depth",
    ] if c in outer_df.columns]
    if cols_to_show:
        print(outer_df[cols_to_show].to_string(index=False))

    return {
        "dataset_id": dataset_info.dataset_id,
        "dataset_name": dataset_info.name,
        "n_samples": dataset_info.n_samples,
        "n_features": dataset_info.n_features,
        "artifact_root": str(artifact_root),
        "outer_summary": outer_summary,
        "pair_counts": pair_counts,
        "all_results": all_results,
    }


def run_all_datasets():
    print("=" * 70)
    print("MBNDT Single-Study HPO Runner (slim reporting)")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Datasets to process: {len(DATASETS)}")
    print(f"  Outer CV splits: {K_OUTER_SPLITS}")
    print(f"  HPO timeout per split: {HPO_TIMEOUT_SEC / 60:.1f} minutes")
    print(f"  Max trials per split: {N_HPO_TRIALS}")
    print(f"  B choices: {B_CHOICES}")
    print(f"  D choices: {D_CHOICES}")
    print(f"  Leaf budget choices by (B, D): {LEAF_BUDGET_CHOICES_BY_BD}")
    print(f"  Monitor metric: {MONITOR_METRIC}")
    print(f"  Stage metrics: {STAGE1_ES_METRIC} / {STAGE1_SELECT_METRIC} / {STAGE2_ES_METRIC}")
    print(f"  HPO use random restart: {HPO_USE_RANDOM_RESTART}")
    print(f"  Batch (fixed rule): target={BATCH_SIZE}, min/max={BATCH_MIN}/{BATCH_MAX}, min_batches/epoch={MIN_BATCHES_PER_EPOCH}")
    print(f"  Artifact base: {ARTIFACT_BASE}")


    if torch.cuda.is_available():
        torch.cuda.set_device(CUDA_NUM)
        device = torch.device(f"cuda:{CUDA_NUM}")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    datasets = load_and_filter_datasets(DATASETS)
    if not datasets:
        print("[ERROR] No valid binary classification datasets found!")
        return

    print(f"\n[INFO] Found {len(datasets)} valid binary classification datasets")

    all_dataset_results = []
    for i, dataset_info in enumerate(datasets):
        print(f"\n\n{'#' * 70}")
        print(f"# DATASET {i+1}/{len(datasets)}")
        print(f"{'#' * 70}")
        try:
            result = run_single_dataset(dataset_info, device)
            all_dataset_results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed on dataset {dataset_info.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    master_summary_path = ARTIFACT_BASE / "multi_dataset_single_hpo_summary.json"
    master_summary = []
    for r in all_dataset_results:
        master_summary.append({
            "dataset_id":   r["dataset_id"],
            "dataset_name": r["dataset_name"],
            "n_samples":    r["n_samples"],
            "n_features":   r["n_features"],
            "artifact_root": r["artifact_root"],
            "outer_summary": r["outer_summary"],
            "pair_counts":   r["pair_counts"],
        })
    save_json(master_summary_path, master_summary)

    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for r in all_dataset_results:
        print(f"  {r['dataset_name']}: results saved under {r['artifact_root']}")

    print(f"\n[DONE] Results saved to: {ARTIFACT_BASE}")
    print(f"[DONE] Master summary: {master_summary_path}")



# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

def parse_dataset_specs_arg(s: str):
    """
    Parse dataset specs from a comma-separated string.

    Accepted formats:
      --datasets uci:46,uci:519,uci:47
      --datasets uci_46,uci_519,openml_1169
      --datasets 46,519,47              # defaults to UCI
      --datasets openml:credit-g         # OpenML name also allowed
    """
    if s is None or str(s).strip() == "":
        return DATASETS

    specs = []
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

        if source not in {"uci", "openml"}:
            raise ValueError(
                f"Unknown dataset source in token '{tok}'. "
                "Use uci:<id> or openml:<id/name>."
            )

        # UCI IDs should be integers.
        if source == "uci":
            try:
                did = int(did)
            except ValueError as e:
                raise ValueError(f"UCI dataset id must be int, got '{did}' from token '{tok}'") from e
        else:
            # OpenML accepts either integer data_id or string name.
            try:
                did = int(did)
            except ValueError:
                did = str(did)

        specs.append((source, did))

    if not specs:
        raise ValueError("No valid dataset specs parsed from --datasets")

    return specs


def parse_loss_types_arg(s: str) -> List[str]:
    """
    Parse comma-separated loss types.

    Examples:
      --loss_types balanced_bce
      --loss_types bce,balanced_bce,focal
    """
    allowed = {"bce", "balanced_bce", "focal"}
    if s is None or str(s).strip() == "":
        return list(LOSS_TYPE_CHOICES)

    vals = []
    for tok in str(s).split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok not in allowed:
            raise ValueError(
                f"Unknown loss type '{tok}'. Allowed values are: {sorted(allowed)}"
            )
        vals.append(tok)

    # Deduplicate preserving order.
    out = []
    seen = set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)

    if not out:
        raise ValueError("No valid loss types parsed from --loss_types")
    return out


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {v}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MBNDT HPO runner with CLI-configurable datasets, metrics, batch sizing, restart metric, output dir, and CUDA device."
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated dataset specs. Examples: "
            "'uci:46,uci:519,uci:47', 'uci_46,uci_519,openml_1169', or '46,519,47' for UCI."
        ),
    )
    parser.add_argument(
        "--monitor_metric",
        type=str,
        default=MONITOR_METRIC,
        choices=EVAL_METRIC_CHOICES,
        help="Metric key used for HPO objective and train/val/test evaluation via evaluate_model().",
    )
    parser.add_argument(
        "--artifact_base",
        type=str,
        default=str(ARTIFACT_BASE),
        help="Output root directory for reports/artifacts.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=(
            "Absolute target batch size for the train loader (NOT tuned). "
            "Also used as the default batch for eval/ES loaders."
        ),
    )
    parser.add_argument(
        "--min_batches_per_epoch",
        type=int,
        default=MIN_BATCHES_PER_EPOCH,
        help=(
            "Small-data guardrail. The train batch is scaled down to at most "
            "n_train // min_batches_per_epoch so tiny datasets never run near "
            "full-batch. Has no effect once n_train is large."
        ),
    )
    parser.add_argument(
        "--batch_min",
        type=int,
        default=BATCH_MIN,
        help="Minimum batch size (lower clamp).",
    )
    parser.add_argument(
        "--batch_max",
        type=int,
        default=BATCH_MAX,
        help="Maximum batch size (upper clamp).",
    )
    parser.add_argument(
        "--restart_metric",
        type=str,
        default=RESTART_METRIC,
        choices=TRAIN_MONITOR_CHOICES,
        help=(
            "Backward-compatible alias stored in TrainingHP.restart_metric only. "
            "For the actual random-restart stages, use --stage1_es_metric, "
            "--stage1_select_metric, and --stage2_es_metric."
        ),
    )
    parser.add_argument(
        "--stage1_es_metric",
        type=str,
        default=STAGE1_ES_METRIC,
        choices=TRAIN_MONITOR_CHOICES,
        help="Early-stopping monitor during the short stage-1 restart-training phase.",
    )
    parser.add_argument(
        "--stage1_select_metric",
        type=str,
        default=STAGE1_SELECT_METRIC,
        choices=TRAIN_MONITOR_CHOICES,
        help="Metric used to choose which random restart/seed is kept after stage 1.",
    )
    parser.add_argument(
        "--stage2_es_metric",
        type=str,
        default=STAGE2_ES_METRIC,
        choices=TRAIN_MONITOR_CHOICES,
        help="Early-stopping monitor during the longer stage-2 continuation/refit phase.",
    )
    parser.add_argument(
        "--hpo_use_random_restart",
        type=str2bool,
        default=HPO_USE_RANDOM_RESTART,
        help="Whether HPO trials use random restarts. Final refit still uses random restart.",
    )
    parser.add_argument(
        "--loss_types",
        type=str,
        default=",".join(LOSS_TYPE_CHOICES),
        help=(
            "Comma-separated MBNDT loss-function choices for HPO. "
            "Use 'balanced_bce' for a fixed balanced-BCE run, or "
            "'bce,balanced_bce,focal' to tune the loss. "
            "Allowed: bce, balanced_bce, focal."
        ),
    )
    parser.add_argument(
        "--cuda_num",
        type=int,
        default=CUDA_NUM,
        help="CUDA device index. Example: --cuda_num 1 uses cuda:1.",
    )

    # Optional but useful for parallel/debug runs.
    parser.add_argument(
        "--n_hpo_trials",
        type=int,
        default=N_HPO_TRIALS,
        help="Maximum number of Optuna trials per outer split.",
    )
    parser.add_argument(
        "--hpo_timeout_sec",
        type=int,
        default=HPO_TIMEOUT_SEC,
        help="Optuna timeout in seconds per outer split.",
    )
    parser.add_argument(
        "--inner_repeats",
        type=int,
        default=INNER_REPEATS,
        help=(
            "Number of inner StratifiedKFold splits used to average the internal HPO objective. "
            "Default: 5."
        ),
    )
    parser.add_argument(
        "--use_mask",
        type=str2bool,
        default=USE_MASK,
        help="Whether to build MBNDT with branch masks. Use --use_mask false for no-mask runs.",
    )

    return parser.parse_args()


def apply_args(args):
    """
    Mutate global config variables from CLI arguments.
    Must be called before run_all_datasets().
    """
    global DATASETS, MONITOR_METRIC, RESTART_METRIC, LOSS_TYPE_CHOICES
    global STAGE1_ES_METRIC, STAGE1_SELECT_METRIC, STAGE2_ES_METRIC, HPO_USE_RANDOM_RESTART
    global ARTIFACT_BASE, CUDA_NUM
    global BATCH_SIZE, BATCH_MIN, BATCH_MAX, MIN_BATCHES_PER_EPOCH
    global N_HPO_TRIALS, HPO_TIMEOUT_SEC, INNER_REPEATS, USE_MASK
    global BDK_CHOICES

    DATASETS = parse_dataset_specs_arg(args.datasets)
    MONITOR_METRIC = str(args.monitor_metric)

    STAGE1_ES_METRIC = str(args.stage1_es_metric)
    STAGE1_SELECT_METRIC = str(args.stage1_select_metric)
    STAGE2_ES_METRIC = str(args.stage2_es_metric)
    HPO_USE_RANDOM_RESTART = bool(args.hpo_use_random_restart)

    # Backward-compatibility only. The stage-specific metrics above are authoritative.
    RESTART_METRIC = str(args.restart_metric)
    LOSS_TYPE_CHOICES = parse_loss_types_arg(args.loss_types)

    ARTIFACT_BASE = Path(args.artifact_base)
    CUDA_NUM = int(args.cuda_num)

    BATCH_SIZE = int(args.batch_size)
    BATCH_MIN = int(args.batch_min)
    BATCH_MAX = int(args.batch_max)
    MIN_BATCHES_PER_EPOCH = int(args.min_batches_per_epoch)

    N_HPO_TRIALS = int(args.n_hpo_trials)
    HPO_TIMEOUT_SEC = int(args.hpo_timeout_sec)
    INNER_REPEATS = int(args.inner_repeats)

    if INNER_REPEATS < 2:
        raise ValueError(f"INNER_REPEATS must be at least 2 for StratifiedKFold, got {INNER_REPEATS}.")

    USE_MASK = bool(args.use_mask)

    # Basic consistency warnings. We do not force equality because loss/loss/target is intentional
    # for balanced-accuracy runs, but the printed config makes the choice explicit.
    if MONITOR_METRIC == "balanced_acc" and STAGE2_ES_METRIC != "val_bacc":
        print(f"[WARN] MONITOR_METRIC=balanced_acc but STAGE2_ES_METRIC={STAGE2_ES_METRIC}. Expected val_bacc for the main balanced-accuracy scheme.")
    if MONITOR_METRIC == "aucroc" and STAGE2_ES_METRIC != "val_auc":
        print(f"[WARN] MONITOR_METRIC=aucroc but STAGE2_ES_METRIC={STAGE2_ES_METRIC}. Expected val_auc for the main AUROC scheme.")

    # If future arguments change B_CHOICES, D_CHOICES, or the K grid, this keeps choices consistent.
    BDK_CHOICES = build_B_depth_leaf_budget_triples()

    if BATCH_MIN <= 0 or BATCH_MAX <= 0:
        raise ValueError("BATCH_MIN and BATCH_MAX must be positive.")
    if BATCH_MIN > BATCH_MAX:
        raise ValueError(f"BATCH_MIN ({BATCH_MIN}) cannot be greater than BATCH_MAX ({BATCH_MAX}).")
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive.")
    if MIN_BATCHES_PER_EPOCH <= 0:
        raise ValueError("MIN_BATCHES_PER_EPOCH must be positive.")

    print("\n[CLI CONFIG APPLIED]")
    print(f"  DATASETS:        {DATASETS}")
    print(f"  MONITOR_METRIC:        {MONITOR_METRIC}")
    print(f"  STAGE1_ES_METRIC:     {STAGE1_ES_METRIC}")
    print(f"  STAGE1_SELECT_METRIC: {STAGE1_SELECT_METRIC}")
    print(f"  STAGE2_ES_METRIC:     {STAGE2_ES_METRIC}")
    print(f"  HPO_USE_RANDOM_RESTART: {HPO_USE_RANDOM_RESTART}")
    print(f"  RESTART_METRIC (compat only): {RESTART_METRIC}")
    print(f"  LOSS_TYPE_CHOICES: {LOSS_TYPE_CHOICES}")
    print(f"  ARTIFACT_BASE:   {ARTIFACT_BASE}")
    print(f"  BATCH (fixed rule): target={BATCH_SIZE}, min/max={BATCH_MIN}/{BATCH_MAX}, min_batches/epoch={MIN_BATCHES_PER_EPOCH}")
    print(f"  CUDA_NUM:        {CUDA_NUM}")
    print(f"  USE_MASK:        {USE_MASK}")
    print(f"  N_HPO_TRIALS:    {N_HPO_TRIALS}")
    print(f"  INNER_REPEATS:   {INNER_REPEATS}")
    print(f"  HPO_TIMEOUT_SEC: {HPO_TIMEOUT_SEC}")
    print(f"  B/D/K triples:   {BDK_CHOICES}")


if __name__ == "__main__":
    args = parse_args()
    apply_args(args)
    run_all_datasets()
