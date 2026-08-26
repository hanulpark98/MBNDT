from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, LabelEncoder

import category_encoders as ce

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def detect_num_classes(y_train, y_valid=None, y_test=None) -> int:
    y_tr = np.asarray(y_train)
    uniq_tr = np.unique(y_tr[~pd.isna(y_tr)])
    n = len(uniq_tr)
    if n < 2:
        raise ValueError(f"Need ≥2 classes in train; got {n}: {uniq_tr}")
    expected = set(range(n))
    if set(uniq_tr.astype(int).tolist()) != expected:
        raise ValueError(
            f"Labels must be {sorted(expected)}, got {sorted(uniq_tr.tolist())}. "
            "Apply LabelEncoder first."
        )
    for name, y in [("valid", y_valid), ("test", y_test)]:
        if y is None: continue
        unseen = set(np.unique(np.asarray(y)).tolist()) - set(uniq_tr.tolist())
        if unseen:
            raise ValueError(f"{name} set has unseen classes: {unseen}")
    return 1 if n == 2 else int(n)


@dataclass
class PreprocessConfig:
    # 1) split
    test_size: float = 0.2
    use_val: bool = True
    val_size: float = 0.2          # proportion of TRAIN that becomes VAL
    random_state: int = 42
    stratify: bool = True

    # 2) imputation
    impute: Union[str, bool] = "median"   # {"median","mean", False}

    # 3) categorical encoding
    encode_categoricals: bool = True
    cardinality_threshold: int = 10       # C: LOO if > C, OHE if <= C
    te_sigma: float = 0.0                 # noise regularization for LOO

    # 4) numerical stability
    numerical_stability: bool = False
    clip_lo: float = 0.001
    clip_hi: float = 0.999
    log1p_skew_thresh: float = 2.0

    # 5) SMOTE
    use_smote: bool = False
    smote_k_neighbors: int = 5

    # 6) scaling
    scale: bool = False
    qt_output_distribution: str = "normal"   # keep fixed; toggle is scale True/False

    # 7) torch
    to_torch: bool = False


@dataclass
class PreprocessResult:
    X_train: Any
    X_valid: Any
    X_test: Any
    y_train: Any
    y_valid: Any
    y_test: Any
    meta: Dict[str, Any]


def _infer_columns(X: pd.DataFrame,
                   cat_cols: Optional[List[str]] = None,
                   num_cols: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    if cat_cols is None and num_cols is None:
        # treat object/category as categorical; numbers + bool as numeric
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        return cat_cols, num_cols

    if cat_cols is None:
        cat_cols = [c for c in X.columns if c not in (num_cols or [])]
    if num_cols is None:
        num_cols = [c for c in X.columns if c not in (cat_cols or [])]
    return list(cat_cols), list(num_cols)

def _fit_apply_imputers(
    X_tr: pd.DataFrame, X_va: Optional[pd.DataFrame], X_te: pd.DataFrame,
    cat_cols: List[str], num_cols: List[str],
    cfg, meta: Dict[str, Any]
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    if cfg.impute is False:
        meta["imputation"] = False
        return X_tr, X_va, X_te
    if cfg.impute not in {"median", "mean"}:
        raise ValueError(f'impute must be "median", "mean", or False, got: {cfg.impute}')

    original_cols = X_tr.columns.tolist()

    # ✅ FIX: Validate that num_cols are actually numeric
    actual_num_cols = []
    misclassified_cols = []
    for col in num_cols:
        if pd.api.types.is_numeric_dtype(X_tr[col]):
            actual_num_cols.append(col)
        else:
            misclassified_cols.append(col)
            cat_cols.append(col)  # Move to categorical

    if misclassified_cols:
        print(f"Warning: Columns {misclassified_cols} were labeled as numeric but contain non-numeric data. Moving to categorical.")

    num_cols = actual_num_cols  # Use validated list

    # --------------------
    # Numeric imputer
    # --------------------
    num_imp = None
    if num_cols:
        num_imp = SimpleImputer(strategy=cfg.impute)
        X_tr_num = pd.DataFrame(num_imp.fit_transform(X_tr[num_cols]), columns=num_cols, index=X_tr.index)
        X_te_num = pd.DataFrame(num_imp.transform(X_te[num_cols]), columns=num_cols, index=X_te.index)
        X_va_num = None
        if X_va is not None:
            X_va_num = pd.DataFrame(num_imp.transform(X_va[num_cols]), columns=num_cols, index=X_va.index)
    else:
        X_tr_num = pd.DataFrame(index=X_tr.index)
        X_te_num = pd.DataFrame(index=X_te.index)
        X_va_num = None if X_va is None else pd.DataFrame(index=X_va.index)

    # --------------------
    # Categorical imputer
    # --------------------
    cat_imp = None
    if cat_cols:
        X_tr_cat = X_tr[cat_cols].copy()
        X_te_cat = X_te[cat_cols].copy()
        X_va_cat = X_va[cat_cols].copy() if X_va is not None else None

        X_tr_cat = X_tr_cat.astype("object")
        X_te_cat = X_te_cat.astype("object")
        if X_va_cat is not None:
            X_va_cat = X_va_cat.astype("object")

        cat_imp = SimpleImputer(strategy="constant", fill_value="__MISSING__")
        X_tr_cat = pd.DataFrame(cat_imp.fit_transform(X_tr_cat), columns=cat_cols, index=X_tr.index)
        X_te_cat = pd.DataFrame(cat_imp.transform(X_te_cat), columns=cat_cols, index=X_te.index)
        if X_va is not None:
            X_va_cat = pd.DataFrame(cat_imp.transform(X_va_cat), columns=cat_cols, index=X_va.index)
    else:
        X_tr_cat = pd.DataFrame(index=X_tr.index)
        X_te_cat = pd.DataFrame(index=X_te.index)
        X_va_cat = None if X_va is None else pd.DataFrame(index=X_va.index)

    # --------------------
    # Recombine in original column order
    # --------------------
    def _recombine(X_num, X_cat, original_cols):
        out = pd.concat([X_num, X_cat], axis=1)
        return out.loc[:, original_cols]

    X_tr2 = _recombine(X_tr_num, X_tr_cat, original_cols)
    X_te2 = _recombine(X_te_num, X_te_cat, original_cols)
    X_va2 = None if X_va is None else _recombine(X_va_num, X_va_cat, original_cols)

    meta["imputation"] = {"numeric": cfg.impute, "categorical": "constant(__MISSING__)"}
    meta["num_imputer"] = num_imp
    meta["cat_imputer"] = cat_imp

    return X_tr2, X_va2, X_te2


# def _fit_apply_cat_encoding(X_tr: pd.DataFrame, X_va: Optional[pd.DataFrame], X_te: pd.DataFrame,
#                             y_tr: np.ndarray,
#                             cat_cols: List[str], cfg: PreprocessConfig, meta: Dict[str, Any]
#                             ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
#     if not cfg.encode_categoricals or not cat_cols:
#         meta["cat_encoding"] = False
#         return X_tr, X_va, X_te

#     # force consistent string representation
#     for c in cat_cols:
#         X_tr[c] = X_tr[c].astype("object").astype(str)
#         X_te[c] = X_te[c].astype("object").astype(str)
#         if X_va is not None:
#             X_va[c] = X_va[c].astype("object").astype(str)

#     # split into high/low-cardinality using TRAIN only
#     card = {c: int(X_tr[c].nunique(dropna=False)) for c in cat_cols}
#     hi = [c for c in cat_cols if card[c] > cfg.cardinality_threshold]
#     lo = [c for c in cat_cols if card[c] <= cfg.cardinality_threshold]

#     meta["cat_encoding"] = {"threshold_C": cfg.cardinality_threshold, "high_card": hi, "low_card": lo}

#     # High-card -> Leave-One-Out encoding
#     if hi:
#         loo = ce.LeaveOneOutEncoder(
#             cols=hi,
#             sigma=cfg.te_sigma,
#             handle_unknown="value",
#             handle_missing="value",
#             random_state=cfg.random_state,
#         )
#         loo.fit(X_tr, y_tr)
#         X_tr = loo.transform(X_tr, y_tr)      # TRAIN uses y (LOO)
#         X_te = loo.transform(X_te)            # VAL/TEST no y
#         if X_va is not None:
#             X_va = loo.transform(X_va)
#         meta["loo_encoder"] = loo

#     # Low-card -> One-hot encoding
#     if lo:
#         ohe = ce.OneHotEncoder(
#             cols=lo,
#             use_cat_names=True,
#             drop_invariant=True,
#             handle_unknown="value",
#             handle_missing="value",
#         )
#         ohe.fit(X_tr)
#         X_tr = ohe.transform(X_tr)
#         X_te = ohe.transform(X_te)
#         if X_va is not None:
#             X_va = ohe.transform(X_va)
#         meta["ohe_encoder"] = ohe

#     return X_tr, X_va, X_te

def _fit_apply_cat_encoding(
    X_tr: pd.DataFrame,
    X_va: Optional[pd.DataFrame],
    X_te: pd.DataFrame,
    y_tr: np.ndarray,
    cat_cols: List[str],
    cfg: PreprocessConfig,
    meta: Dict[str, Any],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    if not cfg.encode_categoricals or not cat_cols:
        meta["cat_encoding"] = False
        return X_tr, X_va, X_te

    # force consistent string representation
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype("object").astype(str)
        X_te[c] = X_te[c].astype("object").astype(str)
        if X_va is not None:
            X_va[c] = X_va[c].astype("object").astype(str)

    # split into high/low-cardinality using TRAIN only
    card = {c: int(X_tr[c].nunique(dropna=False)) for c in cat_cols}
    hi = [c for c in cat_cols if card[c] > cfg.cardinality_threshold]
    lo = [c for c in cat_cols if card[c] <= cfg.cardinality_threshold]

    # detect task type from y_train
    n_classes = int(np.unique(np.asarray(y_tr)).size)
    is_binary = (n_classes <= 2)

    hi_loo = hi if is_binary else []
    hi_count = hi if not is_binary else []

    meta["cat_encoding"] = {
        "threshold_C": cfg.cardinality_threshold,
        "n_classes": n_classes,
        "high_card": hi,
        "high_card_loo": hi_loo,
        "high_card_count": hi_count,
        "low_card": lo,
    }

    # High-card binary -> Leave-One-Out encoding
    if hi_loo:
        loo = ce.LeaveOneOutEncoder(
            cols=hi_loo,
            sigma=cfg.te_sigma,
            handle_unknown="value",
            handle_missing="value",
            random_state=cfg.random_state,
        )
        loo.fit(X_tr, y_tr)
        X_tr = loo.transform(X_tr, y_tr)   # TRAIN uses y
        X_te = loo.transform(X_te)
        if X_va is not None:
            X_va = loo.transform(X_va)
        meta["loo_encoder"] = loo

    # High-card multiclass -> Count encoding
    if hi_count:
        cnt = ce.CountEncoder(
            cols=hi_count,
            handle_unknown="value",
            handle_missing="value",
            normalize=False,
        )
        cnt.fit(X_tr)
        X_tr = cnt.transform(X_tr)
        X_te = cnt.transform(X_te)
        if X_va is not None:
            X_va = cnt.transform(X_va)
        meta["count_encoder"] = cnt

    # Low-card -> One-hot encoding
    if lo:
        ohe = ce.OneHotEncoder(
            cols=lo,
            use_cat_names=True,
            drop_invariant=True,
            handle_unknown="value",
            handle_missing="value",
        )
        ohe.fit(X_tr)
        X_tr = ohe.transform(X_tr)
        X_te = ohe.transform(X_te)
        if X_va is not None:
            X_va = ohe.transform(X_va)
        meta["ohe_encoder"] = ohe

    return X_tr, X_va, X_te


def _apply_numerical_stability(X_tr: pd.DataFrame, X_va: Optional[pd.DataFrame], X_te: pd.DataFrame,
                              cfg: PreprocessConfig, meta: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    if not cfg.numerical_stability:
        meta["numerical_stability"] = False
        return X_tr, X_va, X_te

    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        meta["numerical_stability"] = {"enabled": True, "note": "no numeric columns"}
        return X_tr, X_va, X_te

    # Clip by TRAIN quantiles
    q_lo = X_tr[num_cols].quantile(cfg.clip_lo)
    q_hi = X_tr[num_cols].quantile(cfg.clip_hi)

    def _clip(df):
        df = df.copy()
        df[num_cols] = df[num_cols].clip(lower=q_lo, upper=q_hi, axis=1)
        return df

    X_tr = _clip(X_tr)
    X_te = _clip(X_te)
    if X_va is not None:
        X_va = _clip(X_va)

    # Log1p on very skewed nonnegative columns (decided on TRAIN)
    skew = X_tr[num_cols].skew(numeric_only=True)
    log_cols = [c for c, s in skew.items() if np.isfinite(s) and (s > cfg.log1p_skew_thresh)]

    applied = []
    for c in log_cols:
        ok = (X_tr[c] >= 0).all() and (X_te[c] >= 0).all() and (True if X_va is None else (X_va[c] >= 0).all())
        if ok:
            X_tr[c] = np.log1p(X_tr[c])
            X_te[c] = np.log1p(X_te[c])
            if X_va is not None:
                X_va[c] = np.log1p(X_va[c])
            applied.append(c)

    meta["numerical_stability"] = {
        "enabled": True,
        "clip": (cfg.clip_lo, cfg.clip_hi),
        "log1p_skew_thresh": cfg.log1p_skew_thresh,
        "log1p_applied_cols": applied,
    }
    return X_tr, X_va, X_te


def _ensure_numeric_matrix(df: pd.DataFrame, step_name: str):
    bad = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(
            f"{step_name} requires all features to be numeric, but found non-numeric columns: {bad[:10]}"
            + (" ..." if len(bad) > 10 else "")
        )


def _maybe_label_encode_y(y_tr, y_va, y_te, meta) -> Tuple[np.ndarray, Any, Any]:
    # If y is non-numeric (strings, categories), encode to ints (especially needed for SMOTE).
    y_tr_arr = np.asarray(y_tr)
    needs = not np.issubdtype(y_tr_arr.dtype, np.number)
    if not needs:
        return y_tr_arr, (None if y_va is None else np.asarray(y_va)), np.asarray(y_te)

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr_arr)
    y_te_enc = le.transform(np.asarray(y_te))
    y_va_enc = None if y_va is None else le.transform(np.asarray(y_va))
    meta["label_encoder"] = le
    return y_tr_enc, y_va_enc, y_te_enc


def _apply_smote(X_tr: pd.DataFrame, y_tr: np.ndarray, cfg: PreprocessConfig, meta: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
    if not cfg.use_smote:
        meta["smote"] = False
        return X_tr, y_tr
    if not _HAS_SMOTE:
        raise ImportError("use_smote=True but imblearn is not installed. `pip install imbalanced-learn`")

    _ensure_numeric_matrix(X_tr, "SMOTE")

    # SMOTE needs enough minority samples for k_neighbors
    y_int = np.asarray(y_tr).astype(int)
    _, counts = np.unique(y_int, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 0
    if min_count <= 1:
        raise ValueError(f"SMOTE requires at least 2 samples in every class; got min class count = {min_count}")

    k = min(cfg.smote_k_neighbors, min_count - 1)
    sm = SMOTE(random_state=cfg.random_state, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_tr.values.astype(np.float64), y_int)
    X_res = pd.DataFrame(X_res, columns=X_tr.columns, index=None)

    meta["smote"] = {"enabled": True, "k_neighbors_used": k}
    return X_res, y_res


def _apply_scaling(X_tr: pd.DataFrame, X_va: Optional[pd.DataFrame], X_te: pd.DataFrame,
                   cfg: PreprocessConfig, meta: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    if not cfg.scale:
        meta["scaling"] = False
        return X_tr.values if isinstance(X_tr, pd.DataFrame) else np.asarray(X_tr), \
               None if X_va is None else (X_va.values if isinstance(X_va, pd.DataFrame) else np.asarray(X_va)), \
               X_te.values if isinstance(X_te, pd.DataFrame) else np.asarray(X_te)

    _ensure_numeric_matrix(X_tr, "Scaling")
    if X_va is not None:
        _ensure_numeric_matrix(X_va, "Scaling")
    _ensure_numeric_matrix(X_te, "Scaling")

    Xtr = X_tr.values.astype(np.float64)
    Xte = X_te.values.astype(np.float64)
    Xva = None if X_va is None else X_va.values.astype(np.float64)

    # Jitter TRAIN to reduce ties for quantile mapping
    stds = np.std(Xtr, axis=0, keepdims=True)
    noise_std = 1e-4 / np.maximum(stds, 1e-4)
    rng = np.random.RandomState(cfg.random_state)
    Xtr_j = Xtr + noise_std * rng.randn(*Xtr.shape)

    qt = QuantileTransformer(output_distribution=cfg.qt_output_distribution, random_state=cfg.random_state)
    qt.fit(Xtr_j)

    Xtr_s = qt.transform(Xtr)
    Xte_s = qt.transform(Xte)
    Xva_s = None if Xva is None else qt.transform(Xva)

    meta["scaling"] = {"enabled": True, "type": "QuantileTransformer", "output_distribution": cfg.qt_output_distribution}
    meta["scaler"] = qt
    return Xtr_s, Xva_s, Xte_s


def _to_torch(X_tr, X_va, X_te, y_tr, y_va, y_te, meta):
    if not _HAS_TORCH:
        raise ImportError("to_torch=True but torch is not installed.")

    X_train_t = torch.tensor(np.asarray(X_tr), dtype=torch.float32)
    X_test_t  = torch.tensor(np.asarray(X_te), dtype=torch.float32)
    X_valid_t = None if X_va is None else torch.tensor(np.asarray(X_va), dtype=torch.float32)

    # Single source of truth: trust meta if already populated upstream,
    # else compute now. detect_num_classes returns 1 for binary, C for multiclass.
    if "num_classes" not in meta:
        meta["num_classes"] = detect_num_classes(y_tr)
    is_binary = (meta["num_classes"] == 1)

    if is_binary:
        y_train_t = torch.tensor(np.asarray(y_tr).astype(np.float32), dtype=torch.float32)
        y_test_t  = torch.tensor(np.asarray(y_te).astype(np.float32), dtype=torch.float32)
        y_valid_t = None if y_va is None else \
            torch.tensor(np.asarray(y_va).astype(np.float32), dtype=torch.float32)
        meta["y_torch_dtype"] = "float32(binary)"
    else:
        y_train_t = torch.tensor(np.asarray(y_tr).astype(np.int64), dtype=torch.long)
        y_test_t  = torch.tensor(np.asarray(y_te).astype(np.int64), dtype=torch.long)
        y_valid_t = None if y_va is None else \
            torch.tensor(np.asarray(y_va).astype(np.int64), dtype=torch.long)
        meta["y_torch_dtype"] = f"int64(multiclass, C={meta['num_classes']})"

    return X_train_t, X_valid_t, X_test_t, y_train_t, y_valid_t, y_test_t


def preprocess_dataset(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray, List[Any]],
    cfg: PreprocessConfig,
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
) -> PreprocessResult:
    """
    Toggle-based preprocessing:
      1) split (optional val)
      2) impute (mean/median/False)
      3) categorical encoding (LOO for >C, OHE for <=C) or off
      4) numerical stability (clip + optional log1p) or off
      5) SMOTE or off
      6) scaling (QuantileTransformer->normal) or off
      7) to_torch or off
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    y = pd.Series(y, index=X.index)

    meta: Dict[str, Any] = {"config": cfg}

    # 1) split
    strat = y if cfg.stratify else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=strat
    )
    X_va = y_va = None
    if cfg.use_val:
        strat2 = y_tr if cfg.stratify else None
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_tr, y_tr, test_size=cfg.val_size, random_state=cfg.random_state, stratify=strat2
        )

    meta["split"] = {
        "test_size": cfg.test_size,
        "use_val": cfg.use_val,
        "val_size": cfg.val_size if cfg.use_val else None,
        "n_train": len(X_tr),
        "n_valid": (None if X_va is None else len(X_va)),
        "n_test": len(X_te),
    }

    # infer columns on TRAIN only
    cat_cols, num_cols = _infer_columns(X_tr, cat_cols=cat_cols, num_cols=num_cols)
    meta["cat_cols_inferred"] = cat_cols
    meta["num_cols_inferred"] = num_cols

    # 2) imputation
    X_tr, X_va, X_te = _fit_apply_imputers(X_tr, X_va, X_te, cat_cols, num_cols, cfg, meta)

    # y arrays (keep as-is for now; label encoding may be applied if SMOTE needs it)
    y_tr_arr = np.asarray(y_tr)
    y_te_arr = np.asarray(y_te)
    y_va_arr = None if y_va is None else np.asarray(y_va)

    # 3) categorical encoding
    X_tr, X_va, X_te = _fit_apply_cat_encoding(X_tr, X_va, X_te, y_tr_arr, cat_cols, cfg, meta)

    # 4) numerical stability
    X_tr, X_va, X_te = _apply_numerical_stability(X_tr, X_va, X_te, cfg, meta)

    meta["num_classes"] = detect_num_classes(y_tr_arr, y_va_arr, y_te_arr)

    # 5) SMOTE (requires numeric X and integer y)
    if cfg.use_smote:
        y_tr_arr, y_va_arr, y_te_arr = _maybe_label_encode_y(y_tr_arr, y_va_arr, y_te_arr, meta)
        X_tr, y_tr_arr = _apply_smote(X_tr, y_tr_arr, cfg, meta)

    # record feature names before possible scaling->numpy conversion
    meta["feature_names_after"] = list(X_tr.columns) if isinstance(X_tr, pd.DataFrame) else [f"f{i}" for i in range(X_tr.shape[1])]

    # 6) scaling
    X_tr_out, X_va_out, X_te_out = _apply_scaling(X_tr, X_va, X_te, cfg, meta)

    # 7) torch
    if cfg.to_torch:
        X_tr_out, X_va_out, X_te_out, y_tr_out, y_va_out, y_te_out = _to_torch(
            X_tr_out, X_va_out, X_te_out, y_tr_arr, y_va_arr, y_te_arr, meta
        )
    else:
        y_tr_out, y_va_out, y_te_out = y_tr_arr, y_va_arr, y_te_arr

    return PreprocessResult(
        X_train=X_tr_out, X_valid=X_va_out, X_test=X_te_out,
        y_train=y_tr_out, y_valid=y_va_out, y_test=y_te_out,
        meta=meta
    )




def preprocess_splits(
    X_train: pd.DataFrame,
    y_train,
    cfg: PreprocessConfig,
    X_valid: Optional[pd.DataFrame] = None,
    y_valid=None,
    X_test: Optional[pd.DataFrame] = None,
    y_test=None,
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
) -> PreprocessResult:
    """
    Fit ALL preprocessing on (X_train, y_train) only, then transform X_valid/X_test.
    Does NOT do any splitting.
    """
    meta: Dict[str, Any] = {"config": cfg}

    X_tr = X_train.copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    X_va = None if X_valid is None else (X_valid.copy() if isinstance(X_valid, pd.DataFrame) else pd.DataFrame(X_valid))
    X_te = None if X_test  is None else (X_test.copy()  if isinstance(X_test,  pd.DataFrame) else pd.DataFrame(X_test))

    y_tr = np.asarray(y_train)
    y_va = None if y_valid is None else np.asarray(y_valid)
    y_te = None if y_test  is None else np.asarray(y_test)

    # infer columns from TRAIN only
    cat_cols, num_cols = _infer_columns(X_tr, cat_cols=cat_cols, num_cols=num_cols)
    meta["cat_cols_inferred"] = cat_cols
    meta["num_cols_inferred"] = num_cols

    # 2) imputation
    X_tr, X_va, X_te = _fit_apply_imputers(X_tr, X_va, X_te, cat_cols, num_cols, cfg, meta)

    # 3) categorical encoding (fit on train only)
    if cfg.encode_categoricals and cat_cols:
        # if X_te is None, pass dummy DF through and discard after
        _X_te = X_te if X_te is not None else X_tr.iloc[:0].copy()
        X_tr, X_va, _X_te = _fit_apply_cat_encoding(X_tr, X_va, _X_te, y_tr, cat_cols, cfg, meta)
        X_te = None if X_test is None else _X_te

    # 4) numerical stability
    _X_te = X_te if X_te is not None else X_tr.iloc[:0].copy()
    X_tr, X_va, _X_te = _apply_numerical_stability(X_tr, X_va, _X_te, cfg, meta)
    X_te = None if X_test is None else _X_te

    meta["num_classes"] = detect_num_classes(y_tr, y_va, y_te)

    # 5) SMOTE (train only)
    if cfg.use_smote:
        y_tr2, y_va2, y_te2 = _maybe_label_encode_y(y_tr, y_va, (y_te if y_te is not None else y_tr[:0]), meta)
        X_tr, y_tr2 = _apply_smote(X_tr, y_tr2, cfg, meta)
        y_tr, y_va = y_tr2, y_va2
        if X_test is not None:
            y_te = y_te2

    meta["feature_names_after"] = list(X_tr.columns)

    # 6) scaling
    _X_te = X_te if X_te is not None else X_tr.iloc[:0].copy()
    X_tr_out, X_va_out, _X_te_out = _apply_scaling(X_tr, X_va, _X_te, cfg, meta)
    X_te_out = None if X_test is None else _X_te_out

    # 7) torch
    if cfg.to_torch:
        X_tr_out, X_va_out, X_te_out, y_tr, y_va, y_te = _to_torch(
            X_tr_out,
            X_va_out,
            (X_te_out if X_te_out is not None else np.zeros((0, np.asarray(X_tr_out).shape[1]))),
            y_tr, y_va,
            (y_te if y_te is not None else np.zeros((0,), dtype=np.asarray(y_tr).dtype)),
            meta
        )
        if X_test is None:
            X_te_out, y_te = None, None

    return PreprocessResult(
        X_train=X_tr_out, X_valid=X_va_out, X_test=X_te_out,
        y_train=y_tr, y_valid=y_va, y_test=y_te,
        meta=meta
    )




__all__ = ["PreprocessConfig", "PreprocessResult", "preprocess_dataset",  "preprocess_splits", "detect_num_classes"]
