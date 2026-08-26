import torch
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any

# -------------------------
# Config dataclasses
# -------------------------
@dataclass
class ModelHP:
    n_features: int
    D: int = 5
    B: int = 3
    num_classes: int = 1
    #task: str = "binary"

    # gating choices
    selector_mode: str = "entmax_st"   # {"entmax_st","reinmax","entmax_soft"}
    branch_mode:   str = "st"          # {"st","reinmax","soft"}  (soft = no hard routing)
    use_masks: bool = True

    # temperatures
    tau_cdf: float = 0.6               # your existing CDF temp (t vs x)
    #tau_feat: float      = 1.5,
    #tau_branch: float    = 1.5,

@dataclass
class OptimHP:
    lr_feature: float = 1e-2
    lr_thresh: float  = 1e-2
    lr_leaf: float    = 1e-2
    lr_mask: float    = 1e-2

@dataclass
class TrainingHP:
    random_restart: bool = True
    n_restarts: int      = 5
    base_seed: int       = 2025

    # Optional backward compatibility
    restart_metric: str  = "val_bacc"

    # Explicit metric controls
    stage1_es_metric: str = "val_bacc"
    stage1_select_metric: str = "val_bacc"
    stage2_es_metric: str = "val_bacc"

    stage1_epoch: int    = 40
    stage1_patience: int = 8

    stage2_epoch: int    = 250
    stage2_patience: int = 25

    loss_type: str = "bce"
    verbose: bool = True

# ---- Regularizers (each has on/off + specific knobs) ----
@dataclass
class LoadBalanceHP:
    on: bool = True
    weight: float = 1e-3                 # == lambda
    mode: Literal["kl"] = "kl"            # fix; don't HPO
    warmup_frac: float = 0.10            # optional HPO
    K_gate: float = 2.0                  # fixed
    schedule: Dict[str, Any] = field(default_factory=lambda: {
        "type": "warmup_linear_decay",   # or cosine
        "warmup_frac": 0.10,
    })

@dataclass
class LeafBudgetHP:
    on: bool = True
    K: float = 8.0
    # Supported by patched train_model/default_reg_config:
    #   branch_log:    relu(log L_branch - log K)
    #   branch_linear: relu(L_branch - K)
    #   mass_log:      relu(log L_mass - log K)
    #   mass_linear:   relu(L_mass - K)
    mode: str = "mass_log"

@dataclass
class RegularizersHP:
    load_balance: LoadBalanceHP = field(default_factory=LoadBalanceHP)
    leaf_budget:  LeafBudgetHP  = field(default_factory=LeafBudgetHP)

@dataclass
class MBNDTConfig:
    model: ModelHP
    optim: OptimHP = field(default_factory=OptimHP)
    training: TrainingHP = field(default_factory=TrainingHP)
    regularizers: RegularizersHP = field(default_factory=RegularizersHP)

def reg_to_train_config(B: int, regularizers) -> Dict[str, Any]:
    return {
        "load_balance": {
            "on":          regularizers.load_balance.on,
            "lambda":      regularizers.load_balance.weight,
            "mode":        regularizers.load_balance.mode,
            "warmup_frac": regularizers.load_balance.warmup_frac,
            "K_gate":      float(min(B, regularizers.load_balance.K_gate)),
            "schedule":    regularizers.load_balance.schedule,
        },
        "leaf_budget": {
            "on":   regularizers.leaf_budget.on,
            "K":    float(regularizers.leaf_budget.K),
            "mode": str(getattr(regularizers.leaf_budget, "mode", "mass_log")),
        },
    }

def build_model_from_cfg(NT5, cfg: MBNDTConfig, device):
    model = NT5.MBNDT(
        n_features=cfg.model.n_features,
        D=cfg.model.D,
        B=cfg.model.B,
        num_classes=cfg.model.num_classes,
        #task=cfg.model.task,       # "binary" | "multiclass" | "regression"

        # gating choices
        selector_mode=cfg.model.selector_mode,
        branch_mode=cfg.model.branch_mode,
        use_masks=cfg.model.use_masks,
        # temperatures
        tau_cdf=cfg.model.tau_cdf,
        #tau_feat=cfg.model.tau_feat,
        #tau_branch=cfg.model.tau_branch,

    ).to(device)
    return model

def train_from_cfg(NT5, cfg: MBNDTConfig, train_loader, val_loader, device):
    # ------------------------------------------------------------
    # Regularizers
    # ------------------------------------------------------------
    reg_cfg = reg_to_train_config(B=cfg.model.B, regularizers=cfg.regularizers)

    tau_sched = {
        "type": "warmup_cosine",
        "start": 3.0,
        "end": float(cfg.model.tau_cdf),
        "warmup_frac": 0.10,
        "min": 0.2,
    }

    # ------------------------------------------------------------
    # Metric fallback logic
    # ------------------------------------------------------------
    # These getattr(...) fallbacks keep old configs from crashing.
    # But for the clean version, you should add these fields to TrainingHP.
    stage1_es_metric = getattr(
        cfg.training,
        "stage1_es_metric",
        getattr(cfg.training, "restart_metric", "val_bacc"),
    )

    stage1_select_metric = getattr(
        cfg.training,
        "stage1_select_metric",
        getattr(cfg.training, "restart_metric", "val_bacc"),
    )

    stage2_es_metric = getattr(
        cfg.training,
        "stage2_es_metric",
        getattr(cfg.training, "restart_metric", "val_bacc"),
    )

    # ------------------------------------------------------------
    # Random-restart path
    # ------------------------------------------------------------
    if getattr(cfg.training, "random_restart", False):
        final_model, best_info, rr_hist = NT5.random_restart_train(
            NT5,
            cfg,
            train_loader,
            val_loader,
            device=device,
            test_loader=None,

            n_restarts=cfg.training.n_restarts,
            base_seed=cfg.training.base_seed,

            # Stage 1: short training window
            stage1_epochs=cfg.training.stage1_epoch,
            stage1_patience=cfg.training.stage1_patience,
            stage1_es_monitor=stage1_es_metric,

            # Stage 1: choose best restart / seed
            stage1_select_monitor=stage1_select_metric,

            # Stage 2: continuation training from selected restart
            stage2_epochs=cfg.training.stage2_epoch,
            stage2_patience=cfg.training.stage2_patience,
            stage2_es_monitor=stage2_es_metric,

            verbose=cfg.training.verbose,

            reg_config=reg_cfg,
            continue_after_selection=True,
            return_histories=True,

            # tau_cdf_schedule=tau_sched,
        )

        # Keep stage2 history, but attach RR metadata
        history2 = rr_hist.get("stage2", {})
        if not isinstance(history2, dict):
            history2 = {"stage2_history": history2}

        history = dict(history2)
        history["rr_timing"] = rr_hist.get("timing", None)
        history["rr_best_info"] = rr_hist.get("best_info", best_info)

        # Store all metric choices explicitly
        history["stage1_es_metric"] = stage1_es_metric
        history["stage1_select_metric"] = stage1_select_metric
        history["stage2_es_metric"] = stage2_es_metric

        model = final_model

    # ------------------------------------------------------------
    # Non-random-restart path
    # ------------------------------------------------------------
    else:
        model = build_model_from_cfg(NT5, cfg, device).to(device)

        history = NT5.train_model(
            model,
            train_loader,
            val_loader,
            epochs=cfg.training.stage2_epoch,
            patience=cfg.training.stage2_patience,

            lr_feature=cfg.optim.lr_feature,
            lr_thresh=cfg.optim.lr_thresh,
            lr_leaf=cfg.optim.lr_leaf,
            lr_mask=cfg.optim.lr_mask,

            reg_config=reg_cfg,
            device=device,
            verbose=cfg.training.verbose,
            loss_type=cfg.training.loss_type,

            # IMPORTANT:
            # This makes non-RR training use the same configured ES metric.
            monitor=stage2_es_metric,
            force_load_best_at_end=True,

            # tau_cdf_schedule=tau_sched,
        )

        history["stage2_es_metric"] = stage2_es_metric
        history["random_restart"] = False

    return model, history
