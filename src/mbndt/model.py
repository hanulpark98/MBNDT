import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15
import math
import sys
import copy
import numpy as np
from reinmax import reinmax
from collections import defaultdict
from tqdm.auto import tqdm, trange
import time
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from .config import (
    build_model_from_cfg
)
import os
import matplotlib.pyplot as plt
import random
from pathlib import Path
import json
from collections import Counter
import matplotlib
matplotlib.use("Agg")  # headless 저장 안전
import matplotlib.pyplot as plt
import optuna
from torch.cuda.amp import GradScaler, autocast


# ------------------------------
# fixed internal constants for leaf-budget augmented Lagrangian
# ------------------------------
# LEAF_BUDGET_RHO = 0.3
# LEAF_BUDGET_MU_INIT = 0.0
# LEAF_BUDGET_MU_LR = 0.01
# LEAF_BUDGET_EMA_BETA = 0.0


LEAF_BUDGET_RHO = 0.03
LEAF_BUDGET_MU_LR = 0.003
LEAF_BUDGET_EMA_BETA = 0.5
LEAF_BUDGET_MU_INIT = 0.0
# --------------------------
# --------- utils ----------
# --------------------------
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# stable standard normal CDF via erf
def normal_cdf(z):
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

def inv_softplus(x, eps=1e-6):
    return torch.log(torch.exp(x) - 1.0 + eps)

class StraightThroughOneHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs):
        with torch.no_grad():
            idx = probs.argmax(dim=1)
            out = F.one_hot(idx, num_classes=probs.size(1)).float()
        ctx.save_for_backward(probs)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class StraightThroughOneHot3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs):  # probs: [B, Nnodes, Bbranches]
        with torch.no_grad():
            idx = probs.argmax(dim=-1)         # [B, Nnodes]
            flat = idx.view(-1)
            one_hot = F.one_hot(flat, num_classes=probs.size(-1)).float()
            one_hot = one_hot.view(*probs.shape)
        ctx.save_for_backward(probs)
        return one_hot
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def _val_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = float("nan")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    except Exception:
        spec = float("nan")

    return {"auc": auc, "acc": acc, "bacc": bacc, "f1_macro": f1_macro}

# =============================================================================
# LOSS HELPERS
# =============================================================================

def weighted_bce_from_probs(p, y, w_pos=1.0, w_neg=1.0, eps=1e-8):
    p = p.clamp(eps, 1 - eps)
    return (-(w_pos * y) * torch.log(p) - (w_neg * (1 - y)) * torch.log(1 - p)).mean()


class FocalLossFromProbs(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, eps=1e-8):
        super().__init__()
        self.alpha, self.gamma, self.eps = float(alpha), float(gamma), float(eps)

    def forward(self, p, y):
        p = p.clamp(self.eps, 1 - self.eps)
        pt = p * y + (1 - p) * (1 - y)
        alpha_t = self.alpha * y + (1 - self.alpha) * (1 - y)
        return (-(alpha_t * (1 - pt).pow(self.gamma) * torch.log(pt))).mean()

def weighted_bce_with_logits(logits: torch.Tensor, y: torch.Tensor, w_pos: float = 1.0, w_neg: float = 1.0):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    w = y * w_pos + (1.0 - y) * w_neg
    return (w * bce).mean()

class FocalLossFromLogits(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.float()
        t = (2.0 * y - 1.0) * logits          # y=1 -> logits, y=0 -> -logits
        pt = torch.sigmoid(t)                 # stable
        logpt = -F.softplus(-t)               # log(sigmoid(t)), stable
        alpha_t = y * self.alpha + (1.0 - y) * (1.0 - self.alpha)
        loss = -alpha_t * (1.0 - pt) ** self.gamma * logpt
        return loss.mean()



def infer_class_weights_from_loader(train_loader):
    pos, neg = 0, 0
    for _, yb in train_loader:
        yb_np = yb.detach().cpu().numpy()
        pos += int((yb_np == 1).sum())
        neg += int((yb_np == 0).sum())
    total = pos + neg if (pos + neg) > 0 else 1
    w_pos = total / (2.0 * max(pos, 1))
    w_neg = total / (2.0 * max(neg, 1))
    return float(w_pos), float(w_neg)



def infer_class_weights_multiclass(train_loader, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, yb in train_loader:
        yb_np = yb.detach().cpu().long().numpy()
        for c in range(num_classes):
            counts[c] += int((yb_np == c).sum())
    total = counts.sum().clamp(min=1)
    # inverse-frequency, normalized so mean weight ≈ 1
    w = total / (num_classes * counts.clamp(min=1))
    return w.float()                                   # [C]

class FocalLossMultiClassFromLogits(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer(
            "alpha",
            None if alpha is None else torch.as_tensor(alpha, dtype=torch.float)
        )
    def forward(self, logits, y):                      # logits: [N,C], y: [N] long
        y = y.long()
        log_p  = F.log_softmax(logits, dim=-1)
        log_pt = log_p.gather(1, y.unsqueeze(-1)).squeeze(-1)
        pt     = log_pt.exp()
        loss   = -(1.0 - pt).pow(self.gamma) * log_pt
        if self.alpha is not None:
            loss = self.alpha.to(logits.device)[y] * loss
        return loss.mean()


def _val_metrics_multi(y_true, probs):
    y_pred = probs.argmax(axis=1)
    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return {
        "auc":     auc,
        "acc":     accuracy_score(y_true, y_pred),
        "bacc":    balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# =============================================================================
# OTHER HELPERS
# =============================================================================

def _reach_from_g_soft(g: torch.Tensor, B: int, D: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Differentiable node reach from soft routing probabilities.

    Args:
        g: [Bsz, Nint, B] post-mask routing probabilities in BFS order
        B: branching factor
        D: internal depth
    Returns:
        reach: [Bsz, Nint] soft probability of reaching each internal node
    """
    g = g.clamp_min(eps)
    g = g / (g.sum(dim=-1, keepdim=True) + eps)

    Bsz, Nint, _ = g.shape
    reaches = []
    r = torch.ones((Bsz, 1), device=g.device, dtype=g.dtype)  # root reach

    for depth in range(D):
        n = B ** depth
        reaches.append(r)  # nodes at this depth

        if depth == D - 1:
            break

        start = (B ** depth - 1) // (B - 1)
        g_d = g[:, start:start + n, :]                  # [Bsz, n, B]
        r = (r.unsqueeze(-1) * g_d).reshape(Bsz, n * B)

    return torch.cat(reaches, dim=1)                    # [Bsz, Nint]


def _expected_routable_leaves_soft(aux, model, eps=1e-12):
    g = aux["g_soft"]
    B, D = model.B, model.D

    reach = _reach_from_g_soft(g, B=B, D=D, eps=eps)   # [Bsz, Nint]
    alpha = reach.mean(dim=0)                          # [Nint]

    g_bar = g.mean(dim=0).clamp_min(eps)
    g_bar = g_bar / g_bar.sum(dim=-1, keepdim=True).clamp_min(eps)
    neff  = 1.0 / (g_bar.pow(2).sum(dim=-1) + eps)     # [Nint] in [1, B]

    # multiplicative composition along paths -> sum of weighted log Neff
    log_eff_leaves = (alpha * torch.log(neff)).sum()   # in [0, D log B]
    return log_eff_leaves                              # NOTE: returns LOG leaves


def _leaf_reach_from_g_soft(g: torch.Tensor, model, eps: float = 1e-12) -> torch.Tensor:
    """
    Soft leaf reach probabilities in DFS leaf order.

    Order matches both model._compute_leaf_paths (which is what
    leaf_logits is indexed by) and PostHocMergedMBNDT's
    leaf_id = leaf_id * B + br convention.

    Args:
        g: [Bsz, Nint, B] gating probs (per-node sum to 1 along last dim)
        model: provides B, D
    Returns:
        leaf_reach: [Bsz, B**D]; rows sum to ~1 if g is normalized
    """
    B, D = int(model.B), int(model.D)
    if B < 2:
        return torch.ones((g.shape[0], 1), device=g.device, dtype=g.dtype)

    reach_int = _reach_from_g_soft(g, B=B, D=D, eps=eps)            # [Bsz, Nint]

    last_start = (B ** (D - 1) - 1) // (B - 1)
    n_last     = B ** (D - 1)

    reach_last = reach_int[:, last_start : last_start + n_last]      # [Bsz, n_last]
    g_last     = g[:, last_start : last_start + n_last, :]            # [Bsz, n_last, B]

    leaf_reach = reach_last.unsqueeze(-1) * g_last                    # [Bsz, n_last, B]
    leaf_reach = leaf_reach.reshape(leaf_reach.shape[0], n_last * B)  # [Bsz, B^D]
    return leaf_reach


def _log_eff_leaves_from_leaf_mass(aux, model, eps: float = 1e-12) -> torch.Tensor:
    """
    Effective number of leaves via inverse participation ratio on the
    soft leaf reach mass. Returns log(n_eff_leaves), in [0, D * log B].

    Compared to _expected_routable_leaves_soft (a per-internal-node
    branch-diversity measure), this is computed at the leaf level and
    more directly tracks how many leaves carry meaningful mass — which
    is closer to what MBNDT-pp counts.
    """
    g = aux["g_soft"]
    leaf_reach = _leaf_reach_from_g_soft(g, model, eps=eps)           # [Bsz, num_leaves]
    leaf_mass = leaf_reach.mean(dim=0).clamp_min(eps)
    leaf_mass = leaf_mass / leaf_mass.sum().clamp_min(eps)
    n_eff = 1.0 / (leaf_mass.pow(2).sum() + eps)                       # in [1, num_leaves]
    return torch.log(n_eff + eps)


def _leaf_budget_log_eff(aux, model, mode: str) -> torch.Tensor:
    """Return log effective leaves for the selected leaf-budget surrogate.

    Supported modes:
      - branch_log / branch_linear: branch-diversity surrogate
      - mass_log / mass_linear: soft leaf-mass/IPR surrogate
    """
    mode = str(mode).lower()
    if mode.startswith("branch"):
        return _expected_routable_leaves_soft(aux, model)
    if mode.startswith("mass"):
        return _log_eff_leaves_from_leaf_mass(aux, model)
    raise ValueError(f"Unknown leaf_budget mode: {mode}")


def _leaf_budget_violation_from_log_eff(log_eff: torch.Tensor, K: float, mode: str) -> torch.Tensor:
    """Compute constraint violation either in log scale or linear scale.

    log mode:    relu(log L - log K)    -> ratio-like pressure, gentler
    linear mode: relu(exp(log L) - K)   -> absolute-leaf pressure, stronger
    """
    mode = str(mode).lower()
    K = float(K)
    if K <= 0:
        raise ValueError(f"leaf_budget K must be positive, got {K}")
    if mode.endswith("_log") or mode == "log":
        return F.relu(log_eff - math.log(K))
    if mode.endswith("_linear") or mode == "linear":
        # Clamp only for numerical safety. The theoretical range is <= B**D.
        return F.relu(torch.exp(log_eff.clamp(max=50.0)) - K)
    raise ValueError(f"Unknown leaf_budget mode: {mode}")


class MBNDT(nn.Module):
    def __init__(
        self, n_features, D, B,
        num_classes: int = 1,
        # gating choices
        selector_mode: str = "entmax_st",     # {"entmax_st","reinmax","entmax_soft"}
        branch_mode:   str = "st",     # {"st","reinmax","soft"}
        # temperatures
        tau_cdf: float = 0.6,               # CDF temperature (t vs x)
        tau_feat: float = 1.5,              # selector temperature (ReinMax)
        tau_branch: float = 1.5,            # branch temperature (ReinMax)
        use_masks: bool = False,
    ):
        super().__init__()
        assert B >= 2
        self.D, self.B = int(D), int(B)
        self.num_internal_nodes = (B**D - 1) // (B - 1) if B > 1 else 1
        self.num_leaves = B**D
        self.num_classes = int(num_classes)

        # config
        self.selector_mode = selector_mode
        self.branch_mode   = branch_mode
        self.tau_cdf       = float(tau_cdf)
        self.tau_feat      = float(tau_feat)
        self.tau_branch    = float(tau_branch)
        self.use_masks     = bool(use_masks)

        # (1) feature logits
        self.feature_logits = nn.Parameter(torch.empty(self.num_internal_nodes, n_features))
        nn.init.normal_(self.feature_logits, mean=0.0, std=0.03)

        # (2) ordered thresholds (via softplus gaps)  — fixed init
        self.thresh_deltas = nn.Parameter(torch.empty(self.num_internal_nodes, B - 1))
        nn.init.normal_(self.thresh_deltas, mean=0.0, std=0.03)
        self.t_base = nn.Parameter(torch.zeros(self.num_internal_nodes, 1))

        with torch.no_grad():
            ps = torch.arange(1, B, dtype=torch.float32) / B
            q  = math.sqrt(2.0) * torch.erfinv(2.0 * ps - 1.0)    # quantiles: [B-1], strictly increasing

            # Make t ≈ q by using base = q0 and positive gaps for strict monotonicity.
            base = q[0]                                           # typically negative
            eps  = torch.tensor(1e-6, dtype=q.dtype)              # tiny positive to keep first gap > 0
            gaps = torch.empty_like(q)
            gaps[0]  = eps                                        # so t[0] ≈ base + eps ≈ q[0]
            gaps[1:] = q[1:] - q[:-1]                             # strictly positive

            self.t_base.copy_(base.repeat(self.num_internal_nodes, 1))
            gaps_full = gaps.unsqueeze(0).expand(self.num_internal_nodes, -1)
            self.thresh_deltas.copy_(inv_softplus(gaps_full))     # softplus(inv_softplus(x)) = x (≈ exact for x>0)


        # (3) per-branch masks (applied only if use_masks=True)
        self.mask_logits = nn.Parameter(torch.full((self.num_internal_nodes, B), 3.0))


        # (4) leaf logits
        if self.num_classes <= 1:
            self.leaf_logits = nn.Parameter(torch.empty(self.num_leaves))
        else:
            self.leaf_logits = nn.Parameter(torch.empty(self.num_leaves, self.num_classes))

        nn.init.normal_(self.leaf_logits, mean=0.0, std=1e-2)

        # precompute leaf paths & a flattened (node,branch) index buffer for fast gather
        paths = self._compute_leaf_paths()  # list of [(node_id, branch_id), ...] for each leaf
        nodes = torch.tensor([[nid for (nid, bid) in p] for p in paths], dtype=torch.long)  # [L, D]
        brchs = torch.tensor([[bid for (nid, bid) in p] for p in paths], dtype=torch.long)  # [L, D]
        node_branch_ids = nodes * self.B + brchs  # flatten (node,branch) to a single index
        self.register_buffer("node_branch_ids", node_branch_ids, persistent=False)  # [L, D]

    def enable_masks(self, enabled: bool = True):
        self.use_masks = bool(enabled)

    def _compute_leaf_paths(self):
        if self.B == 1:
            return [[(0, 0)] * self.D]
        paths = []
        def dfs(node_idx, depth, path):
            if depth == self.D:
                paths.append(path)
                return
            for b in range(self.B):
                child_idx = node_idx * self.B + (b + 1)
                dfs(child_idx, depth + 1, path + [(node_idx, b)])
        dfs(0, 0, [])
        return paths

    def _ordered_thresholds(self):
        gaps = F.softplus(self.thresh_deltas)       # [Nnodes, B-1] positive
        t = self.t_base + torch.cumsum(gaps, dim=1) # strictly increasing
        return t                                     # [Nnodes, B-1]

    def _reduce_tree_expectation(self, node_gates, leaf_p):
        Bsz, B, D = node_gates.size(0), self.B, self.D
        if leaf_p.dim() == 1:                              # binary: keep old path
            v = leaf_p.view(1, -1)
            for depth in range(D - 1, -1, -1):
                n = B ** depth
                start = (B ** depth - 1) // (B - 1)
                g = node_gates[:, start:start + n, :]
                v = v.view(-1, n, B)
                v = (g * v).sum(dim=-1)
            return v[:, 0]                                 # [Bsz]
        else:                                              # multiclass
            C = leaf_p.size(-1)
            v = leaf_p.view(1, -1, C)                      # [1, L, C]
            for depth in range(D - 1, -1, -1):
                n = B ** depth
                start = (B ** depth - 1) // (B - 1)
                g = node_gates[:, start:start + n, :]      # [Bsz, n, B]
                v = v.view(-1, n, B, C)
                v = (g.unsqueeze(-1) * v).sum(dim=-2)      # [Bsz, n, C]
            return v[:, 0, :]                              # [Bsz, C]

    @torch.no_grad()
    def _reach_mean_from_g(self, g: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        g: [Bsz, Nint, B] branch probs in BFS-by-depth internal-node order.
        returns: reach_mean [Nint]
        """
        Bsz, Nint, B = g.shape
        assert B == self.B
        # normalize just in case
        g = g.clamp_min(eps)
        g = g / (g.sum(dim=-1, keepdim=True) + eps)

        reach_chunks = []
        r = torch.ones((Bsz, 1), device=g.device, dtype=g.dtype)  # depth 0 (root)

        for depth in range(self.D):
            n = self.B ** depth
            start = (self.B ** depth - 1) // (self.B - 1)

            # r corresponds to nodes at this depth
            reach_chunks.append(r)

            if depth == self.D - 1:
                break

            g_d = g[:, start:start + n, :]             # [Bsz, n, B]
            r = (r.unsqueeze(-1) * g_d).reshape(Bsz, n * self.B)  # [Bsz, B^(depth+1)]

        reach = torch.cat(reach_chunks, dim=1)         # [Bsz, Nint]
        return reach.mean(dim=0)                       # [Nint]


    def forward(self, x, return_aux: bool = False):
        """
        x: [batch, n_features]
        return_aux: if True returns (y_pred, aux_dict)
        """
        Bsz = x.size(0)

        # -----------------------------
        # (A) Feature selection per node  (modes from A)
        # -----------------------------
        if self.selector_mode == "reinmax":
            feat_soft = entmax15(self.feature_logits, dim=-1)  # no tau division needed
            feat_hard, feat_soft = reinmax(self.feature_logits, tau=self.tau_feat)
            feat_w = feat_hard
        elif self.selector_mode == "entmax_soft":
            feat_w = entmax15(self.feature_logits, dim=1)  # soft mixture over features
            feat_soft = feat_w
        elif self.selector_mode == "soft":
            feat_w = self.feature_logits
            feat_soft = self.feature_logits
        else:  # "entmax_st"
            feat_soft = entmax15(self.feature_logits, dim=1)
            feat_w = StraightThroughOneHot.apply(feat_soft)  # one-hot forward, grad via ST

        # Always do matmul (keeps gradients correct for ReinMax/ST)
        x_sel = x @ feat_w.t()  # [B, Nnodes]

        # -----------------------------
        # (B) Branch probabilities via CDF differences
        # -----------------------------
        t = self._ordered_thresholds()                   # [Nnodes, B-1]

        x_exp = x_sel.unsqueeze(-1)                      # [Bsz, Nnodes, 1]
        t_exp = t.unsqueeze(0).expand(x_sel.size(0), -1, -1)

        z = (t_exp - x_exp) / max(self.tau_cdf, 1e-6)
        z = torch.clamp(z, -12.0, 12.0)                  # sigmoid saturates later than normal_cdf anyway
        S = torch.sigmoid(z)                             # [Bsz, Nnodes, B-1]

        g_left  = S[..., :1]
        g_mid   = (S[..., 1:] - S[..., :-1]) if self.B > 2 else None
        g_right = 1.0 - S[..., -1:]
        g_soft  = torch.cat([g_left, g_mid, g_right], dim=-1) if g_mid is not None \
                  else torch.cat([g_left, g_right], dim=-1)   # [Bsz, Nnodes, B]
        g_pre = g_soft  # unmasked routing

        # ---- Apply masks like B (only if enabled), then renormalize
        if self.use_masks:
            m = torch.sigmoid(self.mask_logits)                 # [Nnodes, B]
            g_soft = g_soft * m.unsqueeze(0)                    # broadcast to [B, Nnodes, B]
            g_soft = g_soft / (g_soft.sum(dim=-1, keepdim=True) + 1e-12)

        if return_aux:
            reach_pre_mean  = self._reach_mean_from_g(g_pre).detach()
            reach_post_mean = self._reach_mean_from_g(g_soft).detach()

        # -----------------------------
        # (C) Hard routing options (from A)
        # -----------------------------
        if self.branch_mode == "reinmax":
            g_hard, _ = reinmax(g_soft, tau=1.0)  # g_soft already shaped by tau_cdf
            node_gates = g_hard                                # [B, Nnodes, B]
        elif self.branch_mode == "soft":
            node_gates = g_soft                                  # fully soft
        else:  # "st"
            node_gates = StraightThroughOneHot3D.apply(g_soft)   # one-hot forward, grad via ST

        # -----------------------------
        # (D) Compute Output
        # -----------------------------
        # node_gates = ... (your existing ST/soft/reinmax choice)
        #leaf_p = torch.sigmoid(self.leaf_logits)  # [L]
        y_pred = self._reduce_tree_expectation(node_gates, self.leaf_logits)  # [Bsz]

        if not return_aux:
            return y_pred

        aux = {
            "feat_w": feat_w,                      # [Nnodes, n_features]
            "feat_soft": (feat_soft if 'feat_soft' in locals() else None),
            "x_sel": x_sel,                        # [B, Nnodes]
            "t": t,                                # [Nnodes, B-1]
            "g_pre": g_pre,                        # [Bsz, Nint, B]
            "g_soft": g_soft,                      # [B, Nnodes, B] (after masks if used)
            "reach_pre_mean": reach_pre_mean,
            "reach_post_mean": reach_post_mean,
            "masks": torch.sigmoid(self.mask_logits) if self.use_masks else None
        }
        return y_pred, aux


# -------------------- training with toggled regularizers --------------------

def default_reg_config(B: int):
    return {
        # Keep LB only as early anti-collapse stabilization
        "load_balance": {
            "on": False,
            "lambda": 1e-3,
            "mode": "kl",
            "warmup_frac": 0.10,
            "K_gate": 2.0,
        },

        # New: 1-knob leaf budget regularizer
        # Only exposed hyperparameter = K
        "leaf_budget": {
            "on": False,
            "K": 8.0,
            # {"branch_log", "branch_linear", "mass_log", "mass_linear"}
            # branch_* uses _expected_routable_leaves_soft(aux, model), which returns log L_branch.
            # mass_* uses _log_eff_leaves_from_leaf_mass(aux, model), which returns log L_mass.
            # *_log compares log L to log K; *_linear compares L to K.
            "mode": "mass_log",
        },
    }

def _ndtri(p: torch.Tensor) -> torch.Tensor:
    # standard-normal inverse CDF
    if hasattr(torch.special, "ndtri"):
        return torch.special.ndtri(p)
    return torch.sqrt(torch.tensor(2.0, device=p.device, dtype=p.dtype)) * torch.erfinv(2*p - 1)

def _median_quantile_gap(B: int, device=None, dtype=torch.float32) -> torch.Tensor:
    # Δ_B = median adjacent Normal-quantile gap for a B-way split
    if B <= 2:
        return torch.tensor(1.0, device=device, dtype=dtype)  # harmless scale when no internal gaps
    ks = torch.arange(1, B, device=device, dtype=dtype)      # 1..B-1
    z  = _ndtri(ks / B)
    return (z[1:] - z[:-1]).median()


# ------------------------------
# schedule helpers
# ------------------------------
def _as_int(x, default=0):
    try: return int(x)
    except: return default

def _as_float(x, default=0.0):
    try: return float(x)
    except: return default

def _epoch0_to_frac(epoch0: int, total_epochs: int) -> float:
    # epoch0 is 0-index; frac in [0,1]
    if total_epochs <= 1:
        return 1.0
    return max(0.0, min(1.0, epoch0 / float(total_epochs - 1)))

def _sched_mult(schedule: dict, *, epoch0: int, total_epochs: int) -> float:
    """
    Returns multiplier in [0,1] (or [final_mult,1] for some schedules).
    epoch0: 0-indexed global epoch
    total_epochs: total epochs used for schedule normalization
    """
    if schedule is None:
        return 1.0
    typ = str(schedule.get("type", "constant")).lower()
    t = _epoch0_to_frac(epoch0, total_epochs)

    if typ == "constant":
        return 1.0

    if typ == "warmup_linear_decay":
        # decay from 1 -> final_mult over warmup, then stay at final_mult
        warmup_frac = _as_float(schedule.get("warmup_frac", 0.10), 0.10)
        warmup_epochs = schedule.get("warmup_epochs", None)
        if warmup_epochs is None:
            W = max(1, int(round(warmup_frac * total_epochs)))
        else:
            W = max(1, _as_int(warmup_epochs, 1))
        final_mult = _as_float(schedule.get("final_mult", 0.0), 0.0)

        if epoch0 >= W:
            return final_mult
        # linear from 1 -> final_mult
        u = epoch0 / float(max(1, W))
        return (1.0 - u) * 1.0 + u * final_mult

    if typ == "warmup_cosine_decay":
        # warmup: hold 1.0, then cosine decay 1.0 -> final_mult to end
        warmup_frac = _as_float(schedule.get("warmup_frac", 0.10), 0.10)
        warmup_epochs = schedule.get("warmup_epochs", None)
        if warmup_epochs is None:
            W = max(1, int(round(warmup_frac * total_epochs)))
        else:
            W = max(1, _as_int(warmup_epochs, 1))
        final_mult = _as_float(schedule.get("final_mult", 0.0), 0.0)

        if epoch0 <= W:
            return 1.0
        # progress after warmup
        denom = max(1, (total_epochs - 1) - W)
        u = (epoch0 - W) / float(denom)  # 0..1
        # cosine from 1 -> final_mult
        c = 0.5 * (1.0 + math.cos(math.pi * u))  # 1..0
        return final_mult + (1.0 - final_mult) * c

    if typ == "late_ramp":
        # 0 until start, then ramp to 1 by end (power curve)
        start_frac = _as_float(schedule.get("start_frac", 0.10), 0.10)
        end_frac   = _as_float(schedule.get("end_frac", 0.50), 0.50)
        power      = _as_float(schedule.get("power", 2.0), 2.0)

        if t <= start_frac:
            return 0.0
        if t >= end_frac:
            return 1.0
        u = (t - start_frac) / max(1e-12, (end_frac - start_frac))
        return float(u ** power)

    if typ == "ramp_epochs":
        # absolute epoch schedule
        start_epoch = _as_int(schedule.get("start_epoch", 0), 0)   # 0-index
        ramp_epochs = max(1, _as_int(schedule.get("ramp_epochs", 10), 10))
        power       = _as_float(schedule.get("power", 2.0), 2.0)

        if epoch0 < start_epoch:
            return 0.0
        if epoch0 >= start_epoch + ramp_epochs:
            return 1.0
        u = (epoch0 - start_epoch) / float(ramp_epochs)
        return float(u ** power)

    # fallback
    return 1.0

def _base_weight(cfg: dict) -> float:
    # support either "weight" or "lambda"
    if cfg is None:
        return 0.0
    if "lambda" in cfg:
        return _as_float(cfg.get("lambda", 0.0), 0.0)
    return _as_float(cfg.get("weight", 0.0), 0.0)

def _effective_weight_for_reg(reg_cfg: dict, name: str, *, epoch0: int, total_epochs: int) -> float:
    cfg = reg_cfg.get(name, None)
    if not isinstance(cfg, dict):
        return 0.0
    if not cfg.get("on", False):
        return 0.0
    w0 = _base_weight(cfg)
    if w0 <= 0.0:
        return 0.0

    # If LB has warmup_frac but no schedule, treat it as warmup_linear_decay to 0
    sch = cfg.get("schedule", None)
    if (sch is None) and (name == "load_balance") and ("warmup_frac" in cfg or "warmup_epochs" in cfg):
        sch = {
            "type": "warmup_linear_decay",
            "warmup_frac": cfg.get("warmup_frac", 0.10),
            "warmup_epochs": cfg.get("warmup_epochs", None),
            "final_mult": 0.0,
        }

    mult = _sched_mult(sch, epoch0=epoch0, total_epochs=total_epochs)
    return float(w0 * mult)



def _compute_regularizers_raw(model, aux, reg_cfg):
    losses = {}
    eps = 1e-12
    gate_delta = 0.25
    mask_thr = 1e-3

    g_pre = aux["g_pre"]              # [Bsz, Nint, B]
    m = aux.get("masks", None)
    reach_pre_mean = aux["reach_pre_mean"]

    def rw_mean(per_node, reach_mean):
        w = reach_mean.clamp_min(0.0)
        return (per_node * w).sum() / (w.sum() + eps)

    neff_m = None
    if (m is not None) and getattr(model, "use_masks", False):
        m2 = m.clamp_min(eps)
        p = m2 / (m2.sum(dim=-1, keepdim=True) + eps)
        neff_m = 1.0 / (p.pow(2).sum(dim=-1) + eps)

    if reg_cfg.get("load_balance", {}).get("on", False):
        K_gate = float(reg_cfg["load_balance"].get("K_gate", 2.0))

        bar_g = g_pre.mean(dim=0).clamp_min(eps)     # [Nint, B]
        bar_g = bar_g / (bar_g.sum(dim=-1, keepdim=True) + eps)

        if neff_m is not None:
            active = (m.detach() > mask_thr).float()
            Z = active.sum(dim=-1, keepdim=True).clamp_min(1.0)
            u = active / Z

            g_act = (bar_g * active).clamp_min(eps)
            g_act = g_act / (g_act.sum(dim=-1, keepdim=True) + eps)

            gate = (neff_m.detach() > (K_gate + gate_delta)).float()
        else:
            u = torch.full_like(bar_g, 1.0 / model.B)
            g_act = bar_g
            gate = torch.ones((bar_g.shape[0],), device=bar_g.device, dtype=bar_g.dtype)

        per_node = (u.clamp_min(eps) * (u.clamp_min(eps).log() - g_act.log())).sum(dim=-1)
        per_node = per_node * gate

        losses["load_balance"] = rw_mean(per_node, reach_pre_mean)

    return losses




def _one_line(msg, _state={"last_len": 0}):
    # Clear previous line if new message is shorter
    pad = max(_state["last_len"] - len(msg), 0)
    sys.stdout.write("\r" + msg + " " * pad)
    sys.stdout.flush()
    _state["last_len"] = len(msg)



def train_model(
    model, train_loader, val_loader=None, epochs=1000,
    lr_feature=1e-2, lr_thresh=1e-2, lr_leaf=1e-2, lr_mask=1e-2,
    patience=30, reg_config=None, device=None, verbose=True,
    val_threshold=0.5,
    # ---- classification loss toggles ----
    loss_type="bce",            # "bce" | "balanced_bce" | "focal"
    class_weights=None,         # (w_pos, w_neg) for balanced_bce; if None and auto_class_weights=True, infer
    auto_class_weights=True,    # infer weights from train_loader when class_weights is None
    focal_alpha=0.7, focal_gamma=2.0,
    # ---- Optional: use same criterion for validation loss logging ----
    val_loss_same_as_train=True,
    # ---- NEW: checkpoint control ----
    monitor: str = "val_bacc",            # {"val_loss","val_auc","val_f1_macro","val_bacc","val_acc"}
    force_load_best_at_end: bool = False, # load best even if no early-stop occurred
    return_best_state: bool = False,      # also return the best state_dict (on CPU)

    # ---- NEW: schedule continuity for staged training ----
    epoch_offset: int = 0,                # global epoch offset (0-index)
    total_epochs_for_schedule: int = None, # if set, schedules use this total length

    # ---- FIX #2: allow seeding best score/state from a prior stage ----
    initial_best_score: float = None,     # seed early-stopping baseline
    initial_best_state: dict = None,      # state_dict (on CPU) to fall back to
):
    """
    Train MBNDT with scheduled regularizer weights.
    - reg terms are computed unweighted by _compute_regularizers(...)
    - weights are applied per-epoch according to reg_config[name]["schedule"]
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    if reg_config is None:
        reg_config = default_reg_config(model.B)

    if reg_config.get("leaf_budget", {}).get("on", False) and hasattr(model, "enable_masks"):
        model.enable_masks(True)

    leaf_budget_cfg = reg_config.get("leaf_budget", {})
    leaf_budget_on = bool(leaf_budget_cfg.get("on", False))
    leaf_budget_K = float(leaf_budget_cfg.get("K", 8.0))
    leaf_budget_mode = str(leaf_budget_cfg.get("mode", "branch_log")).lower()
    valid_leaf_budget_modes = {"branch_log", "branch_linear", "mass_log", "mass_linear"}
    if leaf_budget_mode not in valid_leaf_budget_modes:
        raise ValueError(
            f"Unknown leaf_budget mode={leaf_budget_mode}. "
            f"Expected one of {sorted(valid_leaf_budget_modes)}"
        )
    leaf_budget_mu = LEAF_BUDGET_MU_INIT
    leaf_budget_violation_ema = None

    num_batches = max(1, len(train_loader))

    history = defaultdict(list)

    optimizer = torch.optim.Adam([
        {"params": [model.feature_logits], "lr": lr_feature, "name": "feature_logits"},
        {"params": [model.thresh_deltas, model.t_base], "lr": lr_thresh, "name": "thresholds"},
        {"params": [model.mask_logits], "lr": lr_mask, "name": "masks"},
        {"params": [model.leaf_logits], "lr": lr_leaf, "name": "leaf_logits"},
    ])

    # -------------------------
    # LOGITS-BASED LOSS SETUP
    # -------------------------

    is_multi = (getattr(model, "num_classes", 1) > 1)

    if loss_type == "bce":
        if is_multi:
            ce = nn.CrossEntropyLoss()
            train_criterion = lambda logits, y: ce(logits, y.long())
            loss_desc = "CrossEntropy"
        else:
            train_criterion = lambda logits, y: F.binary_cross_entropy_with_logits(logits, y)
            loss_desc = "BCEWithLogits"

    elif loss_type == "balanced_bce":
        if is_multi:
            if class_weights is None and auto_class_weights:
                w = infer_class_weights_multiclass(train_loader, model.num_classes)
            else:
                w = torch.as_tensor(class_weights, dtype=torch.float)
            ce = nn.CrossEntropyLoss(weight=w.to(device))
            train_criterion = lambda logits, y: ce(logits, y.long())
            loss_desc = f"Weighted CE (w={w.tolist()})"
        else:
            if class_weights is None and auto_class_weights:
                w_pos, w_neg = infer_class_weights_from_loader(train_loader)
            else:
                w_pos, w_neg = (class_weights or (1.0, 1.0))
                w_pos, w_neg = float(w_pos), float(w_neg)

            def train_criterion(logits, y):
                return weighted_bce_with_logits(logits, y, w_pos=w_pos, w_neg=w_neg)
            loss_desc = f"Balanced BCEWithLogits (w_pos={w_pos:.4f}, w_neg={w_neg:.4f})"

    elif loss_type == "focal":
        if is_multi:
            if auto_class_weights and class_weights is None:
                alpha = infer_class_weights_multiclass(train_loader, model.num_classes)
                alpha = alpha / alpha.sum()                # normalize to a prior over classes
            else:
                alpha = class_weights
            floss = FocalLossMultiClassFromLogits(alpha=alpha, gamma=focal_gamma)
            train_criterion = lambda logits, y: floss(logits, y.long())
            loss_desc = f"MultiClassFocal (gamma={focal_gamma})"
        else:
            focal_loss = FocalLossFromLogits(alpha=focal_alpha, gamma=focal_gamma)
            def train_criterion(logits, y):
                return focal_loss(logits, y)
            loss_desc = f"FocalFromLogits (alpha={focal_alpha}, gamma={focal_gamma})"

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Val loss: same space (logits)
    val_criterion = train_criterion if val_loss_same_as_train else \
                    (lambda logits, y: F.binary_cross_entropy_with_logits(logits, y))


    if verbose:
        tot_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {tot_params}")
        print(f"[Loss] train={loss_desc}")

    # ---- monitor mapping ----
    def _get_mon(vloss, mets, key: str):
        if key == "val_loss":      return vloss, False     # minimize
        if key == "val_auc":       return mets["auc"], True
        if key == "val_f1_macro":  return mets["f1_macro"], True
        if key == "val_bacc":      return mets["bacc"], True
        if key == "val_acc":       return mets["acc"], True
        raise ValueError(f"Unknown monitor={key}")

    # ---- FIX #2: seed best_score and best_state from prior stage ----
    best_score = initial_best_score       # None if no prior stage
    best_state = None
    if initial_best_state is not None:
        best_state = {k: v.detach().cpu().clone() for k, v in initial_best_state.items()}

    # ---- FIX #1: track which epoch produced the best checkpoint ----
    best_epoch = 0  # 0-indexed global epoch; 0 means "the initial state"

    patience_counter = 0
    maximize = (monitor != "val_loss")

    # schedules normalize to this total
    sched_total = int(total_epochs_for_schedule) if total_epochs_for_schedule is not None else int(epoch_offset + epochs)

    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_reg = defaultdict(float)
        n_samples = 0

        # global epoch index (0-index) used for schedules
        epoch0 = int(epoch_offset) + (epoch - 1)

        # precompute effective weights for this epoch
        eff_w = {
            "load_balance": _effective_weight_for_reg(
                reg_config,
                "load_balance",
                epoch0=epoch0,
                total_epochs=sched_total,
            )
        }

        any_weighted_reg_on = (eff_w["load_balance"] > 0.0)

        # --- start epoch timer ---
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()

        # ---------------------- Train ----------------------
        for bx, by in train_loader:
            bx, by = bx.to(device).float(), by.to(device).float()

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                logits, aux = model(bx, return_aux=True)
                clf_loss = train_criterion(logits, by)

                total_reg = 0.0

                # load balance only
                if any_weighted_reg_on:
                    reg_losses = _compute_regularizers_raw(model, aux, reg_config)
                    lb_val = reg_losses.get("load_balance", 0.0)
                    w_lb = eff_w["load_balance"]
                    if w_lb > 0.0:
                        total_reg = total_reg + (w_lb * lb_val)
                    running_reg["load_balance"] += float(lb_val.detach().cpu()) * bx.size(0)

                # leaf budget only
                leaf_budget_loss = 0.0
                if leaf_budget_on:
                    log_eff_leaves = _leaf_budget_log_eff(aux, model, leaf_budget_mode)
                    violation = _leaf_budget_violation_from_log_eff(
                        log_eff_leaves,
                        leaf_budget_K,
                        leaf_budget_mode,
                    )

                    # batch-dependent surrogate -> do NOT divide by num_batches
                    leaf_budget_loss = (
                        leaf_budget_mu * violation
                        + 0.5 * LEAF_BUDGET_RHO * violation.pow(2)
                    )

                loss = clf_loss + total_reg + leaf_budget_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # ---- FIX #5: accumulate training loss and sample count ----
            running_loss += float(clf_loss.detach().cpu()) * bx.size(0)
            n_samples += bx.size(0)

        if device.type == "cuda": torch.cuda.synchronize()
        epoch_secs = time.perf_counter() - t0
        history.setdefault("epoch_secs", []).append(epoch_secs)

        avg_clf = running_loss / max(n_samples, 1)
        history["train_loss"].append(avg_clf)

        if leaf_budget_on:
            branch_vals, mass_vals, vio_vals = [], [], []

            # GPU-side visited-leaf tracking (no per-batch CPU sync)
            visited_mask = torch.zeros(
                int(model.num_leaves), dtype=torch.bool, device=device
            )

            model.eval()
            with torch.no_grad():
                for bx, _ in train_loader:
                    bx = bx.to(device, non_blocking=True).float()
                    _, aux_eval = model(bx, return_aux=True)

                    # --- diagnostics: always log both soft surrogates on log scale ---
                    log_eff_branch = _expected_routable_leaves_soft(aux_eval, model)
                    log_eff_mass   = _log_eff_leaves_from_leaf_mass(aux_eval, model)
                    branch_vals.append(float(log_eff_branch.detach().cpu()))
                    mass_vals.append(float(log_eff_mass.detach().cpu()))

                    # --- dual-ascent violation uses the selected mode ---
                    log_eff_selected = _leaf_budget_log_eff(aux_eval, model, leaf_budget_mode)
                    vio_b = _leaf_budget_violation_from_log_eff(
                        log_eff_selected,
                        leaf_budget_K,
                        leaf_budget_mode,
                    )
                    vio_vals.append(float(vio_b.detach().cpu()))

                    # --- hard active leaves: actual MBNDT-pp criterion ---
                    # Hard route: argmax at each internal node, then trace to leaf.
                    g_soft = aux_eval["g_soft"]                                  # [Bsz, Nint, B]
                    g_hard = F.one_hot(
                        g_soft.argmax(dim=-1), num_classes=g_soft.shape[-1]
                    ).float()
                    leaf_reach_hard = _leaf_reach_from_g_soft(g_hard, model)     # [Bsz, num_leaves]
                    leaf_idx = leaf_reach_hard.argmax(dim=-1)                    # [Bsz]

                    # accumulate visits on-device (no host sync inside the loop)
                    visited_mask.scatter_(
                        0, leaf_idx, torch.ones_like(leaf_idx, dtype=torch.bool)
                    )

            mean_log_branch = float(np.mean(branch_vals)) if branch_vals else 0.0
            mean_log_mass   = float(np.mean(mass_vals))   if mass_vals   else 0.0
            mean_n_eff_branch = float(np.exp(mean_log_branch))
            mean_n_eff_mass   = float(np.exp(mean_log_mass))
            hard_active     = int(visited_mask.sum().item())   # one sync at the end

            history.setdefault("log_eff_leaves_branch", []).append(mean_log_branch)
            history.setdefault("log_eff_leaves_mass",   []).append(mean_log_mass)
            history.setdefault("n_eff_leaves_branch",   []).append(mean_n_eff_branch)
            history.setdefault("n_eff_leaves_mass",     []).append(mean_n_eff_mass)
            history.setdefault("hard_active_leaves",    []).append(hard_active)

            # --- dual ascent on mu ---
            mean_violation = float(np.mean(vio_vals)) if vio_vals else 0.0
            if leaf_budget_violation_ema is None:
                leaf_budget_violation_ema = mean_violation
            else:
                leaf_budget_violation_ema = (
                    LEAF_BUDGET_EMA_BETA * leaf_budget_violation_ema
                    + (1.0 - LEAF_BUDGET_EMA_BETA) * mean_violation
                )
            leaf_budget_mu = max(
                0.0,
                leaf_budget_mu + LEAF_BUDGET_MU_LR * leaf_budget_violation_ema,
            )

            # Backward compatibility: keep the old key, but store the selected surrogate
            # in log scale for *_log modes and linear scale for *_linear modes.
            selected_log_mean = mean_log_mass if leaf_budget_mode.startswith("mass") else mean_log_branch
            selected_n_eff_mean = mean_n_eff_mass if leaf_budget_mode.startswith("mass") else mean_n_eff_branch
            history.setdefault("leaf_budget_expected_leaves", []).append(
                selected_n_eff_mean if leaf_budget_mode.endswith("_linear") else selected_log_mean
            )
            history.setdefault("leaf_budget_violation",       []).append(mean_violation)
            history.setdefault("leaf_budget_mu",              []).append(leaf_budget_mu)

        else:
            history.setdefault("log_eff_leaves_branch", []).append(0.0)
            history.setdefault("log_eff_leaves_mass",   []).append(0.0)
            history.setdefault("n_eff_leaves_branch",   []).append(0.0)
            history.setdefault("n_eff_leaves_mass",     []).append(0.0)
            history.setdefault("hard_active_leaves",    []).append(0)
            history.setdefault("leaf_budget_expected_leaves", []).append(0.0)
            history.setdefault("leaf_budget_violation",       []).append(0.0)
            history.setdefault("leaf_budget_mu",              []).append(0.0)


        # load balance logging
        denom = max(n_samples, 1)
        if reg_config.get("load_balance", {}).get("on", False):
            history.setdefault("reg_load_balance", []).append(running_reg["load_balance"] / denom)
        else:
            history.setdefault("reg_load_balance", []).append(0.0)

        history.setdefault("w_load_balance", []).append(float(eff_w["load_balance"]))

        # ---------------------- Validate ----------------------
        if val_loader is not None:
            model.eval()
            vloss_sum, vcount = 0.0, 0
            all_probs, all_y = [], []

            with torch.no_grad():
                for vx, vy in val_loader:
                    vx = vx.to(device).float()
                    vy = vy.to(device).float()

                    with autocast(enabled=use_amp):
                        vlogits = model(vx)

                    vloss_sum += float(val_criterion(vlogits, vy).detach().cpu()) * vx.size(0)
                    vcount    += vx.size(0)

                    if is_multi:
                        all_probs.append(F.softmax(vlogits, dim=-1).cpu())
                    else:
                        all_probs.append(torch.sigmoid(vlogits).cpu())
                    all_y.append(vy.cpu())

            vloss = vloss_sum / max(vcount, 1)
            probs = torch.cat(all_probs).numpy()
            y_np  = torch.cat(all_y).numpy().astype(int)
            mets  = _val_metrics_multi(y_np, probs) if is_multi \
                    else _val_metrics(y_np, probs, thr=val_threshold)

            history["val_loss"].append(vloss)
            for k, v in mets.items():
                history[f"val_{k}"].append(v)

            # ---- Monitor-aware early stopping & best checkpoint ----
            score, _ = _get_mon(vloss, mets, monitor)
            is_better = (best_score is None) or ((score > best_score) if maximize else (score < best_score))

            early = False
            if is_better:
                best_score = score
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch0  # FIX #1: record the global epoch of this checkpoint
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    early = True
                    if best_state is not None:
                        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

            if verbose:
                msg = (f"Epoch {epoch}/{epochs} (global {epoch0+1}/{sched_total}) | "
                       f"TrainLoss {avg_clf:.4f} | ValLoss {vloss:.4f} | "
                       f"AUC {mets['auc']:.4f} | Acc {mets['acc']:.4f} | "
                       f"BAcc {mets['bacc']:.4f} | F1_macro {mets['f1_macro']:.4f} | "
                       f"{monitor} {score:.4f} | "
                       f"Time: {epoch_secs:.2f}s" + (" | EarlyStop" if early else ""))
                _one_line(msg)

            if early:
                break
        else:
            if verbose:
                _one_line(f"Epoch {epoch}/{epochs} | TrainLoss {avg_clf:.4f}")

    # ---- Force-load best even if no early-stop happened ----
    if (val_loader is not None) and (best_state is not None) and force_load_best_at_end:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if verbose:
        sys.stdout.write("\n"); sys.stdout.flush()

    if return_best_state:
        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # FIX #1: return best_epoch alongside best_state
        return history, best_state, best_epoch

    return history



def random_restart_train(
    NT5_module, cfg, train_loader, val_loader,
    device,
    test_loader=None,
    n_restarts=5,
    base_seed=0,

    # --- Stage-1 (selection window) ---
    stage1_epochs=None,
    stage1_patience=None,

    stage1_es_monitor="val_bacc",
    stage1_select_monitor="val_bacc",
    stage2_es_monitor="val_bacc",

    # --- Stage-2 (continuation) ---
    stage2_epochs=None,
    stage2_patience=None,
    verbose=True,

    # --- NEW ---
    reg_config=None,
    continue_after_selection=True,
    return_histories: bool = False
):
    # derive reg_config once if not provided
    if reg_config is None:
        try:
            reg_config = reg_to_train_config(B=cfg.model.B, regularizers=cfg.regularizers)
        except Exception:
            reg_config = default_reg_config(cfg.model.B)

    stage1_total_sec = 0.0
    best_stage1_sec  = 0.0
    stage2_sec       = 0.0

    # budgets
    E1 = cfg.training.epochs   if stage1_epochs   is None else stage1_epochs
    P1 = cfg.training.patience if stage1_patience is None else stage1_patience
    E2 = cfg.training.epochs   if stage2_epochs   is None else stage2_epochs
    P2 = cfg.training.patience if stage2_patience is None else stage2_patience

    # schedules normalized to total planned timeline
    TOTAL_SCHED_EPOCHS = int(E1 + E2)

    thr_sel = getattr(cfg.training, "decision_threshold", 0.5)

    higher_is_better = stage1_select_monitor in {"val_auc", "val_acc", "val_bacc", "val_f1_macro"}
    best_sel = -float("inf") if higher_is_better else float("inf")
    best_state_global, best_info = None, None
    best_stage1_best_epoch = 0  # FIX #1: track best epoch, not epochs run
    best_stage1_best_score = None  # FIX #2: track the score to seed stage 2

    stage1_histories, stage2_history = [], None

    def _evaluate(model, loader, thr=thr_sel):
        try:
            return NT5_module.evaluate_model(model, loader)
        except Exception:
            model.eval(); ps, ys = [], []
            with torch.no_grad():
                for vx, vy in loader:
                    logits = model(vx.to(device).float()).cpu()
                    if logits.dim() == 1:
                        ps.append(torch.sigmoid(logits))
                    else:
                        ps.append(F.softmax(logits, dim=-1))
                    ys.append(vy.cpu())

            p = torch.cat(ps).numpy()
            y = torch.cat(ys).numpy().astype(int)

            if p.ndim == 1:
                yhat = (p >= thr).astype(int)
                try:    auc = roc_auc_score(y, p)
                except: auc = float("nan")
            else:
                yhat = p.argmax(axis=1)
                try:    auc = roc_auc_score(y, p, multi_class="ovr", average="macro")
                except: auc = float("nan")

            return {
                "auc": auc,
                "acc": accuracy_score(y, yhat),
                "bacc": balanced_accuracy_score(y, yhat),
                "f1_macro": f1_score(y, yhat, average="macro", zero_division=0),
                "val_loss": float("nan"),
            }

    # -------- Stage-1 --------
    for r in range(n_restarts):
        seed_r = int(base_seed) + r
        set_global_seed(seed_r)

        model = build_model_from_cfg(NT5_module, cfg, device)



        t0 = time.perf_counter()

        # FIX #1: train_model now returns best_epoch as third element
        hist1, best_state_r, best_epoch_r = train_model(
            model, train_loader, val_loader,
            epochs=E1, patience=P1,
            lr_feature=cfg.optim.lr_feature, lr_thresh=cfg.optim.lr_thresh,
            lr_leaf=cfg.optim.lr_leaf, lr_mask=cfg.optim.lr_mask,
            loss_type=cfg.training.loss_type,
            verbose=False,
            monitor=stage1_es_monitor,
            force_load_best_at_end=True,
            return_best_state=True,
            reg_config=reg_config,

            # schedules start at epoch 0 for stage1
            epoch_offset=0,
            total_epochs_for_schedule=TOTAL_SCHED_EPOCHS,
        )

        t1 = time.perf_counter()
        stage1_sec = float(t1 - t0)

        stage1_total_sec += stage1_sec

        val_m = _evaluate(model, val_loader, thr=thr_sel)
        if stage1_select_monitor == "val_loss":
            sel = float(np.nanmin(np.array(hist1["val_loss"], dtype=float))) if "val_loss" in hist1 else float("nan")
        else:
            sel = float(val_m[stage1_select_monitor.replace("val_", "")])

        if return_histories:
            stage1_histories.append({
                "restart": r, "seed": seed_r, "history": hist1, "sel_value": sel
            })

        is_better = (sel > best_sel) if higher_is_better else (sel < best_sel)
        if is_better:
            best_sel = sel
            best_state_global = {k: v.detach().cpu().clone() for k, v in best_state_r.items()}
            best_info = {
                "restart": r,
                "seed": seed_r,
                "stage1_es_monitor": stage1_es_monitor,
                "stage1_select_monitor": stage1_select_monitor,
                "stage2_es_monitor": stage2_es_monitor,
                "val_value": sel,
            }
            best_stage1_best_epoch = best_epoch_r  # FIX #1: actual best epoch
            best_stage1_best_score = sel            # FIX #2: score to seed stage 2
            best_stage1_sec = stage1_sec

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if verbose:
            best_r = best_info["restart"] if best_info is not None else "-"
            print(
                f"[r {r+1}/{n_restarts}] "
                f"{stage1_select_monitor}={sel:.4f}  "
                f"best={best_sel:.4f} (r={best_r})"
            )

    # -------- Stage-2 --------
    final_model = build_model_from_cfg(NT5_module, cfg, device)
    if best_state_global is not None:
        final_model.load_state_dict({k: v.to(device) for k, v in best_state_global.items()})

    stage2_sec = 0.0

    # If stage-1 selection and stage-2 early stopping use different metrics
    # (e.g., val_loss -> val_bacc), their scores are not comparable.
    # In that case, initialize stage 2 from the selected state, but let
    # train_model establish its own best score under stage2_es_monitor.
    same_monitor_for_stage2 = (stage1_select_monitor == stage2_es_monitor)
    initial_score_for_stage2 = best_stage1_best_score if same_monitor_for_stage2 else None
    initial_state_for_stage2 = best_state_global

    if continue_after_selection:
        t2 = time.perf_counter()
        stage2_history = train_model(
            final_model, train_loader, val_loader,
            epochs=E2, patience=P2,
            lr_feature=cfg.optim.lr_feature, lr_thresh=cfg.optim.lr_thresh,
            lr_leaf=cfg.optim.lr_leaf, lr_mask=cfg.optim.lr_mask,
            loss_type=cfg.training.loss_type,
            verbose=cfg.training.verbose,
            monitor=stage2_es_monitor,
            force_load_best_at_end=True,
            return_best_state=False,
            reg_config=reg_config,

            # FIX #1: continue schedule from actual best epoch, not epochs-run
            epoch_offset=best_stage1_best_epoch,
            total_epochs_for_schedule=TOTAL_SCHED_EPOCHS,

            # FIX #2: seed stage 2 with stage-1 best so it can revert
            initial_best_score=initial_score_for_stage2,
            initial_best_state=initial_state_for_stage2,
        )
        t3 = time.perf_counter()
        stage2_sec = float(t3 - t2)

    timing = {
        "stage1_selected_sec": best_stage1_sec,
        "stage1_total_sec": stage1_total_sec,
        "stage2_sec": stage2_sec,
        "fit_time_selected_sec": best_stage1_sec + stage2_sec,
        "fit_time_total_sec": stage1_total_sec + stage2_sec,
        "n_restarts": int(n_restarts),
    }

    histories = {
        "stage1": stage1_histories,
        "stage2": stage2_history,
        "stage1_es_monitor": stage1_es_monitor,
        "stage1_select_monitor": stage1_select_monitor,
        "stage2_es_monitor": stage2_es_monitor,
        "same_monitor_for_stage2": same_monitor_for_stage2,
        "initial_score_for_stage2": initial_score_for_stage2,
        "timing": timing,
        "best_info": best_info,
    }

    return final_model, best_info, histories







# =============================================================================
# EVALUATE MODEL (LOGITS MODEL)
# =============================================================================
def evaluate_model(model, data_loader, threshold=0.5, criterion=None, device=None):
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    is_multi = (getattr(model, "num_classes", 1) > 1)
    C = int(getattr(model, "num_classes", 1))

    if criterion is None:
        criterion = nn.CrossEntropyLoss() if is_multi else nn.BCEWithLogitsLoss()

    model.eval()
    all_probs, all_labels = [], []
    loss_sum, n_samples = 0.0, 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for Xb, yb in data_loader:
            Xb = Xb.to(device).float()
            yb = yb.to(device).long() if is_multi else yb.to(device).float()

            logits = model(Xb)
            loss_sum += criterion(logits, yb).item() * Xb.size(0)
            n_samples += Xb.size(0)

            probs = F.softmax(logits, dim=-1) if is_multi else torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu())
            all_labels.append(yb.detach().cpu())

    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0

    loss   = loss_sum / max(n_samples, 1)
    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)

    if is_multi:
        preds = probs.argmax(axis=1)

        try:
            auc_ovr = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except Exception:
            auc_ovr = float("nan")
        try:
            auc_ovo = roc_auc_score(labels, probs, multi_class="ovo", average="macro")
        except Exception:
            auc_ovo = float("nan")
        try:
            y_oh = np.eye(C)[labels]
            auprc_macro = average_precision_score(y_oh, probs, average="macro")
        except Exception:
            auprc_macro = float("nan")

        accuracy = accuracy_score(labels, preds)
        bal_acc  = balanced_accuracy_score(labels, preds)
        mcc      = matthews_corrcoef(labels, preds)
        f1_mac   = f1_score(labels, preds, average="macro",    zero_division=0)
        f1_wt    = f1_score(labels, preds, average="weighted", zero_division=0)

        lab_range = list(range(C))
        prec_pc = precision_score(labels, preds, average=None, zero_division=0, labels=lab_range)
        rec_pc  = recall_score   (labels, preds, average=None, zero_division=0, labels=lab_range)
        f1_pc   = f1_score       (labels, preds, average=None, zero_division=0, labels=lab_range)
        cm      = confusion_matrix(labels, preds, labels=lab_range)

        return {
            "loss":           f"{loss:.4f}",
            "accuracy":       f"{accuracy:.4f}",
            "balanced_acc":   f"{bal_acc:.4f}",
            "mcc":            f"{mcc:.4f}",
            "auc_ovr_macro":  f"{auc_ovr:.4f}",
            "auc_ovo_macro":  f"{auc_ovo:.4f}",
            "auprc_macro":    f"{auprc_macro:.4f}",
            "f1_macro":       f"{f1_mac:.4f}",
            "f1_weighted":    f"{f1_wt:.4f}",
            "confusion_matrix":    cm.tolist(),
            "per_class_precision": [f"{x:.4f}" for x in prec_pc],
            "per_class_recall":    [f"{x:.4f}" for x in rec_pc],
            "per_class_f1":        [f"{x:.4f}" for x in f1_pc],
            "num_classes":   C,
            "secs":          f"{secs:.4f}",
        }

    # ----- binary branch (unchanged except for syntax fix and num_classes field) -----
    try:
        aucroc = roc_auc_score(labels, probs)
    except Exception:
        aucroc = float("nan")
    try:
        auprc = average_precision_score(labels, probs)
    except Exception:
        auprc = float("nan")

    thr   = float(threshold)
    preds = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    accuracy  = (preds == labels).mean()
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1_mac    = f1_score(labels, preds, average="macro", zero_division=0)
    bal_acc   = balanced_accuracy_score(labels, preds)
    mcc       = matthews_corrcoef(labels, preds)
    ppv       = precision
    npv       = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "loss":         f"{loss:.4f}",
        "accuracy":     f"{accuracy:.4f}",
        "balanced_acc": f"{bal_acc:.4f}",
        "mcc":          f"{mcc:.4f}",
        "aucroc":       f"{aucroc:.4f}",
        "auprc":        f"{auprc:.4f}",
        "f1_macro":     f"{f1_mac:.4f}",
        "ppv":          f"{ppv:.4f}",
        "npv":          f"{npv:.4f}",
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "secs":         f"{secs:.4f}",
    }

# ---------------------------- ---------------------------- ----------------------------
# ---------------------------- ---------------------------- ----------------------------
# ---------------------------- ---------------------------- ----------------------------
# ---------------------------- ---------------------------- ----------------------------
#  ----------------------- HISTORY  ----------------------- ----------------------------
# ---------------------------- ---------------------------- ----------------------------
# ---------------------------- ---------------------------- ----------------------------
# ---------------------------- ---------------------------- ----------------------------
# ---------------------------- ---------------------------- ----------------------------

# ----------------------------
# history validation / summary
# ----------------------------
def validate_history(history: dict, strict: bool = False):
    """
    간단 무결성 체크:
    - list 길이 불일치 경고
    - NaN/inf 존재 경고
    strict=True면 문제 발견 시 ValueError
    """
    if history is None:
        raise ValueError("history is None")

    # 기준 길이: train_loss
    base = len(history.get("train_loss", []))
    problems = []

    for k, v in history.items():
        if not isinstance(v, (list, tuple)):
            continue
        if k == "train_loss":
            continue
        # val_*들은 early stopping 때문에 train_loss보다 짧을 수도 있는데,
        # 여기서는 "너무 이상하게" 긴 경우만 체크
        if len(v) > base:
            problems.append(f"[len] {k} is longer than train_loss ({len(v)} > {base})")

        arr = np.asarray(v, dtype=float) if len(v) > 0 else None
        if arr is not None and len(arr) > 0:
            if np.any(~np.isfinite(arr)):
                problems.append(f"[nan/inf] {k} contains non-finite values")

    if problems:
        msg = "\n".join(problems)
        if strict:
            raise ValueError("History validation failed:\n" + msg)
        else:
            print("[validate_history] warnings:\n" + msg)

def summarize_history(history: dict, monitor="val_loss"):
    """
    best epoch/value 요약 + 마지막 epoch 값 요약
    """
    def _best_idx(y, maximize):
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            return None
        return int(np.nanargmax(y) if maximize else np.nanargmin(y))

    maximize = (monitor != "val_loss")
    out = {"monitor": monitor, "maximize": maximize}

    # best
    if monitor in history and len(history[monitor]) > 0:
        bi = _best_idx(history[monitor], maximize)
        out["best_epoch"] = int(bi + 1)
        out["best_value"] = float(np.asarray(history[monitor], dtype=float)[bi])
    else:
        out["best_epoch"] = None
        out["best_value"] = None

    # last
    for k in ["train_loss", "val_loss", "val_auc", "val_f1_macro", "val_bacc", "val_acc"]:
        if k in history and len(history[k]) > 0:
            out[f"last_{k}"] = float(np.asarray(history[k], dtype=float)[-1])

    return out

# ----------------------------
# plot helpers
# ----------------------------
def _to_1d(a):
    if a is None:
        return None
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        a = a.reshape(1)
    return a

def _moving_avg(y, window=1):
    y = _to_1d(y)
    if y is None or window is None or window <= 1 or len(y) < window:
        return y
    w = int(window)
    ypad = np.pad(y, (w - 1, 0), mode="edge")
    c = np.cumsum(ypad)
    return (c[w:] - c[:-w]) / w

def _savefig(fig, path, dpi=200):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _vline(ax, x, text=None):
    ax.axvline(x, linestyle="--")
    if text is not None:
        ymin, ymax = ax.get_ylim()
        ax.text(x, ymax, text, va="top")

# ----------------------------
# main: train_model history plotter
# ----------------------------
def plot_train_history_all(
    history: dict,
    out_dir: str,
    prefix: str = "run",
    monitor: str = "val_loss",
    smooth_window: int = 1,
    dpi: int = 200,
    make_dashboard: bool = True,
    strict_validate: bool = False,
):
    """
    네 train_model history에 맞춰:
    - loss / metrics / regs / weights / time 각각 PNG 저장
    - summary.json 저장
    - 선택적으로 dashboard(한 장)도 저장
    """
    os.makedirs(out_dir, exist_ok=True)
    validate_history(history, strict=strict_validate)

    summ = summarize_history(history, monitor=monitor)
    with open(os.path.join(out_dir, f"{prefix}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)

    best_epoch = summ.get("best_epoch", None)

    # 1) loss
    fig, ax = plt.subplots(figsize=(7, 4))
    tr = _moving_avg(history.get("train_loss"), smooth_window)
    vl = _moving_avg(history.get("val_loss"), smooth_window)
    if tr is not None and len(tr) > 0:
        ax.plot(np.arange(1, len(tr) + 1), tr, label="train_loss", linewidth=2.0)
    if vl is not None and len(vl) > 0:
        ax.plot(np.arange(1, len(vl) + 1), vl, label="val_loss", linewidth=2.0)
    ax.set_title(f"{prefix} | Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    if best_epoch is not None and monitor == "val_loss":
        _vline(ax, best_epoch, f"best@{best_epoch}")
    _savefig(fig, os.path.join(out_dir, f"{prefix}_loss.png"), dpi=dpi)

    # 2) metrics
    metric_keys = ["val_auc", "val_f1_macro", "val_bacc", "val_acc"]
    if any(k in history and len(history[k]) > 0 for k in metric_keys):
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in metric_keys:
            y = _moving_avg(history.get(k), smooth_window)
            if y is None or len(y) == 0:
                continue
            ax.plot(np.arange(1, len(y) + 1), y, label=k, linewidth=2.0)
        ax.set_title(f"{prefix} | Val metrics")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        if best_epoch is not None and monitor in metric_keys:
            _vline(ax, best_epoch, f"best@{best_epoch}")
        _savefig(fig, os.path.join(out_dir, f"{prefix}_metrics.png"), dpi=dpi)

    # 3) regs (raw)
    reg_keys = sorted([k for k in history.keys() if str(k).startswith("reg_")])
    reg_has = any(len(history.get(k, [])) > 0 and np.any(np.asarray(history[k]) != 0) for k in reg_keys)
    if reg_has:
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in reg_keys:
            y = _moving_avg(history.get(k), smooth_window)
            if y is None or len(y) == 0:
                continue
            ax.plot(np.arange(1, len(y) + 1), y, label=k, linewidth=2.0)
        ax.set_title(f"{prefix} | Regularizers (raw)")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        _savefig(fig, os.path.join(out_dir, f"{prefix}_regs.png"), dpi=dpi)

    # 4) weights (schedule effective weights)
    w_keys = sorted([k for k in history.keys() if str(k).startswith("w_")])
    w_has = any(len(history.get(k, [])) > 0 and np.any(np.asarray(history[k]) != 0) for k in w_keys)
    if w_has:
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in w_keys:
            y = _moving_avg(history.get(k), smooth_window)
            if y is None or len(y) == 0:
                continue
            ax.plot(np.arange(1, len(y) + 1), y, label=k, linewidth=2.0)
        ax.set_title(f"{prefix} | Reg weights (schedule)")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        _savefig(fig, os.path.join(out_dir, f"{prefix}_weights.png"), dpi=dpi)

    # 5) time
    if "epoch_secs" in history and len(history["epoch_secs"]) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        y = _moving_avg(history.get("epoch_secs"), smooth_window)
        ax.plot(np.arange(1, len(y) + 1), y, label="epoch_secs", linewidth=2.0)
        ax.set_title(f"{prefix} | Time per epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Seconds")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        _savefig(fig, os.path.join(out_dir, f"{prefix}_time.png"), dpi=dpi)

    # 6) dashboard (한 장)
    if make_dashboard:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        axes = axes.ravel()

        # (0) loss
        ax = axes[0]
        if tr is not None and len(tr) > 0: ax.plot(np.arange(1, len(tr)+1), tr, label="train_loss", linewidth=2.0)
        if vl is not None and len(vl) > 0: ax.plot(np.arange(1, len(vl)+1), vl, label="val_loss", linewidth=2.0)
        if best_epoch is not None and monitor == "val_loss": _vline(ax, best_epoch)
        ax.set_title("Loss"); ax.grid(True, alpha=0.25); ax.legend(frameon=False)

        # (1) metrics
        ax = axes[1]
        for k in metric_keys:
            y = _moving_avg(history.get(k), smooth_window)
            if y is None or len(y) == 0: continue
            ax.plot(np.arange(1, len(y)+1), y, label=k, linewidth=2.0)
        if best_epoch is not None and monitor in metric_keys: _vline(ax, best_epoch)
        ax.set_title("Val metrics"); ax.grid(True, alpha=0.25); ax.legend(frameon=False)

        # (2) regs
        ax = axes[2]
        if reg_has:
            for k in reg_keys:
                y = _moving_avg(history.get(k), smooth_window)
                if y is None or len(y) == 0: continue
                ax.plot(np.arange(1, len(y)+1), y, label=k, linewidth=2.0)
            ax.legend(frameon=False)
        ax.set_title("Regularizers (raw)"); ax.grid(True, alpha=0.25)

        # (3) weights or time (둘 다 있으면 weights 우선)
        ax = axes[3]
        if w_has:
            for k in w_keys:
                y = _moving_avg(history.get(k), smooth_window)
                if y is None or len(y) == 0: continue
                ax.plot(np.arange(1, len(y)+1), y, label=k, linewidth=2.0)
            ax.set_title("Reg weights (schedule)"); ax.legend(frameon=False)
        elif "epoch_secs" in history and len(history["epoch_secs"]) > 0:
            y = _moving_avg(history.get("epoch_secs"), smooth_window)
            ax.plot(np.arange(1, len(y)+1), y, label="epoch_secs", linewidth=2.0)
            ax.set_title("Time per epoch"); ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)

        fig.suptitle(prefix)
        _savefig(fig, os.path.join(out_dir, f"{prefix}_dashboard.png"), dpi=dpi)

# ----------------------------
# random_restart histories plotter
# ----------------------------
def plot_random_restart_all(
    histories: dict,
    out_dir: str,
    prefix: str = "rr",
    smooth_window: int = 1,
    dpi: int = 200,
    make_global_continuous: bool = True,
):
    """
    histories = {"stage1": [...], "stage2": history_or_None, "monitor": str}
    stage1 overlay + best highlight + stage2 상세 + (옵션) stage1+stage2 연속 플롯
    """
    os.makedirs(out_dir, exist_ok=True)
    monitor = histories.get("monitor", "val_loss")
    stage1 = histories.get("stage1", []) or []
    stage2 = histories.get("stage2", None)

    # ---- stage1 overlay는 return_histories=True일 때만 존재 ----
    if len(stage1) > 0:
        # 어떤 key를 overlay할지 결정
        key = monitor
        for item in stage1:
            h = (item or {}).get("history", {}) or {}
            if key in h and len(h.get(key, [])) > 0:
                break
        else:
            key = "val_loss"

        # best restart by sel_value
        higher_is_better = (monitor != "val_loss")
        best_i, best_sel = None, None
        for i, item in enumerate(stage1):
            sel = item.get("sel_value", None)
            if sel is None or (isinstance(sel, float) and np.isnan(sel)):
                continue
            if best_sel is None:
                best_sel, best_i = float(sel), i
            else:
                if (float(sel) > best_sel) if higher_is_better else (float(sel) < best_sel):
                    best_sel, best_i = float(sel), i

        fig, ax = plt.subplots(figsize=(7, 4))
        for i, item in enumerate(stage1):
            h = (item or {}).get("history", {}) or {}
            y = _moving_avg(h.get(key), smooth_window)
            if y is None or len(y) == 0:
                continue
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, alpha=0.25, linewidth=1.5)

        if best_i is not None:
            hbest = stage1[best_i]["history"]
            yb = _moving_avg(hbest.get(key), smooth_window)
            if yb is not None and len(yb) > 0:
                xb = np.arange(1, len(yb) + 1)
                ax.plot(xb, yb, alpha=0.95, linewidth=3.0, label=f"BEST r{best_i} sel={best_sel:.4f}")
                ax.legend(frameon=False)

        ax.set_title(f"{prefix} | Stage1 overlay ({key})")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        _savefig(fig, os.path.join(out_dir, f"{prefix}_stage1_overlay_{key}.png"), dpi=dpi)

        # stage1 best length for continuous plots
        best_stage1_len = len(stage1[best_i]["history"].get("train_loss", [])) if best_i is not None else None
    else:
        best_i, best_sel, best_stage1_len = None, None, None

    # ---- stage2 상세 ----
    if stage2 is not None:
        plot_train_history_all(
            stage2,
            out_dir=out_dir,
            prefix=f"{prefix}_stage2",
            monitor=monitor,
            smooth_window=smooth_window,
            dpi=dpi,
            make_dashboard=True
        )

    # ---- (옵션) stage1(best) + stage2 를 연속(global epoch)으로 이어붙이기 ----
    if make_global_continuous and (len(stage1) > 0) and (best_i is not None) and (stage2 is not None) and (best_stage1_len is not None):
        h1 = stage1[best_i]["history"]
        h2 = stage2

        def _concat(k):
            a = _to_1d(h1.get(k))
            b = _to_1d(h2.get(k))
            if a is None: a = np.array([], dtype=float)
            if b is None: b = np.array([], dtype=float)
            return np.concatenate([a, b], axis=0)

        # loss continuous
        fig, ax = plt.subplots(figsize=(7,4))
        y = _moving_avg(_concat("train_loss"), smooth_window)
        ax.plot(np.arange(1, len(y)+1), y, label="train_loss", linewidth=2.0)
        yv = _moving_avg(_concat("val_loss"), smooth_window)
        if len(yv) > 0:
            ax.plot(np.arange(1, len(yv)+1), yv, label="val_loss", linewidth=2.0)
        _vline(ax, best_stage1_len, "stage1_end")
        ax.set_title(f"{prefix} | Continuous Loss (best restart + stage2)")
        ax.set_xlabel("Global epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        _savefig(fig, os.path.join(out_dir, f"{prefix}_continuous_loss.png"), dpi=dpi)

        # metrics continuous
        fig, ax = plt.subplots(figsize=(7,4))
        for k in ["val_auc","val_f1_macro","val_bacc","val_acc"]:
            yk = _moving_avg(_concat(k), smooth_window)
            if len(yk) == 0:
                continue
            ax.plot(np.arange(1, len(yk)+1), yk, label=k, linewidth=2.0)
        _vline(ax, best_stage1_len, "stage1_end")
        ax.set_title(f"{prefix} | Continuous Val metrics")
        ax.set_xlabel("Global epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        _savefig(fig, os.path.join(out_dir, f"{prefix}_continuous_metrics.png"), dpi=dpi)









###########################################################################################
###########################################################################################
######## complexity measures
###########################################################################################
###########################################################################################

import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt


def _merge_intervals(intervals, tol=0.0):
    """intervals: list[(lo,hi)] -> merged(sorted) list[(lo,hi)]"""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out = [intervals[0]]
    for lo, hi in intervals[1:]:
        plo, phi = out[-1]
        if lo <= phi + tol:
            out[-1] = (plo, max(phi, hi))
        else:
            out.append((lo, hi))
    return out

def _intersect_unions(a, b, tol=0.0):
    """a,b: merged unions. returns merged intersection."""
    if not a or not b:
        return []
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi - tol:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return _merge_intervals(out, tol=tol)

def _depths_for_internal_nodes(num_internal_nodes: int, B: int):
    """B-ary heap indexing: parent(j)=(j-1)//B"""
    d = [0] * num_internal_nodes
    for j in range(1, num_internal_nodes):
        d[j] = d[(j - 1) // B] + 1
    return torch.tensor(d, dtype=torch.long)


@torch.no_grad()
def mbndt_masked_argmax_regions_per_node(
    model,
    z_min=-6.0,
    z_max=6.0,
    n_grid=4001,
    apply_masks=None,     # None -> model.use_masks
    mask_eps=0.0,         # >0이면 mask floor
    node_chunk=256,
    tol=1e-12,
    device=None,
):
    """
    Returns:
      regions: list length Nnodes; regions[j][b] = union intervals [(lo,hi),...]
               where branch b is argmax of (p_b(z) * mask_b) over z grid.
      feat_idx: [Nnodes] hard-selected feature index used for constraint bookkeeping
      z_grid:   [G] grid on CPU
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    B = int(model.B)
    N = int(model.num_internal_nodes)
    tau = float(model.tau_cdf)

    if apply_masks is None:
        apply_masks = bool(model.use_masks)

    # --- hard feature index per node (analysis용 deterministic choice)
    # forward의 entmax_st(one-hot)와 가장 일치: argmax(entmax15(feature_logits))
    entmax15_fn = None
    if "entmax15" in globals():
        entmax15_fn = globals()["entmax15"]
    else:
        try:
            from entmax import entmax15 as _entmax15
            entmax15_fn = _entmax15
        except Exception:
            entmax15_fn = None

    feat_logits = model.feature_logits.detach().to(device)
    if entmax15_fn is not None:
        feat_prob = entmax15_fn(feat_logits, dim=1)
        feat_idx = torch.argmax(feat_prob, dim=1)  # [N]
    else:
        feat_idx = torch.argmax(feat_logits, dim=1)

    # --- thresholds
    t = model._ordered_thresholds().detach().to(device)  # [N, B-1]

    # --- masks
    if apply_masks:
        m = torch.sigmoid(model.mask_logits.detach().to(device))  # [N,B]
        if mask_eps > 0:
            m = mask_eps + (1.0 - mask_eps) * m
    else:
        m = None

    # --- z grid & boundaries (grid cell boundaries for intervals)
    z = torch.linspace(z_min, z_max, n_grid, device=device, dtype=t.dtype)  # [G]
    mid = (z[:-1] + z[1:]) * 0.5
    boundaries = torch.cat([z[:1], mid, z[-1:]], dim=0)  # [G+1]
    boundaries_cpu = boundaries.detach().cpu()

    regions = [[[] for _ in range(B)] for _ in range(N)]

    for j0 in range(0, N, node_chunk):
        j1 = min(N, j0 + node_chunk)
        tc = t[j0:j1]  # [C, B-1]

        # p(z) via your exact sigmoid-CDF-diff form
        z_exp = z.view(1, -1, 1)      # [1, G, 1]
        t_exp = tc.unsqueeze(1)       # [C, 1, B-1]
        zz = (t_exp - z_exp) / max(tau, 1e-6)
        zz = torch.clamp(zz, -12.0, 12.0)
        S = torch.sigmoid(zz)         # [C, G, B-1]

        g_left = S[..., :1]
        g_right = 1.0 - S[..., -1:]
        if B > 2:
            g_mid = S[..., 1:] - S[..., :-1]
            p = torch.cat([g_left, g_mid, g_right], dim=-1)  # [C,G,B]
        else:
            p = torch.cat([g_left, g_right], dim=-1)         # [C,G,2]

        pm = p * m[j0:j1].unsqueeze(1) if apply_masks else p  # [C,G,B]
        b_hat = torch.argmax(pm, dim=-1).detach().cpu()       # [C,G]

        G = b_hat.size(1)
        for cj in range(j1 - j0):
            j = j0 + cj
            arr = b_hat[cj]  # [G]

            if G > 1:
                change = torch.nonzero(arr[1:] != arr[:-1], as_tuple=False).flatten() + 1
                cuts = torch.cat([torch.tensor([0]), change, torch.tensor([G])])
            else:
                cuts = torch.tensor([0, 1])

            per_branch = [[] for _ in range(B)]
            for s, e in zip(cuts[:-1].tolist(), cuts[1:].tolist()):
                b = int(arr[s].item())
                lo = float(boundaries_cpu[s].item())
                hi = float(boundaries_cpu[e].item())
                if lo < hi - tol:
                    per_branch[b].append((lo, hi))

            for b in range(B):
                regions[j][b] = _merge_intervals(per_branch[b], tol=tol)

    return regions, feat_idx.detach().cpu(), z.detach().cpu()




@torch.no_grad()
def mbndt_routability_no_feature_constraints(model, regions):
    """
    Ignores repeated-feature consistency. Only requires that along a path,
    each chosen branch has non-empty argmax region at that node.

    Returns:
      leaf_count_nf, node_used_nf [N], branch_used_nf [N,B], local_branch_routable [N,B]
    """
    B = int(model.B)
    D = int(model.D)
    N = int(model.num_internal_nodes)

    depth = _depths_for_internal_nodes(N, B)  # [N]

    local_branch_routable = torch.zeros(N, B, dtype=torch.bool)
    for j in range(N):
        for b in range(B):
            if len(regions[j][b]) > 0:
                local_branch_routable[j, b] = True

    # DP arrays
    leaf_count_sub = torch.zeros(N, dtype=torch.long)     # number of feasible leaves in subtree
    has_leaf_sub = torch.zeros(N, dtype=torch.bool)

    # bottom-up (children indices > parent index in this heap)
    for j in range(N - 1, -1, -1):
        dj = int(depth[j].item())
        cnt = 0
        if dj == D - 1:
            # each allowed branch produces a leaf
            cnt = int(local_branch_routable[j].sum().item())
        else:
            for b in range(B):
                if not local_branch_routable[j, b].item():
                    continue
                child = j * B + (b + 1)
                # child should be internal for dj < D-1
                if child < N:
                    cnt += int(leaf_count_sub[child].item())
                else:
                    # safety fallback
                    cnt += 1
        leaf_count_sub[j] = cnt
        has_leaf_sub[j] = (cnt > 0)

    # derive branch_used/node_used (reachable to at least one feasible leaf)
    branch_used_nf = torch.zeros(N, B, dtype=torch.bool)
    node_used_nf = torch.zeros(N, dtype=torch.bool)
    for j in range(N):
        dj = int(depth[j].item())
        used_any = False
        for b in range(B):
            if not local_branch_routable[j, b].item():
                continue
            if dj == D - 1:
                branch_used_nf[j, b] = True
                used_any = True
            else:
                child = j * B + (b + 1)
                if child < N and has_leaf_sub[child].item():
                    branch_used_nf[j, b] = True
                    used_any = True
        node_used_nf[j] = used_any

    leaf_count_nf = int(leaf_count_sub[0].item()) if has_leaf_sub[0].item() else 0
    return leaf_count_nf, node_used_nf, branch_used_nf, local_branch_routable, depth



@torch.no_grad()
def mbndt_routability_with_feature_constraints(
    model,
    regions,
    feat_idx,
    tol=1e-12,
    max_record_leaves=0,
    record_all_leaf_ids=False,
):
    """
    Enforces repeated-feature consistency via intersection of union-interval constraints.

    Returns:
      leaf_count, node_used [N], branch_used [N,B], recorded_leaf_ids(list)
    """
    B = int(model.B)
    D = int(model.D)
    N = int(model.num_internal_nodes)

    node_used = torch.zeros(N, dtype=torch.bool)
    branch_used = torch.zeros(N, B, dtype=torch.bool)

    leaf_count = 0
    recorded_leaf_ids = []

    def dfs(node, depth, cons, leaf_prefix):
        nonlocal leaf_count, recorded_leaf_ids

        if depth == D:
            return True
        if node >= N:
            return False

        f = int(feat_idx[node].item())
        any_feasible = False

        for b in range(B):
            U = regions[node][b]
            if not U:
                continue

            U2 = _intersect_unions(cons[f], U, tol=tol) if f in cons else U
            if not U2:
                continue

            leaf_id = int(leaf_prefix * B + b)

            if depth == D - 1:
                leaf_count += 1
                branch_used[node, b] = True
                any_feasible = True
                if record_all_leaf_ids:
                    recorded_leaf_ids.append(leaf_id)
                elif max_record_leaves and len(recorded_leaf_ids) < max_record_leaves:
                    recorded_leaf_ids.append(leaf_id)
            else:
                child = node * B + (b + 1)
                cons2 = dict(cons)
                cons2[f] = U2
                ok = dfs(child, depth + 1, cons2, leaf_id)
                if ok:
                    branch_used[node, b] = True
                    any_feasible = True

        if any_feasible:
            node_used[node] = True
        return any_feasible

    dfs(0, 0, {}, 0)

    if record_all_leaf_ids:
        recorded_leaf_ids = sorted(set(int(x) for x in recorded_leaf_ids))

    return leaf_count, node_used, branch_used, recorded_leaf_ids



@torch.no_grad()
def mbndt_function_hard_routability_report(
    model,
    z_min=-6.0,
    z_max=6.0,
    n_grid=4001,
    apply_masks=None,
    mask_eps=0.0,
    node_chunk=256,
    tol=1e-12,
    max_record_leaves=0,
    record_all_leaf_ids=False,
    device=None,
    return_details=False,
):
    """
    Full function-level routability report (NO data):
      - local (node-wise) routable branches
      - no-feature-constraints routable (tree-structure only)
      - feature-constraints routable (global feasibility across repeated features)
    """
    regions, feat_idx, zgrid = mbndt_masked_argmax_regions_per_node(
        model=model,
        z_min=z_min, z_max=z_max, n_grid=n_grid,
        apply_masks=apply_masks,
        mask_eps=mask_eps,
        node_chunk=node_chunk,
        tol=tol,
        device=device,
    )

    leaf_nf, node_nf, branch_nf, local_branch_routable, depth = mbndt_routability_no_feature_constraints(
        model, regions
    )

    leaf_fc, node_fc, branch_fc, leaf_ids = mbndt_routability_with_feature_constraints(
        model,
        regions,
        feat_idx,
        tol=tol,
        max_record_leaves=max_record_leaves,
        record_all_leaf_ids=record_all_leaf_ids,
    )

    local_cnt = local_branch_routable.sum(dim=1)
    nf_cnt = branch_nf.sum(dim=1)
    fc_cnt = branch_fc.sum(dim=1)

    def _nodes_where(mask_bool):
        idx = torch.nonzero(mask_bool, as_tuple=False).flatten()
        return [(int(i.item()), int(depth[i].item())) for i in idx]

    nodes_dead_local = _nodes_where(local_cnt == 0)
    nodes_collapsed_local = _nodes_where(local_cnt == 1)
    nodes_collapsed_nf = _nodes_where((node_nf) & (nf_cnt == 1))
    nodes_collapsed_fc = _nodes_where((node_fc) & (fc_cnt == 1))

    nodes_lost_due_to_feat = _nodes_where(node_nf & (~node_fc))
    branches_lost_due_to_feat = int(((branch_nf) & (~branch_fc)).sum().item())
    leaves_lost_due_to_feat = int(leaf_nf - leaf_fc)

    report = {
        "z_min": float(z_min),
        "z_max": float(z_max),
        "n_grid": int(n_grid),
        "apply_masks": bool(model.use_masks) if apply_masks is None else bool(apply_masks),
        "mask_eps": float(mask_eps),
        "tol": float(tol),

        "B": int(model.B),
        "D": int(model.D),
        "num_internal_nodes_nominal": int(model.num_internal_nodes),
        "num_leaves_nominal": int(model.num_leaves),

        "local_routable_nodes": int((local_cnt > 0).sum().item()),
        "local_routable_branches": int(local_branch_routable.sum().item()),

        "function_routable_nodes_no_feat": int(node_nf.sum().item()),
        "function_routable_branches_no_feat": int(branch_nf.sum().item()),
        "function_routable_leaves_no_feat": int(leaf_nf),

        "function_hard_routable_nodes": int(node_fc.sum().item()),
        "function_hard_routable_branches": int(branch_fc.sum().item()),
        "function_hard_routable_leaves": int(leaf_fc),
        "function_hard_routable_leaf_ratio": float(leaf_fc / max(model.num_leaves, 1)),

        "lost_nodes_due_to_feat": int((node_nf & (~node_fc)).sum().item()),
        "lost_branches_due_to_feat": int(branches_lost_due_to_feat),
        "lost_leaves_due_to_feat": int(leaves_lost_due_to_feat),

        "num_dead_nodes_local": len(nodes_dead_local),
        "num_collapsed_nodes_local(cnt==1)": len(nodes_collapsed_local),
        "num_collapsed_nodes_no_feat(cnt==1)": len(nodes_collapsed_nf),
        "num_collapsed_nodes_feat(cnt==1)": len(nodes_collapsed_fc),
    }

    diagnostics = {
        "nodes_dead_local": nodes_dead_local,
        "nodes_collapsed_local": nodes_collapsed_local,
        "nodes_collapsed_no_feat": nodes_collapsed_nf,
        "nodes_collapsed_feat": nodes_collapsed_fc,
        "nodes_lost_due_to_feat": nodes_lost_due_to_feat,
        "recorded_leaf_ids": leaf_ids,
    }

    if return_details:
        details = {
            "regions": regions,
            "feat_idx": feat_idx,
            "zgrid": zgrid,
            "depth": depth,
            "local_branch_routable": local_branch_routable,
            "node_used_no_feat": node_nf,
            "branch_used_no_feat": branch_nf,
            "node_used_feat": node_fc,
            "branch_used_feat": branch_fc,
            "function_hard_routable_leaf_ids": leaf_ids,
        }
        return report, diagnostics, details

    return report, diagnostics



@torch.no_grad()
def mbndt_export_effective_tree_edges_fast(
    model, regions, feat_idx, node_used, branch_used,
    include_feature=True, include_mask=False, include_regions=False
):
    B = int(model.B)

    # iterate only used nodes (your case: 838 nodes instead of 88k)
    used_nodes = torch.nonzero(node_used, as_tuple=False).flatten().cpu().tolist()
    feat = feat_idx.detach().cpu().tolist()

    nodes = [{"node": j, **({"feature": int(feat[j])} if include_feature else {})}
             for j in used_nodes]

    m = None
    if include_mask:
        m = torch.sigmoid(model.mask_logits.detach()).cpu()

    edges = []
    for j in used_nodes:
        used_bs = torch.nonzero(branch_used[j], as_tuple=False).flatten().cpu().tolist()
        for b in used_bs:
            child = j * B + (b + 1)
            e = {"parent": int(j), "branch": int(b), "child": int(child)}
            if include_mask:
                e["mask"] = float(m[j, b].item())
            if include_regions:
                e["z_region_union"] = regions[j][b]
            edges.append(e)

    return nodes, edges



@torch.no_grad()
def mbndt_explain_collapsed_nodes(
    model,
    regions,         # details["regions"]
    feat_idx,        # details["feat_idx"] (cpu tensor)
    tol=1e-12,
    max_nodes=200,   # 너무 많으면 잘라서 반환
):

    B = int(model.B)
    D = int(model.D)
    N = int(model.num_internal_nodes)

    # depth map
    depth = _depths_for_internal_nodes(N, B)

    # provenance-aware constraints:
    # cons[f] = union intervals, prov[f] = list of (node, branch, union_at_that_step)
    explanations = []

    def dfs(node, d, cons, prov):
        nonlocal explanations
        if d == D:
            return True
        if node >= N:
            return False

        f = int(feat_idx[node].item())
        per_branch = []
        any_ok = False

        for b in range(B):
            U_local = regions[node][b]
            if not U_local:
                per_branch.append({
                    "b": b,
                    "local_nonempty": False,
                    "after_intersection_nonempty": False,
                    "child_feasible": False,
                    "reason": "local_argmax_region_empty (masked argmax never selects this branch)",
                })
                continue

            # apply intersection if repeated feature constrained
            if f in cons:
                U_after = _intersect_unions(cons[f], U_local, tol=tol)
                if not U_after:
                    per_branch.append({
                        "b": b,
                        "local_nonempty": True,
                        "after_intersection_nonempty": False,
                        "child_feasible": False,
                        "reason": "empty_after_intersection_with_ancestor_feature_constraint",
                    })
                    continue
            else:
                U_after = U_local

            # recurse / leaf
            if d == D - 1:
                child_ok = True
            else:
                child = node * B + (b + 1)
                cons2 = dict(cons)
                prov2 = {k: list(v) for k, v in prov.items()}

                # update constraint & provenance for this feature
                cons2[f] = U_after
                prov2.setdefault(f, [])
                prov2[f].append((int(node), int(b), U_after))

                child_ok = dfs(child, d + 1, cons2, prov2)

            if child_ok:
                any_ok = True

            per_branch.append({
                "b": b,
                "local_nonempty": True,
                "after_intersection_nonempty": True,
                "child_feasible": bool(child_ok),
                "reason": "ok" if child_ok else "child_subtree_infeasible",
            })

        # node is "used" iff any branch leads to a feasible leaf
        if any_ok:
            feasible_bs = [x["b"] for x in per_branch if x["child_feasible"]]
            if len(feasible_bs) == 1:
                # collapsed node (global feasible 기준)
                info = {
                    "node": int(node),
                    "depth": int(depth[node].item()),
                    "feature": int(f),
                    "feasible_branch": int(feasible_bs[0]),
                    "active_constraint_union_for_feature": (cons.get(f, None)),
                    "provenance_for_feature": prov.get(f, []),
                    "per_branch_status": per_branch,
                }
                explanations.append(info)

                # early stop if too many
                if len(explanations) >= max_nodes:
                    return True

        return any_ok

    dfs(0, 0, {}, {})
    return explanations


from collections import deque

def mbndt_truncate_by_depth(nodes, edges, max_depth: int, root: int = 0, keep_leaf_edges: bool = True):

    max_depth = int(max_depth)
    internal = {n["node"] for n in nodes}

    # build adjacency only through internal children (for depth computation)
    adj_int = {u: [] for u in internal}
    for e in edges:
        u, v = e["parent"], e["child"]
        if u in internal and v in internal:
            adj_int[u].append(v)

    # BFS/DFS to compute depth from root (reachable internal nodes only)
    depth = {}
    if root in internal:
        depth[root] = 0
        q = deque([root])
        while q:
            u = q.popleft()
            du = depth[u]
            if du >= max_depth:
                continue
            for v in adj_int.get(u, []):
                if v not in depth:
                    depth[v] = du + 1
                    q.append(v)

    # keep internal nodes up to max_depth
    kept_internal = {u for u, d in depth.items() if d <= max_depth}

    # filter nodes list
    nodes_cut = [n for n in nodes if n["node"] in kept_internal]

    # filter edges list
    edges_cut = []
    for e in edges:
        u, v = e["parent"], e["child"]
        if u not in kept_internal:
            continue
        du = depth.get(u, None)
        if du is None:
            continue

        if v in internal:
            # internal -> internal edge: only if child is also kept (i.e., within depth)
            if v in kept_internal:
                edges_cut.append(e)
        else:
            # internal -> leaf edge: keep if desired and parent depth <= max_depth
            if keep_leaf_edges and du <= max_depth:
                edges_cut.append(e)

    return nodes_cut, edges_cut, depth


def mbndt_truncate_effective_tree(nodes, edges, max_depth: int, root: int = 0):

    max_depth = int(max_depth)

    # 1) depth_map 만들기: nodes에 depth 있으면 그대로 사용
    depth_map = {}
    has_depth = all(("depth" in n) for n in nodes)
    if has_depth:
        for n in nodes:
            depth_map[int(n["node"])] = int(n["depth"])
    else:
        # depth 없으면 BFS로 계산 (internal graph 기준)
        internal = {int(n["node"]) for n in nodes}
        adj = {u: [] for u in internal}
        for e in edges:
            u = int(e["parent"]); v = int(e["child"])
            if u in internal and v in internal:
                adj[u].append(v)

        depth_map = {}
        if root in internal:
            depth_map[root] = 0
            q = deque([root])
            while q:
                u = q.popleft()
                du = depth_map[u]
                if du >= max_depth:
                    continue
                for v in adj.get(u, []):
                    if v not in depth_map:
                        depth_map[v] = du + 1
                        q.append(v)

    # 2) depth<=max_depth 인 node만 keep
    kept_nodes = [n for n in nodes if depth_map.get(int(n["node"]), 10**9) <= max_depth]
    kept_set = {int(n["node"]) for n in kept_nodes}

    # 3) parent가 keep인 edge만 keep (child는 깊어도 OK: plot에서 leaf로 보임)
    kept_edges = [e for e in edges if int(e["parent"]) in kept_set]

    return kept_nodes, kept_edges


def mbndt_plot_effective_tree(
    nodes, edges,
    title="Effective MBNDT (top-8 depth)",
    draw_leaf=True,
    figsize=(14, 7),
):

    # build node sets
    internal_nodes = {n["node"] for n in nodes}
    # adjacency with synthetic leaf nodes if child not internal
    adj = {u: [] for u in internal_nodes}
    leaf_nodes = set()

    for e in edges:
        u = e["parent"]
        v = e["child"]
        b = e["branch"]
        if u not in adj:
            continue
        if v in internal_nodes:
            adj[u].append((v, b, e))
        else:
            if draw_leaf:
                leaf_id = f"leaf_{u}_{b}"
                leaf_nodes.add(leaf_id)
                adj[u].append((leaf_id, b, e))

    # compute depths
    depth = {}
    depth[0] = 0
    stack = [0]
    while stack:
        u = stack.pop()
        for v, _, _ in adj.get(u, []):
            if v in internal_nodes:
                if v not in depth:
                    depth[v] = depth[u] + 1
                    stack.append(v)

    # assign x by DFS leaf-order
    x = {}
    y = {}

    leaf_order = 0

    def dfs_layout(u):
        nonlocal leaf_order
        children = adj.get(u, [])
        # sort by branch id for consistent layout
        children = sorted(children, key=lambda t: t[1])

        if not children:
            # isolated node
            x[u] = float(leaf_order)
            leaf_order += 1
            return x[u]

        child_xs = []
        for v, _, _ in children:
            if v in internal_nodes:
                child_xs.append(dfs_layout(v))
            else:
                # leaf
                x[v] = float(leaf_order)
                leaf_order += 1
                child_xs.append(x[v])
        x[u] = sum(child_xs) / max(len(child_xs), 1)
        return x[u]

    if 0 in internal_nodes:
        dfs_layout(0)

    # y positions
    for u in internal_nodes:
        y[u] = -float(depth.get(u, 0))
    for v in leaf_nodes:
        # put leaves one level below their parent inferred from id
        parts = v.split("_")
        pu = int(parts[1])
        y[v] = -float(depth.get(pu, 0) + 1)

    # plot
    fig, ax = plt.subplots(figsize=figsize)

    # edges
    for e in edges:
        u = e["parent"]
        b = e["branch"]
        v = e["child"]
        if u not in x:
            continue
        if v not in internal_nodes:
            v_key = f"leaf_{u}_{b}"
            if not draw_leaf or v_key not in x:
                continue
            vx = x[v_key]; vy = y[v_key]
        else:
            if v not in x:
                continue
            vx = x[v]; vy = y[v]
        ax.plot([x[u], vx], [y[u], vy])

        # label branch id lightly
        mx, my = (x[u] + vx) / 2.0, (y[u] + vy) / 2.0
        ax.text(mx, my, f"b={b}", fontsize=8)

    # nodes (internal)
    for n in nodes:
        u = n["node"]
        if u not in x:
            continue
        ax.scatter([x[u]], [y[u]])
        ax.text(x[u], y[u], f"n{u}\nf{n['feature']}", fontsize=8, ha="center", va="bottom")

    # leaves
    if draw_leaf:
        for v in leaf_nodes:
            if v not in x:
                continue
            ax.scatter([x[v]], [y[v]], marker="x")
            ax.text(x[v], y[v], "L", fontsize=8, ha="center", va="bottom")

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    return fig, ax












import torch

@torch.no_grad()
def mbndt_collect_hard_path_counts(
    model,
    loader,
    device="cuda",
    decision_mask=None,      # [N] bool-like (np/torch/list). Decision node := outdeg>=2 in effective tree.
    threeplus_mask=None,     # [N] bool-like. 3+ node := outdeg>=3 in effective tree.
    return_path_stats=False,
    min_branch_hit=1,        # for empirical arity stats (hits>=min_branch_hit)
):

    model.eval()
    model = model.to(device)

    B = int(model.B)
    D = int(model.D)
    N = int(model.num_internal_nodes)
    L = int(model.num_leaves)

    branch_hits = torch.zeros(N, B, dtype=torch.long)  # CPU
    node_hits   = torch.zeros(N, dtype=torch.long)     # CPU
    leaf_hits   = torch.zeros(L, dtype=torch.long)     # CPU

    n_samples = 0

    # Optional path stats accumulators
    if return_path_stats:
        if decision_mask is None or threeplus_mask is None:
            raise ValueError("return_path_stats=True requires decision_mask and threeplus_mask")

        # to torch bool on device
        if not torch.is_tensor(decision_mask):
            decision_mask = torch.tensor(decision_mask, dtype=torch.bool)
        if not torch.is_tensor(threeplus_mask):
            threeplus_mask = torch.tensor(threeplus_mask, dtype=torch.bool)

        decision_mask  = decision_mask.to(device=device)
        threeplus_mask = threeplus_mask.to(device=device)

        # histogram of decision-step counts (0..D)
        pathlen_hist = torch.zeros(D + 1, dtype=torch.long)  # CPU
        any3plus_total = 0

    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device)
        bs = int(x.size(0))
        n_samples += bs

        _, aux = model(x, return_aux=True)
        g_soft = aux["g_soft"]                 # [bs, N, B]

        node = torch.zeros(bs, dtype=torch.long, device=device)  # current internal node id
        leaf_id = torch.zeros(bs, dtype=torch.long, device=device)

        if return_path_stats:
            dec_steps = torch.zeros(bs, dtype=torch.int16, device=device)
            seen3plus = torch.zeros(bs, dtype=torch.bool, device=device)

        for _ in range(D):
            # reached node count
            node_cpu = node.detach().cpu()
            node_hits.scatter_add_(0, node_cpu, torch.ones_like(node_cpu, dtype=torch.long))

            if return_path_stats:
                # count "decision" nodes only => contracted path length proxy
                dec_steps += decision_mask[node].to(dec_steps.dtype)
                seen3plus |= threeplus_mask[node]

            # choose branch only for visited nodes (avoid [bs,N] argmax)
            br = torch.argmax(g_soft[torch.arange(bs, device=device), node, :], dim=-1)  # [bs]

            # branch hit
            flat = (node * B + br).detach().cpu()  # [bs]
            branch_hits.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.long))

            # update leaf id and move to child
            leaf_id = leaf_id * B + br
            node = node * B + (br + 1)

        # final leaf hit
        leaf_cpu = leaf_id.detach().cpu()
        leaf_hits.scatter_add_(0, leaf_cpu, torch.ones_like(leaf_cpu, dtype=torch.long))

        if return_path_stats:
            dc = dec_steps.detach().cpu().to(torch.long)  # [bs]
            h = torch.bincount(dc, minlength=D + 1)
            pathlen_hist += h
            any3plus_total += int(seen3plus.detach().sum().item())

    if not return_path_stats:
        return branch_hits, node_hits, leaf_hits, int(n_samples)

    # ---------- summarize path stats ----------
    hist = pathlen_hist.numpy()
    total = int(hist.sum()) if n_samples > 0 else 0

    def _q_from_hist(hist_arr, q):
        if total <= 0:
            return 0
        cdf = np.cumsum(hist_arr) / float(total)
        return int(np.searchsorted(cdf, q, side="left"))

    mean = float(np.dot(np.arange(len(hist)), hist) / float(total)) if total > 0 else 0.0
    med  = _q_from_hist(hist, 0.5)
    p90  = _q_from_hist(hist, 0.9)

    # empirical arity of reached nodes on this loader (counts how many branches actually taken)
    reached = (node_hits > 0)
    arity_taken = (branch_hits >= int(min_branch_hit)).sum(dim=1)  # [N]
    arity_reached = arity_taken[reached]
    a1 = int((arity_reached == 1).sum().item())
    a2 = int((arity_reached == 2).sum().item())
    a3 = int((arity_reached >= 3).sum().item())
    denom = int(max(int((arity_reached > 0).sum().item()), 1))
    unary_ratio_emp = float(a1 / denom)

    path_stats = {
        "pathlen_hist_decisions": hist.tolist(),
        "pathlen_decisions_mean": float(mean),
        "pathlen_decisions_median": int(med),
        "pathlen_decisions_p90": int(p90),
        "sample_frac_through_3plus_nodes": float(any3plus_total / float(n_samples)) if n_samples > 0 else 0.0,
        "empirical_arity_nodes_1": a1,
        "empirical_arity_nodes_2": a2,
        "empirical_arity_nodes_3plus": a3,
        "empirical_unary_ratio": unary_ratio_emp,
    }

    return branch_hits, node_hits, leaf_hits, int(n_samples), path_stats


def mbndt_export_tree_from_branch_hits(model, branch_hits, min_edge_hits=1):

    B = model.B
    N = model.num_internal_nodes

    # depth (heap indexing)
    depth = _depths_for_internal_nodes(N, B)

    nodes_set = set()
    edges = []

    for j in range(N):
        for b in range(B):
            c = int(branch_hits[j, b].item())
            if c < min_edge_hits:
                continue
            nodes_set.add(j)
            child = j * B + (b + 1)
            # child may be internal or leaf-layer
            edges.append({
                "parent": int(j),
                "branch": int(b),
                "child": int(child),
                "count": int(c),
                "depth": int(depth[j].item()),
            })
            if child < N:
                nodes_set.add(child)

    nodes = [{"node": int(j), "depth": int(depth[j].item())} for j in sorted(nodes_set)]
    return nodes, edges



import matplotlib.pyplot as plt

def mbndt_plot_effective_tree_from_hits(
    nodes, edges,
    title="Empirical hard-path subtree",
    draw_leaf=True,
    figsize=(12, 6),
    edge_width_mode="log",   # {"const","log","sqrt"}
    show_edge_counts=True,
):

    internal_nodes = {n["node"] for n in nodes}
    adj = {u: [] for u in internal_nodes}
    leaf_nodes = set()

    for e in edges:
        u = e["parent"]
        v = e["child"]
        b = e["branch"]
        if u not in adj:
            continue
        if v in internal_nodes:
            adj[u].append((v, b, e))
        else:
            if draw_leaf:
                leaf_id = f"leaf_{u}_{b}"
                leaf_nodes.add(leaf_id)
                adj[u].append((leaf_id, b, e))

    # compute depths by traversal
    depth = {0: 0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v, _, _ in adj.get(u, []):
            if v in internal_nodes and v not in depth:
                depth[v] = depth[u] + 1
                stack.append(v)

    # layout x by DFS leaf-order
    x = {}
    y = {}
    leaf_order = 0

    def dfs_layout(u):
        nonlocal leaf_order
        children = sorted(adj.get(u, []), key=lambda t: t[1])
        if not children:
            x[u] = float(leaf_order)
            leaf_order += 1
            return x[u]
        child_xs = []
        for v, _, _ in children:
            if v in internal_nodes:
                child_xs.append(dfs_layout(v))
            else:
                x[v] = float(leaf_order)
                leaf_order += 1
                child_xs.append(x[v])
        x[u] = sum(child_xs) / max(len(child_xs), 1)
        return x[u]

    if 0 in internal_nodes:
        dfs_layout(0)

    for u in internal_nodes:
        y[u] = -float(depth.get(u, 0))
    for v in leaf_nodes:
        parts = v.split("_")
        pu = int(parts[1])
        y[v] = -float(depth.get(pu, 0) + 1)

    # edge width scaling
    counts = [e["count"] for e in edges] if edges else [1]
    cmax = max(counts)
    def w(c):
        if edge_width_mode == "sqrt":
            return max(0.5, (c ** 0.5))
        if edge_width_mode == "log":
            return max(0.5, math.log1p(c))
        return 1.5

    fig, ax = plt.subplots(figsize=figsize)

    # edges
    for e in edges:
        u = e["parent"]
        b = e["branch"]
        v = e["child"]
        if u not in x:
            continue

        if v not in internal_nodes:
            v_key = f"leaf_{u}_{b}"
            if not draw_leaf or v_key not in x:
                continue
            vx, vy = x[v_key], y[v_key]
        else:
            if v not in x:
                continue
            vx, vy = x[v], y[v]

        ax.plot([x[u], vx], [y[u], vy], linewidth=w(e["count"]))

        if show_edge_counts:
            mx, my = (x[u] + vx)/2.0, (y[u] + vy)/2.0
            ax.text(mx, my, f"b={b}\n{e['count']}", fontsize=8, ha="center", va="center")

    # nodes
    for n in nodes:
        u = n["node"]
        if u not in x:
            continue
        ax.scatter([x[u]], [y[u]])
        ax.text(x[u], y[u], f"n{u}", fontsize=9, ha="center", va="bottom")

    if draw_leaf:
        for v in leaf_nodes:
            if v not in x:
                continue
            ax.scatter([x[v]], [y[v]], marker="x")
            ax.text(x[v], y[v], "L", fontsize=8, ha="center", va="bottom")

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    return fig, ax












#######################################################################################################
#######################################################################################################
#######################################################################################################
########################          Model Complexity Reports          ##############################


# -------------------------------------------------------------------------
# JSON utilities
# -------------------------------------------------------------------------

def _to_jsonable(x):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
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


# -------------------------------------------------------------------------
# Effective tree structural stats (1-branch passthrough nodes are tracked
# separately and excluded from the "effective" branch count)
# -------------------------------------------------------------------------

def effective_tree_struct_stats(model, nodes, edges):
    """
    Returns:
      stats: dict
        - internal_nodes: nodes that have at least one surviving outgoing edge
        - decision_nodes: nodes with outdeg >= 2 (the 'real' decision points)
        - passthrough_nodes: nodes with outdeg == 1 (skipped per spec)
        - arity_2_nodes / arity_3plus_nodes
        - routable_branches_effective: total surviving branches but only at
          nodes with outdeg >= 2. Branches at passthrough nodes contribute 0.
      decision_mask: bool [N]   (outdeg >= 2)
      threeplus_mask: bool [N]  (outdeg >= 3)
      outdeg: int32 [N]
    """
    N = int(model.num_internal_nodes)
    internal_set = {int(n["node"]) for n in nodes if "node" in n}

    outdeg = np.zeros(N, dtype=np.int32)
    for e in edges:
        u = int(e.get("parent"))
        if 0 <= u < N and u in internal_set:
            outdeg[u] += 1

    used = sorted(u for u in internal_set if 0 <= u < N and outdeg[u] > 0)
    deg = outdeg[used] if used else np.array([], dtype=np.int32)

    a1 = int((deg == 1).sum())
    a2 = int((deg == 2).sum())
    a3 = int((deg >= 3).sum())
    n_internal = int(len(used))
    n_decision = int((deg >= 2).sum())
    n_branches_eff = int(deg[deg >= 2].sum()) if used else 0

    decision_mask = np.zeros(N, dtype=np.bool_)
    threeplus_mask = np.zeros(N, dtype=np.bool_)
    for u in used:
        if outdeg[u] >= 2:
            decision_mask[u] = True
        if outdeg[u] >= 3:
            threeplus_mask[u] = True

    stats = {
        "internal_nodes": n_internal,
        "decision_nodes": n_decision,
        "passthrough_nodes": a1,
        "arity_2_nodes": a2,
        "arity_3plus_nodes": a3,
        "routable_branches_effective": n_branches_eff,
    }
    return stats, decision_mask, threeplus_mask, outdeg


# -------------------------------------------------------------------------
# Slim per-split saver for the function-hard MBNDT
# -------------------------------------------------------------------------

@torch.no_grad()
def save_mbndt_outer_artifacts(
    NT5_v7,
    model,
    trainval_loader,        # combined train + ES-val (caller builds it)
    test_loader,
    out_dir: Path,
    device: str = "cuda",
    z_min: float = -6,
    z_max: float = 6,
    n_grid: int = 8001,
    apply_masks: bool = True,
    mask_eps: float = 0.0,
    tol: float = 1e-12,
    min_edge_hits: int = 1,
):
    """
    Slim function-hard MBNDT report. Returns a flat scalar dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model = model.to(device)

    # 1) Function-hard routability report
    report, _diag, details = NT5_v7.mbndt_function_hard_routability_report(
        model,
        z_min=z_min, z_max=z_max, n_grid=n_grid,
        apply_masks=apply_masks,
        mask_eps=mask_eps,
        tol=tol,
        max_record_leaves=0,
        record_all_leaf_ids=True,
        return_details=True,
    )
    function_leaf_ids = sorted({int(x) for x in details.get("function_hard_routable_leaf_ids", [])})

    # 2) Effective tree -> decision_mask used for path-length
    nodes_f, edges_f = NT5_v7.mbndt_export_effective_tree_edges_fast(
        model,
        regions=details["regions"],
        feat_idx=details["feat_idx"],
        node_used=details["node_used_feat"],
        branch_used=details["branch_used_feat"],
        include_feature=True,
        include_mask=False,
        include_regions=False,
    )
    struct_stats, decision_mask, threeplus_mask, _ = effective_tree_struct_stats(
        model, nodes_f, edges_f
    )

    # 3) Path-length stats on (train+val) and test
    _, _, _, n_tv, tv_path = NT5_v7.mbndt_collect_hard_path_counts(
        model, trainval_loader, device=device,
        decision_mask=decision_mask, threeplus_mask=threeplus_mask,
        return_path_stats=True, min_branch_hit=min_edge_hits,
    )
    _, _, _, n_te, te_path = NT5_v7.mbndt_collect_hard_path_counts(
        model, test_loader, device=device,
        decision_mask=decision_mask, threeplus_mask=threeplus_mask,
        return_path_stats=True, min_branch_hit=min_edge_hits,
    )

    scalars = {
        # routability
        "routable_leaves": int(len(function_leaf_ids)),
        "routable_branches_effective": int(struct_stats["routable_branches_effective"]),
        "decision_nodes": int(struct_stats["decision_nodes"]),
        "passthrough_nodes": int(struct_stats["passthrough_nodes"]),
        "arity_2_nodes": int(struct_stats["arity_2_nodes"]),
        "arity_3plus_nodes": int(struct_stats["arity_3plus_nodes"]),
        # path length on (train+val)
        "trainval_pathlen_mean":   float(tv_path["pathlen_decisions_mean"]),
        "trainval_pathlen_median": float(tv_path["pathlen_decisions_median"]),
        "trainval_pathlen_p90":    float(tv_path["pathlen_decisions_p90"]),
        "trainval_n":              int(n_tv),
        # path length on test
        "test_pathlen_mean":   float(te_path["pathlen_decisions_mean"]),
        "test_pathlen_median": float(te_path["pathlen_decisions_median"]),
        "test_pathlen_p90":    float(te_path["pathlen_decisions_p90"]),
        "test_n":              int(n_te),
    }
    save_json(out_dir / "mbndt_report.json", scalars)
    return scalars


# -------------------------------------------------------------------------
# Aggregate scalar reports across outer splits
# -------------------------------------------------------------------------

def aggregate_scalar_reports(reports_list):
    keys = sorted(set().union(*[r.keys() for r in reports_list]))
    out = {}
    for k in keys:
        vals = []
        for r in reports_list:
            v = r.get(k, None)
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                vals.append(float(v))
        if not vals:
            continue
        out[k] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n":    int(len(vals)),
        }
    return out


# -------------------------------------------------------------------------
# Performance-gap helper (script-side convenience)
# -------------------------------------------------------------------------

def compute_perf_gaps(train_metrics: dict, val_metrics: dict, test_metrics: dict):
    """
    For each shared scalar metric key produces:
      gap_train_test/<k>, gap_train_val/<k>, gap_val_test/<k>
    Non-numeric or non-shared keys are ignored.
    """
    out = {}
    common = set(train_metrics) & set(val_metrics) & set(test_metrics)
    for k in common:
        a, b, c = train_metrics[k], val_metrics[k], test_metrics[k]
        if all(
            isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            for v in (a, b, c)
        ):
            out[f"gap_train_test/{k}"] = float(a) - float(c)
            out[f"gap_train_val/{k}"]  = float(a) - float(b)
            out[f"gap_val_test/{k}"]   = float(b) - float(c)
    return out






# # ============================================================================
# # Post-hoc train-path-pruned MBNDT variant (MBNDT-PP)
# # ============================================================================

# def _nearest_active_branch_map(active_bs, B: int, tie_break: str = "left"):
#     """Build remap old_b -> nearest surviving b. Equal-distance ties go left or right."""
#     active_bs = sorted(int(b) for b in active_bs)
#     remap = torch.full((int(B),), -1, dtype=torch.long)
#     if not active_bs:
#         return remap
#     act = torch.tensor(active_bs, dtype=torch.long)
#     for b in range(int(B)):
#         if b in active_bs:
#             remap[b] = int(b)
#             continue
#         dist = torch.abs(act - int(b))
#         dmin = int(dist.min().item())
#         cands = act[dist == dmin]
#         chosen = int(cands.max().item()) if tie_break == "right" else int(cands.min().item())
#         remap[b] = chosen
#     return remap


# @torch.no_grad()
# def mbndt_make_train_prune_spec(
#     model,
#     train_loader,           # typically train + ES-val combined
#     device: str = "cuda",
#     min_branch_hit: int = 1,
# ):
#     """
#     Build a post-hoc prune spec from hard paths on the construction loader.
#     Surviving branches at each visited node are those with
#     branch_hits >= min_branch_hit. Used by PostHocMergedMBNDT, which performs
#     masked-argmax redirection against active_branch_mask — no precomputed
#     redirect table needed.
#     """
#     B = int(model.B)
#     N = int(model.num_internal_nodes)
#     L = int(model.num_leaves)

#     branch_hits, node_hits, leaf_hits, n_samples = mbndt_collect_hard_path_counts(
#         model, train_loader, device=device, return_path_stats=False,
#     )
#     branch_hits = branch_hits.cpu()
#     node_hits   = node_hits.cpu()
#     leaf_hits   = leaf_hits.cpu()

#     node_alive         = (node_hits >= 1)
#     active_branch_mask = (branch_hits >= int(min_branch_hit)) & node_alive.unsqueeze(1)

#     # With min_branch_hit > 1, a visited node may have no branch surviving the
#     # threshold. Catch this rather than silently producing an unroutable node.
#     if int(min_branch_hit) > 1:
#         any_alive = active_branch_mask.any(dim=1)
#         orphaned = ((~any_alive) & node_alive).nonzero(as_tuple=False).flatten().tolist()
#         if orphaned:
#             raise RuntimeError(
#                 f"Nodes visited but lost all branches under "
#                 f"min_branch_hit={min_branch_hit}: {orphaned[:10]}"
#             )

#     sums             = active_branch_mask.sum(dim=1)
#     decision_mask_pp = ((sums >= 2) & node_alive)
#     passthrough_mask = ((sums == 1) & node_alive)
#     n_branches_eff   = int((active_branch_mask & decision_mask_pp.unsqueeze(1)).sum().item())

#     summary = {
#         "n_samples": int(n_samples),
#         "min_branch_hit": int(min_branch_hit),

#         # Data-dependent full-depth facts.
#         # These are NOT compressed/function-hard structural complexity.
#         "train_reached_internal_nodes_full_depth": int(node_alive.sum().item()),
#         "train_allocated_leaves_full_depth": int((leaf_hits > 0).sum().item()),
#         "train_active_branches_full_depth": int(active_branch_mask.sum().item()),

#         # These are also full-depth train-hit masks, not final PP complexity.
#         "trainhit_decision_nodes_full_depth": int(decision_mask_pp.sum().item()),
#         "trainhit_passthrough_nodes_full_depth": int(passthrough_mask.sum().item()),
#         "trainhit_branches_effective_full_depth": int(n_branches_eff),

#         "nominal_internal_nodes": int(N),
#         "nominal_leaves": int(L),
#     }

#     return {
#         "node_hits":          node_hits,
#         "branch_hits":        branch_hits,
#         "leaf_hits":          leaf_hits,
#         "active_branch_mask": active_branch_mask,
#         "node_alive":         node_alive,
#         "decision_mask_pp":   decision_mask_pp,
#         "n_samples":          int(n_samples),
#         "summary":            summary,
#     }


# class PostHocMergedMBNDT(nn.Module):
#     """
#     Frozen post-hoc MBNDT variant. At each node, the next branch is the
#     argmax of the base model's g_soft restricted to alive (train-visited)
#     branches. This is a likelihood-nearest redirect: dead branches are
#     merged into the surviving branch the model itself ranks highest.
#     Still a hard axis-aligned decision tree.
#     """
#     def __init__(self, base_model, prune_spec: dict, freeze_base: bool = True):
#         super().__init__()
#         self.base_model = copy.deepcopy(base_model)
#         self.base_model.eval()
#         if freeze_base:
#             for p in self.base_model.parameters():
#                 p.requires_grad = False

#         self.D = int(self.base_model.D)
#         self.B = int(self.base_model.B)
#         self.num_internal_nodes = int(self.base_model.num_internal_nodes)
#         self.num_leaves = int(self.base_model.num_leaves)

#         # The redirect table is no longer needed - the active mask is enough.
#         self.register_buffer("posthoc_active_branch_mask", prune_spec["active_branch_mask"].bool().clone(), persistent=True)
#         self.register_buffer("posthoc_node_alive",         prune_spec["node_alive"].bool().clone(),         persistent=True)
#         self.register_buffer("posthoc_decision_mask",      prune_spec["decision_mask_pp"].bool().clone(),   persistent=True)
#         self.prune_summary = copy.deepcopy(prune_spec.get("summary", {}))

#     @torch.no_grad()
#     def route_leaf_ids(self, x, return_paths: bool = False):
#         self.base_model.eval()
#         _, aux = self.base_model(x, return_aux=True)
#         g_soft = aux["g_soft"]                                  # [bs, N, B]
#         bs = int(x.size(0))
#         dev = x.device
#         ar = torch.arange(bs, device=dev)

#         node    = torch.zeros(bs, dtype=torch.long, device=dev)
#         leaf_id = torch.zeros(bs, dtype=torch.long, device=dev)

#         node_path = [] if return_paths else None
#         new_path  = [] if return_paths else None

#         for _ in range(self.D):
#             g_node = g_soft[ar, node, :]                                # [bs, B]
#             active = self.posthoc_active_branch_mask[node]              # [bs, B] bool

#             # Defensive: any reached node must have at least one alive branch.
#             # By construction this should always hold (alive node ⇒ at least one
#             # training-visited branch), but the check is cheap.
#             no_alive = ~active.any(dim=-1)
#             if torch.any(no_alive):
#                 bad = torch.nonzero(no_alive, as_tuple=False).flatten().cpu().tolist()
#                 raise RuntimeError(
#                     "PostHocMergedMBNDT reached a node with no surviving branches. "
#                     f"Sample idx (first few): {bad[:10]}, node ids: {node[bad].cpu().tolist()[:10]}"
#                 )

#             # Likelihood-nearest redirect: argmax of g_soft restricted to alive branches.
#             # masked_fill on a fresh tensor avoids touching the cached g_soft.
#             g_masked = g_node.masked_fill(~active, float("-inf"))
#             br_new = torch.argmax(g_masked, dim=-1)                     # [bs]

#             if return_paths:
#                 node_path.append(node.detach().clone())
#                 new_path.append(br_new.detach().clone())

#             leaf_id = leaf_id * self.B + br_new
#             node    = node    * self.B + (br_new + 1)

#         if not return_paths:
#             return leaf_id
#         return {
#             "leaf_id":              leaf_id,
#             "node_path":            torch.stack(node_path, dim=1),
#             "posthoc_branch_path":  torch.stack(new_path,  dim=1),
#         }

#     def forward(self, x, return_aux: bool = False):
#         x = x.float()
#         routed = self.route_leaf_ids(x, return_paths=return_aux)
#         leaf_id = routed["leaf_id"] if return_aux else routed
#         logits = self.base_model.leaf_logits[leaf_id]
#         if not return_aux:
#             return logits
#         aux = {
#             "posthoc_leaf_id":     leaf_id,
#             "node_path":           routed["node_path"],
#             "posthoc_branch_path": routed["posthoc_branch_path"],
#         }
#         return logits, aux


# @torch.no_grad()
# def build_posthoc_merged_mbndt(
#     model,
#     train_loader,
#     device: str = "cuda",
#     min_branch_hit: int = 1,
#     freeze_base: bool = True,
# ):
#     prune_spec = mbndt_make_train_prune_spec(
#         model=model, train_loader=train_loader, device=device,
#         min_branch_hit=min_branch_hit,
#     )
#     pruned = PostHocMergedMBNDT(model, prune_spec=prune_spec, freeze_base=freeze_base)
#     pruned = pruned.to(next(model.parameters()).device)
#     return pruned, prune_spec


# @torch.no_grad()
# def mbndt_collect_posthoc_path_counts(
#     posthoc_model,
#     loader,
#     device: str = "cuda",
#     return_path_stats: bool = True,
# ):
#     """
#     Hard-path counts under PP-redirected routing. With return_path_stats=True,
#     also returns per-sample path-length stats counting only decision nodes
#     (nodes with >=2 surviving branches). 1-branch passthrough nodes don't count.
#     """
#     posthoc_model.eval()
#     posthoc_model = posthoc_model.to(device)

#     B = int(posthoc_model.B)
#     D = int(posthoc_model.D)
#     N = int(posthoc_model.num_internal_nodes)
#     L = int(posthoc_model.num_leaves)

#     branch_hits = torch.zeros(N, B, dtype=torch.long)
#     node_hits   = torch.zeros(N,    dtype=torch.long)
#     leaf_hits   = torch.zeros(L,    dtype=torch.long)
#     n_samples = 0

#     decision_mask = posthoc_model.posthoc_decision_mask.cpu()  # [N] bool
#     pathlens = []

#     for batch in loader:
#         x = batch[0] if isinstance(batch, (tuple, list)) else batch
#         x = x.to(device).float()
#         bs = int(x.size(0))
#         n_samples += bs

#         routed = posthoc_model.route_leaf_ids(x, return_paths=True)
#         leaf_id     = routed["leaf_id"].detach().cpu()
#         node_path   = routed["node_path"].detach().cpu()              # [bs, D]
#         branch_path = routed["posthoc_branch_path"].detach().cpu()    # [bs, D]

#         for d in range(D):
#             nodes_d = node_path[:, d]
#             br_d    = branch_path[:, d]
#             node_hits.scatter_add_(0, nodes_d, torch.ones_like(nodes_d, dtype=torch.long))
#             flat = nodes_d * B + br_d
#             branch_hits.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.long))

#         leaf_hits.scatter_add_(0, leaf_id, torch.ones_like(leaf_id, dtype=torch.long))

#         if return_path_stats:
#             is_dec = decision_mask[node_path]    # [bs, D] bool
#             pathlens.append(is_dec.sum(dim=1))   # [bs]

#     if return_path_stats:
#         if pathlens:
#             pl = torch.cat(pathlens).float().numpy()
#             path_stats = {
#                 "pathlen_decisions_mean":   float(np.mean(pl)),
#                 "pathlen_decisions_median": float(np.median(pl)),
#                 "pathlen_decisions_p90":    float(np.percentile(pl, 90)),
#             }
#         else:
#             path_stats = {
#                 "pathlen_decisions_mean": 0.0,
#                 "pathlen_decisions_median": 0.0,
#                 "pathlen_decisions_p90": 0.0,
#             }
#         return branch_hits, node_hits, leaf_hits, int(n_samples), path_stats

#     return branch_hits, node_hits, leaf_hits, int(n_samples)


# # -------------------------------------------------------------------------
# # Slim per-split saver for MBNDT-PP, symmetric to save_mbndt_outer_artifacts
# # -------------------------------------------------------------------------

# @torch.no_grad()
# def save_mbndt_pp_outer_artifacts(
#     posthoc_model,
#     prune_spec: dict,
#     trainval_loader,
#     test_loader,
#     out_dir: Path,
#     device: str = "cuda",
# ):
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     s = prune_spec.get("summary", {})

#     _, _, _, n_tv, tv_path = mbndt_collect_posthoc_path_counts(
#         posthoc_model, trainval_loader, device=device, return_path_stats=True,
#     )
#     _, _, _, n_te, te_path = mbndt_collect_posthoc_path_counts(
#         posthoc_model, test_loader,     device=device, return_path_stats=True,
#     )

#     scalars = {
#         "routable_leaves":             int(s.get("routable_leaves", 0)),
#         "routable_branches_effective": int(s.get("routable_branches_effective", 0)),
#         "decision_nodes":              int(s.get("decision_nodes", 0)),
#         "passthrough_nodes":           int(s.get("passthrough_nodes", 0)),
#         "trainval_pathlen_mean":   float(tv_path["pathlen_decisions_mean"]),
#         "trainval_pathlen_median": float(tv_path["pathlen_decisions_median"]),
#         "trainval_pathlen_p90":    float(tv_path["pathlen_decisions_p90"]),
#         "trainval_n":              int(n_tv),
#         "test_pathlen_mean":   float(te_path["pathlen_decisions_mean"]),
#         "test_pathlen_median": float(te_path["pathlen_decisions_median"]),
#         "test_pathlen_p90":    float(te_path["pathlen_decisions_p90"]),
#         "test_n":              int(n_te),
#     }
#     save_json(out_dir / "mbndt_pp_report.json", scalars)
#     return scalars





# ============================================================================
# Post-hoc train-path-pruned MBNDT variant (MBNDT-PP)
# Corrected version:
#   - separates full-depth train allocation from compressed structural complexity
#   - allows PP structural report to reuse the SAME raw MBNDT structural counter
# ============================================================================

def _nearest_active_branch_map(active_bs, B: int, tie_break: str = "left"):
    """Build remap old_b -> nearest surviving b. Equal-distance ties go left or right."""
    active_bs = sorted(int(b) for b in active_bs)
    remap = torch.full((int(B),), -1, dtype=torch.long)
    if not active_bs:
        return remap

    act = torch.tensor(active_bs, dtype=torch.long)
    for b in range(int(B)):
        if b in active_bs:
            remap[b] = int(b)
            continue
        dist = torch.abs(act - int(b))
        dmin = int(dist.min().item())
        cands = act[dist == dmin]
        chosen = int(cands.max().item()) if tie_break == "right" else int(cands.min().item())
        remap[b] = chosen

    return remap


@torch.no_grad()
def _compute_compressed_structure_from_allowed_mask(
    *,
    allowed_branch_mask,
    B: int,
    D: int,
    num_internal_nodes: int,
    num_leaves: int,
):
    """
    Fallback compressed structural counter for MBNDT-PP.

    This is NOT a data allocation count.
    It counts the compressed tree induced by allowed_branch_mask:
        - node with 0 surviving branches: unreachable / ignored
        - node with 1 surviving branch: passthrough/compressed
        - node with >=2 surviving branches: decision node
        - leaves are counted after recursively compressing passthrough chains

    For exact apples-to-apples comparison with raw MBNDT, prefer passing the
    original raw MBNDT function-hard structural counter to save_mbndt_pp_outer_artifacts
    through structural_counter_fn.
    """
    allowed = allowed_branch_mask.detach().cpu().bool()
    B = int(B)
    D = int(D)
    N = int(num_internal_nodes)

    visited_internal = set()
    decision_nodes = set()
    passthrough_nodes = set()
    effective_branches = 0
    arity_hist = {}

    def child_node_id(node_id: int, branch_id: int) -> int:
        return node_id * B + (branch_id + 1)

    def rec(node_id: int, depth: int):
        nonlocal effective_branches

        # Reached a nominal leaf.
        if depth >= D:
            return 1

        # Defensive: if somehow outside internal range, treat as no subtree.
        if node_id < 0 or node_id >= N:
            return 0

        active_bs = torch.nonzero(allowed[node_id], as_tuple=False).flatten().tolist()

        # No surviving branch => this node contributes no routable compressed leaves.
        if len(active_bs) == 0:
            return 0

        visited_internal.add(int(node_id))

        child_leaf_counts = []
        for b in active_bs:
            c = child_node_id(int(node_id), int(b))
            child_leaf_counts.append(rec(c, depth + 1))

        # Remove zero-child subtrees defensively.
        child_leaf_counts = [c for c in child_leaf_counts if c > 0]

        if len(child_leaf_counts) == 0:
            return 0

        # After removing impossible children, this node may collapse.
        arity = len(child_leaf_counts)

        if arity == 1:
            passthrough_nodes.add(int(node_id))
            return int(child_leaf_counts[0])

        decision_nodes.add(int(node_id))
        effective_branches += int(arity)
        arity_hist[arity] = arity_hist.get(arity, 0) + 1

        return int(sum(child_leaf_counts))

    routable_leaves = rec(0, 0)

    return {
        "routable_leaves": int(routable_leaves),
        "routable_branches_effective": int(effective_branches),
        "decision_nodes": int(len(decision_nodes)),
        "passthrough_nodes": int(len(passthrough_nodes)),
        "arity_2_nodes": int(arity_hist.get(2, 0)),
        "arity_3plus_nodes": int(sum(v for k, v in arity_hist.items() if int(k) >= 3)),
        "visited_internal_nodes_compressed": int(len(visited_internal)),
        "nominal_internal_nodes": int(num_internal_nodes),
        "nominal_leaves": int(num_leaves),
    }


@torch.no_grad()
def mbndt_make_train_prune_spec(
    model,
    train_loader,           # typically train + ES-val combined
    device: str = "cuda",
    min_branch_hit: int = 1,
):
    """
    Build a post-hoc prune spec from hard paths on the construction loader.

    Important:
    This function only builds train-hit masks and full-depth allocation diagnostics.
    It does NOT define compressed/function-hard structural complexity.
    """
    B = int(model.B)
    N = int(model.num_internal_nodes)
    L = int(model.num_leaves)

    branch_hits, node_hits, leaf_hits, n_samples = mbndt_collect_hard_path_counts(
        model, train_loader, device=device, return_path_stats=False,
    )

    branch_hits = branch_hits.cpu()
    node_hits   = node_hits.cpu()
    leaf_hits   = leaf_hits.cpu()

    node_alive = node_hits >= 1
    active_branch_mask = (branch_hits >= int(min_branch_hit)) & node_alive.unsqueeze(1)

    # With min_branch_hit > 1, a visited node may have no branch surviving.
    if int(min_branch_hit) > 1:
        any_alive = active_branch_mask.any(dim=1)
        orphaned = ((~any_alive) & node_alive).nonzero(as_tuple=False).flatten().tolist()
        if orphaned:
            raise RuntimeError(
                f"Nodes visited but lost all branches under "
                f"min_branch_hit={min_branch_hit}: {orphaned[:10]}"
            )

    sums = active_branch_mask.sum(dim=1)
    decision_mask_pp = (sums >= 2) & node_alive
    passthrough_mask = (sums == 1) & node_alive

    n_trainhit_branches_eff = int(
        (active_branch_mask & decision_mask_pp.unsqueeze(1)).sum().item()
    )

    # Do NOT call these structural routable leaves.
    # These are only train-hit/full-depth diagnostics.
    summary = {
        "n_samples": int(n_samples),
        "min_branch_hit": int(min_branch_hit),

        "train_reached_internal_nodes_full_depth": int(node_alive.sum().item()),
        "train_allocated_leaves_full_depth": int((leaf_hits > 0).sum().item()),
        "train_active_branches_full_depth": int(active_branch_mask.sum().item()),

        "trainhit_decision_nodes_full_depth": int(decision_mask_pp.sum().item()),
        "trainhit_passthrough_nodes_full_depth": int(passthrough_mask.sum().item()),
        "trainhit_branches_effective_full_depth": int(n_trainhit_branches_eff),

        "nominal_internal_nodes": int(N),
        "nominal_leaves": int(L),
    }

    return {
        "node_hits": node_hits,
        "branch_hits": branch_hits,
        "leaf_hits": leaf_hits,
        "active_branch_mask": active_branch_mask,
        "node_alive": node_alive,
        "decision_mask_pp": decision_mask_pp,
        "n_samples": int(n_samples),
        "summary": summary,
    }


class PostHocMergedMBNDT(nn.Module):
    """
    Frozen post-hoc MBNDT variant.

    Prediction rule:
    At each node, choose argmax of base-model g_soft restricted to train-surviving
    branches. This is a redirected full-depth predictor.

    Complexity reporting:
    Do NOT use redirected full-depth leaf counts as structural routable leaves.
    Structural complexity must be computed separately by compressed structural counter.
    """
    def __init__(self, base_model, prune_spec: dict, freeze_base: bool = True):
        super().__init__()

        self.base_model = copy.deepcopy(base_model)
        self.base_model.eval()

        if freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad = False

        self.D = int(self.base_model.D)
        self.B = int(self.base_model.B)
        self.num_internal_nodes = int(self.base_model.num_internal_nodes)
        self.num_leaves = int(self.base_model.num_leaves)

        self.register_buffer(
            "posthoc_active_branch_mask",
            prune_spec["active_branch_mask"].bool().clone(),
            persistent=True,
        )
        self.register_buffer(
            "posthoc_node_alive",
            prune_spec["node_alive"].bool().clone(),
            persistent=True,
        )
        self.register_buffer(
            "posthoc_decision_mask",
            prune_spec["decision_mask_pp"].bool().clone(),
            persistent=True,
        )

        self.prune_summary = copy.deepcopy(prune_spec.get("summary", {}))

    @torch.no_grad()
    def route_leaf_ids(self, x, return_paths: bool = False):
        self.base_model.eval()

        _, aux = self.base_model(x, return_aux=True)
        g_soft = aux["g_soft"]  # [bs, N, B]

        bs = int(x.size(0))
        dev = x.device
        ar = torch.arange(bs, device=dev)

        node = torch.zeros(bs, dtype=torch.long, device=dev)
        leaf_id = torch.zeros(bs, dtype=torch.long, device=dev)

        node_path = [] if return_paths else None
        new_path = [] if return_paths else None

        for _ in range(self.D):
            g_node = g_soft[ar, node, :]                    # [bs, B]
            active = self.posthoc_active_branch_mask[node]  # [bs, B]

            no_alive = ~active.any(dim=-1)
            if torch.any(no_alive):
                bad = torch.nonzero(no_alive, as_tuple=False).flatten().cpu().tolist()
                bad_nodes = node[bad].detach().cpu().tolist()
                raise RuntimeError(
                    "PostHocMergedMBNDT reached a node with no surviving branches. "
                    f"Sample idx first few: {bad[:10]}, node ids first few: {bad_nodes[:10]}"
                )

            # Redirect by choosing the highest-probability surviving branch.
            g_masked = g_node.masked_fill(~active, float("-inf"))
            br_new = torch.argmax(g_masked, dim=-1)

            if return_paths:
                node_path.append(node.detach().clone())
                new_path.append(br_new.detach().clone())

            leaf_id = leaf_id * self.B + br_new
            node = node * self.B + (br_new + 1)

        if not return_paths:
            return leaf_id

        return {
            "leaf_id": leaf_id,
            "node_path": torch.stack(node_path, dim=1),
            "posthoc_branch_path": torch.stack(new_path, dim=1),
        }

    def forward(self, x, return_aux: bool = False):
        x = x.float()

        routed = self.route_leaf_ids(x, return_paths=return_aux)
        leaf_id = routed["leaf_id"] if return_aux else routed

        logits = self.base_model.leaf_logits[leaf_id]

        if not return_aux:
            return logits

        aux = {
            "posthoc_leaf_id": leaf_id,
            "node_path": routed["node_path"],
            "posthoc_branch_path": routed["posthoc_branch_path"],
        }
        return logits, aux


@torch.no_grad()
def build_posthoc_merged_mbndt(
    model,
    train_loader,
    device: str = "cuda",
    min_branch_hit: int = 1,
    freeze_base: bool = True,
):
    prune_spec = mbndt_make_train_prune_spec(
        model=model,
        train_loader=train_loader,
        device=device,
        min_branch_hit=min_branch_hit,
    )

    pruned = PostHocMergedMBNDT(
        model,
        prune_spec=prune_spec,
        freeze_base=freeze_base,
    )

    pruned = pruned.to(next(model.parameters()).device)

    return pruned, prune_spec


@torch.no_grad()
def mbndt_collect_posthoc_path_counts(
    posthoc_model,
    loader,
    device: str = "cuda",
    return_path_stats: bool = True,
):
    """
    Hard-path counts under PP-redirected routing.

    These are full-depth redirected allocation counts.
    They are useful diagnostics, but they are NOT compressed structural complexity.
    """
    posthoc_model.eval()
    posthoc_model = posthoc_model.to(device)

    B = int(posthoc_model.B)
    D = int(posthoc_model.D)
    N = int(posthoc_model.num_internal_nodes)
    L = int(posthoc_model.num_leaves)

    branch_hits = torch.zeros(N, B, dtype=torch.long)
    node_hits = torch.zeros(N, dtype=torch.long)
    leaf_hits = torch.zeros(L, dtype=torch.long)

    n_samples = 0

    decision_mask = posthoc_model.posthoc_decision_mask.cpu()
    pathlens = []

    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device).float()

        bs = int(x.size(0))
        n_samples += bs

        routed = posthoc_model.route_leaf_ids(x, return_paths=True)

        leaf_id = routed["leaf_id"].detach().cpu()
        node_path = routed["node_path"].detach().cpu()
        branch_path = routed["posthoc_branch_path"].detach().cpu()

        for d in range(D):
            nodes_d = node_path[:, d]
            br_d = branch_path[:, d]

            node_hits.scatter_add_(
                0,
                nodes_d,
                torch.ones_like(nodes_d, dtype=torch.long),
            )

            flat = nodes_d * B + br_d
            branch_hits.view(-1).scatter_add_(
                0,
                flat,
                torch.ones_like(flat, dtype=torch.long),
            )

        leaf_hits.scatter_add_(
            0,
            leaf_id,
            torch.ones_like(leaf_id, dtype=torch.long),
        )

        if return_path_stats:
            is_dec = decision_mask[node_path]
            pathlens.append(is_dec.sum(dim=1))

    if return_path_stats:
        if pathlens:
            pl = torch.cat(pathlens).float().numpy()
            path_stats = {
                "pathlen_decisions_mean": float(np.mean(pl)),
                "pathlen_decisions_median": float(np.median(pl)),
                "pathlen_decisions_p90": float(np.percentile(pl, 90)),
            }
        else:
            path_stats = {
                "pathlen_decisions_mean": 0.0,
                "pathlen_decisions_median": 0.0,
                "pathlen_decisions_p90": 0.0,
            }

        return branch_hits, node_hits, leaf_hits, int(n_samples), path_stats

    return branch_hits, node_hits, leaf_hits, int(n_samples)


@torch.no_grad()
def save_mbndt_pp_outer_artifacts(
    posthoc_model,
    prune_spec: dict,
    trainval_loader,
    test_loader,
    out_dir: Path,
    device: str = "cuda",
    structural_counter_fn=None,
    raw_struct_for_assert: dict = None,
):
    """
    Save corrected MBNDT-PP structure/path artifacts.

    structural_counter_fn:
        Optional function that should be the SAME function used for raw MBNDT
        compressed/function-hard structural reporting.

        Expected call signature:
            structural_counter_fn(
                base_model,
                allowed_branch_mask=allowed_branch_mask,
                device=device,
            )

        Expected return keys:
            routable_leaves
            routable_branches_effective
            decision_nodes
            passthrough_nodes
            optionally arity_2_nodes, arity_3plus_nodes

    raw_struct_for_assert:
        Optional raw MBNDT structural scalars. If provided, this function asserts
        that PP compressed structure does not exceed raw compressed structure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model = posthoc_model.base_model
    allowed_branch_mask = prune_spec["active_branch_mask"].bool()

    # ------------------------------------------------------------------
    # 1) Correct compressed/function-hard structural count
    # ------------------------------------------------------------------
    if structural_counter_fn is not None:
        pp_struct = structural_counter_fn(
            base_model,
            allowed_branch_mask=allowed_branch_mask,
            device=device,
        )
    else:
        # Fallback: compressed count from train-surviving branch mask only.
        # This is better than the previous leaf_hits>0 bug, but for exact
        # apples-to-apples reporting, pass the raw MBNDT structural counter.
        pp_struct = _compute_compressed_structure_from_allowed_mask(
            allowed_branch_mask=allowed_branch_mask,
            B=int(posthoc_model.B),
            D=int(posthoc_model.D),
            num_internal_nodes=int(posthoc_model.num_internal_nodes),
            num_leaves=int(posthoc_model.num_leaves),
        )

    # ------------------------------------------------------------------
    # 2) Full-depth redirected path diagnostics
    # ------------------------------------------------------------------
    _, _, tv_leaf_hits, n_tv, tv_path = mbndt_collect_posthoc_path_counts(
        posthoc_model,
        trainval_loader,
        device=device,
        return_path_stats=True,
    )

    _, _, te_leaf_hits, n_te, te_path = mbndt_collect_posthoc_path_counts(
        posthoc_model,
        test_loader,
        device=device,
        return_path_stats=True,
    )

    s = prune_spec.get("summary", {})

    scalars = {
        # Correct compressed structural complexity
        "routable_leaves": int(pp_struct.get("routable_leaves", 0)),
        "routable_branches_effective": int(pp_struct.get("routable_branches_effective", 0)),
        "decision_nodes": int(pp_struct.get("decision_nodes", 0)),
        "passthrough_nodes": int(pp_struct.get("passthrough_nodes", 0)),
        "arity_2_nodes": int(pp_struct.get("arity_2_nodes", 0)),
        "arity_3plus_nodes": int(pp_struct.get("arity_3plus_nodes", 0)),

        # Full-depth train-hit diagnostics from prune construction
        "train_reached_internal_nodes_full_depth": int(
            s.get("train_reached_internal_nodes_full_depth", 0)
        ),
        "train_allocated_leaves_full_depth": int(
            s.get("train_allocated_leaves_full_depth", 0)
        ),
        "train_active_branches_full_depth": int(
            s.get("train_active_branches_full_depth", 0)
        ),
        "trainhit_decision_nodes_full_depth": int(
            s.get("trainhit_decision_nodes_full_depth", 0)
        ),
        "trainhit_passthrough_nodes_full_depth": int(
            s.get("trainhit_passthrough_nodes_full_depth", 0)
        ),
        "trainhit_branches_effective_full_depth": int(
            s.get("trainhit_branches_effective_full_depth", 0)
        ),

        # Full-depth redirected allocation diagnostics under PP predictor
        "pp_trainval_redirected_allocated_leaves_full_depth": int(
            (tv_leaf_hits > 0).sum().item()
        ),
        "pp_test_redirected_allocated_leaves_full_depth": int(
            (te_leaf_hits > 0).sum().item()
        ),

        # PP path length under redirected predictor
        "trainval_pathlen_mean": float(tv_path["pathlen_decisions_mean"]),
        "trainval_pathlen_median": float(tv_path["pathlen_decisions_median"]),
        "trainval_pathlen_p90": float(tv_path["pathlen_decisions_p90"]),
        "trainval_n": int(n_tv),

        "test_pathlen_mean": float(te_path["pathlen_decisions_mean"]),
        "test_pathlen_median": float(te_path["pathlen_decisions_median"]),
        "test_pathlen_p90": float(te_path["pathlen_decisions_p90"]),
        "test_n": int(n_te),
    }

    # ------------------------------------------------------------------
    # 3) Optional monotonicity assertion against raw MBNDT structure
    # ------------------------------------------------------------------
    if raw_struct_for_assert is not None:
        for key in [
            "routable_leaves",
            "routable_branches_effective",
            "decision_nodes",
        ]:
            pp_v = int(scalars.get(key, 0))
            raw_v = int(raw_struct_for_assert.get(key, 0))

            if pp_v > raw_v:
                raise RuntimeError(
                    f"Invalid MBNDT-PP compressed structural count: {key} increased "
                    f"from raw={raw_v} to pp={pp_v}. "
                    "This means PP is not being counted with the same compressed "
                    "function-hard structural convention as raw MBNDT."
                )

    save_json(out_dir / "mbndt_pp_report.json", scalars)

    return scalars
