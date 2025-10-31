from __future__ import annotations
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["make_supervised_criterion", "FocalLoss", "class_balanced_alpha"]

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", alpha if isinstance(alpha, torch.Tensor) else None, persistent=False)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        if self.label_smoothing > 0.0:
            with torch.no_grad():
                true_dist = torch.zeros_like(logits).fill_(self.label_smoothing / (n_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            true_dist = torch.nn.functional.one_hot(targets, num_classes=n_classes).float()
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        ce = -(true_dist * logp).sum(dim=-1)  # cross-entropy per-sample
        mod = (1 - (p * true_dist).sum(dim=-1)).pow(self.gamma)
        if self.alpha is not None:
            a = self.alpha.to(logits.device).index_select(0, targets)
            ce = a * ce
        loss = mod * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def class_balanced_alpha(class_counts: Sequence[int], beta: float = 0.999) -> torch.Tensor:
    import math
    counts = torch.tensor(class_counts, dtype=torch.float32)
    eff_num = 1.0 - counts.pow(beta)
    eff_num = (1.0 - beta) / eff_num.clamp_min(1e-6)
    alpha = eff_num / eff_num.sum()
    return alpha

def make_supervised_criterion(n_classes: int, loss_cfg: dict) -> nn.Module:
    name = (loss_cfg or {}).get("name","cross_entropy").lower()
    ls = float((loss_cfg or {}).get("label_smoothing", 0.0))
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=ls)
    if name in ("focal","class_balanced_focal"):
        gamma = float((loss_cfg or {}).get("gamma", 2.0))
        alpha = None
        if name == "class_balanced_focal":
            cc = (loss_cfg or {}).get("class_counts", None)
            if cc is not None:
                alpha = class_balanced_alpha(cc, beta=float((loss_cfg or {}).get("beta", 0.999)))
        else:
            weights = (loss_cfg or {}).get("class_weights", None)
            if weights is not None:
                alpha = torch.tensor(weights, dtype=torch.float32)
        return FocalLoss(gamma=gamma, alpha=alpha, label_smoothing=ls)
    # fallback
    return nn.CrossEntropyLoss(label_smoothing=ls)
