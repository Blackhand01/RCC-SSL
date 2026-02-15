#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

__all__ = [
    "l2n",
    "cosine_logits",
    "ema_update",
    "copy_weights_and_freeze",
    "move_to",
    "safe_state_dict",
]


def l2n(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalizes on the last dimension, avoiding division by zero."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _safe_tau(tau: float) -> float:
    return max(tau, 1e-8)


def cosine_logits(q: torch.Tensor, k: torch.Tensor, tau: float) -> torch.Tensor:
    """Cosine similarity logits with temperature."""
    return (l2n(q) @ l2n(k).t()) / _safe_tau(tau)


def move_to(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors to device, preserving structure."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = [move_to(v, device) for v in obj]
        return type(obj)(items) if isinstance(obj, tuple) else items
    return obj


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    """Update teacher = m*teacher + (1-m)*student (in-place, no grad)."""
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)


def copy_weights_and_freeze(dst: nn.Module, src: nn.Module) -> None:
    """Copy weights and disable gradients of the destination module."""
    for p_dst, p_src in zip(dst.parameters(), src.parameters()):
        p_dst.data.copy_(p_src.data)
        p_dst.requires_grad = False


def safe_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """State dict ready for saving: tensors detached and on CPU."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}
