# models/dino_v3.py
from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.trainer.backbones import get_backbone, mlp_head, resolve_backbone_from_model_cfg
from src.training.utils.torch_ops import copy_weights_and_freeze, ema_update, l2n
from src.training.trainer.loops import SSLBaseModel

def dino_distill_loss(
    s: torch.Tensor,
    t: torch.Tensor,
    t_temp: float = 0.04,
    s_temp: float = 0.1,
    center: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if center is None:
        center = t.mean(0, keepdim=True)
    t_norm = (t - center) / max(t_temp, 1e-8)
    s_norm = s / max(s_temp, 1e-8)
    pt = t_norm.softmax(dim=-1)
    ls = s_norm.log_softmax(dim=-1)
    return -(pt * ls).sum(dim=-1).mean()

def gram_loss(tokens_s: torch.Tensor, tokens_t: torch.Tensor) -> torch.Tensor:
    ts = l2n(tokens_s); tt = l2n(tokens_t)
    Gs = ts @ ts.transpose(1,2); Gt = tt @ tt.transpose(1,2)
    return F.mse_loss(Gs, Gt)

class DINOv3(SSLBaseModel):
    def __init__(self, stu: torch.nn.Module, tea: torch.nn.Module,
                 head_s_g: nn.Module, head_t_g: nn.Module,
                 head_s_l: nn.Module, head_t_l: nn.Module,
                 t_t: float=0.04, t_s: float=0.1, w_gram: float=1.0, ema_m: float=0.996,
                 center_m: float = 0.9,
                 t_warmup_min: Optional[float] = None,
                 t_warmup_max: Optional[float] = None,
                 t_warmup_steps: int = 0):
        super().__init__()
        self.stu, self.tea = stu, tea
        self.hs_g, self.ht_g = head_s_g, head_t_g
        self.hs_l, self.ht_l = head_s_l, head_t_l
        self.t_t, self.t_s, self.wg, self.m = t_t, t_s, w_gram, ema_m
        self.center_m = float(center_m)
        self.t_min = float(t_warmup_min) if t_warmup_min is not None else float(t_t)
        self.t_max = float(t_warmup_max) if t_warmup_max is not None else float(t_t)
        self.t_warmup_steps = int(max(0, t_warmup_steps))
        self.center = None
        self._bootstrap()

    @classmethod
    def from_config(cls, cfg: Dict[str,Any]) -> "DINOv3":
        m = cfg["model"]["ssl"]
        bname, bopts = resolve_backbone_from_model_cfg(cfg["model"])
        stu = get_backbone(name=bname, pretrained=False, **bopts)
        tea = get_backbone(name=bname, pretrained=False, **bopts)
        dim = stu.out_dim
        head_s_g = mlp_head(dim, 4096, 1024, bn_last_affine=True); head_t_g = mlp_head(dim, 4096, 1024, bn_last_affine=True)
        head_s_l = mlp_head(dim, 2048, 256, bn_last_affine=True); head_t_l = mlp_head(dim, 2048, 256, bn_last_affine=True)
        # Accept both 'teacher_temp_schedule' and 'temp_teacher_schedule'
        # and both naming schemes: {min,max,warmup_steps} or {start,end,warmup_frac}.
        sched = (m.get("temp_teacher_schedule") or m.get("teacher_temp_schedule") or {})
        # Derive warmup steps if only a fraction is provided.
        tr_ssl = (cfg.get("train", {}).get("ssl", {}) or {})
        total_steps = int(tr_ssl.get("epochs", 1)) * int(tr_ssl.get("steps_per_epoch", 1))
        def _get(name, alt, default):
            v = sched.get(name, None)
            if v is None:
                v = sched.get(alt, default)
            return v
        t_min = _get("min",   "start", m.get("temp_teacher", 0.04))
        t_max = _get("max",   "end",   m.get("temp_teacher", 0.04))
        wu_fr = sched.get("warmup_frac", None)
        wu_st = sched.get("warmup_steps", None)
        if wu_st is None and wu_fr is not None:
            try:
                wu_st = int(float(wu_fr) * max(1, total_steps))
            except Exception:
                wu_st = 0
        if wu_st is None:
            wu_st = 0
        return cls(
            stu, tea, head_s_g, head_t_g, head_s_l, head_t_l,
            t_t=m.get("temp_teacher", 0.04),
            t_s=m.get("temp_student", 0.1),
            w_gram=m.get("gram_lambda", 1.0),
            ema_m=cfg["train"]["ssl"].get("ema_momentum", 0.996),
            center_m=m.get("center_momentum", 0.9),
            t_warmup_min=float(t_min),
            t_warmup_max=float(t_max),
            t_warmup_steps=int(wu_st),
        )

    @torch.no_grad()
    def _bootstrap(self):
        copy_weights_and_freeze(self.tea, self.stu)

    def _teacher_temp(self, global_step: int) -> float:
        if self.t_warmup_steps <= 0 or abs(self.t_max - self.t_min) < 1e-8:
            return self.t_max
        alpha = min(1.0, max(0.0, float(global_step) / float(self.t_warmup_steps)))
        return self.t_min + (self.t_max - self.t_min) * alpha

    def training_step(self, batch: Dict[str,Any], global_step: int) -> Dict[str,Any]:
        # images = [G, L] (multi-crop) oppure [G] se non hai locali
        images = batch["images"]
        if len(images) == 2:
            G, L = images[0], images[1]      # G: concat global, L: concat local
        else:
            G, L = images[0], images[0]      # fallback senza locali

        student_global = self.hs_g(self.stu.forward_global(G))
        student_tokens_raw = self.stu.forward_tokens(L)

        with torch.no_grad():
            ema_update(self.tea, self.stu, self.m)
            teacher_global = self.ht_g(self.tea.forward_global(G)).detach()
            teacher_tokens_raw = self.tea.forward_tokens(L)

        mean_t = teacher_global.mean(0, keepdim=True)
        self.center = mean_t if self.center is None else self.center_m * self.center + (1.0 - self.center_m) * mean_t
        t_temp = self._teacher_temp(global_step)
        loss_global = dino_distill_loss(student_global, teacher_global, t_temp, self.t_s, self.center)

        b_tokens, t_tokens, c_tokens = student_tokens_raw.shape
        student_tokens = self.hs_l(student_tokens_raw.reshape(b_tokens * t_tokens, c_tokens)).view(b_tokens, t_tokens, -1)
        teacher_tokens = self.ht_l(teacher_tokens_raw.reshape(b_tokens * t_tokens, c_tokens)).view(b_tokens, t_tokens, -1).detach()
        loss_gram = gram_loss(student_tokens, teacher_tokens)

        loss = loss_global + self.wg * loss_gram
        return {
            "loss_total": loss,
            "loss_components": {
                "loss_global": float(loss_global.detach()),
                "loss_gram": float(loss_gram.detach()),
                "teacher_temp": float(t_temp),
            },
        }
