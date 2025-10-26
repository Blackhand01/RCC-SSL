# models/ibot.py
from __future__ import annotations
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.utils.trainers import (
    SSLBaseModel,
    ResNetBackbone,
    copy_weights_and_freeze,
    ema_update,
    l2n,
    mlp_head,
    predictor_head,
)

@torch.no_grad()
def sinkhorn(logits: torch.Tensor, iters: int=3, eps: float=1e-6) -> torch.Tensor:
    # subtract max per row before exp to avoid infs on CPU runs
    x = logits - logits.max(dim=1, keepdim=True).values
    Q = torch.exp(x).t()  # [K,B]
    s = Q.sum()
    if not torch.isfinite(s):
        Q = torch.nan_to_num(Q, nan=0.0, posinf=1e6, neginf=0.0)
        s = Q.sum().clamp_min(eps)
    Q /= s; K,B = Q.shape
    for _ in range(iters):
        Q /= (Q.sum(dim=1, keepdim=True) + eps)
        Q /= (Q.sum(dim=0, keepdim=True) + eps)
    Q = (Q.t() * K)
    Q = torch.nan_to_num(Q, nan=0.0, posinf=1e6, neginf=0.0)
    return Q.detach()

def make_mask(B:int, T:int, ratio:float, device) -> torch.Tensor:
    k = max(1, int(T*ratio))
    idx = torch.rand(B, T, device=device).argsort(dim=1)[:, :k]
    m = torch.zeros(B, T, dtype=torch.bool, device=device); m.scatter_(1, idx, True); return m

class IBOT(SSLBaseModel):
    def __init__(self, stu: ResNetBackbone, tea: ResNetBackbone,
                 head_cls_s: nn.Module, head_cls_t: nn.Module,
                 head_tok_s: nn.Module, head_tok_t: nn.Module,
                 prototypes: nn.Parameter, t_s: float=0.1, t_t: float=0.04, mask_ratio: float=0.5, ema_m: float=0.996):
        super().__init__()
        self.stu, self.tea = stu, tea
        self.hc_s, self.hc_t = head_cls_s, head_cls_t
        self.hp_s, self.hp_t = head_tok_s, head_tok_t
        self.prototypes = prototypes
        self.t_s, self.t_t, self.r, self.m = t_s, t_t, mask_ratio, ema_m
        self._bootstrap()

    @classmethod
    def from_config(cls, cfg: Dict[str,Any]) -> "IBOT":
        bname = cfg["model"].get("backbone","resnet50"); m = cfg["model"]["ssl"]
        stu, tea = ResNetBackbone(bname, False), ResNetBackbone(bname, False)
        dim = stu.out_dim; head_cls_s = mlp_head(dim, 4096, 256); head_cls_t = mlp_head(dim, 4096, 256)
        head_tok_s = mlp_head(stu.out_dim, 2048, 256); head_tok_t = mlp_head(stu.out_dim, 2048, 256)
        K = m.get("prototypes", 8192); proto = nn.Parameter(l2n(torch.randn(K, 256)), requires_grad=True)
        return cls(stu, tea, head_cls_s, head_cls_t, head_tok_s, head_tok_t, proto,
                   t_s=m.get("temp_student",0.1), t_t=m.get("temp_teacher",0.04),
                   mask_ratio=m.get("mask_ratio",0.5), ema_m=cfg["train"]["ssl"].get("ema_momentum",0.996))

    @torch.no_grad()
    def _bootstrap(self):
        copy_weights_and_freeze(self.tea, self.stu)

    def _loss_cls(self, xg: torch.Tensor) -> torch.Tensor:
        s = self.hc_s(self.stu.forward_global(xg)) / self.t_s
        with torch.no_grad():
            t = self.hc_t(self.tea.forward_global(xg)).detach() / self.t_t
            y = sinkhorn(t)  # soft assignments
        return -(y * F.log_softmax(s, dim=-1)).sum(dim=-1).mean()

    def _loss_tok(self, xl: torch.Tensor) -> torch.Tensor:
        ts = self.stu.forward_tokens(xl)         # [B,T,C]
        with torch.no_grad():
            tt = self.tea.forward_tokens(xl).detach()
        B,T,C = ts.shape; mask = make_mask(B,T,self.r,ts.device)   # [B,T]
        s = self.hp_s(ts[mask])                                    # [B*Tmask, 256]
        t = self.hp_t(tt[mask]).detach()                           # [B*Tmask, 256]
        s = F.normalize(s, dim=-1)
        t = F.normalize(t, dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        s = (s @ proto.t()) / max(self.t_s, 1e-6)
        t = (t @ proto.t()) / max(self.t_t, 1e-6)
        s = torch.clamp(s, -50.0, 50.0)
        t = torch.clamp(t, -50.0, 50.0)
        y = sinkhorn(t)
        return -(y * F.log_softmax(s, dim=-1)).sum(dim=-1).mean()

    def training_step(self, batch: Dict[str,Any], global_step: int) -> Dict[str,Any]:
        images = batch["images"]
        if len(images) < 2:
            raise ValueError("iBOT requires two global views: got len(images) < 2.")
        xg1, xg2 = images[0], images[1]
        xl = xg1  # se non hai locali separati, usa una globale per il token loss

        with torch.no_grad():
            ema_update(self.tea, self.stu, self.m)

        loss_cls = 0.5 * (self._loss_cls(xg1) + self._loss_cls(xg2))
        loss_tok = self._loss_tok(xl)
        loss = loss_cls + loss_tok
        return {"loss_total": loss, "loss_components": {"loss_cls": float(loss_cls.detach()), "loss_tok": float(loss_tok.detach())}}
