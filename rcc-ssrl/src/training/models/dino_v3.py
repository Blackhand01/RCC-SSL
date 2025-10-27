# models/dino_v3.py
from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.trainer.backbones import ResNetBackbone, mlp_head
from src.training.utils.torch_ops import copy_weights_and_freeze, ema_update, l2n
from src.training.trainer.loops import SSLBaseModel

def dino_distill_loss(s: torch.Tensor, t: torch.Tensor, t_temp: float=0.04, s_temp: float=0.1, center: Optional[torch.Tensor]=None) -> torch.Tensor:
    if center is None: center = t.mean(0, keepdim=True)
    pt = ((t - center) / max(t_temp,1e-8)).softmax(dim=-1)
    ls = (s / max(s_temp,1e-8)).log_softmax(dim=-1)
    return -(pt * ls).sum(dim=-1).mean()

def gram_loss(tokens_s: torch.Tensor, tokens_t: torch.Tensor) -> torch.Tensor:
    ts = l2n(tokens_s); tt = l2n(tokens_t)
    Gs = ts @ ts.transpose(1,2); Gt = tt @ tt.transpose(1,2)
    return F.mse_loss(Gs, Gt)

class DINOv3(SSLBaseModel):
    def __init__(self, stu: ResNetBackbone, tea: ResNetBackbone,
                 head_s_g: nn.Module, head_t_g: nn.Module,
                 head_s_l: nn.Module, head_t_l: nn.Module,
                 t_t: float=0.04, t_s: float=0.1, w_gram: float=1.0, ema_m: float=0.996):
        super().__init__()
        self.stu, self.tea = stu, tea
        self.hs_g, self.ht_g = head_s_g, head_t_g
        self.hs_l, self.ht_l = head_s_l, head_t_l
        self.t_t, self.t_s, self.wg, self.m = t_t, t_s, w_gram, ema_m
        self.center = None
        self._bootstrap()

    @classmethod
    def from_config(cls, cfg: Dict[str,Any]) -> "DINOv3":
        bname = cfg["model"].get("backbone","resnet50"); m = cfg["model"]["ssl"]
        stu, tea = ResNetBackbone(bname, False), ResNetBackbone(bname, False)
        dim = stu.out_dim
        head_s_g = mlp_head(dim, 4096, 1024, bn_last_affine=True); head_t_g = mlp_head(dim, 4096, 1024, bn_last_affine=True)
        head_s_l = mlp_head(dim, 2048, 256, bn_last_affine=True); head_t_l = mlp_head(dim, 2048, 256, bn_last_affine=True)
        return cls(stu, tea, head_s_g, head_t_g, head_s_l, head_t_l,
                   t_t=m.get("temp_teacher",0.04), t_s=m.get("temp_student",0.1),
                   w_gram=m.get("gram_lambda",1.0), ema_m=cfg["train"]["ssl"].get("ema_momentum",0.996))

    @torch.no_grad()
    def _bootstrap(self):
        copy_weights_and_freeze(self.tea, self.stu)

    def training_step(self, batch: Dict[str,Any], global_step: int) -> Dict[str,Any]:
    # "images" = [G, L] (multi-crop) oppure [G] se non hai locali
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

        self.center = teacher_global.mean(0, keepdim=True) if self.center is None else 0.9 * self.center + 0.1 * teacher_global.mean(0, keepdim=True)
        loss_global = dino_distill_loss(student_global, teacher_global, self.t_t, self.t_s, self.center)

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
            },
        }
