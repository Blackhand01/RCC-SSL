# models/moco_v3.py
from __future__ import annotations
from typing import Dict, Any
import torch, torch.nn as nn, torch.nn.functional as F
from src.training.utils.trainers import (
    SSLBaseModel,
    ResNetBackbone,
    copy_weights_and_freeze,
    cosine_logits,
    ema_update,
    mlp_head,
    predictor_head,
)

class MoCoV3(SSLBaseModel):
    def __init__(self, backbone_q: ResNetBackbone, backbone_k: ResNetBackbone,
                 proj_q: nn.Module, proj_k: nn.Module, pred_q: nn.Module,
                 tau: float=0.2, momentum: float=0.996):
        super().__init__()
        self.backbone_q, self.backbone_k = backbone_q, backbone_k
        self.proj_q, self.proj_k, self.pred_q = proj_q, proj_k, pred_q
        self.tau, self.m = tau, momentum
        self._bootstrap()

    @classmethod
    def from_config(cls, cfg: Dict[str,Any]) -> "MoCoV3":
        mcfg = cfg["model"]["ssl"]; bname = cfg["model"].get("backbone","resnet50")
        bb_q = ResNetBackbone(name=bname, pretrained=False); bb_k = ResNetBackbone(name=bname, pretrained=False)
        dim = bb_q.out_dim; proj_q = mlp_head(dim, 4096, 256); proj_k = mlp_head(dim, 4096, 256)
        pred_q = predictor_head(256, 4096)
        return cls(bb_q, bb_k, proj_q, proj_k, pred_q, tau=mcfg.get("temperature",0.2), momentum=cfg["train"]["ssl"].get("ema_momentum",0.996))

    @torch.no_grad()
    def _bootstrap(self) -> None:
        copy_weights_and_freeze(self.backbone_k, self.backbone_q)
        copy_weights_and_freeze(self.proj_k, self.proj_q)

    def _info_nce_sym(self, q1, q2, k1, k2) -> torch.Tensor:
        lab = torch.arange(q1.size(0), device=q1.device)
        l12 = F.cross_entropy(cosine_logits(q1, k2, self.tau), lab)
        l21 = F.cross_entropy(cosine_logits(q2, k1, self.tau), lab)
        return (l12 + l21)

    def training_step(self, batch: Dict[str,Any], global_step: int) -> Dict[str,Any]:
        x1, x2 = batch["images"]
        q1 = self.pred_q(self.proj_q(self.backbone_q.forward_global(x1)))
        q2 = self.pred_q(self.proj_q(self.backbone_q.forward_global(x2)))
        with torch.no_grad():
            ema_update(self.backbone_k, self.backbone_q, self.m)
            k1 = self.proj_k(self.backbone_k.forward_global(x1))
            k2 = self.proj_k(self.backbone_k.forward_global(x2))
        loss = self._info_nce_sym(q1, q2, k1.detach(), k2.detach())
        return {"loss_total": loss, "loss_components": {"loss_info_nce": float(loss.detach())}}

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone_q.forward_global(x)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone_q.forward_tokens(x)
