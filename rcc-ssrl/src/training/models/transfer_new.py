# models/transfer.py
from __future__ import annotations
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.training.trainer.backbones import ResNetBackbone, linear_head, load_backbone_from_checkpoint
from src.training.trainer.loops import SLBaseModel
from src.training.utils.losses import make_supervised_criterion

class TransferResNet(SLBaseModel):
    def __init__(self, backbone: ResNetBackbone, head: nn.Module, n_classes: int, *,
                 freeze_backbone: bool=False,
                 freeze_backbone_bn: bool=False,
                 loss_cfg: Dict[str,Any] | None = None,
                 lr_groups: Dict[str,Any] | None = None):
        super().__init__()
        self.backbone, self.head, self.n = backbone, head, n_classes
        self.freeze_backbone = bool(freeze_backbone)
        self.freeze_backbone_bn = bool(freeze_backbone_bn)
        self.criterion = make_supervised_criterion(n_classes, loss_cfg or {})
        self._lr_groups = lr_groups or {}

    @classmethod
    def from_config(cls, cfg: Dict[str,Any]) -> "TransferResNet":
        m = cfg["model"].get("sl",{})
        n = int(m.get("n_classes", 5))
        src = m.get("checkpoint", None)
        if src is None:
            raise ValueError("TransferResNet richiede 'model.sl.checkpoint'")
        bb = load_backbone_from_checkpoint(src)
        head = linear_head(bb.out_dim, n)
        return cls(
            bb, head, n,
            freeze_backbone=bool(m.get("freeze_backbone", False)),
            freeze_backbone_bn=bool(m.get("freeze_backbone_bn", True)),
            loss_cfg=m.get("loss", {}),
            lr_groups=m.get("lr_groups", {}),
        )

    def training_step(self, batch: Dict[str,Any], global_step: int) -> Dict[str,Any]:
        x, y = batch["images"][0], batch["labels"]
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone.forward_global(x)
        else:
            feat = self.backbone.forward_global(x)
        z = self.head(feat)
        loss = self.criterion(z, y)
        return {"loss_total": loss}

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters(): p.requires_grad = False
        return self

    def param_groups(self) -> List[Dict[str,Any]]:
        bb_lr = float(self._lr_groups.get("backbone_lr_mult", 0.1))
        hd_lr = float(self._lr_groups.get("head_lr_mult", 1.0))
        return [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad], "lr_mult": bb_lr},
            {"params": [p for p in self.head.parameters() if p.requires_grad], "lr_mult": hd_lr},
        ]
