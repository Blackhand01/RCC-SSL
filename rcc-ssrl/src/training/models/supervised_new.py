# models/supervised.py
from __future__ import annotations
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.training.trainer.backbones import ResNetBackbone, linear_head
from src.training.trainer.loops import SLBaseModel
from src.training.utils.losses import make_supervised_criterion

class SupervisedResNet(SLBaseModel):
    def __init__(self, backbone: ResNetBackbone, head: nn.Module, n_classes: int, *,
                 loss_cfg: Dict[str,Any] | None = None,
                 freeze_backbone_bn: bool = False,
                 lr_groups: Dict[str,Any] | None = None):
        super().__init__()
        self.backbone, self.head, self.n = backbone, head, n_classes
        self.criterion = make_supervised_criterion(n_classes, loss_cfg or {})
        self.freeze_backbone_bn = bool(freeze_backbone_bn)
        self._lr_groups = lr_groups or {}

    @classmethod
    def from_config(cls, cfg: Dict[str,Any]) -> "SupervisedResNet":
        bname = cfg["model"].get("backbone","resnet50"); m = cfg["model"].get("sl",{})
        n = int(m.get("n_classes", 5))
        bb = ResNetBackbone(bname, pretrained=bool(m.get("pretrained", True)))
        head = linear_head(bb.out_dim, n)
        return cls(
            bb, head, n,
            loss_cfg = m.get("loss", {}),
            freeze_backbone_bn = bool(m.get("freeze_backbone_bn", False)),
            lr_groups = m.get("lr_groups", {}),
        )

    def training_step(self, batch: Dict[str,Any], global_step: int) -> Dict[str,Any]:
        x, y = batch["images"][0], batch["labels"]
        z = self.head(self.backbone.forward_global(x))
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
        """Gruppi per LR discriminativi: backbone vs head."""
        bb_lr = float(self._lr_groups.get("backbone_lr_mult", 1.0))
        hd_lr = float(self._lr_groups.get("head_lr_mult", 1.0))
        return [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad], "lr_mult": bb_lr},
            {"params": [p for p in self.head.parameters() if p.requires_grad], "lr_mult": hd_lr},
        ]
