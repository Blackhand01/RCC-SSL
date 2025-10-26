# models/transfer.py
from __future__ import annotations
import torch.nn as nn
from torchvision import models

def _freeze_except_head(m: nn.Module) -> None:
    for n,p in m.named_parameters():
        if not n.startswith("fc."): p.requires_grad=False

def build_resnet_transfer(name: str, num_classes: int, weights_tag="DEFAULT",
                          freeze_backbone=False, dropout_p: float=0.0, bn_eval_freeze=False):
    if "34" in name:
        weights_enum = models.ResNet34_Weights
        weights = getattr(weights_enum, weights_tag) if weights_tag else None
        m = models.resnet34(weights=weights)
    elif "50" in name:
        weights_enum = models.ResNet50_Weights
        weights = getattr(weights_enum, weights_tag) if weights_tag else None
        m = models.resnet50(weights=weights)
    else:
        raise ValueError("name must contain 34 or 50")
    in_f = m.fc.in_features
    m.fc = nn.Sequential(*( [nn.Dropout(dropout_p)] if dropout_p>0 else [] ), nn.Linear(in_f, num_classes))
    tfm = weights.transforms()
    if freeze_backbone: _freeze_except_head(m)
    if bn_eval_freeze and freeze_backbone:
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d): mod.eval()
    return m, tfm
