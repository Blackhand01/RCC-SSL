# models/supervised.py
from __future__ import annotations
import torch.nn as nn
from torchvision import models

def _head(in_f: int, num_classes: int, p: float=0.0) -> nn.Module:
    layers = ([nn.Dropout(p)] if p>0 else []) + [nn.Linear(in_f, num_classes)]
    return nn.Sequential(*layers)

def build_resnet_scratch(name: str, num_classes: int, dropout_p: float=0.0):
    if name=="resnet34_scratch": m = models.resnet34(weights=None)
    elif name=="resnet50_scratch": m = models.resnet50(weights=None)
    else: raise ValueError("name must be resnet34_scratch|resnet50_scratch")
    m.fc = _head(m.fc.in_features, num_classes, dropout_p)
    return m, None  # (model, transforms) per simmetria con transfer
