#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

__all__ = ["ResNetBackbone", "mlp_head", "predictor_head"]


def _get_resnet_factory(name: str):
    factories = {
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }
    if name not in factories:
        raise ValueError(f"Unsupported ResNet backbone '{name}'.")
    return factories[name]


def _resolve_torchvision_weights(name: str, pretrained: bool):
    if not pretrained:
        return None
    enum_name = "ResNet34_Weights" if "34" in name else "ResNet50_Weights"
    weights_enum = getattr(models, enum_name, None)
    if weights_enum is None:
        raise RuntimeError(f"Pretrained weights for '{name}' not available in this torchvision version.")
    return weights_enum.DEFAULT


class ResNetBackbone(nn.Module):
    """
    ResNet come estrattore di feature con:
      - forward_global: pooled feature [B, D]
      - forward_tokens: token spatiali [B, T, C] da un blocco selezionato
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = False, return_tokens_from: str = "layer4"):
        super().__init__()
        factory = _get_resnet_factory(name)
        weights = _resolve_torchvision_weights(name, pretrained)
        model = factory(weights=weights)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.out_dim = model.fc.in_features
        self.tokens_source = return_tokens_from

    def _forward_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return {"layer1": l1, "layer2": l2, "layer3": l3, "layer4": l4}

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._forward_stages(x)["layer4"]
        return torch.flatten(F.adaptive_avg_pool2d(feats, 1), 1)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._forward_stages(x)[self.tokens_source]
        b, c, h, w = feats.shape
        return feats.flatten(2).transpose(1, 2).contiguous().view(b, h * w, c)


def _bn1d(dim: int, affine: bool = True) -> nn.BatchNorm1d:
    return nn.BatchNorm1d(dim, affine=affine)


def _relu() -> nn.ReLU:
    return nn.ReLU(inplace=True)


def _linear(in_f: int, out_f: int, bias: bool = False) -> nn.Linear:
    return nn.Linear(in_f, out_f, bias=bias)


def mlp_head(in_dim: int, hidden: int, out_dim: int, bn_last_affine: bool = False) -> nn.Sequential:
    """MLP 3-layer con BN e ReLU; ultima BN opzionale affine."""
    return nn.Sequential(
        _linear(in_dim, hidden, bias=False),
        _bn1d(hidden),
        _relu(),
        _linear(hidden, hidden, bias=False),
        _bn1d(hidden),
        _relu(),
        _linear(hidden, out_dim, bias=False),
        _bn1d(out_dim, affine=bn_last_affine),
    )


def predictor_head(dim: int, hidden: int = 4096) -> nn.Sequential:
    """MLP predittore stile BYOL/MoCoV3."""
    return nn.Sequential(
        _linear(dim, hidden, bias=False),
        _bn1d(hidden),
        _relu(),
        _linear(hidden, dim),
    )
