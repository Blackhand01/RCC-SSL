#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    import timm
except Exception:
    timm = None

__all__ = ["ResNetBackbone", "ViTBackbone", "mlp_head", "predictor_head", "get_backbone", "resolve_backbone_from_model_cfg"]


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


class ViTBackbone(nn.Module):
    """Light wrapper for ViT-S/16 using timm (features only)."""
    def __init__(
        self,
        name: str = "vit_small_patch16_224",
        pretrained: bool = False,
        *,
        patch_size: Optional[int] = None,
        freeze_patch_embed: bool = False,
        random_patch_proj: bool = False,
        bn_in_patch: bool = False,
        input_size: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm not available: pip install timm")
        # Accept 'input_size' from configs (e.g., i_jepa ablations). Not all timm models
        # support 'img_size', so fall back gracefully.
        try:
            self.vit = timm.create_model(
                name,
                pretrained=pretrained,
                num_classes=0,
                img_size=input_size,
                dynamic_img_size=True,
            )
        except TypeError:
            self.vit = timm.create_model(
                name,
                pretrained=pretrained,
                num_classes=0,
                dynamic_img_size=True,
            )
        self.out_dim = self.vit.num_features
        patch_embed = getattr(self.vit, "patch_embed", None)
        patch_sz = None
        if patch_embed is not None and hasattr(patch_embed, "patch_size"):
            patch_sz = patch_embed.patch_size
            if isinstance(patch_sz, (tuple, list)):
                patch_sz = patch_sz[0]
        if patch_sz is None:
            patch_sz = patch_size or 16
        self.patch_size = int(patch_sz)
        # Keep a record of intended input size (for downstream logic if needed)
        self.default_img_size = input_size or getattr(self.vit, "img_size", None)
        # opzionale: congela l'embed dei patch per stabilizzare i primi step
        if freeze_patch_embed and hasattr(self.vit, "patch_embed"):
            for p in self.vit.patch_embed.parameters():
                p.requires_grad = False
        # (gli altri flag sono placeholder compatibili)
        # Previeni NaN: clamp LayerScale / attn se presenti
        if hasattr(self.vit, "blocks"):
            for blk in self.vit.blocks:
                if hasattr(blk, "ls1") and hasattr(blk.ls1, "gamma"):
                    blk.ls1.gamma.data.clamp_(min=1e-4)
                if hasattr(blk, "ls2") and hasattr(blk.ls2, "gamma"):
                    blk.ls2.gamma.data.clamp_(min=1e-4)

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vit.forward_features(x)  # [B,T,C] for ViT
        return feats.mean(dim=1) if feats.dim() == 3 else torch.flatten(
            torch.nn.functional.adaptive_avg_pool2d(feats, 1), 1
        )

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vit.forward_features(x)
        return feats if feats.dim() == 3 else feats.flatten(2).transpose(1, 2)


def get_backbone(name: str, pretrained: bool, **kwargs) -> nn.Module:
    n = name.lower()
    if n in ("resnet34","resnet50"):
        return ResNetBackbone(n, pretrained)
    # add common aliases to be robust to configs
    if n in ("vit_s16", "vit_small", "vit_small_16", "vit_small_patch16", "vit_small_patch16_224", "vit_base_patch16_224"):
        return ViTBackbone(name, pretrained, **kwargs)
    raise ValueError(f"Unsupported backbone: {name}")

def resolve_backbone_from_model_cfg(model_cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Accept both:
      model.backbone: "vit_small_patch16_224"
    or:
      model.backbone: { name: "vit_small_patch16_224", patch_size: 16, ... }
    Fallback to model.backbone_opts for the string form.
    """
    spec = model_cfg.get("backbone", "resnet50")
    if isinstance(spec, dict):
        name = spec.get("name") or spec.get("type")
        if not isinstance(name, str) or not name:
            raise ValueError("model.backbone must be a string or a dict with a 'name' (or 'type') field.")
        opts = {k: v for k, v in spec.items() if k not in ("name", "type")}
        return name, opts
    # string form + optional legacy options
    return str(spec), (model_cfg.get("backbone_opts", {}) or {})


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
