#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to build an SSL classifier from:
 - ResNetBackbone or ViTBackbone weights saved inside SSL checkpoints
 - A linear head checkpoint saved by the linear probe trainer.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ---- Gestione dipendenza TIMM per ViT ----
try:
    import timm
    HAVE_TIMM = True
except ImportError:
    HAVE_TIMM = False

# ---- Backbone Definitions ----

class ResNetBackbone(nn.Module):
    def __init__(self, name: str="resnet50", pretrained: bool=False):
        super().__init__()
        from torchvision import models
        if "34" in name: m = models.resnet34(weights=None)
        else:            m = models.resnet50(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.out_dim = m.fc.in_features
        
    def _fwd(self, x): 
        x = self.stem(x); l1 = self.layer1(x); l2 = self.layer2(l1); l3 = self.layer3(l2); l4 = self.layer4(l3); return l4
    
    def forward_global(self, x): 
        feats = self._fwd(x)
        return torch.flatten(F.adaptive_avg_pool2d(feats, 1), 1)

class _VitBackbone(nn.Module):
    def __init__(self, name: str="vit_small_patch16_224"):
        super().__init__()
        if not HAVE_TIMM:
            raise RuntimeError("timm is required for ViT backbones. Install it with `pip install timm`.")
        # dynamic_img_size=True è fondamentale per XAI/Inference su size diverse
        self.model = timm.create_model(name, pretrained=False, num_classes=0, dynamic_img_size=True)
        self.out_dim = self.model.num_features

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)
        # Output timm standard: [B, T, C]. MoCo v3 usa il CLS token (indice 0)
        if feats.dim() == 3:
            return feats[:, 0] 
        # Fallback per architetture che fanno pool interno
        return torch.flatten(torch.nn.functional.adaptive_avg_pool2d(feats, 1), 1)

# ---- Loader Logic ----

_PREFIXES = ("stu.", "backbone_q.", "student.", "backbone.", "module.stu.", "module.backbone_q.", "base_encoder.")

def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

def _best_substate(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # pick the sub-dict with most keys among known prefixes
    cands = [(_strip_prefix(sd, p), p) for p in _PREFIXES]
    # Add the raw dict as a candidate (prefix "")
    cands.append((sd, ""))
    
    best_dict, best_prefix = max(cands, key=lambda x: len(x[0]))
    return best_dict

def _load_torch_state(path: str) -> Dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu") # Fallback older pytorch
        
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    return payload if isinstance(payload, dict) else {}

class SSLLinearClassifier(nn.Module):
    """
    Compose a Backbone (ResNet or ViT) with a linear head.
    """
    def __init__(self, backbone_name: str="resnet50", num_classes: int=5):
        super().__init__()
        if "vit" in backbone_name.lower():
            self.backbone = _VitBackbone(backbone_name)
        else:
            self.backbone = ResNetBackbone(backbone_name, pretrained=False)
            
        self.head = nn.Linear(self.backbone.out_dim, num_classes)

    def load_backbone_from_ssl(self, ssl_backbone_ckpt: str) -> Tuple[int, int]:
        sd = _load_torch_state(ssl_backbone_ckpt)
        sd = _best_substate(sd)
        
        # Gestione specifica per ViT timm vs MoCo naming
        # A volte MoCo salva come 'module.base_encoder.model.blocks...' vs 'blocks...'
        if isinstance(self.backbone, _VitBackbone):
            new_sd = {}
            for k, v in sd.items():
                # Rimuovi 'model.' se presente (comune in wrapper timm salvati male)
                if k.startswith("model."):
                    k = k[6:]
                new_sd[k] = v
            sd = new_sd

        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        return len(missing), len(unexpected)

    def load_head_from_probe(self, ssl_head_ckpt: str) -> Tuple[int, int]:
        hd = torch.load(ssl_head_ckpt, map_location="cpu")
        if isinstance(hd, dict) and "state_dict" in hd:
            hd = hd["state_dict"]
        missing, unexpected = self.head.load_state_dict(hd, strict=False)
        return len(missing), len(unexpected)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_global(x)
        return self.head(feats)
