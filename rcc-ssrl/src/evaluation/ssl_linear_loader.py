# file: /home/mla_group_01/rcc-ssrl/src/evaluation/ssl_linear_loader.py
#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict, Tuple
import torch, torch.nn as nn
import os

try:
    import timm
    HAVE_TIMM = True
except Exception:
    HAVE_TIMM = False

# ---- ResNet backbone (fallback / compat) ----
try:
    from src.training.trainer.backbones import ResNetBackbone as _ResNetBackbone
except Exception:
    from torchvision import models
    import torch.nn.functional as F
    class _ResNetBackbone(nn.Module):
        def __init__(self, name: str="resnet50"):
            super().__init__()
            m = models.resnet50(weights=None) if "50" in name else models.resnet34(weights=None)
            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
            self.out_dim = m.fc.in_features
        def _fwd(self, x):
            x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x); return x
        def forward_global(self, x):
            x = self._fwd(x)
            return torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)

# ---- ViT backbone (timm) ----
class _VitBackbone(nn.Module):
    def __init__(self, name: str="vit_small_patch16_224"):
        super().__init__()
        if not HAVE_TIMM:
            raise RuntimeError("timm is required for ViT backbones.")
        self.model = timm.create_model(name, pretrained=False, num_classes=0, dynamic_img_size=True)
        self.out_dim = self.model.num_features

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)
        if feats.dim() == 3:  # [B, T, C] tipico ViT
            return feats.mean(dim=1)
        # per sicurezza: se è [B, C, H, W], pool 2D
        return torch.flatten(torch.nn.functional.adaptive_avg_pool2d(feats, 1), 1)


_PREFIXES = ("stu.", "student.", "backbone_q.", "backbone.", "module.stu.", "module.backbone_q.")
def _safe_torch_load(path: str):
    import torch
    try:
        # torch>=2.4
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def _load_state(path: str) -> Dict[str, torch.Tensor]:
    sd = _safe_torch_load(path)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd if isinstance(sd, dict) else {}

def _best_substate(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cands = [{k[len(p):]: v for k, v in sd.items() if k.startswith(p)} for p in _PREFIXES]
    best = max(cands, key=lambda x: len(x))
    return best if len(best) else sd

def _looks_like_vit(sd: Dict[str, torch.Tensor]) -> bool:
    ks = list(sd.keys())
    return any(k.startswith(("pos_embed","cls_token","blocks.","patch_embed.","norm.","fc_norm")) for k in ks)

def _strip_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    plen = len(prefix)
    return {k[plen:]: v for k, v in d.items() if k.startswith(prefix)}

def _maybe_strip(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return _strip_prefix(d, prefix) if any(k.startswith(prefix) for k in d.keys()) else d

class SSLLinearClassifier(nn.Module):
    """Compose an SSL backbone (ResNet or ViT) with a linear head."""
    def __init__(self, backbone_name: str="resnet50", num_classes: int=5):
        super().__init__()
        if backbone_name.startswith("vit"):
            self.backbone = _VitBackbone(backbone_name)
        else:
            self.backbone = _ResNetBackbone(backbone_name)
        self.head = nn.Linear(self.backbone.out_dim, num_classes)
    @staticmethod
    def _try_load(module: nn.Module, state: Dict[str, torch.Tensor]) -> Tuple[int, int, Tuple[list, list]]:
        missing, unexpected = module.load_state_dict(state, strict=False)
        miss_l, unexp_l = list(missing), list(unexpected)
        if os.environ.get("EVAL_DEBUG", "0") == "1":
            print(f"[DEBUG] backbone try_load: missing={len(miss_l)} unexpected={len(unexp_l)}")
            if len(miss_l) <= 15:  print("[DEBUG]  missing keys:", miss_l)
            if len(unexp_l) <= 15: print("[DEBUG]  unexpected keys:", unexp_l)
        return len(miss_l), len(unexp_l), (miss_l, unexp_l)
    @staticmethod
    def _swap_key_prefix(d: Dict[str, torch.Tensor], old: str, new: str) -> Dict[str, torch.Tensor]:
        # Esempio: fc_norm.xxx -> norm.xxx (alcuni ViT timm differiscono qui)
        out = {}
        for k, v in d.items():
            if k.startswith(old):
                out[new + k[len(old):]] = v
            else:
                out[k] = v
        return out

    def load_backbone_from_ssl(self, ssl_backbone_ckpt: str, allow_autoswap: bool = True) -> Tuple[int, int]:
        raw = _load_state(ssl_backbone_ckpt)
        sub = _best_substate(raw)

        # Rileva tipo checkpoint
        want_vit = _looks_like_vit(sub)
        have_vit = isinstance(self.backbone, _VitBackbone)

        # Autoswap arch se necessario
        if allow_autoswap and want_vit and not have_vit:
            self.backbone = _VitBackbone("vit_small_patch16_224")
            self.head = nn.Linear(self.backbone.out_dim, self.head.out_features)
            have_vit = True
        elif allow_autoswap and (not want_vit) and have_vit:
            self.backbone = _ResNetBackbone("resnet50")
            self.head = nn.Linear(self.backbone.out_dim, self.head.out_features)
            have_vit = False

        # Rimuovi 'vit.' se presente
        sub = _maybe_strip(sub, "vit.")

        target_module = self.backbone.model if have_vit else self.backbone

        # --- Primo tentativo (quello che già fai) ---
        m, u, _ = self._try_load(target_module, sub)
        best = (m + u, m, u, sub)

        # --- Se non è perfetto, prova rimappi equivalenti comuni ---
        if m + u > 0 and have_vit:
            candidates = []

            # 1) strip 'model.' (alcuni export hanno 'model.' già incluso)
            if any(k.startswith("model.") for k in sub.keys()):
                candidates.append({k[6:]: v for k, v in sub.items() if k.startswith("model.")})

            # 2) aggiungi 'model.' su tutte le chiavi (in caso inverso)
            candidates.append({f"model.{k}": v for k, v in sub.items()})

            # 3) fc_norm <-> norm (varianti timm)
            candidates.append(_swap_key_prefix(sub, "fc_norm.", "norm."))
            candidates.append(_swap_key_prefix(sub, "norm.", "fc_norm."))

            for cand in candidates:
                m2, u2, _ = self._try_load(target_module, cand)
                score = m2 + u2
                if score < best[0]:
                    best = (score, m2, u2, cand)
                    if score == 0:
                        break  # perfetto, basta così

            # Ricarica il migliore
            if best[3] is not sub:
                target_module.load_state_dict(best[3], strict=False)

        # Ritorna i missing/unexpected del best
        return best[1], best[2]


    def load_head_from_probe(self, ssl_head_ckpt: str) -> Tuple[int, int]:
        hd = torch.load(ssl_head_ckpt, map_location="cpu", weights_only=True)
        if isinstance(hd, dict) and "state_dict" in hd:
            hd = hd["state_dict"]
        missing, unexpected = self.head.load_state_dict(hd, strict=False)
        if (missing or unexpected) and os.environ.get("EVAL_DEBUG", "0") == "1":
            print(f"[DEBUG] head load: missing={missing} unexpected={unexpected}")
        return len(missing), len(unexpected)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_global(x)
        return self.head(feats)
