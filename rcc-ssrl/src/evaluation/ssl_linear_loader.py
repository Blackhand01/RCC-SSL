# src/evaluation/ssl_linear_loader.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single source of truth SSL linear classifier loader for BOTH:
  - evaluation stage
  - explainability/XAI stage

Supports:
  - ResNet backbones (training backbone if available; torchvision fallback)
  - ViT backbones (timm)
  - multiple checkpoint wrappers/prefixes (stu/student/backbone_q/base_encoder/module.*)
  - common ViT key variants (vit./model. prefixes, norm <-> fc_norm)
  - optional auto-arch swap (checkpoint looks like ViT but user instantiated ResNet, and vice-versa)

Important:
  - ViT pooling defaults to MEAN over tokens to stay consistent with training (unless overridden via env VIT_POOL=cls).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, Literal, List

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- TIMM for ViT ----
try:
    import timm  # type: ignore
    HAVE_TIMM = True
except Exception:
    HAVE_TIMM = False

# ---- ResNet backbone (prefer training backbone, fallback to torchvision) ----
try:
    from src.training.trainer.backbones import ResNetBackbone as _TrainResNetBackbone  # type: ignore
except Exception:
    _TrainResNetBackbone = None  # type: ignore


class ResNetBackbone(nn.Module):
    def __init__(self, name: str = "resnet50", pretrained: bool = False):
        super().__init__()
        if _TrainResNetBackbone is not None:
            self._impl = _TrainResNetBackbone(name=name, pretrained=pretrained)  # type: ignore
            self.out_dim = int(getattr(self._impl, "out_dim"))
        else:
            from torchvision import models  # type: ignore

            if "34" in name:
                m = models.resnet34(weights=None)
            else:
                m = models.resnet50(weights=None)

            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
            self.out_dim = int(m.fc.in_features)

    def _fwd(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "_impl"):
            raise RuntimeError("Internal error: ResNetBackbone._fwd called while using training backbone.")
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "_impl"):
            return self._impl.forward_global(x)  # type: ignore[attr-defined]
        feats = self._fwd(x)
        return torch.flatten(F.adaptive_avg_pool2d(feats, 1), 1)


VitPool = Literal["mean", "cls"]


class _VitBackbone(nn.Module):
    def __init__(self, name: str = "vit_small_patch16_224", pool: VitPool = "mean"):
        super().__init__()
        if not HAVE_TIMM:
            raise RuntimeError("timm is required for ViT backbones. Install it with `pip install timm`.")
        # dynamic_img_size=True: useful for inference/XAI with variable sizes
        self.model = timm.create_model(name, pretrained=False, num_classes=0, dynamic_img_size=True)  # type: ignore
        self.out_dim = int(self.model.num_features)  # type: ignore[attr-defined]
        self.pool: VitPool = pool

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)  # type: ignore[attr-defined]

        # timm may return tuple/list/dict depending on model/version
        if isinstance(feats, (tuple, list)) and len(feats) > 0:
            feats = feats[0]
        if isinstance(feats, dict):
            feats = feats.get("x", feats.get("pre_logits", next(iter(feats.values()))))

        if not isinstance(feats, torch.Tensor):
            raise RuntimeError(f"Unexpected forward_features output type: {type(feats)}")

        # token output: [B, T, C]
        if feats.dim() == 3:
            return feats[:, 0] if self.pool == "cls" else feats.mean(dim=1)

        # fallback: [B, C, H, W]
        return torch.flatten(F.adaptive_avg_pool2d(feats, 1), 1)


_PREFIXES = (
    "stu.",
    "student.",
    "backbone_q.",
    "backbone.",
    "base_encoder.",
    "encoder.",
    "module.stu.",
    "module.student.",
    "module.backbone_q.",
    "module.backbone.",
    "module.base_encoder.",
    "module.encoder.",
)


def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in sd.items() if isinstance(k, str) and k.startswith(prefix)}


def _best_substate(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cands = [_strip_prefix(sd, p) for p in _PREFIXES]
    cands.append(sd)
    return max(cands, key=lambda d: len(d))


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_torch_state(path: str) -> Dict[str, torch.Tensor]:
    payload = _safe_torch_load(path)
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    return payload if isinstance(payload, dict) else {}


def _maybe_strip(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return _strip_prefix(sd, prefix) if any(isinstance(k, str) and k.startswith(prefix) for k in sd.keys()) else sd


def _swap_key_prefix(d: Dict[str, torch.Tensor], old: str, new: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in d.items():
        if isinstance(k, str) and k.startswith(old):
            out[new + k[len(old):]] = v
        else:
            out[k] = v
    return out


def _dedup_states(items: List[Tuple[str, Dict[str, torch.Tensor]]]) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    seen = set()
    out: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    for label, sd in items:
        keys = frozenset(k for k in sd.keys() if isinstance(k, str))
        if not keys:
            continue
        if keys in seen:
            continue
        seen.add(keys)
        out.append((label, sd))
    return out


def _candidate_substates(sd: Dict[str, torch.Tensor]) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    cands: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    for pfx in _PREFIXES:
        sub = _strip_prefix(sd, pfx)
        if sub:
            cands.append((f"strip:{pfx}", sub))
    if any(isinstance(k, str) and k.startswith("module.") for k in sd.keys()):
        cands.append(("strip:module.", _strip_prefix(sd, "module.")))
    cands.append(("raw", sd))
    return _dedup_states(cands)


def _candidate_variants(sd: Dict[str, torch.Tensor], have_vit: bool) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    variants: List[Tuple[str, Dict[str, torch.Tensor]]] = [("base", sd)]
    prefixes = ("model.", "vit.", "backbone.", "encoder.")
    for pfx in prefixes:
        for label, cur in list(variants):
            if any(isinstance(k, str) and k.startswith(pfx) for k in cur.keys()):
                variants.append((f"{label}|strip:{pfx}", _strip_prefix(cur, pfx)))
    if have_vit:
        for label, cur in list(variants):
            variants.append((f"{label}|swap:fc_norm->norm", _swap_key_prefix(cur, "fc_norm.", "norm.")))
            variants.append((f"{label}|swap:norm->fc_norm", _swap_key_prefix(cur, "norm.", "fc_norm.")))
    return _dedup_states(variants)


def _align_state_dict(
    sd: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Tuple[Optional[Dict[str, torch.Tensor]], str]:
    clean = {k: v for k, v in sd.items() if isinstance(k, str) and torch.is_tensor(v)}
    if not clean:
        return None, "empty"

    target_keys = list(target_state.keys())
    if all(k in clean for k in target_keys):
        aligned = {k: clean[k] for k in target_keys}
        note = "direct" if len(clean) == len(target_keys) else "direct+drop_extra"
        return aligned, note

    key_map: Dict[str, str] = {}
    for tkey in target_keys:
        if tkey in clean:
            key_map[tkey] = tkey
            continue
        suffix = "." + tkey
        matches = [k for k in clean.keys() if k.endswith(suffix)]
        if len(matches) == 1:
            key_map[tkey] = matches[0]
        else:
            return None, f"suffix_fail:{tkey}:{len(matches)}"
    aligned = {tkey: clean[src_key] for tkey, src_key in key_map.items()}
    return aligned, "suffix_map"


def _shape_mismatch(
    aligned: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Optional[str]:
    for k, v in aligned.items():
        tgt = target_state[k]
        if hasattr(tgt, "shape") and hasattr(v, "shape"):
            if tuple(tgt.shape) != tuple(v.shape):
                return f"{k} expected={tuple(tgt.shape)} got={tuple(v.shape)}"
    return None


def _score_key_overlap(
    sd: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
) -> Tuple[int, int, int]:
    target_keys = set(target_state.keys())
    cand_keys = {k for k in sd.keys() if isinstance(k, str)}
    overlap = len(target_keys & cand_keys)
    missing = len(target_keys) - overlap
    unexpected = len(cand_keys) - overlap
    return missing, unexpected, overlap


def _looks_like_vit(sd: Dict[str, torch.Tensor]) -> bool:
    """
    Heuristic robust to wrappers/prefixes: module./model./vit./stu./student./backbone*/encoder...
    """
    for k in sd.keys():
        if not isinstance(k, str):
            continue
        ks = k
        for _ in range(3):
            for pref in (
                "module.",
                "model.",
                "vit.",
                "stu.",
                "student.",
                "backbone.",
                "backbone_q.",
                "base_encoder.",
                "encoder.",
            ):
                if ks.startswith(pref):
                    ks = ks[len(pref):]
        if ks.startswith(("pos_embed", "cls_token", "blocks.", "patch_embed.", "norm.", "fc_norm")):
            return True
        if ".blocks." in ks or ".patch_embed." in ks or ".pos_embed" in ks or ".cls_token" in ks:
            return True
    return False


def _linear_weight_shape_from_state(sd: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    for k, v in sd.items():
        if isinstance(k, str) and k.endswith("weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            return int(v.shape[0]), int(v.shape[1])
    return None


def _try_load(module: nn.Module, state: Dict[str, torch.Tensor]) -> Tuple[int, int, Tuple[list, list]]:
    missing, unexpected = module.load_state_dict(state, strict=False)
    miss_l, unexp_l = list(missing), list(unexpected)
    if os.environ.get("SSL_LOADER_DEBUG", "0") == "1":
        print(f"[DEBUG] ssl_loader try_load: missing={len(miss_l)} unexpected={len(unexp_l)}")
        if len(miss_l) <= 25:
            print("[DEBUG]  missing keys:", miss_l)
        if len(unexp_l) <= 25:
            print("[DEBUG]  unexpected keys:", unexp_l)
    return len(miss_l), len(unexp_l), (miss_l, unexp_l)


class SSLLinearClassifier(nn.Module):
    """Compose a Backbone (ResNet or ViT) with a linear head."""
    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 5,
        vit_pool: Optional[VitPool] = None,
    ):
        super().__init__()

        if vit_pool is None:
            vit_pool = os.environ.get("VIT_POOL", "mean").strip().lower()  # type: ignore[assignment]
        if vit_pool not in ("mean", "cls"):
            vit_pool = "mean"
        self.vit_pool: VitPool = vit_pool  # type: ignore[assignment]

        if backbone_name.lower().startswith("vit"):
            self.backbone: nn.Module = _VitBackbone(backbone_name, pool=self.vit_pool)
        else:
            self.backbone = ResNetBackbone(backbone_name, pretrained=False)

        out_dim = int(getattr(self.backbone, "out_dim"))
        self.head = nn.Linear(out_dim, num_classes)

    def _rebuild_head(self) -> None:
        out_dim = int(getattr(self.backbone, "out_dim"))
        self.head = nn.Linear(out_dim, int(self.head.out_features))

    def load_backbone_from_ssl(self, ssl_backbone_ckpt: str, allow_autoswap: bool = True) -> Tuple[int, int]:
        sd = _load_torch_state(ssl_backbone_ckpt)
        if not sd:
            raise RuntimeError(f"Empty backbone state_dict: {ssl_backbone_ckpt}")

        want_vit = _looks_like_vit(sd)
        have_vit = isinstance(self.backbone, _VitBackbone)

        if allow_autoswap and want_vit and not have_vit:
            self.backbone = _VitBackbone("vit_small_patch16_224", pool=self.vit_pool)
            self._rebuild_head()
            have_vit = True
        elif allow_autoswap and (not want_vit) and have_vit:
            self.backbone = ResNetBackbone("resnet50", pretrained=False)
            self._rebuild_head()
            have_vit = False

        target_module: nn.Module = self.backbone.model if have_vit else self.backbone  # type: ignore[attr-defined]
        target_state = target_module.state_dict()

        best_label: Optional[str] = None
        best_missing = 10**9
        best_unexpected = 10**9
        best_overlap = -1

        for base_label, base_sd in _candidate_substates(sd):
            for var_label, var_sd in _candidate_variants(base_sd, have_vit):
                aligned, note = _align_state_dict(var_sd, target_state)
                if aligned is None:
                    m, u, ov = _score_key_overlap(var_sd, target_state)
                    if ov > best_overlap:
                        best_overlap = ov
                        best_missing, best_unexpected = m, u
                        best_label = f"{base_label}|{var_label}"
                    continue
                shape_err = _shape_mismatch(aligned, target_state)
                if shape_err:
                    if os.environ.get("SSL_LOADER_DEBUG", "0") == "1":
                        print(f"[DEBUG] ssl_loader shape mismatch {shape_err} [{base_label}|{var_label}]")
                    continue

                target_module.load_state_dict(aligned, strict=True)
                if os.environ.get("SSL_LOADER_DEBUG", "0") == "1":
                    print(f"[DEBUG] ssl_loader backbone loaded via {base_label}|{var_label} ({note})")
                return 0, 0

        msg = (
            "Backbone state_dict mismatch after cleaning; refusing to proceed. "
            "Tried known prefixes and key mappings, but no strict match was found."
        )
        if best_label is not None:
            msg += (
                f" Best candidate={best_label} missing={best_missing} unexpected={best_unexpected} "
                f"(overlap={best_overlap}/{len(target_state)})."
            )
        raise RuntimeError(msg)

    def load_head_from_probe(self, ssl_head_ckpt: str) -> Tuple[int, int]:
        hd = _safe_torch_load(ssl_head_ckpt)
        if isinstance(hd, dict) and "state_dict" in hd and isinstance(hd["state_dict"], dict):
            hd = hd["state_dict"]
        if not isinstance(hd, dict):
            return (0, 0)

        # common wrappers
        if any(isinstance(k, str) and k.startswith("module.") for k in hd.keys()):
            hd = {k[7:]: v for k, v in hd.items() if isinstance(k, str) and k.startswith("module.")}

        for pfx in ("head.", "linear.", "fc."):
            if any(isinstance(k, str) and k.startswith(pfx) for k in hd.keys()):
                hd = {k[len(pfx):]: v for k, v in hd.items() if isinstance(k, str) and k.startswith(pfx)}
                break

        # guard against backbone/probe mismatch: strict=False does NOT ignore tensor size mismatch
        shp = _linear_weight_shape_from_state(hd)
        if shp is not None:
            ckpt_out, ckpt_in = shp
            cur_in = int(self.head.in_features)
            cur_out = int(self.head.out_features)
            if ckpt_in != cur_in or ckpt_out != cur_out:
                raise RuntimeError(
                    "Incompatible linear head checkpoint for current model. "
                    f"ckpt weight shape=[{ckpt_out}, {ckpt_in}] vs current head=[{cur_out}, {cur_in}]. "
                    "This usually means backbone feature dim mismatch (e.g., ViT-S=384 vs ResNet50=2048) "
                    "or wrong ckpt pairing."
                )

        missing, unexpected = self.head.load_state_dict(hd, strict=False)
        if (missing or unexpected) and os.environ.get("SSL_LOADER_DEBUG", "0") == "1":
            print(f"[DEBUG] ssl_loader head load: missing={list(missing)} unexpected={list(unexpected)}")
        return len(missing), len(unexpected)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_global(x)  # type: ignore[attr-defined]
        return self.head(feats)
