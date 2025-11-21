# models/i_jepa.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.training.trainer.backbones import get_backbone, resolve_backbone_from_model_cfg
from src.training.utils.torch_ops import copy_weights_and_freeze, ema_update
from src.training.trainer.loops import SSLBaseModel

@torch.no_grad()
def _cosine_to_one(m0: float, t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return 1.0 - (1.0 - float(m0)) * (0.5 * (1.0 + math.cos(math.pi * t)))

@torch.no_grad()
def _sample_ctx(img: torch.Tensor, scale: float=0.92, jitter: float=0.0, ar: Tuple[float,float]=(0.9,1.1)) -> Tuple[int,int,int,int]:
    B,C,H,W = img.shape
    s = float(torch.empty(1, device=img.device).uniform_(max(0.05, scale - jitter), min(0.98, scale + jitter)))
    a = float(torch.empty(1, device=img.device).uniform_(ar[0], ar[1]))
    area = H*W*s
    h = max(16, int((area / a) ** 0.5))
    w = max(16, int((area * a) ** 0.5))
    h = min(h, H); w = min(w, W)
    y = int(torch.randint(0, max(1, H-h+1), (1,), device=img.device)); x = int(torch.randint(0, max(1, W-w+1), (1,), device=img.device))
    return y, x, h, w

def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ay, ax, ah, aw = a; by, bx, bh, bw = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0: return 0.0
    ua = ah*aw + bh*bw - inter
    return float(inter) / float(max(1, ua))

def _sample_targets(img: torch.Tensor, k: int=4, smin: float=0.15, smax: float=0.2,
                    ar: Tuple[float,float]=(0.75,1.5), avoid: Tuple[int,int,int,int] | None = None,
                    max_tries: int = 20, min_hw: int = 16) -> List[Tuple[int,int,int,int]]:
    B,C,H,W = img.shape; outs=[]
    for _ in range(k):
        tries = 0
        while tries < max_tries:
            s = float(torch.empty(1, device=img.device).uniform_(smin, smax))
            a = float(torch.empty(1, device=img.device).uniform_(ar[0], ar[1]))
            area = H*W*s
            h = max(min_hw,int((area / a) ** 0.5))
            w = max(min_hw,int((area * a) ** 0.5))
            h = min(h, H); w = min(w, W)
            y = int(torch.randint(0, max(1,H-h+1), (1,), device=img.device))
            x = int(torch.randint(0, max(1,W-w+1), (1,), device=img.device))
            cand = (y,x,h,w)
            if avoid is None or _iou(cand, avoid) == 0.0:
                outs.append(cand); break
            tries += 1
    return outs

def _crop(img: torch.Tensor, box: Tuple[int,int,int,int]) -> torch.Tensor:
    y,x,h,w = box; return img[..., y:y+h, x:x+w]

def _predictor(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 4*in_dim), nn.GELU(), nn.Linear(4*in_dim, out_dim))

class IJEPA(SSLBaseModel):
    def __init__(self, stu: nn.Module, tea: nn.Module, predictor: nn.Module, ema_m: float=0.996,
                 k_targets: int=4, ctx_scale: float=0.6, tmin: float=0.1, tmax: float=0.4,
                 ctx_jitter: float=0.08, ema_m_min: float=0.99, ema_warmup_steps: int=0,
                 no_overlap: bool = True, ctx_resample: int = 4, min_target_tokens: int = 1):
        super().__init__()
        self.stu, self.tea, self.pred = stu, tea, predictor
        self.m, self.k, self.cs, self.tmin, self.tmax = ema_m, k_targets, ctx_scale, tmin, tmax
        self.ctx_jitter = float(ctx_jitter)
        self.ema_m_min = float(ema_m_min)
        self.ema_warmup_steps = int(max(0, ema_warmup_steps))
        self.no_overlap = bool(no_overlap)
        self.patch_multiple = int(getattr(self.stu, "patch_size", 0)) or None
        self.ctx_resample = int(max(0, ctx_resample))
        self.min_target_tokens = int(max(1, min_target_tokens))
        self._bootstrap()

    @classmethod
    def from_config(cls, cfg: Dict[str,Any]) -> "IJEPA":
        m = cfg["model"]["ssl"]
        bname, bopts = resolve_backbone_from_model_cfg(cfg["model"])
        stu = get_backbone(bname, pretrained=False, **bopts)
        tea = get_backbone(bname, pretrained=False, **bopts)
        pred = _predictor(stu.out_dim, stu.out_dim)
        ema_sched = (m.get("ema_schedule") or {})
        return cls(
            stu, tea, pred,
            ema_m=cfg["train"]["ssl"].get("ema_momentum", 0.996),
            k_targets=int(m.get("targets_per_image", 4)),
            ctx_scale=float(m.get("context_scale", 0.92)),
            tmin=float(m.get("target_scales", [0.15, 0.2])[0]),
            tmax=float(m.get("target_scales", [0.15, 0.2])[1]),
            ctx_jitter=float(m.get("context_jitter", 0.0)),
            ema_m_min=ema_sched.get("min", 0.99),
            ema_warmup_steps=int(ema_sched.get("warmup_steps", 0)),
            no_overlap=bool(m.get("no_overlap", True)),
            ctx_resample=int(m.get("max_context_resample", 4)),
            min_target_tokens=int(m.get("min_target_tokens", 1)),
        )

    @torch.no_grad()
    def _bootstrap(self):
        copy_weights_and_freeze(self.tea, self.stu)

    def _ema_momentum(self, global_step: int) -> float:
        if self.ema_warmup_steps <= 0 or abs(self.m - self.ema_m_min) < 1e-8:
            return self.m
        t = min(1.0, max(0.0, float(global_step) / float(self.ema_warmup_steps)))
        return _cosine_to_one(self.ema_m_min, t)

    def training_step(self, batch: Dict[str,Any], global_step: int) -> Dict[str,Any]:
        x = batch["images"][0]
        min_hw = 16
        if self.patch_multiple and self.min_target_tokens > 1:
            min_side_tokens = max(1, int(math.sqrt(self.min_target_tokens)))
            min_hw = max(min_hw, self.patch_multiple * min_side_tokens)
        attempts = max(1, self.ctx_resample + 1)
        ctx_box, tgt_boxes = None, []
        for _ in range(attempts):
            ctx_box = _sample_ctx(x, self.cs, self.ctx_jitter)
            tgt_boxes = _sample_targets(
                x, self.k, self.tmin, self.tmax,
                avoid=(ctx_box if self.no_overlap else None),
                min_hw=min_hw,
            )
            if tgt_boxes:
                break
        if not tgt_boxes:
            zero = self._zero_loss()
            return {"loss_total": zero, "loss_components": {"skipped_empty_targets": 1.0}}
        xc = self._prepare_crop(_crop(x, ctx_box))
        c_lat = self.stu.forward_global(xc)                       # [B,D]
        with torch.no_grad():
            m_curr = self._ema_momentum(global_step)
            ema_update(self.tea, self.stu, m_curr)
            t_lat = [self.tea.forward_global(self._prepare_crop(_crop(x, b))).detach() for b in tgt_boxes]
        preds = [self.pred(c_lat) for _ in tgt_boxes]
        pairs = [(p, t) for p, t in zip(preds, t_lat) if p is not None and t is not None]
        if not pairs:
            zero = self._zero_loss()
            return {"loss_total": zero, "loss_components": {"skipped_empty_targets": 1.0}}
        loss = torch.stack([F.mse_loss(p, t) for p,t in pairs]).mean()
        return {"loss_total": loss, "loss_components": {"loss_ijepa": float(loss.detach()), "ema_momentum": float(m_curr)}}

    def _prepare_crop(self, crop: torch.Tensor) -> torch.Tensor:
        """Ensure crops are compatible with the ViT patch size."""
        if not self.patch_multiple:
            return crop
        _, _, h, w = crop.shape
        target_h = int(math.ceil(h / self.patch_multiple) * self.patch_multiple)
        target_w = int(math.ceil(w / self.patch_multiple) * self.patch_multiple)
        if target_h == h and target_w == w:
            return crop
        return F.interpolate(crop, size=(target_h, target_w), mode="bilinear", align_corners=False)

    def _zero_loss(self) -> torch.Tensor:
        """Return a differentiable zero scalar to keep AMP/backward happy when skipping steps."""
        param = next(self.parameters())
        return (param.sum() * 0.0)
