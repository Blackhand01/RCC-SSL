# models/i_jepa.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from src.training.utils.trainers import SSLBaseModel, ResNetBackbone, copy_weights_and_freeze, ema_update

def _sample_ctx(img: torch.Tensor, scale: float=0.6) -> Tuple[int,int,int,int]:
    B,C,H,W = img.shape; h = max(16, int(H*scale)); w = max(16, int(W*scale))
    y = int(torch.randint(0, max(1,H-h+1), (1,), device=img.device)); x = int(torch.randint(0, max(1,W-w+1), (1,), device=img.device))
    return y, x, h, w

def _sample_targets(img: torch.Tensor, k: int=4, smin: float=0.1, smax: float=0.4) -> List[Tuple[int,int,int,int]]:
    B,C,H,W = img.shape; outs=[]
    for _ in range(k):
        s = float(torch.empty(1, device=img.device).uniform_(smin, smax))
        h = max(16,int(H*s)); w = max(16,int(W*s))
        y = int(torch.randint(0, max(1,H-h+1), (1,), device=img.device)); x = int(torch.randint(0, max(1,W-w+1), (1,), device=img.device))
        outs.append((y,x,h,w))
    return outs

def _crop(img: torch.Tensor, box: Tuple[int,int,int,int]) -> torch.Tensor:
    y,x,h,w = box; return img[..., y:y+h, x:x+w]

def _predictor(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 4*in_dim), nn.GELU(), nn.Linear(4*in_dim, out_dim))

class IJEPA(SSLBaseModel):
    def __init__(self, stu: ResNetBackbone, tea: ResNetBackbone, predictor: nn.Module, ema_m: float=0.996,
                 k_targets: int=4, ctx_scale: float=0.6, tmin: float=0.1, tmax: float=0.4):
        super().__init__()
        self.stu, self.tea, self.pred = stu, tea, predictor
        self.m, self.k, self.cs, self.tmin, self.tmax = ema_m, k_targets, ctx_scale, tmin, tmax
        self._bootstrap()

    @classmethod
    def from_config(cls, cfg: Dict[str,Any]) -> "IJEPA":
        bname = cfg["model"].get("backbone","resnet50"); m = cfg["model"]["ssl"]
        stu, tea = ResNetBackbone(bname, False), ResNetBackbone(bname, False)
        pred = _predictor(stu.out_dim, stu.out_dim)
        return cls(stu, tea, pred, ema_m=cfg["train"]["ssl"].get("ema_momentum",0.996),
                   k_targets=m.get("targets_per_image",4), ctx_scale=m.get("context_scale",0.6),
                   tmin=m.get("target_scales",[0.1,0.4])[0], tmax=m.get("target_scales",[0.1,0.4])[1])

    @torch.no_grad()
    def _bootstrap(self):
        copy_weights_and_freeze(self.tea, self.stu)

    def training_step(self, batch: Dict[str,Any], global_step: int) -> Dict[str,Any]:
        x = batch["images"][0]
        ctx_box = _sample_ctx(x, self.cs); tgt_boxes = _sample_targets(x, self.k, self.tmin, self.tmax)
        xc = _crop(x, ctx_box)
        c_lat = self.stu.forward_global(xc)                       # [B,D]
        with torch.no_grad():
            ema_update(self.tea, self.stu, self.m)
            t_lat = [self.tea.forward_global(_crop(x, b)).detach() for b in tgt_boxes]
        preds = [self.pred(c_lat) for _ in tgt_boxes]
        loss = torch.stack([F.mse_loss(p, t) for p,t in zip(preds, t_lat)]).mean()
        return {"loss_total": loss, "loss_components": {"loss_ijepa": float(loss.detach())}}
