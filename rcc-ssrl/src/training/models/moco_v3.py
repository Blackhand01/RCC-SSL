# models/moco_v3.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from src.training.trainer.backbones import ResNetBackbone, mlp_head, predictor_head
from src.training.utils.torch_ops import copy_weights_and_freeze, cosine_logits, ema_update
from src.training.trainer.loops import SSLBaseModel
from src.training.engine.schedules import CosineWithWarmup
import math
import hashlib

class MoCoV3(SSLBaseModel):
    def __init__(self, backbone_q: ResNetBackbone, backbone_k: ResNetBackbone,
                 proj_q: nn.Module, proj_k: nn.Module, pred_q: nn.Module,
                 tau: float=0.2, momentum: float=0.996,
                 *,
                 temp_teacher_sched: Optional[CosineWithWarmup]=None,
                 ema_to_one: bool=True,
                 use_multicrop: bool=False,
                 wsi_debias: Optional[Dict[str,Any]]=None,
                 total_steps: int=10000):
        super().__init__()
        self.backbone_q, self.backbone_k = backbone_q, backbone_k
        self.proj_q, self.proj_k, self.pred_q = proj_q, proj_k, pred_q
        self.Ts = tau
        self.m0 = momentum
        self.m = momentum
        self.Tsched = temp_teacher_sched
        self.ema_to_one = bool(ema_to_one)
        self.use_multicrop = bool(use_multicrop)
        self.total_steps = int(max(1, total_steps))
        self._step = 0

        # WSI-aware debias: {'enabled': bool, 'mode': 'filter'|'downweight', 'downweight': 0.25}
        self.debias = {"enabled": False, "mode": "filter", "downweight": 0.25}
        if wsi_debias:
            self.debias.update(wsi_debias)

        dim = self.proj_q[-1].out_features if hasattr(self.proj_q[-1], "out_features") else 256
        self.register_buffer("queue", torch.randn(dim, self.K))
        self.queue = l2n(self.queue.t()).t()
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # memorizza hash WSI per ciascun elemento in queue (int64); -1 se sconosciuto
        self.register_buffer("queue_wsi", -torch.ones(self.K, dtype=torch.long), persistent=True)

        self._bootstrap()

    @classmethod
    def from_config(cls, cfg: Dict[str,Any]) -> "MoCoV3":
        mcfg = cfg["model"]["ssl"]; bname = cfg["model"].get("backbone","resnet50")
        bb_q = ResNetBackbone(name=bname, pretrained=False); bb_k = ResNetBackbone(name=bname, pretrained=False)
        dim = bb_q.out_dim; proj_q = mlp_head(dim, 4096, 256); proj_k = mlp_head(dim, 4096, 256)
        pred_q = predictor_head(256, 4096)
        tr_ssl = (cfg.get("train",{}).get("ssl",{}) or {})
        steps_per_epoch = int(tr_ssl.get("steps_per_epoch", 1000))
        epochs = int(tr_ssl.get("epochs", 10))
        total_steps = max(1, steps_per_epoch * epochs)
        sched = (mcfg.get("temp_teacher_schedule") or {})
        Ts = None
        if sched:
            Ts = CosineWithWarmup(
                float(sched.get("start", mcfg.get("temperature", 0.2))),
                float(sched.get("end",   mcfg.get("temperature", 0.2))),
                warmup_frac=float(sched.get("warmup_frac", 0.0))
            )
        return cls(
            bb_q, bb_k, proj_q, proj_k, pred_q,
            tau=mcfg.get("temperature",0.2),
            momentum=cfg["train"]["ssl"].get("ema_momentum",0.996),
            temp_teacher_sched=Ts,
            ema_to_one=True,
            use_multicrop=bool(mcfg.get("use_multicrop", False)),
            wsi_debias=mcfg.get("wsi_debias", None),
            total_steps=total_steps,
        )

    @torch.no_grad()
    def _bootstrap(self) -> None:
        copy_weights_and_freeze(self.backbone_k, self.backbone_q)
        copy_weights_and_freeze(self.proj_k, self.proj_q)

    @torch.no_grad()
    def _ema_momentum(self) -> float:
        if not self.ema_to_one or self.total_steps <= 0:
            return self.m
        t = min(1.0, self._step / float(self.total_steps))
        # cosine to 1.0
        return 1.0 - (1.0 - self.m0) * 0.5 * (1.0 + math.cos(math.pi * t))

    def _teacher_temp(self) -> float:
        if self.Tsched is None or self.total_steps <= 0:
            return self.Ts
        t = min(1.0, self._step / float(self.total_steps))
        return float(self.Tsched.at(t))

    @staticmethod
    def _hash_wsi_id(meta_item: Any) -> int:
        # meta può essere dict o altro; cerchiamo 'wsi_id'
        try:
            wsi = str(meta_item.get("wsi_id", ""))
        except Exception:
            wsi = ""
        if not wsi:
            return -1
        # hash stabile (sha1 → int64)
        h = int(hashlib.sha1(wsi.encode("utf-8")).hexdigest()[:16], 16)
        return h & ((1<<63)-1)

    def _info_nce_sym(self, q1, q2, k1, k2) -> torch.Tensor:
        lab = torch.arange(q1.size(0), device=q1.device)
        Tt = self._teacher_temp()
        l12 = F.cross_entropy(cosine_logits(q1, k2, Tt), lab)
        l21 = F.cross_entropy(cosine_logits(q2, k1, Tt), lab)
        return (l12 + l21)

    def training_step(self, batch: Dict[str,Any], global_step: int) -> Dict[str,Any]:
        self._step = global_step
        imgs, meta = batch["images"], batch.get("meta", None)
        # Views: (a) due global standard, (b) opzionali local extra se use_multicrop
        if self.use_multicrop:
            G = imgs[0]; L = imgs[1] if len(imgs) > 1 else imgs[0]     # compat: multicrop builder usa [G,L]
            # split due global
            if G.size(0) % 2 != 0:
                raise ValueError("MoCo multi-crop richiede batch globale divisibile per 2.")
            x1, x2 = G.chunk(2, dim=0)
            extra_pos: List[torch.Tensor] = [L] if L is not None else []
        else:
            if len(imgs) < 2:
                raise ValueError("MoCo v3 requires two global views.")
            x1, x2 = imgs[0], imgs[1]
            extra_pos = []

        q1 = self.pred_q(self.proj_q(self.backbone_q.forward_global(x1)))
        q2 = self.pred_q(self.proj_q(self.backbone_q.forward_global(x2)))
        with torch.no_grad():
            ema_update(self.backbone_k, self.backbone_q, self._ema_momentum())
            k1 = self.proj_k(self.backbone_k.forward_global(x1))
            k2 = self.proj_k(self.backbone_k.forward_global(x2))
        loss_main = self._info_nce_sym(q1, q2, k1.detach(), k2.detach())

        # ulteriori positivi da local views (facoltativo)
        aux_losses = []
        if self.use_multicrop and len(extra_pos) > 0:
            with torch.no_grad():
                k_loc = self.proj_k(self.backbone_k.forward_global(extra_pos[0])).detach()  # treat as extra keys
            q_loc = self.pred_q(self.proj_q(self.backbone_q.forward_global(extra_pos[0])))
            aux_losses.append(self._info_nce_sym(q_loc, q_loc, k_loc, k_loc))  # dummy, adjust as needed

        loss = loss_main + (sum(aux_losses) / len(aux_losses) if aux_losses else 0.0)

        # metriche diagnostiche
        with torch.no_grad():
            Tt = self._teacher_temp()
            # pos/neg sim
            pos_sim = float((q1 * k2).mean().item())
            # entropy media dei softmax sui negativi (dummy for now)
            queue_entropy = 0.0

        return {
            "loss_total": loss,
            "loss_components": {
                "loss_main": float(loss_main.detach()),
                "aux_losses": float(sum([x.detach() for x in aux_losses]) / max(1, len(aux_losses))) if aux_losses else 0.0,
                "t_teacher": float(Tt),
                "ema_m": float(self._ema_momentum()),
                "pos_sim": pos_sim,
                "queue_entropy": queue_entropy,
            },
        }

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone_q.forward_global(x)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone_q.forward_tokens(x)
