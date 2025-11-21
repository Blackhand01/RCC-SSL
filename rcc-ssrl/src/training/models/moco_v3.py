# models/moco_v3.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from src.training.trainer.backbones import get_backbone, mlp_head, predictor_head, resolve_backbone_from_model_cfg
from src.training.utils.torch_ops import copy_weights_and_freeze, cosine_logits, ema_update
from src.training.trainer.loops import SSLBaseModel

class CosineWithWarmup:
    """Cosine annealing from start->end with warmup in [0, warmup_frac]."""
    def __init__(self, start: float, end: float, warmup_frac: float = 0.1):
        self.start, self.end, self.warmup = float(start), float(end), float(warmup_frac)

    def at(self, t: float) -> float:
        t = min(max(t, 0.0), 1.0)
        if t < self.warmup:
            return self.start + (self.end - self.start) * (t / max(self.warmup, 1e-8))
        # cosine on the remaining segment
        import math
        tc = (t - self.warmup) / max(1.0 - self.warmup, 1e-8)
        return self.end + 0.5 * (self.start - self.end) * (1 + math.cos(math.pi * tc))
import math

class MoCoV3(SSLBaseModel):
    """
    MoCo v3 “no-queue”: due viste globali, encoder a momentum (teacher) + predictor.
    Negativi = in-batch. Loss simmetrizzata: ctr(q1,k2) + ctr(q2,k1).
    """
    def __init__(self, backbone_q: nn.Module, backbone_k: nn.Module,
                 proj_q: nn.Module, proj_k: nn.Module, pred_q: nn.Module,
                 tau: float = 0.2, momentum: float = 0.996,
                 *,
                 temp_teacher_sched: Optional[CosineWithWarmup]=None,
                 ema_to_one: bool=True,
                 use_multicrop: bool=False,
                 total_steps: int=10000,
                 clip_qk: float = 50.0,
                 sync_bn: bool = False):
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
        self.clip_qk = float(clip_qk)
        self._sync_bn_enabled = bool(sync_bn)

        self._bootstrap()

    @classmethod
    def from_config(cls, cfg: Dict[str,Any]) -> "MoCoV3":
        mcfg = cfg["model"]["ssl"]
        bname, bopts = resolve_backbone_from_model_cfg(cfg["model"])
        bb_q = get_backbone(name=bname, pretrained=False, **bopts)
        bb_k = get_backbone(name=bname, pretrained=False, **bopts)
        dim = bb_q.out_dim
        proj_dim = int(mcfg.get("proj_dim", 256))
        hid = int(mcfg.get("hidden_dim", 4096))
        proj_q = mlp_head(dim, hid, proj_dim); proj_k = mlp_head(dim, hid, proj_dim)
        pred_q = predictor_head(proj_dim, hid)
        tr_ssl = (cfg.get("train",{}).get("ssl",{}) or {})
        steps_per_epoch = int(tr_ssl.get("steps_per_epoch", 1000))
        epochs = int(tr_ssl.get("epochs", 10))
        total_steps = max(1, steps_per_epoch * epochs)
        # ---- Robust scheduling & numeric coercions (defensive vs. null/None) ----
        sched = (mcfg.get("temp_teacher_schedule") or {})
        Ts = None
        if sched:
            t_default = mcfg.get("temperature", 0.2)
            t_start  = sched.get("start", t_default)
            t_end    = sched.get("end",   t_default)
            t_warm   = sched.get("warmup_frac", 0.0)
            # Coercioni sicure: se i campi sono None o stringhe vuote, usa i default
            def _safe_float(v, d): 
                try:
                    return float(v if v is not None and v != "" else d)
                except (TypeError, ValueError):
                    return float(d)
            Ts = CosineWithWarmup(
                _safe_float(t_start, t_default),
                _safe_float(t_end,   t_default),
                warmup_frac=_safe_float(t_warm, 0.0),
            )
        clip_qk_val = mcfg.get("clip_qk", 50.0)
        # Se è None/invalid, ripiega su 50.0 per stabilità numerica
        try:
            clip_qk_val = 50.0 if clip_qk_val is None else float(clip_qk_val)
        except (TypeError, ValueError):
            clip_qk_val = 50.0
        return cls(
            bb_q, bb_k, proj_q, proj_k, pred_q,
            tau=float(mcfg.get("temperature", 0.2) or 0.2),
            momentum=cfg["train"]["ssl"].get("ema_momentum",0.996),
            temp_teacher_sched=Ts,
            ema_to_one=bool(mcfg.get("ema_to_one", True)),
            use_multicrop=bool(mcfg.get("use_multicrop", False)),  # accettato ma ignoriamo le local per MoCo
            total_steps=total_steps,
            clip_qk=float(clip_qk_val),
            sync_bn=bool(mcfg.get("sync_bn", False)),
        )

    @torch.no_grad()
    def _bootstrap(self) -> None:
        copy_weights_and_freeze(self.backbone_k, self.backbone_q)
        copy_weights_and_freeze(self.proj_k, self.proj_q)
        # Converti BN in SyncBN solo su projector/predictor quando in DDP
        if self._sync_bn_enabled:
            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    self.proj_q = nn.SyncBatchNorm.convert_sync_batchnorm(self.proj_q)
                    self.proj_k = nn.SyncBatchNorm.convert_sync_batchnorm(self.proj_k)
                    self.pred_q = nn.SyncBatchNorm.convert_sync_batchnorm(self.pred_q)
            except Exception:
                pass

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

    def _info_nce_sym(self, q1, q2, k1, k2) -> torch.Tensor:
        lab = torch.arange(q1.size(0), device=q1.device)
        Tt = self._teacher_temp()
        # clamp per stabilità numerica
        l12 = F.cross_entropy(torch.clamp(cosine_logits(q1, k2, Tt), -self.clip_qk, self.clip_qk), lab)
        l21 = F.cross_entropy(torch.clamp(cosine_logits(q2, k1, Tt), -self.clip_qk, self.clip_qk), lab)
        return (l12 + l21)

    def training_step(self, batch: Dict[str,Any], global_step: int) -> Dict[str,Any]:
        self._step = global_step
        imgs, meta = batch["images"], batch.get("meta", None)
        # Two global views required. If multicrop is enabled, imgs == [G, L] where
        # G is the stack of BOTH globals (shape: 2*B, C, H, W) and L are local crops.
        if self.use_multicrop:
            G = imgs[0]
            if G.dim() != 4 or (G.size(0) % 2 != 0):
                raise ValueError(f"MoCo v3 (multicrop) expects stacked 2 globals; got {tuple(G.shape)}.")
            # Split stacked globals into x1/x2
            x1, x2 = torch.chunk(G, 2, dim=0)
        else:
            if len(imgs) < 2:
                raise ValueError("MoCo v3 requires two global views.")
            x1, x2 = imgs[0], imgs[1]

        q1 = self.pred_q(self.proj_q(self.backbone_q.forward_global(x1)))
        q2 = self.pred_q(self.proj_q(self.backbone_q.forward_global(x2)))
        with torch.no_grad():
            ema_update(self.backbone_k, self.backbone_q, self._ema_momentum())
            k1 = self.proj_k(self.backbone_k.forward_global(x1))
            k2 = self.proj_k(self.backbone_k.forward_global(x2))
        # normalizza e clampa per robustezza
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1.detach(), dim=-1)
        k2 = torch.nn.functional.normalize(k2.detach(), dim=-1)
        loss_main = self._info_nce_sym(q1, q2, k1, k2)
        loss = loss_main

        # metriche diagnostiche
        with torch.no_grad():
            Tt = self._teacher_temp()
            pos_sim = float((q1 * k2).sum(dim=-1).mean().item())
            # media SOLO sugli off-diagonali (negativi)
            sim_mat = q1 @ k2.t()
            off = ~torch.eye(sim_mat.shape[0], dtype=torch.bool, device=sim_mat.device)
            neg_sim = float(sim_mat[off].mean().item())

        return {
            "loss_total": loss,
            "loss_components": {
                "loss_main": float(loss_main.detach()),
                "t_teacher": float(Tt),
                "ema_m": float(self._ema_momentum()),
                "pos_sim": pos_sim,
                "neg_sim": neg_sim,
            },
        }

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone_q.forward_global(x)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone_q.forward_tokens(x)
