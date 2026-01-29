# models/ibot.py
from __future__ import annotations
from typing import Any, Dict, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.trainer.backbones import get_backbone, mlp_head, resolve_backbone_from_model_cfg
from src.training.utils.torch_ops import copy_weights_and_freeze, ema_update
from src.training.trainer.loops import SSLBaseModel


def make_mask(B: int, T: int, ratio: float, device: torch.device) -> torch.Tensor:
    """
    Create a boolean mask [B, T] with about ratio*T tokens masked per image.
    - if ratio <= 0: no tokens masked
    - if ratio >= 1: all tokens masked
    """
    m = torch.zeros(B, T, dtype=torch.bool, device=device)
    if ratio <= 0.0:
        return m
    if ratio >= 1.0:
        m[:] = True
        return m
    k = max(1, int(T * ratio))
    idx = torch.rand(B, T, device=device).argsort(dim=1)[:, :k]
    m.scatter_(1, idx, True)
    return m



class IBOT(SSLBaseModel):
    """
    Simplified iBOT (ICLR 2022) implementation, without multi-crop.

    - Student / Teacher: backbone (ViT/ResNet) + MLP head shared for CLS and patches.
    - L[CLS]: self-distillation cross-view on global token (DINO style).
    - L_MIM: self-distillation on masked patches (blockwise masking in image).
    - Teacher, head_teacher and centers updated via EMA.
    """

    def __init__(
        self,
        stu: nn.Module,
        tea: nn.Module,
        head_s: nn.Module,
        head_t: nn.Module,
        *,
        t_cls_s: float = 0.1,
        t_cls_t: float = 0.07,
        t_patch_s: float = 0.1,
        t_patch_t: float = 0.07,
        mask_ratio: float = 0.3,
        mask_ratio_min: float = 0.1,
        mask_ratio_max: float = 0.5,
        ema_base_m: float = 0.996,
        ema_final_m: float = 1.0,
        total_steps: int = 0,
        center_m: float = 0.9,
        center_m_patch: float = 0.9,
    ):
        super().__init__()
        self.stu = stu
        self.tea = tea
        self.head_s = head_s
        self.head_t = head_t

        # temperature for CLS and patches
        self.t_cls_s = float(t_cls_s)
        self.t_cls_t = float(t_cls_t)
        self.t_patch_s = float(t_patch_s)
        self.t_patch_t = float(t_patch_t)

        # masking
        self.mask_ratio = float(mask_ratio)  # used only as default/diagnostics
        self.mask_ratio_min = float(mask_ratio_min)
        self.mask_ratio_max = float(mask_ratio_max)

        # EMA schedule: m_t goes from ema_base_m -> ema_final_m (typically 0.996 -> 1.0)
        self.ema_base_m = float(ema_base_m)
        self.ema_final_m = float(ema_final_m)
        self.total_steps = int(total_steps)

        self.center_m = float(center_m)
        self.center_m_patch = float(center_m_patch)

        self.patch_size: int = int(getattr(self.stu, "patch_size", 16))

        # center for [CLS] and patch tokens (dim = K = out_dim of head)
        out_dim = getattr(self.head_s[-1], "num_features", None)
        if out_dim is None:
            # fallback: try to infer from second-to-last Linear
            for mod in reversed(self.head_s):
                if isinstance(mod, nn.Linear):
                    out_dim = mod.out_features
                    break
        if out_dim is None:
            raise RuntimeError("Unable to infer head output dimension for centers.")
        out_dim = int(out_dim)

        self.register_buffer("center_cls", torch.zeros(out_dim))
        self.register_buffer("center_patch", torch.zeros(out_dim))

        self._bootstrap()

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "IBOT":
        m = cfg["model"]["ssl"]
        bname, bopts = resolve_backbone_from_model_cfg(cfg["model"])
        stu = get_backbone(bname, pretrained=False, **bopts)
        tea = get_backbone(bname, pretrained=False, **bopts)

        dim = getattr(stu, "out_dim", None)
        if dim is None:
            raise ValueError("Backbone used in IBOT must expose 'out_dim'.")

        hidden_dim = int(m.get("hidden_dim", 4096))
        K = int(m.get("num_prototypes", 8192))

        head_s = mlp_head(dim, hidden_dim, K)
        head_t = mlp_head(dim, hidden_dim, K)

        t_cls_s = float(m.get("temp_student", m.get("temperature", 0.1)))
        t_cls_t = float(m.get("temp_teacher", 0.07))
        t_patch_s = float(m.get("temp_student_patch", t_cls_s))
        t_patch_t = float(m.get("temp_teacher_patch", t_cls_t))

        # masking
        mask_ratio = float(m.get("mask_ratio", 0.3))
        mask_ratio_min = float(m.get("mask_ratio_min", 0.1))
        mask_ratio_max = float(m.get("mask_ratio_max", 0.5))

        # train config for total_steps and EMA
        ssl_cfg = (cfg.get("train", {}) or {}).get("ssl", {}) or {}
        epochs = int(ssl_cfg.get("epochs", 1))
        steps_per_epoch = int(ssl_cfg.get("steps_per_epoch", 1))
        total_steps = max(1, epochs * steps_per_epoch)

        ema_base_m = float(ssl_cfg.get("ema_momentum", ssl_cfg.get("ema_m", 0.996)))
        ema_final_m = float(m.get("ema_momentum_final", 1.0))

        center_m = float(m.get("center_momentum", 0.9))
        center_m_patch = float(m.get("center_momentum_patch", center_m))

        return cls(
            stu,
            tea,
            head_s,
            head_t,
            t_cls_s=t_cls_s,
            t_cls_t=t_cls_t,
            t_patch_s=t_patch_s,
            t_patch_t=t_patch_t,
            mask_ratio=mask_ratio,
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            ema_base_m=ema_base_m,
            ema_final_m=ema_final_m,
            total_steps=total_steps,
            center_m=center_m,
            center_m_patch=center_m_patch,
        )


    # ------------------------------------------------------------------ helpers
    @torch.no_grad()
    def _bootstrap(self) -> None:
        """Initialize teacher = student and freeze teacher weights."""
        copy_weights_and_freeze(self.tea, self.stu)
        copy_weights_and_freeze(self.head_t, self.head_s)
        
    
    def _ema_m_for_step(self, global_step: int) -> float:
        """
        Teacher momentum m_t as in DINO/iBOT:
        starts from ema_base_m and approaches ema_final_m with cosine schedule.
        """
        if self.total_steps <= 0:
            return self.ema_base_m
        t = min(max(global_step, 1), self.total_steps)
        # cosine goes from 1 -> -1, remapped to [0,1]
        cos_term = (1.0 + math.cos(math.pi * t / self.total_steps)) / 2.0
        # standard form: m_t = m_final - (m_final - m_base) * cos_term
        m_t = self.ema_final_m - (self.ema_final_m - self.ema_base_m) * cos_term
        return float(m_t)

        
    def _sample_mask_ratio(self) -> float:
        """
        50%: r = 0 (only L[CLS]).
        50%: r ~ U[mask_ratio_min, mask_ratio_max].
        """
        if torch.rand(()) < 0.5:
            return 0.0
        low = self.mask_ratio_min
        high = self.mask_ratio_max
        if high <= low:
            return float(low)
        return float(low + (high - low) * torch.rand(()))


    def _blockwise_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply blockwise masking in image space.
        Returns:
          - x_hat: masked image
          - m: mask token [B, T] with T = (H/patch_size)*(W/patch_size)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        gh, gw = H // p, W // p
        T = gh * gw

        r = self._sample_mask_ratio()

        if r <= 0.0:
            # DINO-like case: no masked patches, no MIM
            mask_tokens = torch.zeros(B, T, dtype=torch.bool, device=x.device)
            x_hat = x
        else:
            mask_tokens = make_mask(B, T, r, x.device)
            mask_img = mask_tokens.view(B, 1, gh, gw).float()
            mask_img = F.interpolate(mask_img, size=(H, W), mode="nearest")
            x_hat = x * (1.0 - mask_img)  # zero-out masked patches

        return x_hat, mask_tokens


    def _tokens_no_cls(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        If present, remove CLS token assuming order [CLS, patch1, patch2, ...].
        Use H, W and patch_size to estimate number of patches.
        """
        B, T, C = tokens.shape
        p = self.patch_size
        gh, gw = H // p, W // p
        patch_T = gh * gw
        if T == patch_T + 1:
            return tokens[:, 1:, :]
        return tokens

    def _project_tokens(self, head: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        """Apply head to tokens [B, T, C] -> [B, T, K]."""
        B, T, C = tokens.shape
        # use reshape to handle non-contiguous tensors (channels_last, etc.)
        flat = tokens.reshape(B * T, C)
        out = head(flat)
        return out.reshape(B, T, -1)


    def _forward_student(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits_cls, logits_patch) of student after head.
        """
        B, C, H, W = x.shape
        if hasattr(self.stu, "vit"):
            tok_all = self.stu.forward_tokens(x)               # [B, T_all, D]
            cls_feat = tok_all[:, 0, :]                        # [B, D]
            tok_feat = self._tokens_no_cls(tok_all, H, W)      # [B, T, D]
        else:
            cls_feat = self.stu.forward_global(x)              # [B, D]
            tok_feat = self.stu.forward_tokens(x)              # [B, T, D]
        cls_logits = self.head_s(cls_feat)                     # [B, K]
        patch_logits = self._project_tokens(self.head_s, tok_feat)  # [B, T, K]
        return cls_logits, patch_logits

    @torch.no_grad()
    def _forward_teacher(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits_cls, logits_patch) of teacher after head.
        """
        B, C, H, W = x.shape
        if hasattr(self.tea, "vit"):
            tok_all = self.tea.forward_tokens(x)
            cls_feat = tok_all[:, 0, :]
            tok_feat = self._tokens_no_cls(tok_all, H, W)
        else:
            cls_feat = self.tea.forward_global(x)
            tok_feat = self.tea.forward_tokens(x)
        cls_logits = self.head_t(cls_feat)
        patch_logits = self._project_tokens(self.head_t, tok_feat)
        return cls_logits, patch_logits

    def _H(self, s: torch.Tensor, t: torch.Tensor, center: torch.Tensor, t_s: float, t_t: float) -> torch.Tensor:
        """
        Implementation of function H(s, t, c, τ_s, τ_t) from Algorithm 1 of iBOT.
        Returns loss vector per sample (or token) along last dim K.
        """
        t = t.detach()
        # s: log-softmax con temperatura student
        log_s = F.log_softmax(s / max(t_s, 1e-6), dim=-1)
        # t: softmax centrato + sharpen
        centered_t = (t - center.view(1, -1)) / max(t_t, 1e-6)
        prob_t = F.softmax(centered_t, dim=-1)
        loss = -(prob_t * log_s).sum(dim=-1)
        return loss

    def _cls_loss(
        self,
        s_u_cls: torch.Tensor,
        s_v_cls: torch.Tensor,
        t_u_cls: torch.Tensor,
        t_v_cls: torch.Tensor,
    ) -> torch.Tensor:
        """
        L[CLS] = 0.5 * ( H(s(u^), t(v)) + H(s(v^), t(u)) ), average over batch.
        """
        loss_uv = self._H(s_u_cls, t_v_cls, self.center_cls, self.t_cls_s, self.t_cls_t)
        loss_vu = self._H(s_v_cls, t_u_cls, self.center_cls, self.t_cls_s, self.t_cls_t)
        return 0.5 * (loss_uv.mean() + loss_vu.mean())

    def _patch_loss(
        self,
        s_patch: torch.Tensor,
        t_patch: torch.Tensor,
        mask_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        L_MIM for one view, following Algorithm 1:
          L_MIM = (m * H(s_patch, t_patch)).sum / m.sum, average over batch.
        """
        B, T, K = s_patch.shape
        # use reshape to avoid contiguity issues
        s_flat = s_patch.reshape(B * T, K)
        t_flat = t_patch.reshape(B * T, K)
        h_flat = self._H(s_flat, t_flat, self.center_patch, self.t_patch_s, self.t_patch_t)  # [B*T]
        h = h_flat.reshape(B, T)  # [B, T]

        m = mask_tokens.to(h.device).float()
        masked = (m * h).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(1.0)
        per_img = masked / denom
        return per_img.mean()


    @torch.no_grad()
    def _update_centers(
        self,
        t_u_cls: torch.Tensor,
        t_v_cls: torch.Tensor,
        t_u_patch: torch.Tensor,
        t_v_patch: torch.Tensor,
    ) -> None:
        """Update C and C0 via EMA on teacher logits."""
        # center [CLS]
        cls_batch = torch.cat([t_u_cls, t_v_cls], dim=0).mean(dim=0)
        self.center_cls.mul_(self.center_m).add_(cls_batch, alpha=(1.0 - self.center_m))

        # center patch tokens
        patch_batch = torch.cat([t_u_patch, t_v_patch], dim=0).mean(dim=(0, 1))
        self.center_patch.mul_(self.center_m_patch).add_(patch_batch, alpha=(1.0 - self.center_m_patch))

    # ------------------------------------------------------------------ training
    def training_step(self, batch: Dict[str, Any], global_step: int) -> Dict[str, Any]:
        images = batch["images"]
        if len(images) < 2:
            raise ValueError("iBOT requires at least two global views (got len(images) < 2).")
        u, v = images[0], images[1]

        # update teacher (EMA) at the beginning of the step, as in DINO/iBOT
        with torch.no_grad():
            ema_m = self._ema_m_for_step(global_step)
            ema_update(self.tea, self.stu, ema_m)
            ema_update(self.head_t, self.head_s, ema_m)

        # blockwise masking only for student
        u_masked, mu = self._blockwise_mask(u)
        v_masked, mv = self._blockwise_mask(v)

        # forward student (masked) and teacher (unmasked)
        s_u_cls, s_u_patch = self._forward_student(u_masked)
        s_v_cls, s_v_patch = self._forward_student(v_masked)

        with torch.no_grad():
            t_u_cls, t_u_patch = self._forward_teacher(u)
            t_v_cls, t_v_patch = self._forward_teacher(v)

        # compute losses
        loss_cls = self._cls_loss(s_u_cls, s_v_cls, t_u_cls, t_v_cls)
        loss_tok_u = self._patch_loss(s_u_patch, t_u_patch, mu)
        loss_tok_v = self._patch_loss(s_v_patch, t_v_patch, mv)
        loss_tok = 0.5 * (loss_tok_u + loss_tok_v)
        loss_total = loss_cls + loss_tok

        # update centers
        with torch.no_grad():
            self._update_centers(t_u_cls, t_v_cls, t_u_patch, t_v_patch)

        return {
            "loss_total": loss_total,
            "loss_components": {
                "loss_cls": float(loss_cls.detach()),
                "loss_tok": float(loss_tok.detach()),
            },
        }
