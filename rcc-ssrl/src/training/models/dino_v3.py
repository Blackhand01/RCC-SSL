from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from src.training.trainer.backbones import (
    get_backbone,
    resolve_backbone_from_model_cfg,
)
from src.training.utils.torch_ops import copy_weights_and_freeze, ema_update
from src.training.trainer.loops import SSLBaseModel

# --- IMPORT OFFICIAL LOSSES ---
# Make sure you have copied the files from dinov3/loss/ repo to src/training/loss/
from src.training.loss.dino_clstoken_loss import DINOLoss
from src.training.loss.ibot_patch_loss import iBOTPatchLoss
from src.training.loss.koleo_loss import KoLeoLoss
from src.training.loss.gram_loss import GramLoss


class DINOHead(nn.Module):
    """
    Standard DINOv3 Head (MLP + optional Weight Norm).
    Replaces mlp_head for compliance with official initialization.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        layers = [nn.Linear(in_dim, hidden_dim, bias=mlp_bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=mlp_bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=mlp_bias))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_init)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


class SimpleMaskGenerator:
    """Genera maschere casuali per iBOT (simil-MAE)"""
    def __init__(self, input_size=224, patch_size=16, mask_ratio=(0.1, 0.5)):
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.mask_ratio = mask_ratio

    def __call__(self, batch_size, device):
        # Sample random ratio
        ratio = torch.empty(1).uniform_(*self.mask_ratio).item()
        num_masked = int(self.num_patches * ratio)
        
        # Generate noise and sort to select patches to mask
        noise = torch.rand(batch_size, self.num_patches, device=device)
        mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
        
        # First 'num_masked' indices are masked (True)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_masked = ids_shuffle[:, :num_masked]
        mask.scatter_(1, ids_masked, True)
        return mask


class DINOv3(SSLBaseModel):
    def __init__(
        self,
        student_backbone: nn.Module,
        teacher_backbone: nn.Module,
        embed_dim: int,
        # Config parameters
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        teacher_temp: float = 0.07,
        student_temp: float = 0.1,
        # Loss weights
        w_dino: float = 1.0,
        w_ibot: float = 1.0,
        w_koleo: float = 0.1,
        w_gram: float = 1.0,
        # EMA
        ema_momentum: float = 0.996,
        # Gram
        gram_start_frac: float = 0.7,
        gram_teacher_update_every: int = 10000,
        # Masking
        patch_size: int = 16,
        input_size: int = 224,
    ) -> None:
        super().__init__()

        self.stu = student_backbone
        self.tea = teacher_backbone
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # --- Heads (Shared Global/Local) ---
        # DINO head (CLS token)
        self.dino_head = DINOHead(embed_dim, out_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
        self.dino_head_tea = copy.deepcopy(self.dino_head)
        
        # iBOT head (Patch tokens)
        self.ibot_head = DINOHead(embed_dim, out_dim, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
        self.ibot_head_tea = copy.deepcopy(self.ibot_head)

        # --- Losses ---
        # Sinkhorn-Knopp for DINO
        self.dino_loss_fn = DINOLoss(out_dim, student_temp=student_temp)
        # iBOT Loss
        self.ibot_loss_fn = iBOTPatchLoss(out_dim, student_temp=student_temp)
        # KoLeo Loss
        self.koleo_loss_fn = KoLeoLoss()
        # Gram Loss
        self.gram_loss_fn = GramLoss()
        self.dino_loss_fn.init_weights()
        self.ibot_loss_fn.init_weights()

        # Weights
        self.w_dino = w_dino
        self.w_ibot = w_ibot
        self.w_koleo = w_koleo
        self.w_gram = w_gram

        # Teacher parameters
        self.teacher_temp = teacher_temp
        self.ema_m = ema_momentum

        # Gram Parameters
        self.gram_teacher: Optional[nn.Module] = None
        self.gram_start_frac = gram_start_frac
        self.gram_teacher_update_every = gram_teacher_update_every
        self.total_steps: Optional[int] = None
        
        # Mask generator
        # Fallback mask generator if dataloader doesn't provide masks
        self.mask_gen = SimpleMaskGenerator(input_size=input_size, patch_size=patch_size)

        self._bootstrap()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "DINOv3":
        m = cfg["model"]["ssl"]
        bname, bopts = resolve_backbone_from_model_cfg(cfg["model"])
        
        # Backbone creation
        stu = get_backbone(name=bname, pretrained=False, **bopts)
        tea = get_backbone(name=bname, pretrained=False, **bopts)
        
        # Parsing config
        return cls(
            student_backbone=stu,
            teacher_backbone=tea,
            embed_dim=stu.out_dim,
            out_dim=m.get("head_n_prototypes", 65536),
            hidden_dim=m.get("head_hidden_dim", 2048),
            bottleneck_dim=m.get("head_bottleneck_dim", 256),
            teacher_temp=m.get("teacher_temp", 0.07),
            student_temp=m.get("student_temp", 0.1),
            w_dino=m.get("loss_weight", 1.0),
            w_ibot=m.get("ibot_loss_weight", 1.0),
            w_koleo=m.get("koleo_loss_weight", 0.1),
            w_gram=float((m.get("gram") or {}).get("lambda", 1.0)),
            ema_momentum=cfg["train"]["ssl"].get("ema_momentum", 0.996),
            patch_size=bopts.get("patch_size", 16),
            input_size=bopts.get("input_size", 224) or 224,
        )

    @torch.no_grad()
    def _bootstrap(self):
        # 1) initialization of student's heads
        self.dino_head.init_weights()
        self.ibot_head.init_weights()

        # 2) copy backbone and head to teacher + freeze
        copy_weights_and_freeze(self.tea, self.stu)
        copy_weights_and_freeze(self.dino_head_tea, self.dino_head)
        copy_weights_and_freeze(self.ibot_head_tea, self.ibot_head)

        self.tea.eval()
        self.dino_head_tea.eval()
        self.ibot_head_tea.eval()


    def set_total_steps(self, total_steps: int):
        self.total_steps = int(total_steps)

    @torch.no_grad()
    def _update_teacher(self):
        ema_update(self.tea, self.stu, self.ema_m)
        ema_update(self.dino_head_tea, self.dino_head, self.ema_m)
        ema_update(self.ibot_head_tea, self.ibot_head, self.ema_m)

    @torch.no_grad()
    def _ensure_gram_teacher(self):
        if self.gram_teacher is None:
            self.gram_teacher = copy.deepcopy(self.tea)
            self.gram_teacher.eval()
            for p in self.gram_teacher.parameters():
                p.requires_grad_(False)

    def _apply_pixel_masking(self, images, masks):
        """
        Apply masking at pixel level since timm backbones don't natively support token dropping.
        images: [B, 3, H, W]
        masks: [B, N_patches] (boolean, True=masked)
        """
        B, C, H, W = images.shape
        P = self.patch_size
        # Reshape mask to spatial [B, 1, H/P, W/P]
        m = masks.reshape(B, 1, H//P, W//P).float()
        # Upsample mask to pixel level [B, 1, H, W] via nearest neighbor
        m_pixel = F.interpolate(m, size=(H, W), mode='nearest')

        # Apply mask: masked pixels -> 0 (or mean, here we use 0)
        # iBOT logic: student sees masked image
        return images * (1 - m_pixel)

    def training_step(self, batch: Dict[str, Any], global_step: int) -> Dict[str, Any]:
        images = batch["images"]

        # input: [Global, Local]
        if isinstance(images, (list, tuple)):
            global_crops = images[0]                # [B_global, 3, 224, 224]
            local_crops = images[1] if len(images) > 1 else None  # [B_local, 3, 96, 96] or None
        else:
            global_crops = images
            local_crops = None

        B_global = global_crops.shape[0]
        device = global_crops.device

        # 1) masked patches for iBOT (only on global)
        masks = self.mask_gen(B_global, device)  # [B_global, N_patches] with N_patches = (224/16)^2 = 196

        # 2) Update teacher with EMA
        self._update_teacher()

        # ------------------------------------------------------------------
        # 3) TEACHER: global features + patch tokens (without CLS)
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Global Features (for DINO CLS loss)
            tea_feat_g = self.tea.forward_global(global_crops)   # [B_global, D]

            # Token (ViT: include CLS + patch) → patch only for iBOT/Gram
            tea_tokens_all = self.tea.forward_tokens(global_crops)  # [B_global, T, D] con T=197 per ViT-S/16
            if tea_tokens_all.shape[1] == self.mask_gen.num_patches + 1:
                # ViT-style: primo token = CLS, remain = patch
                tea_tokens_p = tea_tokens_all[:, 1:, :]   # [B_global, 196, D]
            else:
                # ResNet or other backbone without CLS
                tea_tokens_p = tea_tokens_all            # [B_global, N_patches_qualunque, D]

            # ---- DINO teacher (CLS head) ----
            tea_cls_out = self.dino_head_tea(tea_feat_g)  # [B_global, K]
            tea_cls_probs = self.dino_loss_fn.sinkhorn_knopp_teacher(
                tea_cls_out, teacher_temp=self.teacher_temp
            )  # [B_global, K]

            # ---- iBOT teacher (patch head) ----
            tea_ibot_out = self.ibot_head_tea(tea_tokens_p)  # [B_global, N_patches, K]
            flat_ibot_out = tea_ibot_out.flatten(0, 1)       # [B_global * N_patches, K]

            tea_ibot_probs = self.ibot_loss_fn.sinkhorn_knopp_teacher(
                flat_ibot_out, teacher_temp=self.teacher_temp
            ).view(B_global, -1, tea_ibot_out.shape[-1])     # [B_global, N_patches, K]

        # ------------------------------------------------------------------
        # 4) STUDENT: global features (masked + local) + patch tokens (masked)
        # ------------------------------------------------------------------
        # 4a) Applied mask on global for iBOT (zeroing pixels)
        masked_global = self._apply_pixel_masking(global_crops, masks)  # [B_global, 3, 224, 224]

        # 4b) Global features student (masked global + local)
        stu_feat_g = self.stu.forward_global(masked_global)  # [B_global, D]

        if local_crops is not None:
            stu_feat_l = self.stu.forward_global(local_crops)  # [B_local, D]
            stu_feat_all = torch.cat([stu_feat_g, stu_feat_l], dim=0)  # [B_global + B_local, D]
        else:
            stu_feat_l = None
            stu_feat_all = stu_feat_g  # [B_global, D]

        # 4c) Token patch student for iBOT/Gram (only global masked)
        stu_tokens_all = self.stu.forward_tokens(masked_global)  # [B_global, T, D]
        if stu_tokens_all.shape[1] == self.mask_gen.num_patches + 1:
            stu_tokens_p = stu_tokens_all[:, 1:, :]  # [B_global, N_patches, D] → 196
        else:
            stu_tokens_p = stu_tokens_all            # [B_global, N_patches_qualunque, D]

        # 4d) Head DINO (CLS) + head iBOT (patch)
        stu_cls_out = self.dino_head(stu_feat_all)      # [B_global + B_local, K]
        stu_ibot_out = self.ibot_head(stu_tokens_p)     # [B_global, N_patches, K]

        # ------------------------------------------------------------------
        # 5) LOSS DINO (global + local with average teacher) 
        # ------------------------------------------------------------------
        loss_dict: Dict[str, float] = {}

        # Student logits 
        stu_cls_g = stu_cls_out[:B_global]                 # [B_global, K]
        stu_cls_l = stu_cls_out[B_global:] if stu_feat_l is not None else None

        # 5a) Global-to-Global 
        loss_dino = self.dino_loss_fn(
            stu_cls_g.unsqueeze(0),        # [1, B_global, K]
            tea_cls_probs.unsqueeze(0),    # [1, B_global, K]
        )

        # 5b) Local-to-Global: use average teacher as target for local
        if stu_cls_l is not None and stu_cls_l.numel() > 0:
            # tea_cls_probs: [B_global, K] → average over batch
            tea_mean = tea_cls_probs.mean(dim=0, keepdim=True)       # [1, K]
            tea_mean_exp = tea_mean.expand(stu_cls_l.shape[0], -1)   # [B_local, K]

            loss_dino_local = self.dino_loss_fn(
                stu_cls_l.unsqueeze(0),      # [1, B_local, K]
                tea_mean_exp.unsqueeze(0),   # [1, B_local, K]
            )
            loss_dino = 0.5 * (loss_dino + loss_dino_local)

        loss_total = self.w_dino * loss_dino
        loss_dict["dino_loss"] = float(loss_dino.detach())

        # ------------------------------------------------------------------
        # 6) LOSS iBOT (patch) – ONLY PATCH TOKENS, SHAPE 
        # ------------------------------------------------------------------
        # masks: [B_global, N_patches] (bool) → same dim of stu_ibot_out / tea_ibot_probs
        student_masks = masks  # [B_global, N_patches], True = masked

        loss_ibot = self.ibot_loss_fn(
            stu_ibot_out,           # [B_global, N_patches, K]
            tea_ibot_probs,         # [B_global, N_patches, K]
            student_masks_flat=student_masks,  # [B_global, N_patches]
        )
        loss_total += self.w_ibot * loss_ibot
        loss_dict["ibot_loss"] = float(loss_ibot.detach())

        # ------------------------------------------------------------------
        # 7) KoLeo Loss on global features (CLS)
        # ------------------------------------------------------------------
        loss_koleo = self.koleo_loss_fn(stu_feat_all)  # [B_global(+local), D]
        loss_total += self.w_koleo * loss_koleo
        loss_dict["koleo_loss"] = float(loss_koleo.detach())

        # ------------------------------------------------------------------
        # 8) Gram Loss on patch tokens (teacher vs student)
        # ------------------------------------------------------------------
        gram_active = False
        if self.w_gram > 0.0 and self.total_steps:
            gram_active = global_step >= int(self.gram_start_frac * self.total_steps)

        if gram_active:
            self._ensure_gram_teacher()
            if global_step % self.gram_teacher_update_every == 0:
                self.gram_teacher.load_state_dict(self.tea.state_dict())

            with torch.no_grad():
                tea_gram_tokens_all = self.gram_teacher.forward_tokens(global_crops)  # [B_global, T, D]
                if tea_gram_tokens_all.shape[1] == self.mask_gen.num_patches + 1:
                    tea_gram_tokens = tea_gram_tokens_all[:, 1:, :]   # [B_global, N_patches, D]
                else:
                    tea_gram_tokens = tea_gram_tokens_all             # [B_global, N_patches_qualunque, D]

            # Use the same patch tokens of the student used for iBOT
            stu_gram_tokens = stu_tokens_p  # [B_global, N_patches, D]

            # If size mismatch (e.g., different #patches), interpolate teacher tokens
            if stu_gram_tokens.shape[1] != tea_gram_tokens.shape[1]:
                Dim = stu_gram_tokens.shape[-1]
                Side_S = int(math.sqrt(stu_gram_tokens.shape[1]))
                Side_T = int(math.sqrt(tea_gram_tokens.shape[1]))

                tea_reshaped = tea_gram_tokens.transpose(1, 2).reshape(B_global, Dim, Side_T, Side_T)
                tea_interp = F.interpolate(
                    tea_reshaped, size=(Side_S, Side_S), mode="bicubic", align_corners=False
                )
                tea_gram_tokens = tea_interp.flatten(2).transpose(1, 2)

            loss_gram = self.gram_loss_fn(stu_gram_tokens, tea_gram_tokens)
            loss_total += self.w_gram * loss_gram
            loss_dict["gram_loss"] = float(loss_gram.detach())

        # ------------------------------------------------------------------
        return {
            "loss_total": loss_total,
            "loss_components": loss_dict,
        }
