#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.trainer.backbones import resolve_backbone_from_model_cfg, get_backbone


class IJEPA(nn.Module):
    """
    I-JEPA adattato alla tua pipeline:
    - Teacher: vede tutta l'immagine (full tokens).
    - Student: vede SOLO i token di contesto (masking prima dei blocchi ViT).
    - Loss pesata con maschera di tessuto (ignora patch di puro sfondo).
    """

    def __init__(
        self,
        backbone_name: str,
        backbone_params: Dict[str, Any],
        predictor_depth: int = 6,
        predictor_embed_dim: int = 384,
        predictor_num_heads: int = 12,
        num_target_masks: int = 4,
        target_scale_range: Tuple[float, float] = (0.15, 0.2),
        context_scale_range: Tuple[float, float] = (0.85, 1.0),
        ema_decay: float = 0.996,
        img_size: int = 224,
        patch_size: int = 16,
        background_std_threshold: float = 0.02,
    ):
        super().__init__()

        # 1. Student (Context Encoder)
        self.stu = get_backbone(backbone_name, pretrained=False, **backbone_params)
        self.embed_dim = self.stu.out_dim

        # Richiede ViTBackbone (con .vit)
        if not hasattr(self.stu, "vit"):
            raise TypeError(
                f"IJEPA requires a ViTBackbone with attribute 'vit', "
                f"got {type(self.stu).__name__}"
            )

        # 2. Teacher (Target Encoder)
        self.tea = copy.deepcopy(self.stu)
        self.tea.eval()
        for p in self.tea.parameters():
            p.requires_grad = False

        # 3. Predictor
        self.predictor = IJEPA_Predictor(
            input_dim=self.embed_dim,
            depth=predictor_depth,
            embed_dim=predictor_embed_dim,
            num_heads=predictor_num_heads,
        )

        # 4. Positional Embeddings per il predictor
        self.num_patches = (img_size // patch_size) ** 2
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, predictor_embed_dim)
        )
        torch.nn.init.trunc_normal_(self.predictor_pos_embed, std=0.02)

        # Parametri vari
        self.num_target_masks = num_target_masks
        self.target_scale_range = target_scale_range
        self.context_scale_range = context_scale_range
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.ema_decay = ema_decay
        self.background_std_threshold = background_std_threshold

    # ------------------------------------------------------------------ factory da config
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> IJEPA:
        model_cfg = cfg["model"]
        ssl_cfg = model_cfg.get("ssl", {})
        backbone_name, backbone_opts = resolve_backbone_from_model_cfg(model_cfg)
        ijepa_cfg = ssl_cfg.get("i_jepa", {})
        target_scale_range = tuple(ijepa_cfg.get("target_scale_range", (0.15, 0.2)))
        context_scale_range = tuple(ijepa_cfg.get("context_scale_range", (0.85, 1.0)))
        img_size = (
            cfg.get("data", {})
            .get("webdataset", {})
            .get("transform_train", {})
            .get("resize", 224)
        )
        if isinstance(img_size, list):
            img_size = img_size[0]

        return cls(
            backbone_name=backbone_name,
            backbone_params=backbone_opts,
            predictor_depth=ijepa_cfg.get("predictor_depth", 6),
            predictor_embed_dim=ijepa_cfg.get("predictor_embed_dim", 384),
            predictor_num_heads=ijepa_cfg.get("predictor_num_heads", 6),
            num_target_masks=ijepa_cfg.get("num_target_masks", 4),
            target_scale_range=target_scale_range,
            context_scale_range=context_scale_range,
            ema_decay=ssl_cfg.get("ema_m", 0.996),
            img_size=int(img_size),
            patch_size=backbone_opts.get("patch_size", 16),
            background_std_threshold=ijepa_cfg.get(
                "background_std_threshold", 0.02
            ),
        )

    # ------------------------------------------------------------------ EMA teacher
    @torch.no_grad()
    def update_teacher(self) -> None:
        for param_q, param_k in zip(self.stu.parameters(), self.tea.parameters()):
            param_k.data.mul_(self.ema_decay).add_(
                (1.0 - self.ema_decay) * param_q.data
            )

    # ------------------------------------------------------------------ API trainer
    def training_step(self, batch: Dict[str, Any], global_step: int) -> Dict[str, Any]:
        images = batch["images"]
        if isinstance(images, list):  # compat con altri loader
            images = images[0]
        metrics = self(images)
        return {
            "loss_total": metrics["loss"],
            "loss_components": metrics,
        }

    # ------------------------------------------------------------------ maschera tessuto vs sfondo
    def _get_tissue_mask(self, images: torch.Tensor) -> torch.Tensor:
        """
        Ritorna una maschera [B, Num_Patches] = 1 se il patch contiene tessuto, 0 se sfondo.
        """
        B, C, H, W = images.shape
        P = self.patch_size

        patches = images.unfold(2, P, P).unfold(3, P, P)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            B, self.num_patches, -1
        )
        patch_std = patches.std(dim=-1)
        tissue_mask = (patch_std > self.background_std_threshold).float()
        return tissue_mask  # [B, N]

    # ------------------------------------------------------------------ forward student mascherato
    def _forward_masked_student(
        self, images: torch.Tensor, context_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward dello studente SOLO sui patch di contesto.
        - images: [B, 3, H, W]
        - context_indices: [B, N_ctx] indici di patch da tenere (0..N_patches-1)
        Ritorna: [B, N_ctx, D]
        """
        vit = self.stu.vit
        B = images.shape[0]

        # 1. Patch embedding (timm gestisce già forma corretta)
        x = vit.patch_embed(images)  # tipicamente [B, N, D]

        # 2. Positional embedding + (eventuale) CLS + pos_drop
        # usa direttamente la logica interna di timm (gestisce anche resize dinamici)
        if hasattr(vit, "_pos_embed"):
            x = vit._pos_embed(x)
        else:
            # fallback ultra difensivo (non dovrebbe servire per ViT timm)
            pos_embed = vit.pos_embed
            if getattr(vit, "cls_token", None) is not None:
                cls_token = vit.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_token, x), dim=1)
                if pos_embed is not None:
                    x = x + pos_embed
            elif pos_embed is not None:
                x = x + pos_embed

        if hasattr(vit, "pos_drop"):
            x = vit.pos_drop(x)

        # 3. Separa CLS / patch tokens
        has_cls = getattr(vit, "cls_token", None) is not None
        if has_cls:
            cls_tok = x[:, :1, :]       # [B, 1, D]
            patch_tok = x[:, 1:, :]     # [B, N, D]
        else:
            cls_tok = None
            patch_tok = x               # [B, N, D]

        # 4. Mascheramento via gather (teniamo solo i token di contesto)
        N_ctx = context_indices.shape[1]
        D = patch_tok.shape[-1]
        gather_idx = context_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, N_ctx, D]
        ctx_tok = torch.gather(patch_tok, dim=1, index=gather_idx)    # [B, N_ctx, D]

        # 5. Ricostruisci sequenza da dare ai blocchi
        if has_cls:
            x_ctx = torch.cat([cls_tok, ctx_tok], dim=1)  # [B, 1+N_ctx, D]
        else:
            x_ctx = ctx_tok  # [B, N_ctx, D]

        # 6. Transformer blocks + norm
        for blk in vit.blocks:
            x_ctx = blk(x_ctx)
        if hasattr(vit, "norm") and vit.norm is not None:
            x_ctx = vit.norm(x_ctx)

        # 7. Rimuovi CLS (ritorniamo solo patch di contesto)
        if has_cls:
            x_ctx = x_ctx[:, 1:, :]

        return x_ctx  # [B, N_ctx, D]

    # ------------------------------------------------------------------ forward JEPA
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = images.device
        B = images.shape[0]

        if self.training:
            self.update_teacher()
            self.tea.eval()

        # 1. Maschera tessuto
        tissue_mask = self._get_tissue_mask(images)  # [B, N_patches]

        # 2. Maschere di contesto e target (indici per-sample)
        context_indices, target_indices_list = self._generate_masks(B, device)

        # 3. Teacher: full image tokens (senza leakage)
        with torch.no_grad():
            full_teacher_tokens = self.tea.forward_tokens(images)  # [B, T, D]
            full_teacher_tokens = F.layer_norm(
                full_teacher_tokens, (full_teacher_tokens.size(-1),)
            )
            # Gestione CLS vs no-CLS
            if full_teacher_tokens.shape[1] == self.num_patches + 1:
                full_teacher_tokens = full_teacher_tokens[:, 1:, :]
            elif full_teacher_tokens.shape[1] != self.num_patches:
                raise ValueError(
                    f"Unexpected teacher tokens: {full_teacher_tokens.shape[1]} "
                    f"(expected {self.num_patches} or {self.num_patches + 1})"
                )

        # 4. Student: masked context forward
        context_tokens = self._forward_masked_student(images, context_indices)
        # context_tokens: [B, N_ctx, D]

        # 5. Predictor loop con filtro tessuto
        loss = 0.0
        valid_targets_count = 0
        pred_tokens: Optional[torch.Tensor] = None  # per dummy loss

        for target_indices in target_indices_list:
            # target_indices: [B, N_tgt]
            D = full_teacher_tokens.shape[-1]
            gather_idx = target_indices.unsqueeze(-1).expand(-1, -1, D)
            target_tokens = torch.gather(
                full_teacher_tokens, dim=1, index=gather_idx
            )  # [B, N_tgt, D]

            # Positional embedding per il predictor
            pred_D = self.predictor.embed_dim
            pos_idx = target_indices.unsqueeze(-1).expand(-1, -1, pred_D)
            pos_embed_expand = self.predictor_pos_embed.expand(B, -1, -1)
            curr_target_pos = torch.gather(
                pos_embed_expand, dim=1, index=pos_idx
            )  # [B, N_tgt, pred_D]

            # Predizione
            pred_tokens = self.predictor(context_tokens, curr_target_pos)  # [B, N_tgt, D]

            # Filtro tessuto: gather sulla tissue_mask
            curr_tissue_mask = torch.gather(
                tissue_mask, dim=1, index=target_indices
            )  # [B, N_tgt]

            mse_per_token = F.mse_loss(
                pred_tokens, target_tokens, reduction="none"
            ).mean(dim=-1)  # [B, N_tgt]

            masked_loss = mse_per_token * curr_tissue_mask  # [B, N_tgt]
            num_tissue_tokens = curr_tissue_mask.sum()

            if num_tissue_tokens > 0:
                loss += masked_loss.sum() / num_tissue_tokens
                valid_targets_count += 1

        if valid_targets_count > 0:
            loss = loss / valid_targets_count
        else:
            # immagine tutta sfondo: dummy loss per mantenere il grafo
            if pred_tokens is None:
                loss = (full_teacher_tokens * 0.0).sum()
            else:
                loss = (pred_tokens * 0.0).sum()

        return {"loss": loss, "ssl_loss": loss}

    # ------------------------------------------------------------------ maschere target/context per-sample
    def _generate_masks(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Genera:
        - context_indices: [B, N_ctx]
        - target_indices_list: lista di num_target_masks tensori [B, N_tgt]

        Strategia:
        - stessa dimensione dei blocchi (scale/aspect) per tutto il batch,
          ma posizioni top/left diverse per ogni sample.
        """
        H = W = self.grid_size
        N = self.num_patches

        # ---- Target masks
        union_targets = torch.zeros(
            (batch_size, N), dtype=torch.bool, device=device
        )
        target_indices_list: List[torch.Tensor] = []

        for _ in range(self.num_target_masks):
            min_s, max_s = self.target_scale_range
            scale = min_s + torch.rand(1, device=device).item() * (max_s - min_s)
            aspect = 0.75 + torch.rand(1, device=device).item() * 0.75
            area = max(1, int(N * scale))

            bh = max(1, min(int(math.sqrt(area * aspect)), H))
            bw = max(1, min(int(math.sqrt(area / aspect)), W))

            tops = torch.randint(0, H - bh + 1, (batch_size,), device=device)
            lefts = torch.randint(0, W - bw + 1, (batch_size,), device=device)

            y = torch.arange(bh, device=device).view(1, -1, 1)  # [1, bh, 1]
            x = torch.arange(bw, device=device).view(1, 1, -1)  # [1, 1, bw]

            ys = tops.view(batch_size, 1, 1) + y  # [B, bh, 1]
            xs = lefts.view(batch_size, 1, 1) + x  # [B, 1, bw]

            idx = (ys * W + xs).view(batch_size, -1)  # [B, bh*bw]
            target_indices_list.append(idx)

            union_targets.scatter_(1, idx, True)

        # ---- Context mask
        min_s, max_s = self.context_scale_range
        scale = min_s + torch.rand(1, device=device).item() * (max_s - min_s)
        area = max(1, int(N * scale))
        aspect = 1.0

        ch = max(1, min(int(math.sqrt(area * aspect)), H))
        cw = max(1, min(int(math.sqrt(area / aspect)), W))

        tops = torch.randint(0, H - ch + 1, (batch_size,), device=device)
        lefts = torch.randint(0, W - cw + 1, (batch_size,), device=device)

        y = torch.arange(ch, device=device).view(1, -1, 1)
        x = torch.arange(cw, device=device).view(1, 1, -1)

        ys = tops.view(batch_size, 1, 1) + y
        xs = lefts.view(batch_size, 1, 1) + x

        ctx_idx_all = (ys * W + xs).view(batch_size, -1)  # [B, ch*cw]

        # Costruiamo una maschera booleana dei possibili context
        possible_mask = torch.zeros(
            (batch_size, N), dtype=torch.bool, device=device
        )
        possible_mask.scatter_(1, ctx_idx_all, True)

        # Evita overlap con target
        valid_mask = possible_mask & (~union_targets)

        num_valid = valid_mask.sum(dim=1)  # [B]
        min_valid = int(num_valid.min().item())

        if min_valid <= 0:
            # fallback: usiamo ctx_idx_all (può contenere overlap)
            context_indices = ctx_idx_all
        else:
            # Prendiamo i primi min_valid indici validi per ciascun sample
            scores = valid_mask.float()  # True=1, False=0
            sorted_idx = torch.argsort(scores, dim=1, descending=True)
            context_indices = sorted_idx[:, :min_valid]  # [B, min_valid]

        return context_indices, target_indices_list


class IJEPA_Predictor(nn.Module):
    def __init__(self, input_dim: int, depth: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_in = nn.Linear(input_dim, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * 4),
            dropout=0.0,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.proj_out = nn.Linear(embed_dim, input_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, context_tokens: torch.Tensor, target_pos_embeds: torch.Tensor
    ) -> torch.Tensor:
        # context_tokens: [B, N_ctx, D]
        # target_pos_embeds: [B, N_tgt, D_pred]
        x = self.proj_in(context_tokens)  # [B, N_ctx, D_pred]
        full_seq = torch.cat([x, target_pos_embeds], dim=1)
        out = self.blocks(full_seq)
        out = self.norm(out)
        target_out = out[:, context_tokens.shape[1] :, :]
        return self.proj_out(target_out)  # [B, N_tgt, D]
