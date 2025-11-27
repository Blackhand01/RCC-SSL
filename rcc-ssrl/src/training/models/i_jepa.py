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
        # Nuovo parametro per il filtraggio sfondo
        background_std_threshold: float = 0.02 
    ):
        super().__init__()
        
        # 1. Student (Context Encoder)
        self.stu = get_backbone(backbone_name, pretrained=False, **backbone_params)
        self.embed_dim = self.stu.out_dim
        
        # 2. Teacher (Target Encoder)
        self.tea = copy.deepcopy(self.stu)
        for p in self.tea.parameters():
            p.requires_grad = False
            
        # 3. Predictor
        self.predictor = IJEPA_Predictor(
            input_dim=self.embed_dim,
            depth=predictor_depth,
            embed_dim=predictor_embed_dim,
            num_heads=predictor_num_heads
        )
        
        # 4. Positional Embeddings
        self.num_patches = (img_size // patch_size) ** 2
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, predictor_embed_dim)
        )
        torch.nn.init.trunc_normal_(self.predictor_pos_embed, std=0.02)
        
        # Parametri
        self.num_target_masks = num_target_masks
        self.target_scale_range = target_scale_range
        self.context_scale_range = context_scale_range
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.ema_decay = ema_decay
        
        # Soglia per considerare un patch "vuoto" (solo sfondo)
        self.background_std_threshold = background_std_threshold

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> IJEPA:
        model_cfg = cfg["model"]
        ssl_cfg = model_cfg.get("ssl", {})
        backbone_name, backbone_opts = resolve_backbone_from_model_cfg(model_cfg)
        ijepa_cfg = ssl_cfg.get("i_jepa", {})
        
        img_size = cfg.get("data", {}).get("webdataset", {}).get("transform_train", {}).get("resize", 224)
        if isinstance(img_size, list): img_size = img_size[0]
        
        return cls(
            backbone_name=backbone_name,
            backbone_params=backbone_opts,
            predictor_depth=ijepa_cfg.get("predictor_depth", 6),
            predictor_embed_dim=ijepa_cfg.get("predictor_embed_dim", 384),
            predictor_num_heads=ijepa_cfg.get("predictor_num_heads", 6),
            num_target_masks=ijepa_cfg.get("num_target_masks", 4),
            ema_decay=ssl_cfg.get("ema_m", 0.996),
            img_size=int(img_size),
            patch_size=backbone_opts.get("patch_size", 16),
            # Puoi aggiungere questo parametro al config yaml sotto i_jepa se vuoi tunarlo
            background_std_threshold=ijepa_cfg.get("background_std_threshold", 0.02)
        )

    @torch.no_grad()
    def update_teacher(self):
        for param_q, param_k in zip(self.stu.parameters(), self.tea.parameters()):
            param_k.data.mul_(self.ema_decay).add_((1 - self.ema_decay) * param_q.data)
            
    def training_step(self, batch: Dict[str, Any], global_step: int) -> Dict[str, Any]:
        images = batch["images"]
        if isinstance(images, list):
            images = images[0]
        metrics = self(images)
        return {
            "loss_total": metrics["loss"],
            "loss_components": metrics
        }

    def _get_tissue_mask(self, images: torch.Tensor) -> torch.Tensor:
        """
        Analizza i pixel dell'immagine per creare una maschera che vale:
        1.0 se il patch contiene tessuto (informazione utile)
        0.0 se il patch è sfondo piatto (bianco/vuoto)
        
        Input: [B, 3, H, W] -> Output: [B, Num_Patches]
        """
        B, C, H, W = images.shape
        P = self.patch_size
        
        # 1. Dividi in patch [B, 3, H//P, P, W//P, P] -> [B, N, 3*P*P]
        # Unfold manuale efficiente
        patches = images.unfold(2, P, P).unfold(3, P, P) 
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, self.num_patches, -1)
        
        # 2. Calcola Deviazione Standard per ogni patch
        # Se std è basso, il patch è piatto (sfondo bianco o nero uniforme)
        # Se std è alto, c'è texture (tessuto)
        patch_std = patches.std(dim=-1)
        
        # 3. Crea maschera binaria (float per moltiplicazione)
        tissue_mask = (patch_std > self.background_std_threshold).float()
        
        return tissue_mask

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = images.device
        B = images.shape[0]
        
        if self.training:
            self.update_teacher()

        # 1. Calcola Maschera Tessuto vs Sfondo
        tissue_mask = self._get_tissue_mask(images) # [B, N_patches]

        # 2. Generazione Maschere (Context e Target spaziali)
        context_masks, target_masks_list = self._generate_masks(B, device)

        # 3. Forward Teacher (Target Encoder)
        with torch.no_grad():
            full_teacher_tokens = self.tea.forward_tokens(images) 
            full_teacher_tokens = F.layer_norm(full_teacher_tokens, (full_teacher_tokens.size(-1),))
            if full_teacher_tokens.shape[1] == self.num_patches + 1:
                spatial_teacher_tokens = full_teacher_tokens[:, 1:, :]
            else:
                spatial_teacher_tokens = full_teacher_tokens

        # 4. Forward Student (Context Encoder)
        full_student_tokens = self.stu.forward_tokens(images)
        if full_student_tokens.shape[1] == self.num_patches + 1:
            spatial_student_tokens = full_student_tokens[:, 1:, :]
        else:
            spatial_student_tokens = full_student_tokens

        # Selezione Context Tokens
        context_tokens = spatial_student_tokens[context_masks].view(B, -1, self.embed_dim)

        # 5. Predictor Loop con Filtraggio Sfondo
        loss = 0.0
        valid_targets_count = 0 # Per evitare divisione per zero se tutto è sfondo
        
        for i, target_mask in enumerate(target_masks_list):
            # Estrai target token
            target_tokens = spatial_teacher_tokens[target_mask].view(B, -1, self.embed_dim)
            
            # Estrai pos embeddings
            target_pos_embeds = self.predictor_pos_embed.expand(B, -1, -1)
            curr_target_pos = target_pos_embeds[target_mask].view(B, -1, self.predictor.embed_dim)
            
            # Predizione
            pred_tokens = self.predictor(context_tokens, curr_target_pos)
            
            # --- CALCOLO LOSS CON FILTRO TESSUTO ---
            # 1. Recuperiamo la maschera tessuto per QUESTI specifici target
            # [B, N_patches] -> [B, N_patches_kept] (appiattito o view dipendente dalla maschera)
            # Poiché usiamo Batch Unified, possiamo usare .view()
            curr_tissue_mask = tissue_mask[target_mask].view(B, -1) # [B, N_targets_per_batch]
            
            # 2. Calcola MSE element-wise (senza mediare ancora)
            # [B, N_targets, D] -> mean su D -> [B, N_targets]
            mse_per_token = F.mse_loss(pred_tokens, target_tokens, reduction='none').mean(dim=-1)
            
            # 3. Applica filtro: Azzera loss dove è sfondo
            masked_loss = mse_per_token * curr_tissue_mask
            
            # 4. Somma e normalizza solo per i token validi (tessuto)
            num_tissue_tokens = curr_tissue_mask.sum()
            
            if num_tissue_tokens > 0:
                loss += masked_loss.sum() / num_tissue_tokens
                valid_targets_count += 1
            
            # Se num_tissue_tokens è 0 (tutto sfondo), loss += 0 (corretto, saltiamo questo target)

        # Media finale sui blocchi target validi
        if valid_targets_count > 0:
            loss = loss / valid_targets_count
        else:
            # Caso limite: immagine interamente bianca/sfondo -> Loss 0 ma con gradiente
            # Usiamo una dummy loss per non rompere DDP
            loss = (pred_tokens * 0.0).sum()

        return {"loss": loss, "ssl_loss": loss}

    def _generate_masks(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Batch Unified Masking Strategy"""
        h = w = self.grid_size
        
        def _get_block_mask_batch_unified(scale_range):
            min_s, max_s = scale_range
            scale = min_s + torch.rand(1).item() * (max_s - min_s)
            aspect = 0.75 + torch.rand(1).item() * 0.75
            num_patches = int(self.num_patches * scale)
            bh = max(min(int(math.sqrt(num_patches * aspect)), h), 1)
            bw = max(min(int(math.sqrt(num_patches / aspect)), w), 1)
            top = torch.randint(0, h - bh + 1, (1,), device=device).item()
            left = torch.randint(0, w - bw + 1, (1,), device=device).item()
            mask = torch.zeros((h, w), dtype=torch.bool, device=device)
            mask[top:top+bh, left:left+bw] = True
            return mask.view(1, -1).expand(batch_size, -1)

        target_masks = []
        union_target_mask = torch.zeros((batch_size, self.num_patches), dtype=torch.bool, device=device)
        
        for _ in range(self.num_target_masks):
            tm = _get_block_mask_batch_unified(self.target_scale_range)
            target_masks.append(tm)
            union_target_mask = union_target_mask | tm
            
        raw_context_mask = _get_block_mask_batch_unified(self.context_scale_range)
        context_mask = raw_context_mask & (~union_target_mask)
        
        if context_mask[0].sum() == 0:
            context_mask = raw_context_mask
        
        return context_mask, target_masks


class IJEPA_Predictor(nn.Module):
    def __init__(self, input_dim, depth, embed_dim, num_heads):
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
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.proj_out = nn.Linear(embed_dim, input_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context_tokens, target_pos_embeds):
        x = self.proj_in(context_tokens)
        full_seq = torch.cat([x, target_pos_embeds], dim=1)
        out = self.blocks(full_seq)
        out = self.norm(out)
        target_out = out[:, context_tokens.shape[1]:, :]
        return self.proj_out(target_out)