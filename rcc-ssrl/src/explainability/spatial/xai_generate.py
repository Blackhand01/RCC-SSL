#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial XAI on RCC test set (TP/FP/FN selection via predictions.csv).

Generates:
- GradCAM / IG / Occlusion (if enabled and dependencies are available)
- Attention Rollout for ViT (via monkey patching timm Attention blocks).

This script is config-driven and can be:
- run standalone: python xai_generate.py --config CONFIG_PATH
- called programmatically from the orchestrator:
    from explainability.spatial.xai_generate import main as spatial_xai_main
    spatial_xai_main(["--config", str(config_path)])
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402

from explainability.common.eval_utils import (
    setup_logger,
    set_seed,
    build_preprocess,
    tensor_to_pil,
    load_eval_artifacts,
    select_items,
    make_wds_loader_with_keys,
    make_imgfolder_loader,
)
from explainability.common.ssl_linear_loader import SSLLinearClassifier

# Optional dependencies
try:
    import webdataset as wds  # noqa: F401

    HAVE_WDS = True
except Exception:
    HAVE_WDS = False

try:
    from torchcam.methods import GradCAM  # noqa: F401

    HAVE_TCAM = True
except Exception:
    HAVE_TCAM = False

try:
    from captum.attr import IntegratedGradients, Occlusion  # noqa: F401

    HAVE_CAPTUM = True
except Exception:
    HAVE_CAPTUM = False


# -------------------------------------------------------------------------
# Heatmap overlay utilities
# -------------------------------------------------------------------------
def overlay_heatmap(pil_img: Image.Image, heatmap, alpha: float = 0.5) -> Image.Image:
    """Overlay a normalized heatmap on top of a PIL image."""
    heatmap = np.array(heatmap)
    if heatmap.ndim > 2:
        heatmap = np.squeeze(heatmap)
    if heatmap.ndim == 3:
        if heatmap.shape[0] in (1, 3, 4):
            heatmap = heatmap.mean(axis=0)
        else:
            heatmap = heatmap.mean(axis=-1)

    if heatmap.ndim != 2:
        return pil_img.convert("RGBA")

    hmin, hmax = float(heatmap.min()), float(heatmap.max())
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)

    heat_rgba = Image.fromarray((cm.jet(heatmap) * 255).astype(np.uint8)).convert("RGBA")
    base = pil_img.convert("RGBA")
    if heat_rgba.size != base.size:
        heat_rgba = heat_rgba.resize(base.size, Image.BILINEAR)

    return Image.blend(base, heat_rgba, alpha=alpha)


def _parse_maybe_json_or_literal(s):
    import ast

    if isinstance(s, (bytes, bytearray)):
        s = s.decode("utf-8")
    if isinstance(s, str):
        s = s.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            return ast.literal_eval(s)
        except Exception:
            return s
    return s


# -------------------------------------------------------------------------
# ViT Attention Rollout (monkey patching)
# -------------------------------------------------------------------------
class ViTAttentionRollout:
    """
    Compute Attention Rollout for timm ViT models via Monkey Patching.

    It replaces the forward() of attention blocks to manually compute and
    capture the attention matrices, bypassing Flash Attention / SDPA paths
    that hide the weights.
    """

    def __init__(self, model, head_fusion: str = "mean", discard_ratio: float = 0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []
        self.original_forwards = {}  # type: ignore[var-annotated]

    def _manual_attention_forward(self, module, x, attn_mask=None, **kwargs):
        """
        Replacement forward method for timm Attention blocks.
        Replicates standard ViT attention logic but captures the weights.
        """
        B, N, C = x.shape

        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # [B, H, N, D]

        if hasattr(module, "q_norm") and module.q_norm is not None:
            q = module.q_norm(q)
        if hasattr(module, "k_norm") and module.k_norm is not None:
            k = module.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * module.scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)

        self.attentions.append(attn.detach().cpu())

        attn = module.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = module.proj(x)
        x = module.proj_drop(x)
        return x

    def _patch_model(self) -> None:
        self.attentions = []
        self.original_forwards = {}

        for _, m in self.model.named_modules():
            if hasattr(m, "qkv") and hasattr(m, "scale"):
                self.original_forwards[m] = m.forward

                def make_wrapper(mod):
                    def wrapped(*args, **kwargs):
                        if not args:
                            raise RuntimeError(
                                "Attention forward called without positional input tensor."
                            )
                        x = args[0]
                        return self._manual_attention_forward(mod, x, **kwargs)

                    return wrapped

                m.forward = make_wrapper(m)

    def _unpatch_model(self) -> None:
        for m, original in self.original_forwards.items():
            m.forward = original
        self.original_forwards = {}

    def __call__(self, input_tensor: torch.Tensor):
        self._patch_model()
        try:
            with torch.no_grad():
                _ = self.model(input_tensor)
        finally:
            self._unpatch_model()

        if not self.attentions:
            print("[DEBUG] Rollout Error: No attention blocks captured via patching.")
            return None

        all_attn = torch.stack(self.attentions).squeeze(1)  # [L, H, T, T]

        if self.head_fusion == "mean":
            all_attn = torch.mean(all_attn, dim=1)
        elif self.head_fusion == "max":
            all_attn = torch.max(all_attn, dim=1)[0]
        elif self.head_fusion == "min":
            all_attn = torch.min(all_attn, dim=1)[0]

        num_tokens = all_attn.shape[1]
        eye = torch.eye(num_tokens).to(all_attn.device)
        joint_attentions = eye

        for layer_attn in all_attn:
            if self.discard_ratio > 0:
                flat = layer_attn.view(num_tokens, -1)
                val, _ = torch.topk(
                    flat,
                    int(flat.shape[1] * (1 - self.discard_ratio)),
                    dim=1,
                )
                threshold = val[:, -1].unsqueeze(1)
                layer_attn = torch.where(
                    layer_attn >= threshold,
                    layer_attn,
                    torch.zeros_like(layer_attn),
                )

            layer_attn = layer_attn / (layer_attn.sum(dim=-1, keepdims=True) + 1e-9)
            aug_attn = 0.5 * layer_attn + 0.5 * eye
            joint_attentions = torch.matmul(aug_attn, joint_attentions)

        mask = joint_attentions[0, 1:]
        grid_size = int(np.sqrt(mask.shape[0]))
        if grid_size * grid_size != mask.shape[0]:
            if grid_size * grid_size == mask.shape[0] - 1:
                mask = mask[1:]
            else:
                return None

        mask = mask.reshape(grid_size, grid_size).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)
        return mask


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    import yaml

    logger = setup_logger("xai_spatial")

    parser = argparse.ArgumentParser(description="Spatial XAI for SSL RCC model")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(open(args.config, "r"))

    set_seed(int(cfg["experiment"]["seed"]))

    device = torch.device(
        cfg.get("runtime", {}).get("device", "cuda")
        if torch.cuda.is_available()
        else "cpu"
    )

    out_root = (
        Path(cfg["experiment"]["outputs_root"])
        / cfg["model"]["name"]
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"[Spatial XAI] Output dir: {out_root}")

    # Eval artifacts
    y_true, y_pred, conf, keys, meta_rows = load_eval_artifacts(
        cfg["evaluation_inputs"]["eval_run_dir"],
        cfg["evaluation_inputs"]["predictions_csv"],
        cfg["evaluation_inputs"]["logits_npy"],
        logger,
    )

    arch_hint = cfg["model"].get("arch_hint", "ssl_linear").lower()
    if arch_hint != "ssl_linear":
        logger.error("Only arch_hint=ssl_linear is supported in this script.")
        return

    class_order: List[str] = cfg["labels"]["class_order"]
    num_classes = len(class_order)

    # Model
    model = SSLLinearClassifier(
        backbone_name=cfg["model"].get("backbone_name", "resnet50"),
        num_classes=num_classes,
    )
    mb, ub = model.load_backbone_from_ssl(cfg["model"]["ssl_backbone_ckpt"])
    mh, uh = model.load_head_from_probe(cfg["model"]["ssl_head_ckpt"])
    logger.info(
        f"Loaded SSL backbone (missing={mb}, unexpected={ub}) "
        f"and linear head (missing={mh}, unexpected={uh})."
    )

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)

    img_size = int(cfg["data"]["img_size"])
    imagenet_norm = bool(cfg["data"].get("imagenet_norm", False))
    preprocess_fn = build_preprocess(img_size, imagenet_norm)

    if cfg["data"]["backend"].lower() == "webdataset":
        w = cfg["data"]["webdataset"]
        loader = make_wds_loader_with_keys(
            w["test_dir"],
            w["pattern"],
            w["image_key"],
            w["meta_key"],
            preprocess_fn,
            int(cfg["data"]["num_workers"]),
        )
    else:
        _, loader = make_imgfolder_loader(
            cfg["data"]["imagefolder"]["test_dir"],
            preprocess_fn,
            int(cfg["data"]["batch_size"]),
            int(cfg["data"]["num_workers"]),
        )

    n_classes = num_classes

    targets, sel_reasons = select_items(
        y_true,
        y_pred,
        conf,
        keys,
        n_classes,
        cfg["selection"],
        logger,
    )
    target_set = set(targets)
    logger.info(f"[Spatial XAI] Targets selected: {len(target_set)}")

    import time
    t_start = time.time()
    total_targets = len(target_set)
    if total_targets == 0:
        logger.warning("[Spatial XAI] No targets selected, exiting early.")
        return

    methods = set(cfg["xai"]["methods"])
    use_ig = HAVE_CAPTUM and ("ig" in methods)
    use_occ = HAVE_CAPTUM and ("occlusion" in methods)
    has_gradcam = HAVE_TCAM and (("gradcam" in methods) or ("gradcam++" in methods))
    use_rollout = "attn_rollout" in methods

    if use_rollout:
        logger.info("Attention Rollout ENABLED (monkey patching mode).")

    if use_ig:
        ig = IntegratedGradients(model)  # type: ignore[valid-type]
    if use_occ:
        occl = Occlusion(model)  # noqa: F841

    target_layer = None
    if has_gradcam:
        tname = cfg["xai"]["gradcam"]["target_layer"]
        try:
            modules_dict = dict(model.named_modules())
            if tname in modules_dict:
                target_layer = modules_dict[tname]
            else:
                curr = model
                for part in tname.split("."):
                    curr = getattr(curr, part)
                target_layer = curr
        except Exception:
            target_layer = None
        if target_layer is None:
            logger.warning(f"Target layer {tname} not found for GradCAM.")

    index_csv = open(out_root / "index.csv", "w", newline="")
    writer = csv.writer(index_csv)
    writer.writerow(
        [
            "global_idx",
            "wds_key",
            "true",
            "pred",
            "conf",
            "methods",
            "png_paths",
            "selection_reason",
        ]
    )

    produced = 0
    global_idx = 0

    row_by_key = {r["wds_key"]: r for r in (meta_rows or [])} if meta_rows else {}
    idx_by_key = {k: i for i, k in enumerate(keys)} if keys is not None else {}
    class_order = cfg["labels"]["class_order"]

    rollout_instance = None
    if use_rollout:
        try:
            rollout_instance = ViTAttentionRollout(
                model.backbone.model,
                discard_ratio=cfg["xai"]["attn_rollout"]["discard_ratio"],
            )
        except AttributeError as e:
            logger.error(f"Cannot initialize Rollout: model structure error. {e}")
            use_rollout = False

    for batch in loader:
        if cfg["data"]["backend"].lower() == "webdataset":
            img_t, meta_any, key = batch
            if keys is not None and key not in target_set:
                continue

            meta = meta_any if isinstance(meta_any, dict) else {}
            if isinstance(meta_any, (str, bytes)):
                meta = _parse_maybe_json_or_literal(meta_any) or {}

            row = row_by_key.get(key, {})

            true_id = int(row.get("y_true", -1)) if row else -1
            true_txt = (
                class_order[true_id]
                if 0 <= true_id < n_classes
                else meta.get("class_label", "")
            )

            pred_id = int(row.get("y_pred", -1)) if row else -1
            pred_txt = (
                class_order[pred_id] if 0 <= pred_id < n_classes else str(pred_id)
            )

            idx_eval = idx_by_key.get(key, None)
            if conf is not None and idx_eval is not None:
                prob = float(conf[idx_eval])
            else:
                prob = float("nan")

            sel_reason_list = sel_reasons.get(key, []) if keys is not None else []
            sel_reason_str = "|".join(sel_reason_list) if sel_reason_list else ""
        else:
            img_t, lbl = batch
            key = None
            true_id = int(lbl.item())
            true_txt = class_order[true_id]
            pred_id = -1
            pred_txt = ""
            prob = float("nan")
            sel_reason_str = ""

        x = img_t.to(device).unsqueeze(0) if img_t.ndim == 3 else img_t.to(device)
        x.requires_grad_(True)

        logits = model(x)

        if pred_id >= 0:
            target_class = pred_id
        else:
            target_class = int(torch.argmax(logits, dim=1).item())

        out_dir = out_root / f"idx_{global_idx:07d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        pil_in = tensor_to_pil(x[0], imagenet_norm=imagenet_norm)
        orig_path = out_dir / "input.png"
        pil_in.save(orig_path)

        if sel_reason_str:
            (out_dir / "selection_reason.txt").write_text(sel_reason_str + "\n")

        png_paths, used = [], []

        if use_ig:
            try:
                attr = ig.attribute(
                    x,
                    target=target_class,
                    n_steps=cfg["xai"]["ig"]["steps"],
                )
                heat = (
                    attr.abs()
                    .mean(dim=1)
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )
                over = overlay_heatmap(pil_in, heat, alpha=0.5)
                path = out_dir / "ig.png"
                over.save(path)
                png_paths.append(str(path))
                used.append("ig")
            except Exception as e:
                logger.warning(f"IG failed: {e}")

        if has_gradcam and target_layer is not None:
            try:
                cam_method = GradCAM(model, target_layer=target_layer)
                with torch.enable_grad():
                    model.zero_grad()
                    sc = model(x)
                maps = cam_method(class_idx=target_class, scores=sc)
                raw_map = maps[0].detach().cpu()
                if raw_map.ndim == 1:
                    seq_len = raw_map.numel()
                    grid = int(np.sqrt(seq_len))
                    if grid * grid == seq_len:
                        raw_map = raw_map.reshape(grid, grid)
                    elif grid * grid == seq_len - 1:
                        raw_map = raw_map[1:].reshape(grid, grid)
                heat = raw_map.numpy()
                over = overlay_heatmap(pil_in, heat, alpha=0.5)
                path = out_dir / "gradcam.png"
                over.save(path)
                png_paths.append(str(path))
                used.append("gradcam")
                for h in cam_method.hook_handles:
                    h.remove()
            except Exception as e:
                logger.warning(f"GradCAM failed: {e}")

        if use_rollout and rollout_instance is not None:
            try:
                mask = rollout_instance(x)
                if mask is not None:
                    over = overlay_heatmap(pil_in, mask, alpha=0.6)
                    path = out_dir / "attn_rollout.png"
                    over.save(path)
                    png_paths.append(str(path))
                    used.append("rollout")
            except Exception as e:
                logger.warning(f"Rollout failed: {e}")

        writer.writerow(
            [
                global_idx,
                key,
                true_txt,
                pred_txt,
                prob,
                ";".join(used),
                ";".join(png_paths),
                sel_reason_str,
            ]
        )


        global_idx += 1
        produced += 1

        # progress + ETA
        if produced % 10 == 0 or produced == total_targets:
            elapsed = time.time() - t_start
            avg_per_item = elapsed / max(1, produced)
            remaining = total_targets - produced
            eta = remaining * avg_per_item
            logger.info(
                f"[PROGRESS] Spatial XAI: {produced}/{total_targets} "
                f"({100.0*produced/total_targets:.1f}%), "
                f"elapsed={elapsed/60:.1f} min, "
                f"avg={avg_per_item:.2f} s/item, "
                f"ETA~{eta/60:.1f} min"
            )

        if keys is not None and produced >= len(targets):
            break

    index_csv.close()
    total_elapsed = time.time() - t_start
    logger.info(
        f"[Spatial XAI] Done. Produced {produced} cases in {total_elapsed/60:.1f} min."
    )


if __name__ == "__main__":
    main()
