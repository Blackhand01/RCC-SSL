attention_rollout_job.sbatch codice <<
#!/usr/bin/env bash
#SBATCH --job-name=attn_rollout
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/xai/attn_rollout.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/xai/attn_rollout.%j.err
#SBATCH -p gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --exclude=compute-5-14,compute-5-11,compute-3-12

set -euo pipefail

PROJECT_ROOT="/home/mla_group_01/rcc-ssrl"
mkdir -p "${PROJECT_ROOT}/src/logs/xai" || true

module purge || true

# -----------------------------
# REQUIRED: eval + ckpts (i-JEPA abl01)
# -----------------------------
ABL_ROOT="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/models/exp_20251109_181534_i_jepa/exp_i_jepa_abl01"

EVAL_RUN_DIR="${EVAL_RUN_DIR:-${ABL_ROOT}/eval/i_jepa_ssl_linear_best/20251113_120919}"
SSL_BACKBONE_CKPT="${SSL_BACKBONE_CKPT:-${ABL_ROOT}/checkpoints/i_jepa__ssl_best.pt}"
SSL_HEAD_CKPT="${SSL_HEAD_CKPT:-${ABL_ROOT}/checkpoints/i_jepa__ssl_linear_best.pt}"

# -----------------------------
# DATA (WebDataset test)
# -----------------------------
DATA_BACKEND="${DATA_BACKEND:-webdataset}"

TEST_WDS_DIR="${TEST_WDS_DIR:-/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test}"
WDS_PATTERN="${WDS_PATTERN:-*.tar}"
# FIX: le chiavi reali nel tar sono "img.jpg" e "meta.json" (fallback compat: "jpg"/"json")
WDS_IMAGE_KEY="${WDS_IMAGE_KEY:-img.jpg;jpg}"
WDS_META_KEY="${WDS_META_KEY:-meta.json;json}"

# -----------------------------
# MODEL
# -----------------------------
MODEL_NAME="${MODEL_NAME:-i_jepa_ssl_vit}"
BACKBONE_NAME="${BACKBONE_NAME:-vit_small_patch16_224}"

# -----------------------------
# OUTPUTS
# -----------------------------
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs/xai_spatial}"
RUN_ID="${RUN_ID:-i_jepa_abl01_$(date +%Y%m%d_%H%M%S)}"

# -----------------------------
# XAI controls
# -----------------------------
XAI_METHODS="${XAI_METHODS:-attn_rollout}"
ATNN_DISCARD_RATIO="${ATNN_DISCARD_RATIO:-0.9}"

# subset vs full-test
FULL_TEST="${FULL_TEST:-0}"
TOPK_TP="${TOPK_TP:-6}"
TOPK_FN="${TOPK_FN:-6}"
TOPK_FP="${TOPK_FP:-6}"
TOPK_LOWCONF="${TOPK_LOWCONF:-20}"

# perf/runtime
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMG_SIZE="${IMG_SIZE:-224}"
IMAGENET_NORM="${IMAGENET_NORM:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-1337}"

# venv (opzionale)
VENV_PATH="${VENV_PATH:-/home/mla_group_01/rcc-ssrl/.venvs/xai}"
if [[ -n "${VENV_PATH}" && -d "${VENV_PATH}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

# -----------------------------
# Sanity checks (fail fast)
# -----------------------------
if [[ ! -d "${EVAL_RUN_DIR}" ]]; then
  echo "[FATAL] EVAL_RUN_DIR non esiste: ${EVAL_RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_RUN_DIR}/predictions.csv" ]] || [[ ! -f "${EVAL_RUN_DIR}/logits_test.npy" ]]; then
  echo "[FATAL] EVAL_RUN_DIR deve contenere predictions.csv e logits_test.npy: ${EVAL_RUN_DIR}" >&2
  ls -la "${EVAL_RUN_DIR}" || true
  exit 3
fi
if [[ ! -f "${SSL_BACKBONE_CKPT}" ]]; then
  echo "[FATAL] SSL_BACKBONE_CKPT non trovato: ${SSL_BACKBONE_CKPT}" >&2
  exit 4
fi
if [[ ! -f "${SSL_HEAD_CKPT}" ]]; then
  echo "[FATAL] SSL_HEAD_CKPT non trovato: ${SSL_HEAD_CKPT}" >&2
  exit 5
fi
if [[ "${DATA_BACKEND}" != "webdataset" ]]; then
  echo "[FATAL] Questo job e' configurato per webdataset. DATA_BACKEND=${DATA_BACKEND}" >&2
  exit 6
fi
if [[ ! -d "${TEST_WDS_DIR}" ]]; then
  echo "[FATAL] TEST_WDS_DIR non esiste: ${TEST_WDS_DIR}" >&2
  exit 7
fi

# -----------------------------
# Run
# -----------------------------
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

export PROJECT_ROOT VENV_PATH
export EVAL_RUN_DIR SSL_BACKBONE_CKPT SSL_HEAD_CKPT
export DATA_BACKEND TEST_WDS_DIR WDS_PATTERN WDS_IMAGE_KEY WDS_META_KEY
export MODEL_NAME BACKBONE_NAME
export OUTPUTS_ROOT RUN_ID
export XAI_METHODS ATNN_DISCARD_RATIO
export FULL_TEST TOPK_TP TOPK_FN TOPK_FP TOPK_LOWCONF
export DEVICE NUM_WORKERS IMG_SIZE IMAGENET_NORM BATCH_SIZE SEED

exec "${PROJECT_ROOT}/src/explainability/spatial/run_attention_rollout.sh"
>>

attention_rollout.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial XAI on RCC test set (TP/FP/FN selection via predictions.csv).

Generates:
- GradCAM / IG / Occlusion (if enabled and dependencies are available)
- Attention Rollout for ViT (via monkey patching timm Attention blocks).

This script is config-driven and can be:
- run standalone: python attention_rollout.py --config CONFIG_PATH
- called programmatically from the orchestrator:
    from explainability.spatial.xai_generate import main as spatial_xai_main
    spatial_xai_main(["--config", str(config_path)])
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402

from explainability.spatial.eval_utils import (
    setup_logger,
    set_seed,
    build_preprocess,
    tensor_to_pil,
    load_eval_artifacts,
    select_items,
    make_wds_loader_with_keys,
    make_imgfolder_loader,
    canonicalize_key,
)
from explainability.spatial.ssl_linear_loader import SSLLinearClassifier

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
            logging.getLogger(__name__).warning(
                "Rollout Error: no attention blocks captured via patching."
            )
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

    logger = setup_logger("attention_rollout.sh")

    parser = argparse.ArgumentParser(description="Spatial XAI for SSL RCC model")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    set_seed(int(cfg["experiment"]["seed"]))

    device = torch.device(
        cfg.get("runtime", {}).get("device", "cuda")
        if torch.cuda.is_available()
        else "cpu"
    )

    run_id = cfg["experiment"].get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(cfg["experiment"]["outputs_root"]) / cfg["model"]["name"] / run_id
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
            batch_size=int(cfg["data"].get("batch_size", 1)),
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

    index_csv = (out_root / "index.csv").open("w", newline="", encoding="utf-8")
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
    global_idx = 0  # output index (0..produced-1)
    dataset_idx = 0  # only used for imagefolder (selection by numeric index)

    row_by_key: Dict[str, Dict[str, Any]] = {}
    if meta_rows:
        for r in meta_rows:
            k = canonicalize_key(r.get("wds_key", ""))
            if k:
                row_by_key[k] = r
    idx_by_key = {canonicalize_key(k): i for i, k in enumerate(keys)} if keys is not None else {}
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

    # Conteggi per validazione/summary
    method_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    def _iter_samples_from_batch(batch_any):
        """
        Normalize DataLoader batch to an iterator of (img_t, meta_any, key_str_or_none).
        Supports webdataset batch_size==1 and batch_size>1.
        """
        if cfg["data"]["backend"].lower() != "webdataset":
            img_t, lbl = batch_any
            yield img_t, {"class_id": int(lbl)}, None
            return

        imgs, metas, keys_any = batch_any
        if torch.is_tensor(imgs) and imgs.ndim == 3:
            yield imgs, metas, canonicalize_key(keys_any)
            return

        if torch.is_tensor(imgs) and imgs.ndim == 4:
            bsz = int(imgs.shape[0])
            keys_list = list(keys_any) if isinstance(keys_any, (list, tuple)) else [keys_any] * bsz
            metas_list = list(metas) if isinstance(metas, (list, tuple)) else [metas] * bsz
            for i in range(bsz):
                yield imgs[i], metas_list[i], canonicalize_key(keys_list[i])
            return

        return

    for batch in loader:
        for img_t, meta_any, key in _iter_samples_from_batch(batch):
            if cfg["data"]["backend"].lower() == "webdataset":
                if key is None or key == "":
                    continue
                if keys is not None and key not in target_set:
                    continue
            else:
                if targets and dataset_idx not in target_set:
                    dataset_idx += 1
                    continue
                dataset_idx += 1

            meta = meta_any if isinstance(meta_any, dict) else {}
            if isinstance(meta_any, (str, bytes)):
                meta = _parse_maybe_json_or_literal(meta_any) or {}

            row = row_by_key.get(key or "", {}) if key else {}

            true_id = int(row.get("y_true", -1)) if row else -1
            true_txt = (
                class_order[true_id]
                if 0 <= true_id < n_classes
                else str(meta.get("class_label", ""))
            )

            pred_id = int(row.get("y_pred", -1)) if row else -1
            pred_txt = (
                class_order[pred_id] if 0 <= pred_id < n_classes else str(pred_id)
            )

            idx_eval = idx_by_key.get(key or "", None) if key else None
            if conf is not None and idx_eval is not None:
                prob = float(conf[idx_eval])
            else:
                prob = float("nan")

            sel_reason_list = sel_reasons.get(key, []) if (keys is not None and key) else []
            sel_reason_str = "|".join(sel_reason_list) if sel_reason_list else ""
            for r in sel_reason_list:
                reason_counts[r] += 1

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
                (out_dir / "selection_reason.txt").write_text(sel_reason_str + "\n", encoding="utf-8")

            png_paths: List[str] = []
            used: List[str] = []

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
                    for h in getattr(cam_method, "hook_handles", []):
                        h.remove()
                except Exception as e:
                    logger.warning(f"GradCAM failed: {e}")

            if use_rollout and rollout_instance is not None:
                try:
                    mask = rollout_instance(x)
                    if mask is not None:
                        mask_np = np.array(mask.detach().cpu() if hasattr(mask, "detach") else mask)
                        over = overlay_heatmap(pil_in, mask_np, alpha=0.6)
                        path = out_dir / "attn_rollout.png"
                        over.save(path)
                        np.save(out_dir / "attn_rollout.npy", mask_np)
                        png_paths.append(str(path))
                        used.append("attn_rollout")
                except Exception as e:
                    logger.warning(f"Rollout failed: {e}")

            for m in used:
                method_counts[m] += 1

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

            if produced >= total_targets:
                break
        if produced >= total_targets:
            break

    index_csv.close()
    total_elapsed = time.time() - t_start
    logger.info(
        f"[Spatial XAI] Done. Produced {produced} cases in {total_elapsed/60:.1f} min."
    )

    # ----------------- VALIDAZIONE / SEGNALAZIONI -----------------
    if produced == 0:
        logger.warning(
            "[Spatial XAI] No outputs produced (produced=0). "
            "Controlla selection config / eval artifacts."
        )
    else:
        if method_counts:
            logger.info("[Spatial XAI] Method usage:")
            for m, cnt in method_counts.items():
                logger.info(f"  - {m}: {cnt} patches")
        if reason_counts:
            logger.info("[Spatial XAI] Selection reasons distribution:")
            for r, cnt in reason_counts.items():
                logger.info(f"  - {r}: {cnt} patches")


if __name__ == "__main__":
    main()
>>

eval_utils.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Common utilities for explainability:
 - logging and seeding
 - basic image preprocessing
 - evaluation artifacts loading (predictions + logits)
 - selection of TP/FP/FN and low-confidence cases
 - WebDataset loader with keys (supports batch_size>1)
 - atomic writers (json/csv) for idempotent pipelines
"""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms, datasets

try:
    import webdataset as wds
    HAVE_WDS = True
except Exception:
    HAVE_WDS = False


# -------------------------------------------------------------------------
# Logging / reproducibility
# -------------------------------------------------------------------------
def setup_logger(name: str = "explainability") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonicalize_key(k: Any) -> str:
    """
    Canonicalize keys coming from predictions.csv and WebDataset __key__.
    - decode bytes
    - strip common dataset prefixes like "test::", "val::", "train::"
    """
    if k is None:
        return ""
    if isinstance(k, (bytes, bytearray)):
        try:
            k = k.decode("utf-8")
        except Exception:
            k = str(k)
    s = str(k).strip()
    for pfx in ("test::", "val::", "train::"):
        if s.startswith(pfx):
            s = s[len(pfx):]
            break
    return s


# -------------------------------------------------------------------------
# Atomic writers (avoid partial files in HPC preemptions)
# -------------------------------------------------------------------------
def ensure_dir(p: Union[str, Path]) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def atomic_write_text(path: Union[str, Path], text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding=encoding) as f:
        f.write(text)
        tmp = Path(f.name)
    tmp.replace(path)


def atomic_write_json(path: Union[str, Path], obj: Any, *, indent: int = 2) -> None:
    atomic_write_text(Path(path), json.dumps(obj, indent=indent, ensure_ascii=False) + "\n")


def atomic_write_csv(path: Union[str, Path], rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), newline="") as f:
        tmp = Path(f.name)
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    tmp.replace(path)


# -------------------------------------------------------------------------
# Image transforms
# -------------------------------------------------------------------------
def build_preprocess(img_size: int, imagenet_norm: bool = True) -> transforms.Compose:
    ops: List[Any] = [
        transforms.Resize(
            img_size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ]
    if imagenet_norm:
        ops.append(
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            )
        )
    return transforms.Compose(ops)


def tensor_to_pil(t: torch.Tensor, imagenet_norm: bool = True) -> Image.Image:
    t = t.detach().cpu()
    if imagenet_norm:
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        t = t * std + mean
    t = t.clamp(0.0, 1.0)
    return transforms.ToPILImage()(t)


# -------------------------------------------------------------------------
# Eval artifacts (predictions.csv + logits_test.npy)
# -------------------------------------------------------------------------
def load_eval_artifacts(
    eval_dir: Union[str, Path],
    pred_csv: str,
    logits_npy: str,
    logger: logging.Logger,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[List[str]],
    Optional[List[Dict[str, Any]]],
]:
    """Load y_true / y_pred / confidence / wds_key / full rows from eval output."""
    eval_dir = Path(eval_dir)
    y_true = y_pred = conf = None
    keys: Optional[List[str]] = None
    meta_rows: Optional[List[Dict[str, Any]]] = None

    pcsv = eval_dir / pred_csv
    if pcsv.exists():
        yt, yp, kk, rows = [], [], [], []
        with pcsv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            has_key = "wds_key" in fields
            for row in reader:
                t = row.get("y_true", "")
                yt.append(int(t) if str(t).strip() != "" else -1)
                yp.append(int(row["y_pred"]))
                if has_key:
                    row["wds_key"] = canonicalize_key(row.get("wds_key", ""))
                    kk.append(row["wds_key"])
                else:
                    kk.append(None)
                rows.append(row)
        y_true = np.array(yt)
        y_pred = np.array(yp)
        keys = kk if any(k is not None for k in kk) else None
        meta_rows = rows
        logger.info(f"Loaded predictions.csv with {len(yp)} rows from {pcsv}")
    else:
        logger.warning(f"predictions.csv not found: {pcsv}")

    plog = eval_dir / logits_npy
    if plog.exists():
        logits = np.load(plog)
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = ex / ex.sum(axis=1, keepdims=True)
        conf = probs.max(axis=1)
        logger.info(f"Loaded logits from: {plog}")
    else:
        logger.warning(f"logits npy not found: {plog}")

    return y_true, y_pred, conf, keys, meta_rows


# -------------------------------------------------------------------------
# Selection logic (TP / FP / FN / low-confidence)
# -------------------------------------------------------------------------
def select_items(
    y_true: Optional[np.ndarray],
    y_pred: Optional[np.ndarray],
    conf: Optional[np.ndarray],
    keys: Optional[List[str]],
    n_classes: int,
    cfg_sel: Dict[str, Any],
    logger: logging.Logger,
):
    """
    Select indices to explain and track selection reasons.

    Returns
    -------
    targets : List[str] or List[int]
        Selected wds_keys (if keys is not None) or raw indices.
    reasons : Dict[str, List[str]] or Dict[int, List[str]]
        Map from wds_key (or index) -> list of selection reasons.
    """
    if y_pred is None:
        logger.warning("No predictions available; selection is empty.")
        return [], {}

    def pick(arr, k, by_conf=None, reverse=True):
        if len(arr) == 0 or k <= 0:
            return []
        idx = np.asarray(arr, dtype=int)
        if by_conf is not None:
            # safety: conf shape check
            if by_conf.shape[0] <= idx.max():
                logger.warning(
                    "Confidence array shorter than indices; ignoring confidence ordering."
                )
            else:
                order = np.argsort(by_conf[idx])
                if reverse:
                    order = order[::-1]
                idx = idx[order]
        return idx[:k].tolist()

    items: List[int] = []
    reason_by_idx: Dict[int, set[str]] = {}

    def add_reason(idx: int, reason: str):
        if idx not in reason_by_idx:
            reason_by_idx[idx] = set()
        reason_by_idx[idx].add(reason)

    # ------------------------------------------------------------------
    # Per-class TP / FP / FN
    # ------------------------------------------------------------------
    for c in range(n_classes):
        idx_c = np.where(y_true == c)[0] if y_true is not None else np.array([], dtype=int)

        # High-confidence TP
        if idx_c.size > 0:
            tpc = idx_c[y_pred[idx_c] == c]
        else:
            tpc = np.array([], dtype=int)

        chosen_tp = pick(
            tpc,
            cfg_sel["per_class"].get("topk_tp", 0),
            by_conf=conf,
            reverse=True,
        )
        items += chosen_tp
        for i in chosen_tp:
            add_reason(i, "tp_high_conf")

        # FN
        if idx_c.size > 0:
            fnc = idx_c[y_pred[idx_c] != c]
        else:
            fnc = np.array([], dtype=int)

        chosen_fn = pick(
            fnc,
            cfg_sel["per_class"].get("topk_fn", 0),
            by_conf=conf,
            reverse=False,  # lowest confidence among wrong
        )
        items += chosen_fn
        for i in chosen_fn:
            # reverse=False => low confidence among wrong predictions
            add_reason(i, "fn_low_conf")

        # FP
        idx_pred_c = np.where(y_pred == c)[0]
        if y_true is not None and idx_pred_c.size > 0:
            fpc = idx_pred_c[y_true[idx_pred_c] != c]
        else:
            fpc = idx_pred_c

        chosen_fp = pick(
            fpc,
            cfg_sel["per_class"].get("topk_fp", 0),
            by_conf=conf,
            reverse=True,
        )
        items += chosen_fp
        for i in chosen_fp:
            add_reason(i, "fp_high_conf")

    # ------------------------------------------------------------------
    # Globally low-confidence cases (optional)
    # ------------------------------------------------------------------
    if conf is not None and "global_low_conf" in cfg_sel:
        n_low = cfg_sel["global_low_conf"].get("topk", 0)
        if n_low > 0:
            order = np.argsort(conf)  # ascending → lowest confidence first
            chosen_low = order[:n_low].tolist()
            items += chosen_low
            for i in chosen_low:
                add_reason(i, "low_conf")

    # Dedup preserving order
    seen = set()
    unique_items: List[int] = []
    for i in items:
        if i not in seen:
            seen.add(i)
            unique_items.append(i)

    if keys is not None:
        targets = [keys[i] for i in unique_items]
        reasons = {
            keys[i]: sorted(list(reason_by_idx.get(i, [])))
            for i in unique_items
        }
    else:
        targets = unique_items
        reasons = {
            i: sorted(list(reason_by_idx.get(i, [])))
            for i in unique_items
        }

    logger.info(f"Selected {len(targets)} items for XAI.")
    return targets, reasons


# -------------------------------------------------------------------------
# WebDataset helper: filter by key set in one streaming pass
# -------------------------------------------------------------------------
def iter_wds_filtered_by_keys(
    loader,
    wanted: set[str],
    *,
    key_prefix_strip: Optional[str] = None,
):
    """
    Iterate a WDS loader and yield only samples whose key is in `wanted`.

    Notes
    -----
    - Works for batch_size==1 (single sample) and batch_size>1 (batched).
    - key_prefix_strip: if provided, strips e.g. "test::" from keys before matching.
    """
    if not wanted:
        return

    def _canon_key(k: Any) -> str:
        s = str(k)
        if key_prefix_strip and s.startswith(key_prefix_strip):
            s = s[len(key_prefix_strip):]
        return s

    for batch in loader:
        if batch is None:
            continue
        # batch_size==1: (img, meta, key)
        if isinstance(batch, (list, tuple)) and len(batch) == 3 and not torch.is_tensor(batch[2]):
            img, meta, key = batch
            kk = _canon_key(key)
            if kk in wanted:
                yield img, meta, kk
            continue

        # batch_size>1: (imgs[B,...], metas, keys)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            imgs, metas, keys = batch
            # keys might be list[str] or tuple[str]
            for i, k in enumerate(list(keys)):
                kk = _canon_key(k)
                if kk in wanted:
                    mi = metas[i] if isinstance(metas, (list, tuple)) else metas
                    yield imgs[i], mi, kk


# -------------------------------------------------------------------------
# Data loaders
# -------------------------------------------------------------------------
def _upgrade_wds_field_key(k: str, kind: str) -> str:
    """
    Backward-compatible key upgrade:
    - legacy configs often use "jpg"/"json"
    - dataset uses multi-extension fields "img.jpg" and "meta.json"
    WebDataset supports alternative extensions via ';' (e.g., "img.jpg;jpg").
    """
    kk = str(k).strip()
    if kind == "image" and kk == "jpg":
        return "img.jpg;jpg"
    if kind == "meta" and kk == "json":
        return "meta.json;json"
    return kk


def make_wds_loader_with_keys(
    test_dir: str,
    pattern: str,
    image_key: str,
    meta_key: str,
    preprocess_fn,
    num_workers: int,
    batch_size: int = 1,
):
    """
    Create a WebDataset loader that yields:
      - batch_size==1: (image_tensor, meta, key)
      - batch_size>1 : (images_tensor[B,...], metas, keys)
    """
    if not HAVE_WDS:
        raise RuntimeError("webdataset not available; install it for explainability.")
    import glob

    image_key = _upgrade_wds_field_key(image_key, "image")
    meta_key = _upgrade_wds_field_key(meta_key, "meta")

    shard_glob = str(Path(test_dir) / pattern)
    shards = sorted(glob.glob(shard_glob))
    if not shards:
        raise FileNotFoundError(f"No shards found: {shard_glob}")

    ds = (
        wds.WebDataset(
            shards,
            shardshuffle=False,
            handler=wds.warn_and_continue,
            empty_check=False,
        )
        .decode("pil")
        .to_tuple(image_key, meta_key, "__key__", handler=wds.warn_and_continue)
        .map_tuple(preprocess_fn, lambda x: x, lambda x: x)
    )

    def _collate_first(batch):
        if not batch:
            return None
        return batch[0]

    collate_fn = _collate_first if int(batch_size) == 1 else None
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(batch_size),
        num_workers=min(num_workers, len(shards)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    return loader


def make_imgfolder_loader(
    test_dir: str, preprocess_fn, batch_size: int, num_workers: int
):
    """Fallback loader for ImageFolder datasets (not WebDataset)."""
    ds = datasets.ImageFolder(test_dir, transform=preprocess_fn)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return ds, loader
>>

__init__.py codice <<
# empty on purpose – marks "common" as a package
>>

run_attention_rollout.sh codice <<
#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# User-tunable (override via env)
# -----------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/home/mla_group_01/rcc-ssrl}"
VENV_PATH="${VENV_PATH:-/home/mla_group_01/rcc-ssrl/.venvs/xai}"
PYTHON_BIN="${PYTHON_BIN:-python}"

ATTN_SCRIPT="${ATTN_SCRIPT:-$PROJECT_ROOT/src/explainability/spatial/attention_rollout.py}"

# Eval artifacts
EVAL_RUN_DIR="${EVAL_RUN_DIR:-}"                         # REQUIRED
PREDICTIONS_CSV="${PREDICTIONS_CSV:-predictions.csv}"
LOGITS_NPY="${LOGITS_NPY:-logits_test.npy}"

# Checkpoints
MODEL_NAME="${MODEL_NAME:-ssl_vit}"
BACKBONE_NAME="${BACKBONE_NAME:-vit_small_patch16_224}"
SSL_BACKBONE_CKPT="${SSL_BACKBONE_CKPT:-}"               # REQUIRED
SSL_HEAD_CKPT="${SSL_HEAD_CKPT:-}"                       # REQUIRED

# Data
DATA_BACKEND="${DATA_BACKEND:-webdataset}"               # webdataset | imagefolder
TEST_WDS_DIR="${TEST_WDS_DIR:-}"                         # REQUIRED if webdataset
WDS_PATTERN="${WDS_PATTERN:-*.tar}"
# FIX: dataset usa chiavi multi-estensione (es. "img.jpg", "meta.json")
WDS_IMAGE_KEY="${WDS_IMAGE_KEY:-img.jpg;jpg}"
WDS_META_KEY="${WDS_META_KEY:-meta.json;json}"
TEST_IMAGEFOLDER_DIR="${TEST_IMAGEFOLDER_DIR:-}"         # REQUIRED if imagefolder

IMG_SIZE="${IMG_SIZE:-224}"
IMAGENET_NORM="${IMAGENET_NORM:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Runtime
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1337}"

# Outputs
OUTPUTS_ROOT="${OUTPUTS_ROOT:-$PROJECT_ROOT/outputs/xai_spatial}"
RUN_ID="${RUN_ID:-}"

# XAI
XAI_METHODS="${XAI_METHODS:-attn_rollout}"
ATNN_DISCARD_RATIO="${ATNN_DISCARD_RATIO:-0.9}"
GRADCAM_TARGET_LAYER="${GRADCAM_TARGET_LAYER:-backbone.model.blocks.11}"
IG_STEPS="${IG_STEPS:-32}"

# Selection
FULL_TEST="${FULL_TEST:-0}"
TOPK_TP="${TOPK_TP:-6}"
TOPK_FN="${TOPK_FN:-6}"
TOPK_FP="${TOPK_FP:-6}"
TOPK_LOWCONF="${TOPK_LOWCONF:-20}"

# Labels
CLASS_ORDER_JSON="${CLASS_ORDER_JSON:-[\"ccRCC\",\"pRCC\",\"chRCC\",\"oncocytoma\",\"unclassified\"]}"

# -----------------------------
# Basic checks
# -----------------------------
if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "[FATAL] PROJECT_ROOT not found: $PROJECT_ROOT" >&2
  exit 2
fi
if [[ ! -f "$ATTN_SCRIPT" ]]; then
  echo "[FATAL] attention_rollout.py not found: $ATTN_SCRIPT" >&2
  exit 2
fi
if [[ -z "${EVAL_RUN_DIR}" || ! -d "${EVAL_RUN_DIR}" ]]; then
  echo "[FATAL] Set EVAL_RUN_DIR to a valid eval directory containing predictions/logits." >&2
  echo "        Current EVAL_RUN_DIR: ${EVAL_RUN_DIR:-<empty>}" >&2
  echo "        Expected files: ${PREDICTIONS_CSV} and ${LOGITS_NPY}" >&2
  exit 2
fi
if [[ -z "${SSL_BACKBONE_CKPT}" || ! -f "${SSL_BACKBONE_CKPT}" ]]; then
  echo "[FATAL] Set SSL_BACKBONE_CKPT to an existing file." >&2
  exit 2
fi
if [[ -z "${SSL_HEAD_CKPT}" || ! -f "${SSL_HEAD_CKPT}" ]]; then
  echo "[FATAL] Set SSL_HEAD_CKPT to an existing file." >&2
  exit 2
fi

if [[ "${DATA_BACKEND}" == "webdataset" ]]; then
  if [[ -z "${TEST_WDS_DIR}" || ! -d "${TEST_WDS_DIR}" ]]; then
    echo "[FATAL] DATA_BACKEND=webdataset requires TEST_WDS_DIR (directory with .tar shards)." >&2
    exit 2
  fi
elif [[ "${DATA_BACKEND}" == "imagefolder" ]]; then
  if [[ -z "${TEST_IMAGEFOLDER_DIR}" || ! -d "${TEST_IMAGEFOLDER_DIR}" ]]; then
    echo "[FATAL] DATA_BACKEND=imagefolder requires TEST_IMAGEFOLDER_DIR." >&2
    exit 2
  fi
else
  echo "[FATAL] Unknown DATA_BACKEND=${DATA_BACKEND} (use webdataset|imagefolder)." >&2
  exit 2
fi

# -----------------------------
# Optional tee logs
# -----------------------------
if [[ -n "${LOG_DIR:-}" ]]; then
  mkdir -p "${LOG_DIR}"
  exec > >(tee -a "${LOG_DIR}/run_attention_rollout.${RUN_ID:-local}.$(date +%Y%m%d_%H%M%S).out") 2>&1
fi

# -----------------------------
# Env / venv
# -----------------------------
cd "$PROJECT_ROOT"
if [[ -n "${VENV_PATH}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# -----------------------------
# Build config YAML (generated)
# -----------------------------
mkdir -p "${OUTPUTS_ROOT}"
CFG_DIR="${OUTPUTS_ROOT}/_configs"
mkdir -p "${CFG_DIR}"

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

CFG_PATH="${CFG_DIR}/attn_rollout_${MODEL_NAME}_${RUN_ID}.yaml"

# EXPORT: tutto ciò che il blocco Python legge da os.environ
export PROJECT_ROOT DATA_BACKEND XAI_METHODS FULL_TEST TOPK_TP TOPK_FN TOPK_FP TOPK_LOWCONF
export CLASS_ORDER_JSON SEED RUN_ID OUTPUTS_ROOT DEVICE MODEL_NAME BACKBONE_NAME
export SSL_BACKBONE_CKPT SSL_HEAD_CKPT EVAL_RUN_DIR PREDICTIONS_CSV LOGITS_NPY
export IMG_SIZE IMAGENET_NORM BATCH_SIZE NUM_WORKERS
export TEST_WDS_DIR WDS_PATTERN WDS_IMAGE_KEY WDS_META_KEY TEST_IMAGEFOLDER_DIR
export ATNN_DISCARD_RATIO GRADCAM_TARGET_LAYER IG_STEPS
export CFG_PATH

"${PYTHON_BIN}" - <<'PY'
import json, os
from pathlib import Path
import yaml

def env(name, default=None):
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default

def env_int(name, default):
    return int(env(name, str(default)))

def env_float(name, default):
    return float(env(name, str(default)))

def env_bool01(name, default):
    v = str(env(name, str(default))).strip().lower()
    return v in ("1","true","yes","y","on")

data_backend = str(env("DATA_BACKEND", "webdataset")).lower()
xai_methods = [m.strip() for m in str(env("XAI_METHODS", "attn_rollout")).split(",") if m.strip()]

full_test = env_bool01("FULL_TEST", 0)
if full_test:
    topk_tp = 10**9
    topk_fn = 10**9
    topk_fp = 10**9
    topk_low = 0
else:
    topk_tp = env_int("TOPK_TP", 6)
    topk_fn = env_int("TOPK_FN", 6)
    topk_fp = env_int("TOPK_FP", 6)
    topk_low = env_int("TOPK_LOWCONF", 20)

class_order = json.loads(str(env("CLASS_ORDER_JSON", "[]")))
if not class_order:
    raise SystemExit("CLASS_ORDER_JSON is empty; set it to the correct class order.")

cfg = {
  "experiment": {
    "seed": env_int("SEED", 1337),
    "run_id": str(env("RUN_ID")),
    "outputs_root": str(env("OUTPUTS_ROOT")),
  },
  "runtime": {"device": str(env("DEVICE", "cuda"))},
  "model": {
    "name": str(env("MODEL_NAME", "ssl_vit")),
    "arch_hint": "ssl_linear",
    "backbone_name": str(env("BACKBONE_NAME", "vit_small_patch16_224")),
    "ssl_backbone_ckpt": str(env("SSL_BACKBONE_CKPT")),
    "ssl_head_ckpt": str(env("SSL_HEAD_CKPT")),
  },
  "evaluation_inputs": {
    "eval_run_dir": str(env("EVAL_RUN_DIR")),
    "predictions_csv": str(env("PREDICTIONS_CSV", "predictions.csv")),
    "logits_npy": str(env("LOGITS_NPY", "logits_test.npy")),
  },
  "labels": {"class_order": class_order},
  "data": {
    "backend": data_backend,
    "img_size": env_int("IMG_SIZE", 224),
    "imagenet_norm": env_bool01("IMAGENET_NORM", 1),
    "batch_size": env_int("BATCH_SIZE", 1),
    "num_workers": env_int("NUM_WORKERS", 8),
  },
  "selection": {
    "per_class": {"topk_tp": topk_tp, "topk_fn": topk_fn, "topk_fp": topk_fp},
    "global_low_conf": {"topk": topk_low},
  },
  "xai": {
    "methods": xai_methods,
    "attn_rollout": {"discard_ratio": env_float("ATNN_DISCARD_RATIO", 0.9)},
    "gradcam": {"target_layer": str(env("GRADCAM_TARGET_LAYER", "backbone.model.blocks.11"))},
    "ig": {"steps": env_int("IG_STEPS", 32)},
  },
}

if data_backend == "webdataset":
    cfg["data"]["webdataset"] = {
      "test_dir": str(env("TEST_WDS_DIR")),
      "pattern": str(env("WDS_PATTERN", "*.tar")),
      "image_key": str(env("WDS_IMAGE_KEY", "img.jpg;jpg")),
      "meta_key": str(env("WDS_META_KEY", "meta.json;json")),
    }
else:
    cfg["data"]["imagefolder"] = {"test_dir": str(env("TEST_IMAGEFOLDER_DIR"))}

out = Path(str(env("CFG_PATH")))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(str(out))
PY

echo "[INFO] Config written: ${CFG_PATH}"
echo "[INFO] Running: ${PYTHON_BIN} ${ATTN_SCRIPT} --config ${CFG_PATH}"
exec "${PYTHON_BIN}" "${ATTN_SCRIPT}" --config "${CFG_PATH}"
>>

ssl_linear_loader.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to build an SSL classifier from:
 - ResNetBackbone or ViTBackbone weights saved inside SSL checkpoints
 - A linear head checkpoint saved by the linear probe trainer.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ---- Gestione dipendenza TIMM per ViT ----
try:
    import timm
    HAVE_TIMM = True
except ImportError:
    HAVE_TIMM = False

# ---- Backbone Definitions ----

class ResNetBackbone(nn.Module):
    def __init__(self, name: str="resnet50", pretrained: bool=False):
        super().__init__()
        from torchvision import models
        if "34" in name: m = models.resnet34(weights=None)
        else:            m = models.resnet50(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.out_dim = m.fc.in_features
        
    def _fwd(self, x): 
        x = self.stem(x); l1 = self.layer1(x); l2 = self.layer2(l1); l3 = self.layer3(l2); l4 = self.layer4(l3); return l4
    
    def forward_global(self, x): 
        feats = self._fwd(x)
        return torch.flatten(F.adaptive_avg_pool2d(feats, 1), 1)

class _VitBackbone(nn.Module):
    def __init__(self, name: str="vit_small_patch16_224"):
        super().__init__()
        if not HAVE_TIMM:
            raise RuntimeError("timm is required for ViT backbones. Install it with `pip install timm`.")
        # dynamic_img_size=True è fondamentale per XAI/Inference su size diverse
        self.model = timm.create_model(name, pretrained=False, num_classes=0, dynamic_img_size=True)
        self.out_dim = self.model.num_features

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)
        # Output timm standard: [B, T, C]. MoCo v3 usa il CLS token (indice 0)
        if feats.dim() == 3:
            return feats[:, 0] 
        # Fallback per architetture che fanno pool interno
        return torch.flatten(torch.nn.functional.adaptive_avg_pool2d(feats, 1), 1)

# ---- Loader Logic ----

_PREFIXES = ("stu.", "backbone_q.", "student.", "backbone.", "module.stu.", "module.backbone_q.", "base_encoder.")

def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

def _best_substate(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # pick the sub-dict with most keys among known prefixes
    cands = [(_strip_prefix(sd, p), p) for p in _PREFIXES]
    # Add the raw dict as a candidate (prefix "")
    cands.append((sd, ""))
    
    best_dict, best_prefix = max(cands, key=lambda x: len(x[0]))
    return best_dict

def _load_torch_state(path: str) -> Dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu") # Fallback older pytorch
        
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    return payload if isinstance(payload, dict) else {}

class SSLLinearClassifier(nn.Module):
    """
    Compose a Backbone (ResNet or ViT) with a linear head.
    """
    def __init__(self, backbone_name: str="resnet50", num_classes: int=5):
        super().__init__()
        if "vit" in backbone_name.lower():
            self.backbone = _VitBackbone(backbone_name)
        else:
            self.backbone = ResNetBackbone(backbone_name, pretrained=False)
            
        self.head = nn.Linear(self.backbone.out_dim, num_classes)

    def load_backbone_from_ssl(self, ssl_backbone_ckpt: str) -> Tuple[int, int]:
        sd = _load_torch_state(ssl_backbone_ckpt)
        sd = _best_substate(sd)
        
        # Gestione specifica per ViT timm vs MoCo naming
        # A volte MoCo salva come 'module.base_encoder.model.blocks...' vs 'blocks...'
        if isinstance(self.backbone, _VitBackbone):
            new_sd = {}
            for k, v in sd.items():
                # Rimuovi 'model.' se presente (comune in wrapper timm salvati male)
                if k.startswith("model."):
                    k = k[6:]
                new_sd[k] = v
            sd = new_sd

        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        return len(missing), len(unexpected)

    def load_head_from_probe(self, ssl_head_ckpt: str) -> Tuple[int, int]:
        hd = torch.load(ssl_head_ckpt, map_location="cpu")
        if isinstance(hd, dict) and "state_dict" in hd:
            hd = hd["state_dict"]
        missing, unexpected = self.head.load_state_dict(hd, strict=False)
        return len(missing), len(unexpected)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_global(x)
        return self.head(feats)
>>

