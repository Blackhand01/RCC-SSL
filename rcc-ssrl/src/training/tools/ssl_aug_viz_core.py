#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core utilities to visualize SSL augmentations as a single matrix image.

- Rebuilds the exact SSL transforms used by the training pipeline.
- Samples two images per class from the training WebDataset.
- Composes a 4x4 matrix with paired classes per row:
    Row1: [ccRCC #1  orig | aug] [pRCC   #1  orig | aug]
    Row2: [ccRCC #2  orig | aug] [pRCC   #2  orig | aug]
    Row3: [ONCO  #1  orig | aug] [CHROMO #1  orig | aug]
    Row4: [ONCO  #2  orig | aug] [CHROMO #2  orig | aug]
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import random

import json
import torch
from PIL import Image, ImageDraw
from torchvision.utils import make_grid, save_image

# Project imports: keep aligned with your repo
from src.training.launch_training import read_yaml, paths as resolve_paths, inject_paths_into_cfg
from src.training.data.webdataset import list_shards, make_wds
from src.training.datasets.labels import normalise_label
from src.training.datasets.transforms import (
    two_view_transform,
    multicrop_transform,
    ijepa_input_transform,
    coerce_to_pil_rgb,
)


# -----------------------------
# Config helpers
# -----------------------------
def _model_family(cfg: Dict) -> str:
    return str(cfg["model"]["ssl"]["name"]).lower().strip()

def _abl_code(cfg: Dict) -> str:
    name = str(cfg.get("experiment", {}).get("name", ""))
    for tok in name.split("_"):
        if tok.startswith("abl"):
            return tok
    return "ablNA"

def _load_and_inject(cfg_path: Union[str, Path]) -> Dict:
    cfg = read_yaml(str(cfg_path))
    resolved = resolve_paths()
    cfg = inject_paths_into_cfg(cfg, resolved)
    return cfg


# -----------------------------
# Transform builders (exactly mirroring training)
# -----------------------------
def _build_ssl_transform(cfg: Dict):
    """
    Returns a callable(img_pil) -> aug_out
    aug_out:
      - two-views: (x1, x2)
      - multicrop: [g1, g2, l1, l2, ...]
      - i-JEPA:    x
    """
    ds_cfg = cfg.get("data", {}) or {}
    img_size = int(ds_cfg.get("img_size", 224))
    aug_top = cfg.get("aug", {}) or {}
    mname = _model_family(cfg)

    # Feature flags (if present in your YAML)
    ssl_cfg = (cfg.get("model", {}).get("ssl", {}) or {})
    use_mc = bool(ssl_cfg.get("use_multicrop", False))

    if mname in ("moco_v3", "ibot") and not use_mc:
        # 2-view path (MoCo legacy / iBOT without multicrop)
        jtr = float(ssl_cfg.get("jitter", 0.4))
        blur_p = float(ssl_cfg.get("blur_prob", 0.1))
        gray_p = float(ssl_cfg.get("gray_prob", 0.2))
        solar_p = float(ssl_cfg.get("solarize_prob", 0.0))
        return two_view_transform(
            img_size,
            jtr,
            blur_prob=blur_p,
            gray_prob=gray_p,
            solarize_prob=solar_p,
            cfg_aug=aug_top,
        )

    if mname == "dino_v3" or (mname in ("moco_v3", "ibot") and use_mc):
        dino_cfg = ds_cfg.get("dino_v3", {}) or {}
        return multicrop_transform(
            int(dino_cfg.get("global_size", img_size)),
            int(dino_cfg.get("local_size", 96)),
            int(dino_cfg.get("n_local", 6)),
            float(dino_cfg.get("jitter", 0.4)),
            global_scale=tuple(dino_cfg.get("global_scale", (0.14, 1.0))),
            local_scale=tuple(dino_cfg.get("local_scale", (0.05, 0.14))),
            blur_prob=float(dino_cfg.get("blur_prob", 0.5)),
            solarize_prob=float(dino_cfg.get("solarize_prob", 0.0)),
            solarize_prob_g2=float(dino_cfg.get("solarize_prob_g2", 0.2)),
            cfg_aug=aug_top,
        )

    if mname == "i_jepa":
        return ijepa_input_transform(img_size, aug_top)

    raise ValueError(f"Unsupported SSL model family '{mname}'.")


# -----------------------------
# Dataset sampling
# -----------------------------
def _collect_two_per_class(train_dir: str, class_names: List[str], *, seed: int = 1337) -> Dict[str, List[Image.Image]]:
    """
    Iterate the WebDataset and collect up to two PIL images per class in class_names.
    Returns a dict[class_name] -> [img1, img2] (less if unavailable).
    """
    # Prepare target
    normalized_targets = [normalise_label(c) for c in class_names]
    want = {c: 2 for c in normalized_targets}
    got: Dict[str, List[Image.Image]] = {c: [] for c in normalized_targets}

    shards = list_shards(train_dir)
    # Deterministic shard order for reproducibility; older make_wds() has no 'seed' arg
    rng = random.Random(seed)
    rng.shuffle(shards)
    ds = make_wds(shards, shuffle_shards=64, shuffle_samples=2000)

    for img, meta in ds:
        cls = normalise_label((meta or {}).get("class_label", ""))
        if cls not in want:
            continue
        if len(got[cls]) < want[cls]:
            got[cls].append(coerce_to_pil_rgb(img))
        if all(len(v) >= 2 for v in got.values()):
            break
    return got


# -----------------------------
# Image utilities
# -----------------------------
def _to_pil_tile(x: Union[torch.Tensor, Image.Image], size: Tuple[int, int]) -> Image.Image:
    """Convert a tensor or PIL to a PIL tile of fixed size."""
    if isinstance(x, Image.Image):
        return x.resize(size, Image.BILINEAR)
    t = x.detach().cpu()
    if t.dtype.is_floating_point:
        t = (t.clamp(0, 1) * 255.0).to(torch.uint8)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if t.size(0) == 1:
        t = t.expand(3, -1, -1)
    pil = Image.fromarray(t.permute(1, 2, 0).numpy())
    return pil.resize(size, Image.BILINEAR)

def _blank(size: Tuple[int, int]) -> Image.Image:
    return Image.new("RGB", size, color=(245, 245, 245))

def _aug_panel(aug_out, tile_size: Tuple[int, int], max_locals: int = 4) -> Image.Image:
    """
    Build a single 'Augmented' panel:
      - two-views: grid over (v1, v2) in one row
      - multicrop: grid over [g1,g2,l1..] in one row (locals capped)
      - single:    the only augmented tensor
    """
    if isinstance(aug_out, tuple):
        tensors = [aug_out[0], aug_out[1]]
    elif isinstance(aug_out, list):
        globals_ = aug_out[:2]
        locals_ = aug_out[2:2 + max_locals]
        tensors = list(globals_) + list(locals_)
    else:
        tensors = [aug_out]

    grid = make_grid([t.clamp(0, 1).cpu() for t in tensors], nrow=len(tensors), padding=2)
    return _to_pil_tile(grid, tile_size)

def _label_on(img: Image.Image, text: str) -> Image.Image:
    """Draw a small black label rectangle with white text in the top-left corner."""
    draw = ImageDraw.Draw(img)
    pad = 6
    # Use textbbox for newer PIL versions
    bbox = draw.textbbox((0, 0), text)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([0, 0, w + 2 * pad, h + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255))
    return img


# -----------------------------
# Composition: 4x4 paired layout
# -----------------------------
def compose_matrix_paired_4x4(
    cfg: Dict,
    out_path: Union[str, Path],
    *,
    tile_size: Tuple[int, int] = (224, 224),
    max_locals: int = 4,
    seed: int = 1337,
    class_pairs: List[Tuple[str, str]] = (("ccRCC", "pRCC"), ("ONCO", "CHROMO")),
) -> Path:
    """
    Compose a 4x4 matrix with two classes per row.
    Rows 1-2 use pair[0] (ccRCC) on the left and pair[1] (pRCC) on the right;
    Rows 3-4 use pair[0] (ONCO) on the left and pair[1] (CHROMO) on the right.
    Each side uses two samples: row A -> sample #1, row B -> sample #2.

    NOT_TUMOR is intentionally excluded.
    """
    # Basic seeding for determinism across runs (augmentations and sampling)
    torch.manual_seed(seed)
    random.seed(seed)
    # Resolve dataset & transform
    train_dir = str(cfg["data"]["webdataset"]["train_dir"])
    tfm = _build_ssl_transform(cfg)

    # Collect two images for the four tumor classes
    flat_classes = [c for pair in class_pairs for c in pair]
    per_class = _collect_two_per_class(train_dir, flat_classes, seed=seed)

    # Canvas 4x4
    cols, rows = 4, 4
    W, H = tile_size
    canvas = Image.new("RGB", (cols * W, rows * H), color=(255, 255, 255))

    def get_sample(cls: str, idx: int) -> Image.Image:
        lst = per_class.get(normalise_label(cls), [])
        if idx < len(lst):
            return lst[idx]
        return _blank(tile_size)

    # Row 1-2: (ccRCC | pRCC), Row 3-4: (ONCO | CHROMO)
    for block, (left_cls, right_cls) in enumerate(class_pairs):
        for k in range(2):  # k=0 -> sample#1; k=1 -> sample#2
            r = block * 2 + k  # row index 0..3
            y = r * H

            # Left class (cols 0-1)
            orig_L = _to_pil_tile(get_sample(left_cls, k), tile_size)
            try:
                aug_L = _aug_panel(tfm(orig_L), tile_size, max_locals=max_locals)
            except Exception:
                aug_L = _blank(tile_size)

            # Right class (cols 2-3)
            orig_R = _to_pil_tile(get_sample(right_cls, k), tile_size)
            try:
                aug_R = _aug_panel(tfm(orig_R), tile_size, max_locals=max_locals)
            except Exception:
                aug_R = _blank(tile_size)

            # Labels on originals
            orig_L = _label_on(orig_L.copy(), left_cls)
            orig_R = _label_on(orig_R.copy(), right_cls)

            # Paste: [orig_L, aug_L, orig_R, aug_R]
            canvas.paste(orig_L, (0 * W, y))
            canvas.paste(aug_L,  (1 * W, y))
            canvas.paste(orig_R, (2 * W, y))
            canvas.paste(aug_R,  (3 * W, y))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")

    # small metadata (optional)
    meta = {
        "layout": "paired_4x4",
        "class_pairs": class_pairs,
        "tile_size": tile_size,
        "max_locals": max_locals,
        "train_dir": train_dir,
    }
    (out_path.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    return out_path


# -----------------------------
# Public API
# -----------------------------
def dump_matrix_for_cfg_path(
    cfg_path: Union[str, Path],
    out_root: Union[str, Path],
    *,
    tile_size: int = 224,
    max_locals: int = 4,
    seed: int = 1337,
) -> Path:
    """
    High-level helper: load YAML, resolve paths, derive model/abl, and write matrix+json.
    """
    cfg = _load_and_inject(cfg_path)
    mname = _model_family(cfg)
    abl = _abl_code(cfg)

    out_dir = Path(out_root) / mname / abl
    out_png = out_dir / "aug_matrix.png"

    return compose_matrix_paired_4x4(
        cfg,
        out_png,
        tile_size=(tile_size, tile_size),
        max_locals=max_locals,
        seed=seed,
        class_pairs=[("ccRCC", "pRCC"), ("ONCO", "CHROMO")],
    )
