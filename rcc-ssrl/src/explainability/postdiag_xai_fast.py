#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postdiag_xai_fast.py

XAI post-diagnostics (attention-rollout + concept scores) for models/ablations.

Expected input (per item):
  attention_rollout_concept/<run>/items/idx_xxxxxxxx/
    - input.png
    - attn_rollout.npy  (H x W) or (h x w)
    - roi_bbox.json     (bbox in pixels on input)
    - roi.png           (optional, crop)
    - concept_scores.json (dict concept->score)

Output:
  out_root/
    postdiag_global.csv
    postdiag_global.json
    <exp_name>/<abl_name>/
      summary.json
      summary.csv
      figures/
        idx_xxxxxxxx_panel.png
      paper/
        fig_01_idx_xxxxxxxx.png
        fig_01_idx_xxxxxxxx.json   <-- sidecar with ROI + concepts (Top-k)
      montage.png

Usage:
  PYTHONPATH=/path/to/src /path/to/python postdiag_xai_fast.py \
    --models-root /path/to/models \
    --out-root /path/to/output \
    --selection-mode concept_top \
    --per-class-figures 6 \
    --class-source true

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None


# -------------------------
# Utils
# -------------------------

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)

def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2

def parse_bbox(obj: dict) -> Tuple[int, int, int, int]:
    """
    Supports various common formats:
      {x1,y1,x2,y2} / {xmin,ymin,xmax,ymax} / {left,top,right,bottom} / {x,y,w,h}
      {x0,y0,x1,y1}
      or list [x1,y1,x2,y2]
    """
    if isinstance(obj, list) and len(obj) >= 4:
        return int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])

    keys = set(obj.keys())
    if {"x1", "y1", "x2", "y2"}.issubset(keys):
        return int(obj["x1"]), int(obj["y1"]), int(obj["x2"]), int(obj["y2"])
    if {"x0", "y0", "x1", "y1"}.issubset(keys):
        return int(obj["x0"]), int(obj["y0"]), int(obj["x1"]), int(obj["y1"])
    if {"xmin", "ymin", "xmax", "ymax"}.issubset(keys):
        return int(obj["xmin"]), int(obj["ymin"]), int(obj["xmax"]), int(obj["ymax"])
    if {"left", "top", "right", "bottom"}.issubset(keys):
        return int(obj["left"]), int(obj["top"]), int(obj["right"]), int(obj["bottom"])
    if {"x", "y", "w", "h"}.issubset(keys):
        x1, y1 = int(obj["x"]), int(obj["y"])
        return x1, y1, x1 + int(obj["w"]), y1 + int(obj["h"])

    raise ValueError(f"Unrecognized bbox format: keys={sorted(list(keys))}")

def try_read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def is_number(s: object) -> bool:
    try:
        float(str(s))
        return True
    except Exception:
        return False

def safe_float(s: object, default: float = float("nan")) -> float:
    try:
        return float(str(s))
    except Exception:
        return default

def _parse_quantile(q: object, default: float = 0.90) -> float:
    try:
        v = float(q)
    except Exception:
        return default
    if v > 1.0:
        v = v / 100.0
    if v <= 0.0 or v > 1.0:
        return default
    return v

_CONCEPT_PREFIXES_TO_STRIP = (
    "chrcc_",
    "onco_",
    "ccrcc_",
    "prcc_",
    "notumor_",
)

def concept_display_name(short_name: str) -> str:
    """
    User-facing display: strip class prefixes (chrcc_/onco_/ccrcc_/prcc_/notumor_)
    and replace underscores with spaces.
    """
    s = str(short_name)
    for p in _CONCEPT_PREFIXES_TO_STRIP:
        if s.startswith(p):
            s = s[len(p):]
            break
    s = s.replace("_", " ").strip()
    return s


# -------------------------
# Data model
# -------------------------

@dataclass
class ItemPaths:
    idx: str
    dir: Path
    input_png: Path
    attn_npy: Path
    roi_bbox_json: Path
    roi_png: Optional[Path]
    concept_scores_json: Optional[Path]

@dataclass
class ItemMetrics:
    idx: str
    ok: bool
    reason: str
    input_size: Tuple[int, int]
    attn_size: Tuple[int, int]
    roi_bbox_xyxy: Tuple[int, int, int, int]
    attention_mass_in_roi: float
    peak_in_roi: bool
    top_concepts: List[Tuple[str, float]]


# -------------------------
# Discovery
# -------------------------

def discover_experiments(models_root: Path) -> List[Path]:
    if not models_root.exists():
        return []
    return sorted([p for p in models_root.iterdir() if p.is_dir() and p.name.startswith("exp_")])

def discover_ablations(exp_dir: Path) -> List[Path]:
    out = []
    for p in exp_dir.iterdir():
        if not p.is_dir():
            continue
        # ablation dirs: exp_*_ablXX
        if p.name.startswith("exp_") and "_abl" in p.name:
            out.append(p)
    return sorted(out)

def resolve_run_dir(attn_dir: Path) -> Optional[Path]:
    run_latest = attn_dir / "run_latest"
    if run_latest.exists():
        return run_latest
    runs = [p for p in attn_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]

def discover_items(run_dir: Path) -> List[Path]:
    items_dir = run_dir / "items"
    if not items_dir.exists():
        return []
    return sorted([p for p in items_dir.iterdir() if p.is_dir() and p.name.startswith("idx_")])

def discover_selection(run_dir: Path) -> List[Path]:
    """
    If selection/ contains idx_* (symlink or folders), we use it as shortlist.
    """
    sel = run_dir / "selection"
    if not sel.exists() or not sel.is_dir():
        return []
    idxs = sorted([p for p in sel.iterdir() if p.name.startswith("idx_")])
    if idxs:
        return idxs

    sel_json = sel / "xai_selection.json"
    if sel_json.exists():
        try:
            obj = read_json(sel_json)
        except Exception:
            return []
        indices = obj.get("selected_indices")
        if isinstance(indices, list) and indices:
            items_dir = run_dir / "items"
            chosen: List[Path] = []
            seen: set = set()
            for i in indices:
                try:
                    gi = int(i)
                except Exception:
                    continue
                cand = items_dir / f"idx_{gi:08d}"
                if not cand.exists():
                    cand = items_dir / f"idx_{gi:07d}"
                if cand.exists() and cand not in seen:
                    chosen.append(cand)
                    seen.add(cand)
            return chosen

    return []


def _item_class_from_meta(meta: dict, class_source: str) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    if class_source == "true":
        return meta.get("true_class") or None
    if class_source == "pred":
        return meta.get("pred_class") or None
    return meta.get("true_class") or meta.get("pred_class") or None


def _extract_top_scores(concept_scores: dict) -> List[float]:
    topk = concept_scores.get("topk") if isinstance(concept_scores, dict) else None
    scores: List[float] = []
    if isinstance(topk, list) and topk:
        for row in topk:
            if isinstance(row, dict):
                s = row.get("score")
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                s = row[1]
            else:
                s = None
            if is_number(s):
                scores.append(float(s))
        return scores
    scores_map = concept_scores.get("scores") if isinstance(concept_scores, dict) else None
    if isinstance(scores_map, dict):
        for v in scores_map.values():
            if is_number(v):
                scores.append(float(v))
    return scores

def build_item_paths(idx_dir: Path) -> Optional[ItemPaths]:
    input_png = idx_dir / "input.png"
    attn_npy = idx_dir / "attn_rollout.npy"
    roi_bbox = idx_dir / "roi_bbox.json"
    if not (input_png.exists() and attn_npy.exists() and roi_bbox.exists()):
        return None
    roi_png = idx_dir / "roi.png"
    concept_scores = idx_dir / "concept_scores.json"
    return ItemPaths(
        idx=idx_dir.name,
        dir=idx_dir,
        input_png=input_png,
        attn_npy=attn_npy,
        roi_bbox_json=roi_bbox,
        roi_png=roi_png if roi_png.exists() else None,
        concept_scores_json=concept_scores if concept_scores.exists() else None,
    )


# -------------------------
# Metrics
# -------------------------

def compute_metrics(item: ItemPaths) -> ItemMetrics:
    try:
        im = Image.open(item.input_png).convert("RGB")
        w, h = im.size
        bbox_obj = read_json(item.roi_bbox_json)
        x1, y1, x2, y2 = parse_bbox(bbox_obj)
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)

        attn = np.load(item.attn_npy)
        if attn.ndim == 3:
            attn = attn.squeeze()
        attn = attn.astype(np.float32, copy=False)
        ah, aw = int(attn.shape[0]), int(attn.shape[1])

        # Map bbox into attention space if sizes differ
        sx = aw / float(w)
        sy = ah / float(h)
        ax1 = int(math.floor(x1 * sx))
        ay1 = int(math.floor(y1 * sy))
        ax2 = int(math.ceil(x2 * sx))
        ay2 = int(math.ceil(y2 * sy))
        ax1, ay1, ax2, ay2 = clamp_bbox(ax1, ay1, ax2, ay2, aw, ah)

        attn01 = normalize01(attn)
        total = float(np.sum(attn01)) + 1e-8
        roi_sum = float(np.sum(attn01[ay1:ay2, ax1:ax2]))
        mass_in_roi = roi_sum / total

        peak_y, peak_x = np.unravel_index(int(np.argmax(attn01)), attn01.shape)
        peak_in = (ax1 <= peak_x < ax2) and (ay1 <= peak_y < ay2)

        top_concepts: List[Tuple[str, float]] = []
        if item.concept_scores_json is not None:
            cs = read_json(item.concept_scores_json)
            if isinstance(cs, dict):
                topk = cs.get("topk", None)
                if isinstance(topk, list) and topk:
                    pairs: List[Tuple[str, float]] = []
                    for row in topk:
                        if isinstance(row, dict):
                            name = (
                                row.get("concept_short_name")
                                or row.get("concept_name")
                                or row.get("concept")
                                or row.get("name")
                            )
                            score = row.get("score")
                        elif isinstance(row, (list, tuple)) and len(row) >= 2:
                            name, score = row[0], row[1]
                        else:
                            continue
                        if name is None or not is_number(score):
                            continue
                        pairs.append((str(name), float(score)))
                    if pairs:
                        top_concepts = pairs[:10]

                if not top_concepts:
                    scores_map = cs.get("scores", None)
                    if isinstance(scores_map, dict):
                        pairs = [(str(k), float(v)) for k, v in scores_map.items() if is_number(v)]
                        pairs.sort(key=lambda kv: kv[1], reverse=True)
                        top_concepts = pairs[:10]

                if not top_concepts:
                    pairs = [(str(k), float(v)) for k, v in cs.items() if is_number(v)]
                    pairs.sort(key=lambda kv: kv[1], reverse=True)
                    top_concepts = pairs[:10]

        return ItemMetrics(
            idx=item.idx,
            ok=True,
            reason="ok",
            input_size=(w, h),
            attn_size=(aw, ah),
            roi_bbox_xyxy=(x1, y1, x2, y2),
            attention_mass_in_roi=mass_in_roi,
            peak_in_roi=bool(peak_in),
            top_concepts=top_concepts,
        )
    except Exception as e:
        return ItemMetrics(
            idx=item.idx,
            ok=False,
            reason=f"{type(e).__name__}: {e}",
            input_size=(0, 0),
            attn_size=(0, 0),
            roi_bbox_xyxy=(0, 0, 0, 0),
            attention_mass_in_roi=float("nan"),
            peak_in_roi=False,
            top_concepts=[],
        )


# -------------------------
# Rendering (PIL-only, fast)
# -------------------------

def downscale_rgb(im: Image.Image, max_pixels: int) -> Tuple[Image.Image, float]:
    w, h = im.size
    m = max(w, h)
    if m <= max_pixels:
        return im, 1.0
    scale = max_pixels / float(m)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return im.resize((nw, nh), resample=Image.BILINEAR), scale

def draw_bbox(im: Image.Image, bbox: Tuple[int, int, int, int], width: int = 4) -> Image.Image:
    out = im.copy()
    d = ImageDraw.Draw(out)
    x1, y1, x2, y2 = bbox
    for i in range(width):
        d.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=(255, 0, 0))
    return out

def attn_to_heatmap_rgb(attn: np.ndarray) -> Image.Image:
    """
    Converts a [0,1] map to RGB using a simple LUT (no Matplotlib figure).
    Custom "inferno-lite" palette to avoid extra dependencies.
    """
    a = (np.clip(attn, 0.0, 1.0) * 255.0).astype(np.uint8)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        r = int(255 * min(1.0, 1.2 * t))
        g = int(255 * (t ** 1.7) * 0.8)
        b = int(255 * (t ** 3.0) * 0.3)
        lut[i] = (r, g, b)
    rgb = lut[a]
    return Image.fromarray(rgb, mode=None).convert("RGB")

def blend_heatmap(im_rgb: Image.Image, heat_rgb: Image.Image, alpha: float = 0.45) -> Image.Image:
    heat = heat_rgb.resize(im_rgb.size, resample=Image.BILINEAR)
    return Image.blend(im_rgb, heat, alpha=alpha)


def _ensure_2d_mask(mask_any: np.ndarray) -> np.ndarray:
    m = np.asarray(mask_any, dtype=np.float32)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    if m.ndim == 2:
        return m
    m = np.squeeze(m)
    if m.ndim == 2:
        return m
    if m.ndim == 3:
        if m.shape[0] in (1, 3, 4) and m.shape[1] != m.shape[0]:
            return m.mean(axis=0).astype(np.float32)
        if m.shape[-1] in (1, 3, 4) and m.shape[-2] != m.shape[-1]:
            return m.mean(axis=-1).astype(np.float32)
        return m.mean(axis=0).astype(np.float32)
    while m.ndim > 2:
        m = m.mean(axis=0)
    return np.asarray(m, dtype=np.float32)


def _build_rollout_mask_binary(mask_2d: np.ndarray, q: float) -> np.ndarray:
    m = np.asarray(mask_2d, dtype=np.float32)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.min(m)) if m.size else 0.0
    mx = float(np.max(m)) if m.size else 0.0
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) <= 1e-12:
        m = np.zeros_like(m, dtype=np.float32)
    else:
        m = (m - mn) / (mx - mn)
    thr = float(np.quantile(m, q)) if m.size else 1.0
    bw = (m >= thr)
    if not bw.any() and m.size:
        bw = (m == float(np.max(m)))
    return bw.astype(bool)


def _resize_mask(mask_bool: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    m_img = Image.fromarray(mask_bool.astype(np.uint8) * 255).resize((w, h), resample=Image.NEAREST)
    return (np.asarray(m_img) > 0)


def _shift_mask(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = m.shape
    out = np.zeros_like(m, dtype=bool)
    y0 = max(0, dy)
    y1 = h + min(0, dy)
    x0 = max(0, dx)
    x1 = w + min(0, dx)
    out[y0:y1, x0:x1] = m[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def _mask_edges(mask_bool: np.ndarray) -> np.ndarray:
    if mask_bool.size == 0:
        return mask_bool
    m = mask_bool.astype(bool)
    eroded = m.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            eroded &= _shift_mask(m, dy, dx)
    return m & (~eroded)


def _dilate_mask(mask_bool: np.ndarray, steps: int = 1) -> np.ndarray:
    out = mask_bool.astype(bool)
    for _ in range(max(0, steps)):
        expanded = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                expanded |= _shift_mask(out, dy, dx)
        out = expanded
    return out


def _overlay_mask(im: Image.Image, mask_bool: np.ndarray, color: Tuple[int, int, int], alpha: int) -> Image.Image:
    base = im.convert("RGBA")
    h, w = mask_bool.shape
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[mask_bool] = (color[0], color[1], color[2], alpha)
    overlay = Image.fromarray(arr)
    return Image.alpha_composite(base, overlay).convert("RGB")


def _overlay_edges(im: Image.Image, edges_bool: np.ndarray, color: Tuple[int, int, int], alpha: int = 255) -> Image.Image:
    base = im.convert("RGBA")
    h, w = edges_bool.shape
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[edges_bool] = (color[0], color[1], color[2], alpha)
    overlay = Image.fromarray(arr)
    return Image.alpha_composite(base, overlay).convert("RGB")


def _get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        bb = draw.textbbox((0, 0), text, font=font)
        return max(1, bb[2] - bb[0]), max(1, bb[3] - bb[1])
    return draw.textsize(text, font=font)


def render_paper_figure(
    exp_name: str,
    abl_name: str,
    item: ItemPaths,
    metrics: ItemMetrics,
    out_png: Path,
    max_pixels: int,
    topk: int = 5,
) -> None:
    """
    Paper-friendly figure: 4 panels only.
    Concepts + ROI/crop-related info are written to a sidecar JSON:
      out_png.with_suffix(".json")
    """
    im = Image.open(item.input_png).convert("RGB")
    im_ds, _scale = downscale_rgb(im, max_pixels=max_pixels)

    attn = _ensure_2d_mask(np.load(item.attn_npy))
    attn01 = normalize01(attn)
    heat = attn_to_heatmap_rgb(attn01).resize(im_ds.size, resample=Image.BILINEAR)

    bbox_obj = read_json(item.roi_bbox_json)
    q = _parse_quantile(bbox_obj.get("quantile", bbox_obj.get("threshold", 0.90)))

    mask = _build_rollout_mask_binary(attn, q)
    mask_im = _resize_mask(mask, im_ds.size)

    # Panels (4 only)
    p1 = im_ds
    p2 = _overlay_mask(im_ds, mask_im, color=(255, 0, 0), alpha=80)
    p3 = blend_heatmap(im_ds, heat, alpha=0.45)

    edges = _mask_edges(mask_im)
    edges = _dilate_mask(edges, steps=1)
    p4 = _overlay_edges(p3, edges, color=(120, 0, 140), alpha=255)

    panels = [p1, p2, p3, p4]
    titles = ["Input", "Input + ROI mask", "Input + Attn rollout", "Attn + ROI contour"]

    # Layout
    n = len(panels)
    w, h = panels[0].size
    outer = 12
    col_gap = 10
    header_h = 26
    title_h = 20
    row_gap = 8
    total_w = outer * 2 + n * w + (n - 1) * col_gap
    total_h = outer * 2 + header_h + title_h + row_gap + h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    header = f"{exp_name} | {abl_name} | {item.idx}"
    header_font = _get_font(18)
    title_font = _get_font(14)
    d.text((outer, outer), header, fill=(0, 0, 0), font=header_font)

    y_titles = outer + header_h
    x = outer
    for t in titles:
        tw = _text_size(d, t, font=title_font)[0]
        tx = x + (w - tw) // 2
        d.text((tx, y_titles), t, fill=(0, 0, 0), font=title_font)
        x += w + col_gap

    y_panels = outer + header_h + title_h + row_gap
    x = outer
    for p in panels:
        canvas.paste(p, (x, y_panels))
        x += w + col_gap

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, format="PNG", optimize=True)

    # Sidecar JSON (concepts + ROI info)
    concept_meta = {}
    if item.concept_scores_json is not None and item.concept_scores_json.exists():
        try:
            cs = read_json(item.concept_scores_json)
            if isinstance(cs, dict):
                meta = cs.get("meta")
                if isinstance(meta, dict):
                    # keep only lightweight metadata (avoid huge dumps)
                    for k in ("name", "version", "registry", "concept_set", "concepts_name", "concepts_version"):
                        if k in meta:
                            concept_meta[k] = meta.get(k)
        except Exception:
            pass

    concepts_payload = []
    for k, v in (metrics.top_concepts[:topk] if metrics.top_concepts else []):
        concepts_payload.append({
            "short_name": str(k),
            "display_name": concept_display_name(str(k)),
            "score": float(v),
            "score_2dp": f"{float(v):.2f}",
        })

    sidecar = {
        "exp": exp_name,
        "ablation": abl_name,
        "idx": item.idx,
        "figure_png": str(out_png),
        "figure_json": str(out_png.with_suffix(".json")),
        "paths": {
            "idx_dir": str(item.dir),
            "input_png": str(item.input_png),
            "attn_rollout_npy": str(item.attn_npy),
            "roi_bbox_json": str(item.roi_bbox_json),
            "concept_scores_json": str(item.concept_scores_json) if item.concept_scores_json is not None else None,
        },
        "roi_bbox_xyxy": list(metrics.roi_bbox_xyxy),
        "attention_mass_in_roi": float(metrics.attention_mass_in_roi),
        "peak_in_roi": bool(metrics.peak_in_roi),
        "rollout_quantile": float(q),
        "concepts_meta": concept_meta,
        "topk": int(topk),
        "concepts": concepts_payload,
    }
    write_json(out_png.with_suffix(".json"), sidecar)


# -------------------------
# Summaries
# -------------------------

def pick_items_for_figures(
    run_dir: Path,
    all_item_dirs: List[Path],
    per_ablation_figures: int,
    seed: int,
    selection_mode: str = "selection",
    per_class_figures: int = 0,
    class_source: str = "true",
    min_concept_score: float = 0.0,
    min_concept_margin: float = 0.0,
) -> List[Path]:
    random.seed(seed)

    if selection_mode == "concept_top":
        items_dir = run_dir / "items"
        per_class: Dict[str, List[Tuple[float, Path]]] = {}
        for d in all_item_dirs:
            ip = build_item_paths(d)
            if ip is None or ip.concept_scores_json is None:
                continue
            try:
                cs = read_json(ip.concept_scores_json)
            except Exception:
                continue
            meta = cs.get("meta", {}) if isinstance(cs, dict) else {}
            src = class_source if class_source in ("true", "pred") else "auto"
            cls = _item_class_from_meta(meta, src)
            if not cls:
                continue
            scores = _extract_top_scores(cs)
            if not scores:
                continue
            scores.sort(reverse=True)
            top1 = scores[0]
            top2 = scores[1] if len(scores) > 1 else 0.0
            margin = top1 - top2
            if top1 < min_concept_score or margin < min_concept_margin:
                continue
            per_class.setdefault(cls, []).append((margin, items_dir / ip.idx))

        chosen: List[Path] = []
        for cls, rows in per_class.items():
            rows.sort(key=lambda r: r[0], reverse=True)
            take = per_class_figures if per_class_figures > 0 else per_ablation_figures
            for _margin, p in rows[:take]:
                chosen.append(p)
        return chosen

    if selection_mode == "random":
        if len(all_item_dirs) <= per_ablation_figures:
            return all_item_dirs
        return random.sample(all_item_dirs, k=per_ablation_figures)

    sel = discover_selection(run_dir)
    if sel:
        chosen = [p for p in sel if p.name.startswith("idx_")]
        return chosen[:per_ablation_figures]

    if len(all_item_dirs) <= per_ablation_figures:
        return all_item_dirs
    return random.sample(all_item_dirs, k=per_ablation_figures)

def summarize_metrics(metrics: List[ItemMetrics]) -> dict:
    okm = [m for m in metrics if m.ok]
    if not okm:
        return {
            "n_total": len(metrics),
            "n_ok": 0,
            "n_fail": len(metrics),
            "fail_reasons": {},
        }

    fail_reasons: Dict[str, int] = {}
    for m in metrics:
        if m.ok:
            continue
        fail_reasons[m.reason] = fail_reasons.get(m.reason, 0) + 1

    masses = [m.attention_mass_in_roi for m in okm if not math.isnan(m.attention_mass_in_roi)]
    peaks = [1 if m.peak_in_roi else 0 for m in okm]
    concept_counts: Dict[str, int] = {}
    for m in okm:
        for k, _v in m.top_concepts[:10]:
            concept_counts[k] = concept_counts.get(k, 0) + 1

    top_concepts = sorted(concept_counts.items(), key=lambda kv: kv[1], reverse=True)[:30]

    def q(x: List[float], p: float) -> float:
        if not x:
            return float("nan")
        xs = sorted(x)
        i = int(round((len(xs) - 1) * p))
        return float(xs[i])

    return {
        "n_total": len(metrics),
        "n_ok": len(okm),
        "n_fail": len(metrics) - len(okm),
        "fail_reasons": fail_reasons,
        "attention_mass_in_roi": {
            "mean": float(statistics.mean(masses)) if masses else float("nan"),
            "median": float(statistics.median(masses)) if masses else float("nan"),
            "p10": q(masses, 0.10),
            "p90": q(masses, 0.90),
        },
        "peak_in_roi_rate": float(statistics.mean(peaks)) if peaks else float("nan"),
        "top_concepts_freq": [{"concept": k, "count": v} for k, v in top_concepts],
    }

def write_csv_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------------
# Main pipeline
# -------------------------

def run_postdiag(
    models_root: Path,
    out_root: Path,
    per_ablation_figures: int,
    max_metrics_items: int,
    max_pixels: int,
    seed: int,
    selection_mode: str,
    per_class_figures: int,
    class_source: str,
    min_concept_score: float,
    min_concept_margin: float,
) -> int:
    exps = discover_experiments(models_root)
    if not exps:
        print(f"No exp_* found in: {models_root}")
        return 2

    global_rows: List[dict] = []
    global_json: dict = {"models_root": str(models_root), "experiments": []}

    for exp in exps:
        exp_obj = {"exp": exp.name, "ablations": []}
        abls = discover_ablations(exp)
        for abl in abls:
            attn_dir = abl / "attention_rollout_concept"
            if not attn_dir.exists():
                continue

            run_dir = resolve_run_dir(attn_dir)
            if run_dir is None:
                continue

            item_dirs = discover_items(run_dir)
            if not item_dirs:
                continue

            metrics_sample = item_dirs
            if len(metrics_sample) > max_metrics_items:
                random.seed(seed)
                metrics_sample = random.sample(metrics_sample, k=max_metrics_items)

            items: List[ItemPaths] = []
            for d in metrics_sample:
                ip = build_item_paths(d)
                if ip is not None:
                    items.append(ip)

            metrics: List[ItemMetrics] = [compute_metrics(ip) for ip in items]
            summ = summarize_metrics(metrics)

            out_abl = out_root / exp.name / abl.name
            write_json(out_abl / "summary.json", summ)

            per_item_rows: List[dict] = []
            for m in metrics:
                per_item_rows.append({
                    "exp": exp.name,
                    "ablation": abl.name,
                    "idx": m.idx,
                    "ok": int(m.ok),
                    "reason": m.reason,
                    "input_w": m.input_size[0],
                    "input_h": m.input_size[1],
                    "attn_w": m.attn_size[0],
                    "attn_h": m.attn_size[1],
                    "attention_mass_in_roi": m.attention_mass_in_roi,
                    "peak_in_roi": int(m.peak_in_roi),
                    "top_concept_1": m.top_concepts[0][0] if m.top_concepts else "",
                    "top_concept_1_score": m.top_concepts[0][1] if m.top_concepts else float("nan"),
                })
            write_csv_rows(out_abl / "summary.csv", per_item_rows)

            chosen_dirs = pick_items_for_figures(
                run_dir,
                item_dirs,
                per_ablation_figures,
                seed,
                selection_mode=selection_mode,
                per_class_figures=per_class_figures,
                class_source=class_source,
                min_concept_score=min_concept_score,
                min_concept_margin=min_concept_margin,
            )

            paper_dir = out_abl / "paper"
            montage_tiles: List[Image.Image] = []

            m_by_idx = {m.idx: m for m in metrics}

            for fig_idx, d in enumerate(chosen_dirs, start=1):
                ip = build_item_paths(d)
                if ip is None:
                    continue
                mm = m_by_idx.get(ip.idx) or compute_metrics(ip)

                paper_png = paper_dir / f"fig_{fig_idx:02d}_{ip.idx}.png"
                render_paper_figure(
                    exp_name=exp.name,
                    abl_name=abl.name,
                    item=ip,
                    metrics=mm,
                    out_png=paper_png,
                    max_pixels=max_pixels,
                    topk=5,
                )

                # montage thumbnail from paper figure
                try:
                    t = Image.open(paper_png).convert("RGB")
                    t = t.resize((900, int(900 * (t.size[1] / t.size[0]))), resample=Image.BILINEAR)
                    montage_tiles.append(t)
                except Exception:
                    pass

            if montage_tiles:
                w0 = max(im.size[0] for im in montage_tiles)
                h0 = sum(im.size[1] for im in montage_tiles) + 12 * (len(montage_tiles) - 1)
                canvas = Image.new("RGB", (w0, h0), (255, 255, 255))
                y = 0
                for im in montage_tiles:
                    canvas.paste(im, (0, y))
                    y += im.size[1] + 12
                canvas.save(out_abl / "montage.png", format="PNG", optimize=True)

            row = {
                "exp": exp.name,
                "ablation": abl.name,
                "n_items": len(item_dirs),
                "n_metrics_sample": len(metrics),
                "n_ok": summ.get("n_ok", 0),
                "n_fail": summ.get("n_fail", 0),
                "mass_roi_mean": summ.get("attention_mass_in_roi", {}).get("mean", float("nan")),
                "mass_roi_median": summ.get("attention_mass_in_roi", {}).get("median", float("nan")),
                "peak_in_roi_rate": summ.get("peak_in_roi_rate", float("nan")),
                "out_dir": str(out_abl),
            }
            global_rows.append(row)
            exp_obj["ablations"].append({"ablation": abl.name, "summary": summ, "out_dir": str(out_abl)})

        if exp_obj["ablations"]:
            global_json["experiments"].append(exp_obj)

    write_csv_rows(out_root / "postdiag_global.csv", global_rows)
    write_json(out_root / "postdiag_global.json", global_json)

    print(f"OK: wrote report in {out_root}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", type=str, required=True, help="Root containing exp_* (timestamp).")
    ap.add_argument("--out-root", type=str, required=True, help="Output root for report and figures.")
    ap.add_argument("--per-ablation-figures", type=int, default=6, help="Number of figures per ablation.")
    ap.add_argument("--max-metrics-items", type=int, default=200, help="Max items per ablation for metrics (sample).")
    ap.add_argument("--max-pixels", type=int, default=1024, help="Downscale max side for figures (speed).")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--selection-mode",
        type=str,
        default="selection",
        choices=["selection", "concept_top", "random"],
        help="Selection method: selection (xai_selection), concept_top (per class), random.",
    )
    ap.add_argument(
        "--per-class-figures",
        type=int,
        default=0,
        help="If >0 and selection-mode=concept_top, number of patches per class.",
    )
    ap.add_argument(
        "--class-source",
        type=str,
        default="true",
        choices=["true", "pred", "auto"],
        help="Class used for selection (true/pred/auto).",
    )
    ap.add_argument(
        "--min-concept-score",
        type=float,
        default=0.0,
        help="Minimum filter on top-1 concept score (selection-mode=concept_top).",
    )
    ap.add_argument(
        "--min-concept-margin",
        type=float,
        default=0.0,
        help="Minimum filter on margin (top1 - top2) concept (selection-mode=concept_top).",
    )
    args = ap.parse_args()

    models_root = Path(args.models_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    return run_postdiag(
        models_root=models_root,
        out_root=out_root,
        per_ablation_figures=max(0, args.per_ablation_figures),
        max_metrics_items=max(1, args.max_metrics_items),
        max_pixels=max(128, args.max_pixels),
        seed=args.seed,
        selection_mode=args.selection_mode,
        per_class_figures=max(0, args.per_class_figures),
        class_source=args.class_source,
        min_concept_score=max(0.0, args.min_concept_score),
        min_concept_margin=max(0.0, args.min_concept_margin),
    )


if __name__ == "__main__":
    raise SystemExit(main())
