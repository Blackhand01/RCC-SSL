#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postdiag_xai_fast.py

Post-diagnostica XAI (attention-rollout + concept scores) per modelli/ablation.

Input atteso (per item):
  attention_rollout_concept/<run>/items/idx_xxxxxxxx/
    - input.png
    - attn_rollout.npy  (H x W) o (h x w)
    - roi_bbox.json     (bbox in pixel su input)
    - roi.png           (opzionale, crop)
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
      montage.png

Uso:
  python postdiag_xai_fast.py \
    --models-root /beegfs-scratch/.../models \
    --out-root /home/.../explainability/output \
    --per-ablation-figures 6 \
    --max-metrics-items 200 \
    --max-pixels 1024

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Evita falsi allarmi su immagini grandi (tipico in pipeline WSI/patch).
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
    Supporta vari formati comuni:
      {x1,y1,x2,y2} / {xmin,ymin,xmax,ymax} / {left,top,right,bottom} / {x,y,w,h}
      oppure lista [x1,y1,x2,y2]
    """
    if isinstance(obj, list) and len(obj) >= 4:
        return int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])

    keys = set(obj.keys())
    if {"x1", "y1", "x2", "y2"}.issubset(keys):
        return int(obj["x1"]), int(obj["y1"]), int(obj["x2"]), int(obj["y2"])
    if {"xmin", "ymin", "xmax", "ymax"}.issubset(keys):
        return int(obj["xmin"]), int(obj["ymin"]), int(obj["xmax"]), int(obj["ymax"])
    if {"left", "top", "right", "bottom"}.issubset(keys):
        return int(obj["left"]), int(obj["top"]), int(obj["right"]), int(obj["bottom"])
    if {"x", "y", "w", "h"}.issubset(keys):
        x1, y1 = int(obj["x"]), int(obj["y"])
        return x1, y1, x1 + int(obj["w"]), y1 + int(obj["h"])

    raise ValueError(f"Formato bbox non riconosciuto: keys={sorted(list(keys))}")

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
    Se selection/ contiene idx_* (symlink o cartelle), lo usiamo come shortlist.
    """
    sel = run_dir / "selection"
    if not sel.exists() or not sel.is_dir():
        return []
    idxs = sorted([p for p in sel.iterdir() if p.name.startswith("idx_")])
    return idxs

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

        # Porta bbox nello spazio della mappa attenzione se dimensioni diverse
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

def overlay_roi_mask(im: Image.Image, bbox: Tuple[int, int, int, int], alpha: int = 80) -> Image.Image:
    base = im.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    x1, y1, x2, y2 = bbox
    d.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, alpha))
    return Image.alpha_composite(base, overlay).convert("RGB")

def attn_to_heatmap_rgb(attn: np.ndarray) -> Image.Image:
    """
    Converte una mappa [0,1] in RGB usando una LUT semplice (no Matplotlib figure).
    Palette tipo "inferno-lite" custom per evitare dipendenze extra.
    """
    a = (np.clip(attn, 0.0, 1.0) * 255.0).astype(np.uint8)
    # LUT 256x3 (gradiente scuro->chiaro con tinta calda)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        r = int(255 * min(1.0, 1.2 * t))
        g = int(255 * (t ** 1.7) * 0.8)
        b = int(255 * (t ** 3.0) * 0.3)
        lut[i] = (r, g, b)
    rgb = lut[a]
    return Image.fromarray(rgb, mode=None).convert("RGB")  # mode=None -> niente 'mode=' deprecato

def blend_heatmap(im_rgb: Image.Image, heat_rgb: Image.Image, alpha: float = 0.45) -> Image.Image:
    heat = heat_rgb.resize(im_rgb.size, resample=Image.BILINEAR)
    return Image.blend(im_rgb, heat, alpha=alpha)

def text_block(lines: List[str], width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    y = 8
    for ln in lines:
        d.text((8, y), ln, fill=(0, 0, 0), font=font)
        y += 20
        if y > height - 20:
            break
    return canvas

def panel_2x2(a: Image.Image, b: Image.Image, c: Image.Image, dimg: Image.Image, pad: int = 12) -> Image.Image:
    # uniforma dimensioni usando la max
    w = max(a.size[0], b.size[0], c.size[0], dimg.size[0])
    h = max(a.size[1], b.size[1], c.size[1], dimg.size[1])

    def fit(im: Image.Image) -> Image.Image:
        if im.size == (w, h):
            return im
        return im.resize((w, h), resample=Image.BILINEAR)

    a2, b2, c2, d2 = fit(a), fit(b), fit(c), fit(dimg)
    out = Image.new("RGB", (2 * w + pad, 2 * h + pad), (255, 255, 255))
    out.paste(a2, (0, 0))
    out.paste(b2, (w + pad, 0))
    out.paste(c2, (0, h + pad))
    out.paste(d2, (w + pad, h + pad))
    return out

def render_item_panel(item: ItemPaths, metrics: ItemMetrics, out_png: Path, max_pixels: int, topk: int = 6) -> None:
    im = Image.open(item.input_png).convert("RGB")
    im_ds, scale = downscale_rgb(im, max_pixels=max_pixels)

    x1, y1, x2, y2 = metrics.roi_bbox_xyxy
    # scala bbox
    x1s = int(round(x1 * scale)); y1s = int(round(y1 * scale))
    x2s = int(round(x2 * scale)); y2s = int(round(y2 * scale))
    x1s, y1s, x2s, y2s = clamp_bbox(x1s, y1s, x2s, y2s, im_ds.size[0], im_ds.size[1])
    bbox_s = (x1s, y1s, x2s, y2s)

    # attn load + normalize + resize nello spazio downscaled (per evitare costi)
    attn = np.load(item.attn_npy)
    if attn.ndim == 3:
        attn = attn.squeeze()
    attn01 = normalize01(attn)
    heat = attn_to_heatmap_rgb(attn01).resize(im_ds.size, resample=Image.BILINEAR)

    # pannelli
    p1 = draw_bbox(im_ds, bbox_s)
    p2 = draw_bbox(heat, bbox_s)
    p3 = draw_bbox(blend_heatmap(im_ds, heat, alpha=0.45), bbox_s)

    # ROI crop + concetti
    crop = im_ds.crop(bbox_s)
    crop = crop.resize((max(256, crop.size[0]), max(256, crop.size[1])), resample=Image.BILINEAR)
    concept_lines = [
        f"{item.idx}",
        f"mass_in_roi={metrics.attention_mass_in_roi:.3f}",
        f"peak_in_roi={int(metrics.peak_in_roi)}",
        "",
        "Top concepts:",
    ]
    for k, v in metrics.top_concepts[:topk]:
        concept_lines.append(f"- {k}: {v:.3f}")
    tb = text_block(concept_lines, width=360, height=crop.size[1])
    p4 = Image.new("RGB", (crop.size[0] + tb.size[0], crop.size[1]), (255, 255, 255))
    p4.paste(crop, (0, 0))
    p4.paste(tb, (crop.size[0], 0))

    fig = panel_2x2(p1, p2, p3, p4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out_png, format="PNG", optimize=True)


# -------------------------
# Summaries
# -------------------------

def pick_items_for_figures(
    run_dir: Path,
    all_item_dirs: List[Path],
    per_ablation_figures: int,
    seed: int,
) -> List[Path]:
    random.seed(seed)

    sel = discover_selection(run_dir)
    if sel:
        # selection potrebbe contenere symlink o cartelle
        chosen = [p for p in sel if p.name.startswith("idx_")]
        return chosen[:per_ablation_figures]

    # fallback: prendi subset casuale (robusto anche senza xai_summary)
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
) -> int:
    exps = discover_experiments(models_root)
    if not exps:
        print(f"Nessuna exp_* trovata in: {models_root}")
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

            # campiona per metriche (evita di macinare decine di migliaia di item)
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

            # write per-ablation outputs
            out_abl = out_root / exp.name / abl.name
            write_json(out_abl / "summary.json", summ)

            # per-item csv (solo campione)
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

            # figure selection (shortlist)
            chosen_dirs = pick_items_for_figures(run_dir, item_dirs, per_ablation_figures, seed)
            figs_dir = out_abl / "figures"
            montage_tiles: List[Image.Image] = []

            # per evitare doppio compute: mappa idx->metrics (solo se presente nel campione)
            m_by_idx = {m.idx: m for m in metrics}

            for d in chosen_dirs:
                ip = build_item_paths(d)
                if ip is None:
                    continue
                if ip.idx in m_by_idx:
                    mm = m_by_idx[ip.idx]
                else:
                    mm = compute_metrics(ip)

                out_png = figs_dir / f"{ip.idx}_panel.png"
                render_item_panel(ip, mm, out_png=out_png, max_pixels=max_pixels, topk=6)

                # per montage: thumbnail
                try:
                    t = Image.open(out_png).convert("RGB")
                    t = t.resize((900, int(900 * (t.size[1] / t.size[0]))), resample=Image.BILINEAR)
                    montage_tiles.append(t)
                except Exception:
                    pass

            # semplice montage verticale
            if montage_tiles:
                w = max(im.size[0] for im in montage_tiles)
                h = sum(im.size[1] for im in montage_tiles) + 12 * (len(montage_tiles) - 1)
                canvas = Image.new("RGB", (w, h), (255, 255, 255))
                y = 0
                for im in montage_tiles:
                    canvas.paste(im, (0, y))
                    y += im.size[1] + 12
                canvas.save(out_abl / "montage.png", format="PNG", optimize=True)

            # global rows
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

    print(f"OK: scritto report in {out_root}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", type=str, required=True, help="Root che contiene exp_* (timestamp).")
    ap.add_argument("--out-root", type=str, required=True, help="Output root per report e figure.")
    ap.add_argument("--per-ablation-figures", type=int, default=6, help="Numero figure per ablation.")
    ap.add_argument("--max-metrics-items", type=int, default=200, help="Max item per ablation per metriche (campione).")
    ap.add_argument("--max-pixels", type=int, default=1024, help="Downscale max lato per figure (velocità).")
    ap.add_argument("--seed", type=int, default=1337)
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
    )


if __name__ == "__main__":
    raise SystemExit(main())
