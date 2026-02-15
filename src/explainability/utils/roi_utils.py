#!/usr/bin/env python3
from __future__ import annotations

"""
ROI utilities:
  - robust bbox extraction from 2D rollout masks (supports low-res 14x14 etc)
  - safe scaling to image pixel coordinates
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class RoiBox:
    x0: int
    y0: int
    x1: int
    y1: int
    method: str
    threshold: float

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return int(self.x0), int(self.y0), int(self.x1), int(self.y1)


def _normalize_mask(m: np.ndarray) -> np.ndarray:
    """
    Normalize a rollout/heatmap mask to [0,1] and coerce to 2D.
    Accepts common variants: (H,W), (1,H,W), (H,W,1), (B,H,W), (B,1,H,W), etc.
    """
    m = np.asarray(m, dtype=np.float32)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

    if m.ndim == 2:
        pass
    else:
        m = np.squeeze(m)
        if m.ndim == 2:
            pass
        elif m.ndim == 3:
            # Heuristic: if looks like (C,H,W) or (H,W,C), average channels; else average batch-like dim.
            if m.shape[0] in (1, 3, 4) and m.shape[1] != m.shape[0]:
                m = m.mean(axis=0)
            elif m.shape[-1] in (1, 3, 4) and m.shape[-2] != m.shape[-1]:
                m = m.mean(axis=-1)
            else:
                m = m.mean(axis=0)
        else:
            # collapse any remaining leading dims until 2D
            while m.ndim > 2:
                m = m.mean(axis=0)

    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D after coercion, got shape={m.shape}")
    mn = float(np.nanmin(m))
    mx = float(np.nanmax(m))
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) <= 1e-12:
        return np.zeros_like(m, dtype=np.float32)
    out = (m - mn) / (mx - mn)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def extract_bbox_from_mask(
    mask_2d: np.ndarray,
    *,
    img_w: int,
    img_h: int,
    quantile: float = 0.90,
    min_area_frac: float = 0.01,
    pad_frac: float = 0.05,
) -> RoiBox:
    """
    Extract a single bbox around the most salient region.

    - quantile: threshold at q-quantile of normalized mask.
    - min_area_frac: if bbox is too small, fall back to full image.
    - pad_frac: expand bbox by this fraction of its size (clamped).
    """
    if int(img_w) <= 0 or int(img_h) <= 0:
        raise ValueError(f"Invalid image size img_w={img_w} img_h={img_h}")

    m = _normalize_mask(mask_2d)
    thr = float(np.quantile(m, quantile)) if m.size > 0 else 1.0
    bw = (m >= thr)
    ys, xs = np.where(bw)

    # fallback: no pixels above threshold
    if xs.size == 0 or ys.size == 0:
        return RoiBox(0, 0, img_w - 1, img_h - 1, method="fallback_full", threshold=thr)

    # bbox in mask coords
    x0m, x1m = int(xs.min()), int(xs.max())
    y0m, y1m = int(ys.min()), int(ys.max())

    mh, mw = m.shape[0], m.shape[1]
    sx = float(img_w) / float(max(1, mw))
    sy = float(img_h) / float(max(1, mh))

    x0 = int(np.floor(x0m * sx))
    x1 = int(np.ceil((x1m + 1) * sx) - 1)
    y0 = int(np.floor(y0m * sy))
    y1 = int(np.ceil((y1m + 1) * sy) - 1)

    # clamp
    x0 = max(0, min(img_w - 1, x0))
    x1 = max(0, min(img_w - 1, x1))
    y0 = max(0, min(img_h - 1, y0))
    y1 = max(0, min(img_h - 1, y1))

    # padding
    bwx = max(1, x1 - x0 + 1)
    bwy = max(1, y1 - y0 + 1)
    px = int(round(pad_frac * bwx))
    py = int(round(pad_frac * bwy))
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - py)
    x1 = min(img_w - 1, x1 + px)
    y1 = min(img_h - 1, y1 + py)

    area = float((x1 - x0 + 1) * (y1 - y0 + 1))
    full = float(img_w * img_h)
    if full > 0 and (area / full) < float(min_area_frac):
        return RoiBox(0, 0, img_w - 1, img_h - 1, method="fallback_small_bbox", threshold=thr)

    return RoiBox(x0, y0, x1, y1, method="quantile_bbox", threshold=thr)
