from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageColor, ImageDraw


def _as_numpy(arr: Union[np.ndarray, "torch.Tensor", Iterable]) -> np.ndarray:  # type: ignore[name-defined]
    try:
        import torch

        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(arr)


def _ensure_mask_shape(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    if mask.shape[1] == target_size[0] and mask.shape[0] == target_size[1]:
        return mask
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = mask_img.resize(target_size, resample=Image.NEAREST)
    return (np.asarray(resized) > 0).astype(bool)


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _expand_bbox(
    bbox: Tuple[int, int, int, int], min_size: int, max_w: int, max_h: int
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    if w >= min_size and h >= min_size:
        return x0, y0, x1, y1
    pad_w = max(0, (min_size - w) // 2)
    pad_h = max(0, (min_size - h) // 2)
    x0 = max(0, x0 - pad_w)
    y0 = max(0, y0 - pad_h)
    x1 = min(max_w - 1, x1 + pad_w + ((min_size - w) % 2))
    y1 = min(max_h - 1, y1 + pad_h + ((min_size - h) % 2))
    return x0, y0, x1, y1


def roi_from_rollout(
    rollout_map: Union[np.ndarray, "torch.Tensor"], *, quantile: float = 0.9, min_size_px: int = 64
) -> np.ndarray:
    """
    Build a binary ROI mask from an attention rollout map.
    """
    arr = _as_numpy(rollout_map).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    thr = np.quantile(arr, quantile)
    mask = arr >= thr
    if not mask.any():
        mask = arr == arr.max()
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        h, w = mask.shape
        cy, cx = h // 2, w // 2
        half = max(1, min_size_px // 2)
        y0, y1 = max(0, cy - half), min(h - 1, cy + half)
        x0, x1 = max(0, cx - half), min(w - 1, cx + half)
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
        return mask
    x0, y0, x1, y1 = _expand_bbox(bbox, min_size_px, mask.shape[1], mask.shape[0])
    out = np.zeros_like(mask, dtype=bool)
    out[y0 : y1 + 1, x0 : x1 + 1] = mask[y0 : y1 + 1, x0 : x1 + 1]
    return out


def crop_from_mask(image: Image.Image, mask: np.ndarray, min_size_px: int = 32) -> Image.Image:
    """
    Crop a PIL image to the bounding box of the mask. Falls back to the input image if
    the mask is empty.
    """
    mask_bool = mask.astype(bool)
    mask_bool = _ensure_mask_shape(mask_bool, image.size)
    bbox = _bbox_from_mask(mask_bool)
    if bbox is None:
        return image
    x0, y0, x1, y1 = _expand_bbox(bbox, min_size_px, image.size[0], image.size[1])
    return image.crop((x0, y0, x1 + 1, y1 + 1))


def overlay_contour(
    image: Image.Image, mask: np.ndarray, color: Union[str, Tuple[int, int, int]] = "blue"
) -> Image.Image:
    """
    Draw a 1px contour of the mask on top of the image (ablation / visualization only).
    """
    mask_bool = _ensure_mask_shape(mask.astype(bool), image.size)
    if isinstance(color, str):
        color_rgb = ImageColor.getrgb(color)
    else:
        color_rgb = tuple(int(c) for c in color)

    # Boundary = mask pixels with at least one non-mask neighbour
    up = np.roll(mask_bool, 1, axis=0)
    down = np.roll(mask_bool, -1, axis=0)
    left = np.roll(mask_bool, 1, axis=1)
    right = np.roll(mask_bool, -1, axis=1)
    boundary = mask_bool & ~(up & down & left & right)

    if not boundary.any():
        return image

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    ys, xs = np.where(boundary)
    for y, x in zip(ys.tolist(), xs.tolist()):
        draw.point((int(x), int(y)), fill=(*color_rgb, 255))

    base = image.convert("RGBA")
    composed = Image.alpha_composite(base, overlay)
    return composed.convert("RGB")
