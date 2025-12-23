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
