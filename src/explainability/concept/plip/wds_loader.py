#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

try:
    import webdataset as wds
except Exception as e:
    raise RuntimeError("webdataset is required for PLIP pipeline (pip install webdataset)") from e


def parse_meta(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, (bytes, bytearray)):
        try:
            return json.loads(x.decode("utf-8"))
        except Exception:
            return {}
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def list_shards(split_dir: Path, pattern: str) -> List[str]:
    split_dir = Path(split_dir)

    # Path("") becomes "." -> a very common misconfiguration when YAML has split_dir: ""
    if str(split_dir).strip() in ("", "."):
        raise ValueError(
            "WebDataset split_dir is empty. "
            "Set data.webdataset.split_dir in the YAML, or export WDS_TRAIN_DIR/WDS_TEST_DIR (or WDS_DIR) "
            "and let the caller inject it."
        )

    # Support passing a single shard tar directly
    if split_dir.is_file():
        return [str(split_dir)]

    if not split_dir.is_dir():
        raise FileNotFoundError(f"WebDataset split_dir not found: {split_dir}")

    # If the pattern is a non-glob URL/template, forward as-is (glob won't expand it).
    # WebDataset can still handle such URL-style specs internally.
    if any(tok in pattern for tok in ("{", "}", "pipe:", "::")):
        return [str(split_dir / pattern)]

    shard_glob = str(split_dir / pattern)
    shards = sorted(glob.glob(shard_glob))
    if shards:
        return shards

    # Helpful error for pattern mismatch: show what tar files exist.
    any_tars = sorted(glob.glob(str(split_dir / "*.tar")))
    if any_tars:
        ex = ", ".join(Path(p).name for p in any_tars[:8])
        raise FileNotFoundError(
            f"No shards matched pattern '{pattern}' under {split_dir}. "
            f"Found .tar files e.g. {ex}. "
            "Fix data.webdataset.pattern in the config to match your shard naming."
        )
    raise FileNotFoundError(f"No .tar shards found under {split_dir} (pattern '{pattern}').")


def build_wds_loader(
    split_dir: Path,
    pattern: str,
    image_key: str,
    meta_key: str,
    preprocess: Optional[Callable],
    batch_size: int,
    num_workers: int,
    *,
    return_raw: bool = False,
) -> Iterable[Tuple[Optional[torch.Tensor], List[Dict[str, Any]], List[str], Optional[List[Any]]]]:
    """
    Build a WebDataset loader that can optionally return raw PIL images alongside
    preprocessed tensors. The preprocess callable is applied only if provided.

    Yields tuples:
      - images: Tensor [B,3,H,W] if preprocess is set, otherwise None
      - metas: list of dict
      - keys: list of shard keys (str)
      - raw_images: list of PIL images if return_raw=True else None
    """
    shards = list_shards(split_dir, pattern)

    def _map_img(img):
        pil_img = img
        proc_img = preprocess(pil_img) if preprocess is not None else pil_img
        if return_raw:
            return proc_img, pil_img
        return proc_img

    ds = (
        wds.WebDataset(
            shards,
            shardshuffle=False,
            handler=wds.warn_and_continue,
            empty_check=False,
        )
        .decode("pil")
        .to_tuple(image_key, meta_key, "__key__", handler=wds.warn_and_continue)
        .map_tuple(_map_img, parse_meta, lambda k: k)
    )

    # Let WebDataset do batching; then use WebLoader (DataLoader wrapper)
    ds = ds.batched(int(batch_size), partial=True)

    try:
        loader = wds.WebLoader(ds, batch_size=None, num_workers=int(num_workers))
    except Exception:
        loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=int(num_workers))

    def _iter():
        for batch in loader:
            if batch is None:
                continue
            imgs, metas, keys = batch
            raw_imgs = None

            if return_raw:
                # WebDataset batching may return either a list of (proc_img, pil_img)
                # or a tuple (proc_batch, raw_batch) depending on collation.
                if isinstance(imgs, tuple) and len(imgs) == 2:
                    proc_part, raw_part = imgs
                    imgs = proc_part
                    raw_imgs = list(raw_part) if isinstance(raw_part, (list, tuple)) else [raw_part]
                else:
                    raw_imgs = [im[1] for im in imgs]
                    imgs = [im[0] for im in imgs]

            imgs_tensor: Optional[torch.Tensor]
            if preprocess is None:
                imgs_tensor = None
            else:
                # WebDataset may already emit a Tensor batch (common case).
                if isinstance(imgs, torch.Tensor):
                    imgs_tensor = imgs
                else:
                    imgs_tensor = torch.stack(list(imgs), dim=0)

            yield imgs_tensor, list(metas), [str(k) for k in keys], raw_imgs

    return _iter()
