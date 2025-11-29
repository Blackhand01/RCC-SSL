#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build concept candidate CSV for RCC concepts directly from TRAIN WebDataset.

Dataset-level (project-level), NOT experiment-level.

Input:
- train_dir: root of WebDataset train split
- pattern: glob for shards (e.g. 'shard-*.tar')
- image_key: e.g. 'img.jpg;jpg;jpeg;png'
- meta_key: e.g. 'meta.json;json'

Output:
- concept_candidates_rcc.csv with columns:
    image_path, wds_key, class_label

Additionally exports PNG crops to an images_root directory so the VLM
can read them from filesystem.

NOTE:
- This is STAGE 0, dataset-level, independent from a specific experiment.
- run_full_xai.sh will call this with default paths for the RCC project.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import webdataset as wds
from PIL import Image

from explainability.common.eval_utils import setup_logger, set_seed


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build RCC concept candidate CSV from TRAIN WebDataset."
    )
    p.add_argument(
        "--train-dir",
        required=True,
        type=Path,
        help="Root of train WebDataset split (folder with shard-*.tar).",
    )
    p.add_argument(
        "--pattern",
        default="shard-*.tar",
        type=str,
        help="Glob pattern for shards (default: shard-*.tar).",
    )
    p.add_argument(
        "--image-key",
        default="img.jpg;jpg;jpeg;png",
        type=str,
        help="WebDataset image key(s) (default: img.jpg;jpg;jpeg;png).",
    )
    p.add_argument(
        "--meta-key",
        default="meta.json;json",
        type=str,
        help="WebDataset metadata key(s) (default: meta.json;json).",
    )
    p.add_argument(
        "--out-csv",
        required=True,
        type=Path,
        help="Output CSV path (concept_candidates_rcc.csv).",
    )
    p.add_argument(
        "--images-root",
        required=True,
        type=Path,
        help="Root directory where PNG crops for VLM will be stored.",
    )
    p.add_argument(
        "--max-patches-per-class",
        type=int,
        default=20, # 2000
        help="Maximum number of candidate patches per class_label.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed (used mostly for shuffle).",
    )
    return p.parse_args(argv)


def _parse_meta(meta_raw) -> Dict:
    if isinstance(meta_raw, dict):
        return meta_raw
    if isinstance(meta_raw, (bytes, bytearray)):
        try:
            return json.loads(meta_raw.decode("utf-8"))
        except Exception:
            return {}
    if isinstance(meta_raw, str):
        try:
            return json.loads(meta_raw)
        except Exception:
            return {}
    return {}


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    log = setup_logger("build_concept_candidates_train")
    set_seed(args.seed)

    shard_glob = str(args.train_dir / args.pattern)
    log.info(f"Reading train shards from: {shard_glob}")

    shards = list(Path(args.train_dir).glob(args.pattern))
    if not shards:
        raise FileNotFoundError(f"No shards found matching {shard_glob}")

    log.info(f"Found {len(shards)} shards.")

    ds = (
        wds.WebDataset(
            [str(s) for s in shards],
            shardshuffle=True,
            handler=wds.warn_and_continue,
        )
        .shuffle(10000)
        .decode("pil")
        .to_tuple(args.image_key, args.meta_key, "__key__", handler=wds.warn_and_continue)
    )

    args.images_root.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    import time
    t_start = time.time()
    log_every = 200  # stampa log ogni N esempi

    total_seen = 0
    rows: List[Dict[str, str]] = []
    class_counts: Dict[str, int] = {}

    for img, meta_raw, key in ds:
        total_seen += 1
        if total_seen % log_every == 0:
            elapsed = time.time() - t_start
            eps = elapsed / max(1, total_seen)
            msg = (
                f"[PROGRESS] seen={total_seen} examples, "
                f"class_counts={class_counts}, "
                f"elapsed={elapsed/60:.1f} min, "
                f"~{eps:.3f} s/example"
            )
            log.info(msg)

        meta = _parse_meta(meta_raw)
        cls = str(meta.get("class_label", "")).strip()
        if not cls:
            continue

        cnt = class_counts.get(cls, 0)
        if cnt >= args.max_patches_per_class:
            continue  # already enough for this class

        safe_key = key.replace("/", "_")
        class_dir = args.images_root / cls
        class_dir.mkdir(parents=True, exist_ok=True)

        out_img_path = class_dir / f"{safe_key}.png"

        if isinstance(img, Image.Image):
            pil_img = img.convert("RGB")
        else:
            pil_img = Image.fromarray(img)
        pil_img.save(out_img_path)

        rows.append(
            {
                "image_path": str(out_img_path),
                "wds_key": key,
                "class_label": cls,
            }
        )
        class_counts[cls] = cnt + 1

    if not rows:
        log.warning(
            "No candidate patches collected; nothing to write "
            f"(total_seen={total_seen}, class_counts={class_counts})."
        )
        return

    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "wds_key", "class_label"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    log.info(
        f"Wrote {len(rows)} candidate rows to {args.out_csv} "
        f"(per-class counts: {class_counts})"
    )

    total_elapsed = time.time() - t_start
    log.info(
        f"[SUMMARY] concept_candidates_rcc: rows={len(rows)}, "
        f"classes={list(class_counts.keys())}, "
        f"total_elapsed={total_elapsed/60:.1f} min"
    )


if __name__ == "__main__":
    main()
