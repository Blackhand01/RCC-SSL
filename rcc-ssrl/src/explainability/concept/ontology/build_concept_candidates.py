#!/usr/bin/env python3
"""
Build RCC concept candidate CSV from a WebDataset split.

Given a train WebDataset (shard-*.tar) with entries like:
  <wds_key>.img.jpg
  <wds_key>.meta.json   (must contain class_label)

This script:
  - extracts up to N patches per class into PNGs under --images-root/<class_label>/
  - writes a CSV with columns: image_path, wds_key, class_label

No VLM calls happen here: this is Stage 0a (dataset-level candidates). Stage 0b
will consume the CSV to query the VLM and build the concept bank.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
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
        default=500, 
        help="Maximum number of candidate patches per class_label (default: 2000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed (currently unused, kept for compatibility).",
    )
    return p.parse_args()


def _suffixes(raw: str) -> List[str]:
    """Return list of suffixes with leading dot, splitting on ';'."""
    out: List[str] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        out.append(part if part.startswith(".") else f".{part}")
    return out


def _find_meta_member(tf: tarfile.TarFile, base: str, meta_suffixes: Sequence[str]) -> tarfile.TarInfo | None:
    for suf in meta_suffixes:
        candidate = f"{base}{suf}"
        try:
            return tf.getmember(candidate)
        except KeyError:
            continue
    return None


def _iter_image_members(tf: tarfile.TarFile, image_suffixes: Sequence[str]) -> Iterable[tarfile.TarInfo]:
    for member in tf.getmembers():
        for suf in image_suffixes:
            if member.name.endswith(suf):
                yield member
                break


# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    shards = sorted(args.train_dir.glob(args.pattern))
    if not shards:
        raise FileNotFoundError(f"No shards found under {args.train_dir} matching {args.pattern}")

    image_suffixes = _suffixes(args.image_key)
    meta_suffixes = _suffixes(args.meta_key)

    args.images_root.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    per_class_counts: Dict[str, int] = {}
    written_rows = 0

    with args.out_csv.open("w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["image_path", "wds_key", "class_label"])

        for shard in shards:
            print(f"[INFO] Processing shard {shard.name}")
            with tarfile.open(shard) as tf:
                for img_member in _iter_image_members(tf, image_suffixes):
                    # strip the image suffix to get base key
                    for suf in image_suffixes:
                        if img_member.name.endswith(suf):
                            base = img_member.name[: -len(suf)]
                            break
                    else:
                        continue

                    meta_member = _find_meta_member(tf, base, meta_suffixes)
                    if meta_member is None:
                        continue

                    with tf.extractfile(meta_member) as mf:
                        if mf is None:
                            continue
                        try:
                            meta = json.load(mf)
                        except Exception:
                            continue

                    class_label = str(meta.get("class_label", "")).strip()
                    if not class_label:
                        continue

                    # enforce per-class cap
                    current = per_class_counts.get(class_label, 0)
                    if args.max_patches_per_class > 0 and current >= args.max_patches_per_class:
                        continue

                    # derive wds_key and output path
                    wds_key = base
                    out_dir = args.images_root / class_label
                    out_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"{wds_key.replace('/', '_')}.png"
                    out_img_path = out_dir / filename

                    with tf.extractfile(img_member) as imf:
                        if imf is None:
                            continue
                        try:
                            img = Image.open(io.BytesIO(imf.read())).convert("RGB")
                        except Exception:
                            continue

                    img.save(out_img_path, format="PNG")

                    writer.writerow([out_img_path.as_posix(), wds_key, class_label])
                    per_class_counts[class_label] = current + 1
                    written_rows += 1

    if written_rows == 0:
        raise RuntimeError(
            f"No candidate patches were written to {args.out_csv}. "
            "Check that shards contain the expected keys."
        )

    print(
        f"[OK] concept_candidates CSV written to {args.out_csv} "
        f"({written_rows} rows, classes={len(per_class_counts)})"
    )


if __name__ == "__main__":
    main()
