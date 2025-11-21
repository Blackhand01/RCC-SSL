#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump a few augmented samples per class for a given SSL config.
It rebuilds the exact transforms used in the training loaders, so what you see
matches what the model receives.

Usage:
  python -m src.training.tools.dump_augmentations \
      --config src/training/configs/ablations/dino_v3/exp_dino_v3_abl21.yaml \
      --per-class 2

Optional:
  --out-root /custom/path
  --classes ccRCC pRCC CHROMO ONCO NOT_TUMOR
  --seed 1337
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torchvision.utils import make_grid, save_image

# Project imports (match your repo layout)
from src.training.launch_training import read_yaml, paths as resolve_paths, inject_paths_into_cfg
from src.training.data.webdataset import list_shards, make_wds
from src.training.datasets.labels import normalise_label
from src.training.datasets.transforms import (
    two_view_transform,
    multicrop_transform,
    ijepa_input_transform,
    coerce_to_pil_rgb,
)
DEFAULT_OUT_ROOT = "/home/mla_group_01/rcc-ssrl/src/training/configs/ablations/augms"


def _model_family(cfg: Dict) -> str:
    return str(cfg["model"]["ssl"]["name"]).lower().strip()


def _abl_code(cfg: Dict) -> str:
    """
    Try to extract 'ablNN' from experiment.name (e.g., 'exp_dino_v3_abl21').
    Fallback to a safe token.
    """
    name = str(cfg.get("experiment", {}).get("name", ""))
    for tok in name.split("_"):
        if tok.startswith("abl"):
            return tok
    return "ablNA"


def _class_filter(allowed: Optional[List[str]]):
    if not allowed:
        return lambda s: True
    allowed_norm = {normalise_label(c) for c in allowed}
    return lambda s: normalise_label((s or "")) in allowed_norm


def _build_transform(cfg: Dict):
    """
    Rebuild the exact transform used by SSL loader based on model family and cfg.
    Returns (callable, mode_str) where mode_str in {"two_views","multicrop","single"}.
    """
    ds_cfg = cfg.get("data", {})
    img_size = int(ds_cfg.get("img_size", 224))
    aug_top = cfg.get("aug", {}) or {}
    mname = _model_family(cfg)
    # Flags as in builders.build_ssl_loader
    use_mc_ibot = bool((cfg.get("model", {}).get("ssl", {}) or {}).get("use_multicrop", False))
    use_mc_moco = bool((cfg.get("model", {}).get("ssl", {}) or {}).get("use_multicrop", False)) if mname == "moco_v3" else False

    if (mname in ("moco_v3",) and not use_mc_moco) or (mname == "ibot" and not use_mc_ibot):
        # legacy/2-view path with top-level aug
        ssl_aug_legacy = (cfg.get("model", {}).get("ssl", {}) or {}).get("aug", {})
        jitter = float(ssl_aug_legacy.get("jitter", 0.4))
        blur_p = float(ssl_aug_legacy.get("blur_prob", 0.1))
        gray_p = float(ssl_aug_legacy.get("gray_prob", 0.2))
        solar_p = float(ssl_aug_legacy.get("solarize_prob", 0.0))
        tfm = two_view_transform(
            img_size,
            jitter,
            blur_prob=blur_p,
            gray_prob=gray_p,
            solarize_prob=solar_p,
            cfg_aug=aug_top,
        )
        return tfm, "two_views"

    if mname == "dino_v3" or (mname == "ibot" and use_mc_ibot) or (mname == "moco_v3" and use_mc_moco):
        dino_cfg = ds_cfg.get("dino_v3", {})
        tfm = multicrop_transform(
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
        return tfm, "multicrop"

    if mname == "i_jepa":
        tfm = ijepa_input_transform(img_size, aug_top)
        return tfm, "single"

    raise ValueError(f"Unsupported SSL model family '{mname}'.")


def _collect_samples_per_class(train_dir: str, wanted: Dict[str, int], limit_per_class: int) -> List[Tuple]:
    """
    Iterate the WebDataset and collect raw (PIL image, meta) samples up to limit_per_class per class.
    """
    shards = list_shards(train_dir)
    ds = make_wds(shards, shuffle_shards=32, shuffle_samples=2000)
    # ds yields (img, meta) with img as PIL (decode('pil') in make_wds)
    out = []
    counters = {k: 0 for k in wanted.keys()}
    for img, meta in ds:
        cls = normalise_label((meta or {}).get("class_label", ""))
        if cls not in wanted:
            continue
        if counters[cls] >= limit_per_class:
            # check if all full
            if all(counters[k] >= limit_per_class for k in counters):
                break
            continue
        out.append((img, meta))
        counters[cls] += 1
    return out


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_original(img_pil, out_dir: Path, stem: str) -> None:
    path = out_dir / f"{stem}_orig.png"
    coerce_to_pil_rgb(img_pil).save(path)


def _save_aug_grid(mode: str, aug_out, out_dir: Path, stem: str, max_locals: int = 4) -> None:
    """
    Convert the augmentation output to a grid tensor and save it.
    - two_views: (x1, x2)
    - multicrop: ([g1,g2], [l1,...])
    - single:    x
    """
    tiles: List[torch.Tensor] = []
    if mode == "two_views":
        x1, x2 = aug_out
        tiles = [x1, x2]
    elif mode == "multicrop":
        Gs, Ls = aug_out
        tiles = list(Gs)
        # Resize locals to global size for grid
        if Gs and Ls:
            g_h, g_w = Gs[0].shape[1], Gs[0].shape[2]
            for l in Ls[:max_locals]:
                resized_l = torch.nn.functional.interpolate(
                    l.unsqueeze(0), size=(g_h, g_w), mode='bilinear', align_corners=False
                ).squeeze(0)
                tiles.append(resized_l)
    elif mode == "single":
        tiles = [aug_out]
    else:
        raise ValueError(mode)
    if not tiles:
        return
    # Clamp just in case; inputs are [0,1]
    tiles = [t.clamp(0, 1).cpu() for t in tiles]
    grid = make_grid(torch.stack(tiles, 0), nrow=min(4, len(tiles)))
    save_image(grid, out_dir / f"{stem}_aug_grid.png")


def dump_from_config(cfg: Dict, *, out_root: str = DEFAULT_OUT_ROOT, per_class: int = 2,
                     only_classes: Optional[List[str]] = None, seed: Optional[int] = None,
                     quiet: bool = False) -> Dict[str, str]:
    """
    Core entrypoint: given an already-resolved cfg (with data.webdataset.train_dir),
    dump a few augmented samples per class under the target layout and return a summary.
    """
    if seed is not None:
        torch.manual_seed(int(seed))
    mname = _model_family(cfg)
    abl = _abl_code(cfg)

    # Resolve WebDataset train_dir from cfg (after inject_paths_into_cfg)
    train_dir = str(cfg["data"]["webdataset"]["train_dir"])
    classes_map = (cfg.get("data", {}).get("webdataset", {}) or {}).get("class_to_id", {})
    class_names = list(classes_map.keys())
    class_sel = class_names if not only_classes else only_classes
    allowed = {normalise_label(c): per_class for c in class_sel}
    samples = _collect_samples_per_class(train_dir, allowed, per_class)

    # Build transform
    tfm, mode = _build_transform(cfg)

    # Output folder
    base = Path(out_root) / mname / abl
    _ensure_dir(base)

    # Metadata readme
    meta = {
        "model": mname,
        "ablation": abl,
        "mode": mode,
        "per_class": per_class,
        "train_dir": train_dir,
        "classes": class_sel,
    }
    (base / "readme.json").write_text(json.dumps(meta, indent=2))

    # Iterate and write
    written = 0
    per_cls_counters: Dict[str, int] = {}
    for img_pil, meta in samples:
        cls = normalise_label((meta or {}).get("class_label", ""))
        idx = per_cls_counters.get(cls, 0) + 1
        per_cls_counters[cls] = idx
        target_dir = _ensure_dir(base / cls)
        stem = f"{cls.lower()}_{idx:02d}"
        try:
            _save_original(img_pil, target_dir, stem)
            aug_out = tfm(img_pil)
            _save_aug_grid(mode, aug_out, target_dir, stem)
            written += 1
        except Exception as e:
            if not quiet:
                print(f"[dump] Skip sample {cls} #{idx}: {e}")

    if not quiet:
        print(f"[dump] Wrote {written} samples to {base}")
    return {"out_dir": str(base), "written": str(written)}


def _load_and_inject(cfg_path: str) -> Dict:
    """
    Load YAML and inject resolved paths (train/val/test) using project's resolver.
    """
    cfg = read_yaml(cfg_path)
    resolved = resolve_paths()
    cfg = inject_paths_into_cfg(cfg, resolved)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML ablation config")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--classes", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = _load_and_inject(args.config)
    dump_from_config(
        cfg,
        out_root=args.out_root,
        per_class=int(args.per_class),
        only_classes=args.classes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
