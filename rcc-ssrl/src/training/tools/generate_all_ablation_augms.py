#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch generator: enumerate exp_*_abl*.yaml under a model's ablations folder
and generate the 4x4 augmentation matrix for each config.
"""
import argparse
from pathlib import Path
from typing import List

from src.training.tools.ssl_aug_viz_core import dump_matrix_for_cfg_path

DEFAULT_BASE = "/home/mla_group_01/rcc-ssrl/src/training/configs/ablations"
DEFAULT_OUT = "/home/mla_group_01/rcc-ssrl/src/training/configs/ablations/augms"

def _glob_cfgs(abl_root: Path, pattern: str) -> List[Path]:
    return sorted(abl_root.glob(pattern))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dino_v3", "i_jepa", "ibot", "moco_v3"])
    ap.add_argument("--abl-root", default=None, help="Root of ablations for model")
    ap.add_argument("--pattern", default="exp_*_abl*.yaml")
    ap.add_argument("--out-root", default=DEFAULT_OUT)
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--max-locals", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    abl_root = Path(args.abl_root) if args.abl_root else Path(DEFAULT_BASE) / args.model
    cfgs = _glob_cfgs(abl_root, args.pattern)

    ok, fail = 0, 0
    for cfg in cfgs:
        try:
            out = dump_matrix_for_cfg_path(
                cfg,
                args.out_root,
                tile_size=args.tile,
                max_locals=args.max_locals,
                seed=args.seed,
            )
            print(f"[OK] {cfg.name} -> {out}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {cfg.name}: {e}")
            fail += 1

    print(f"[SUMMARY] model={args.model} total={len(cfgs)} ok={ok} fail={fail}")

if __name__ == "__main__":
    main()
