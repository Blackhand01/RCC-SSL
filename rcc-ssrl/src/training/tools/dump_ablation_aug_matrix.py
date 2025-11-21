#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI: given one ablation YAML, write the paired 4x4 augmentation matrix.

Output:
  /home/mla_group_01/rcc-ssrl/src/training/configs/ablations/augms/{model}/{ablNN}/aug_matrix.png
"""
import argparse
from pathlib import Path
from src.training.tools.ssl_aug_viz_core import dump_matrix_for_cfg_path

DEFAULT_OUT_ROOT = "/home/mla_group_01/rcc-ssrl/src/training/configs/ablations/augms"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to ablation YAML (exp_*_ablNN.yaml)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="Root folder for outputs")
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--max-locals", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out = dump_matrix_for_cfg_path(
        args.config,
        args.out_root,
        tile_size=args.tile,
        max_locals=args.max_locals,
        seed=args.seed,
    )
    print(f"[OK] Wrote: {out}")

if __name__ == "__main__":
    main()
