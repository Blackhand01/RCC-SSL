# utils/reproducibility.py
from __future__ import annotations
import os, json, random, shutil, hashlib
from typing import Any, Dict

try:  # pragma: no cover - optional dependency guard
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _require_library(lib, name: str) -> None:
    if lib is None:
        raise RuntimeError(f"Missing optional dependency '{name}'. Install it to use the reproducibility utilities.")

def set_global_seed(seed: int = 1337) -> None:
    _require_library(np, "numpy")
    _require_library(torch, "torch")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # True => più lento
    # ---- PERFORMANCE (FP32) ------------------------------------------------
    # Abilita autotuning dei kernel conv per forme fisse (più veloce).
    torch.backends.cudnn.benchmark = True
    # TF32 accelera matmul/conv mantenendo un percorso FP32 pragmaticamente accurato.
    # Puoi disabilitarlo esportando ALLOW_TF32=0.
    allow_tf32 = os.environ.get("ALLOW_TF32", "1") != "0"
    try:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high" if allow_tf32 else "medium")
    except Exception:
        pass

def _hash_dict(d: Dict[str, Any]) -> str:
    b = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha1(b).hexdigest()[:10]

def snapshot_config(cfg: Dict[str, Any], out_root: str) -> str:
    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, f"config_snapshot_{_hash_dict(cfg)}.json")
    with open(path, "w") as f: json.dump(cfg, f, indent=2)
    return path

def copy_code_snapshot(src_root: str, out_root: str, include=("models","utils",".","launch_training.py")) -> None:
    snap_dir = os.path.join(out_root, "code_snapshot"); os.makedirs(snap_dir, exist_ok=True)
    for p in include:
        sp = os.path.join(src_root, p)
        if os.path.isdir(sp):
            shutil.copytree(sp, os.path.join(snap_dir, os.path.basename(p)), dirs_exist_ok=True)
        elif os.path.isfile(sp):
            shutil.copy2(sp, snap_dir)
