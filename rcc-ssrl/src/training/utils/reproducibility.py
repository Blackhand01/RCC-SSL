# utils/reproducibility.py
from __future__ import annotations
import os, json, random, shutil, hashlib
from typing import Any, Dict, Iterable
import sys

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
    torch.use_deterministic_algorithms(False)  # True => slower
    # ---- PERFORMANCE (FP32) ------------------------------------------------
    # Enable autotuning of conv kernels for fixed shapes (faster).
    torch.backends.cudnn.benchmark = True
    # TF32 accelerates matmul/conv while maintaining a pragmatically accurate FP32 path.
    # You can disable it by setting ALLOW_TF32=0.
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

def _default_snapshot_excludes() -> Iterable[str]:
    return (
        ".git", ".gitignore", ".gitattributes",
        ".venv", "venv", "__pycache__", "*.pyc",
        "outputs", "mlruns", "wandb",
        "dist", "build", "*.egg-info", "*.pth", "*.pt",
        # any nested site-packages in accidental copies
        "site-packages", "*.dist-info", "*.egg-info",
    )

def copy_code_snapshot(src_dir: str, dst_dir: str, *, excludes: Iterable[str] = ()) -> None:
    """
    Copy a slim snapshot of the training code into `dst_dir`.
    Heavy/irrelevant folders are excluded by default.
    """
    if os.environ.get("DISABLE_CODE_SNAPSHOT", "0") == "1":
        return
    os.makedirs(dst_dir, exist_ok=True)
    # If already populated, do nothing (idempotent per run dir)
    if os.listdir(dst_dir):
        return
    patterns = list(_default_snapshot_excludes()) + list(excludes or ())
    shutil.copytree(
        src_dir,
        dst_dir,
        ignore=shutil.ignore_patterns(*patterns),
        dirs_exist_ok=True,
    )
    # Optional: freeze environment for traceability
    try:
        req_out = os.path.join(dst_dir, "pip-freeze.txt")
        os.system(f"{sys.executable} -m pip freeze > '{req_out}'")
    except Exception:
        pass
