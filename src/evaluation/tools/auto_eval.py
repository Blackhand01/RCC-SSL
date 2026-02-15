# file: /home/mla_group_01/rcc-ssrl/src/evaluation/tools/auto_eval.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-discover SSL runs from mlruns, build eval YAMLs, and (optionally) submit SLURM jobs.

- Accepts either the experiment folder (containing exp_*/ runs) or a single run folder.
- Fixed defaults for WebDataset test set (override via env RCC_WDS_TEST_DIR or CLI).
- Prefers class_order and backbone from experiment_snapshot.yaml.
- Outputs eval results inside the *same run folder*: <run_dir>/eval.
- Writes generated eval configs to /home/mla_group_01/rcc-ssrl/src/evaluation/auto_configs.
"""

import os
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple

# ---------- Paths anchored to this file (cwd-agnostic) ----------
SCRIPT_DIR = Path(__file__).resolve().parent              # .../scripts/05_evaluation/tools
EVAL_DIR   = SCRIPT_DIR.parent                            # .../scripts/05_evaluation
CFG_OUTPUT_DIR = (EVAL_DIR / "auto_configs").resolve()    # where .yaml will be written
SBATCH_DEFAULT = (EVAL_DIR / "eval_models.sbatch").resolve()
# print("script dir:", SCRIPT_DIR)
# print("eval dir:", EVAL_DIR)
# print("cfg output dir:", CFG_OUTPUT_DIR)
# print("sbatch default:", SBATCH_DEFAULT)


# ---------- Defaults (overridable by env) ----------
DEFAULT_WDS_TEST_DIR = os.environ.get(
    "RCC_WDS_TEST_DIR",
    "/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test",
)
DEFAULT_WDS_PATTERN = "shard-*.tar"
DEFAULT_IMAGE_KEY = "img.jpg;jpg;jpeg;png"
DEFAULT_META_KEY = "meta.json;json"

# Optional dataset registry keyed by dataset_key in snapshot
DATASET_REGISTRY = {
    "rcc_final_ablation": {
        "test_dir": DEFAULT_WDS_TEST_DIR,
        "pattern": DEFAULT_WDS_PATTERN,
        "image_key": DEFAULT_IMAGE_KEY,
        "meta_key": DEFAULT_META_KEY,
    },
    "rcc_v2": {
        "test_dir": DEFAULT_WDS_TEST_DIR,
        "pattern": DEFAULT_WDS_PATTERN,
        "image_key": DEFAULT_IMAGE_KEY,
        "meta_key": DEFAULT_META_KEY,
    },
}

# ---------- I/O ----------
def read_json(p: Path):
    return json.load(open(p)) if p.is_file() else {}

def read_yaml(p: Path):
    import yaml as _y
    return _y.safe_load(open(p)) if p.is_file() else {}

# ---------- Helpers ----------
def guess_model_name(run_dir: Path) -> str:
    """Infer model short name from checkpoint file or run folder name."""
    for ck in (run_dir / "checkpoints").glob("*__ssl_best.pt"):
        return ck.name.split("__", 1)[0]
    parts = run_dir.name.split("_")
    return parts[1] if len(parts) >= 2 else "ssl_model"

def _first_ckpt_match(run_dir: Path, pat: str) -> Path:
    """Resolve checkpoint path; support wildcards and fallback to any in checkpoints/."""
    cand = list(run_dir.glob(pat))
    if not cand and run_dir.joinpath(pat).exists():
        cand = [run_dir.joinpath(pat)]
    if not cand:
        cand = list(run_dir.glob("checkpoints/*"))
    if not cand:
        raise FileNotFoundError(f"Cannot resolve: {pat} in {run_dir}")
    return cand[0]

def detect_backbone_name(snapshot: Path, ssl_ckpt: Path) -> str:
    """
    Decide the backbone name for eval:
    1) Infer from checkpoint keys (source of truth).
    2) If snapshot agrees, keep snapshot's name; otherwise force a safe default of that family.
    """
    snap_name: Optional[str] = None
    try:
        snap = read_yaml(snapshot) or {}
        snap_name = str(snap.get("model", {}).get("backbone", {}).get("name", "")).strip() or None
    except Exception:
        snap_name = None

    want_vit: Optional[bool] = None
    try:
        import torch
        sd = torch.load(ssl_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        ks = list(sd.keys()) if isinstance(sd, dict) else []
        want_vit = any(k.startswith(("pos_embed", "blocks.", "patch_embed.")) for k in ks)
    except Exception:
        want_vit = None

    if want_vit is True:
        # ViT family
        return snap_name if (snap_name and snap_name.startswith("vit")) else "vit_small_patch16_224"
    if want_vit is False:
        # ResNet family
        return snap_name if (snap_name and not snap_name.startswith("vit")) else "resnet50"
    # Unknown → fallback: prefer snapshot, else resnet50
    return snap_name or "resnet50"

def _torch_load_weights(path: str):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # torch>=2.4
    except TypeError:
        return torch.load(path, map_location="cpu")

def _read_head_in_features(head_ckpt: Path) -> Optional[int]:
    """Return in_features of linear head from its checkpoint, if detectable."""
    import torch
    hd = _torch_load_weights(str(head_ckpt))
    if isinstance(hd, dict) and "state_dict" in hd and isinstance(hd["state_dict"], dict):
        hd = hd["state_dict"]
    if not isinstance(hd, dict):
        return None
    # pick any 2D '...weight' tensor as linear weight
    for k, v in hd.items():
        if k.endswith("weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            return int(v.shape[1])
    return None

def _map_in_features_to_backbone(in_features: int) -> Optional[str]:
    """Common mappings from probe dim to canonical backbones."""
    # ViT family
    if in_features == 384:  return "vit_small_patch16_224"
    if in_features == 768:  return "vit_base_patch16_224"
    if in_features == 1024: return "vit_large_patch16_224"
    # ResNet family (most common)
    if in_features == 2048: return "resnet50"
    if in_features == 512:  return "resnet34"
    # Unknown → None
    return None

def _resolve_wds_params(run_dir: Path, args):
    """Choose test set params: registry → CLI overrides → env/defaults."""
    test_dir = DEFAULT_WDS_TEST_DIR
    pattern = DEFAULT_WDS_PATTERN
    image_key = DEFAULT_IMAGE_KEY
    meta_key  = DEFAULT_META_KEY

    snap = read_yaml(run_dir / "configuration" / "experiment_snapshot.yaml") or {}
    dskey = None
    try:
        dskey = snap["data"]["webdataset"]["dataset_key"]
    except Exception:
        pass

    if dskey and dskey in DATASET_REGISTRY:
        reg = DATASET_REGISTRY[dskey]
        test_dir = reg.get("test_dir", test_dir)
        pattern  = reg.get("pattern", pattern)
        image_key = reg.get("image_key", image_key)
        meta_key  = reg.get("meta_key", meta_key)

    # CLI overrides (if provided)
    if args.wds_test_dir: test_dir = args.wds_test_dir
    if args.wds_pattern:  pattern  = args.wds_pattern
    if args.image_key:    image_key = args.image_key
    if args.meta_key:     meta_key  = args.meta_key
    return test_dir, pattern, image_key, meta_key

def build_eval_cfg(run_dir: Path, test_root: str, pattern: str, image_key: str, meta_key: str, labels: list) -> Path:
    """Compose a self-contained eval YAML for eval.py, saved under CFG_OUTPUT_DIR."""
    fm = read_json(run_dir / "metrics" / "final_metrics.json")
    snap_path = run_dir / "configuration" / "experiment_snapshot.yaml"

    ssl_backbone_rel = fm.get("ssl_backbone_path", "checkpoints/*__ssl_best.pt")
    ssl_head_rel     = fm.get("ssl_linear_ckpt_path", "checkpoints/*__ssl_linear_best.pt")

    ssl_backbone = _first_ckpt_match(run_dir, ssl_backbone_rel)
    ssl_head     = _first_ckpt_match(run_dir, ssl_head_rel)

    model_name     = f"{guess_model_name(run_dir)}_ssl_linear_best"
    # 1) Try to decide from the head dim (source of truth for the probe)
    head_in = _read_head_in_features(ssl_head)
    backbone_name = _map_in_features_to_backbone(head_in) if head_in is not None else None
    # 2) If still unknown, fall back to ckpt/snapshot heuristic
    if backbone_name is None:
        backbone_name = detect_backbone_name(snap_path, ssl_backbone)

    out_root       = (run_dir / "eval").resolve()

    labels_from_snapshot = None
    snap = read_yaml(snap_path)
    try:
        c2i = snap["data"]["webdataset"]["class_to_id"]
        labels_from_snapshot = [k for k, _ in sorted(c2i.items(), key=lambda kv: kv[1])]
    except Exception:
        pass
    labels = labels_from_snapshot or labels

    cfg = {
        "experiment": {"name": f"eval_{run_dir.name}", "seed": 1337, "outputs_root": str(out_root)},
        "data": {
            "backend": "webdataset", "img_size": 224, "imagenet_norm": False,
            "num_workers": 8, "batch_size": 256,
            "webdataset": {
                "test_dir": test_root, "pattern": pattern, "image_key": image_key, "meta_key": meta_key
            }
        },
        "labels": {"class_order": labels},
        "model": {
            "name": model_name, "arch_hint": "ssl_linear",
            "backbone_name": backbone_name,
            "ssl_backbone_ckpt": str(ssl_backbone.resolve()),
            "ssl_head_ckpt":     str(ssl_head.resolve()),
            "strict_load": False,
            "allow_arch_autoswap": False
        },
        "evaluation": {
            "save_logits": True, "save_embeddings": True, "save_preds_csv": True,
            "umap": {"enabled": True, "source": "features", "n_neighbors": 15, "min_dist": 0.1, "random_state": 1337}
        },
        "runtime": {"device": "cuda", "precision": "fp32"}
    }

    CFG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = CFG_OUTPUT_DIR / f"{run_dir.parent.name}__{run_dir.name}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path

def _discover_runs(root: Path):
    """Return [root] if root is itself a run (has metrics/final_metrics.json), else scan children exp_*."""
    if (root / "metrics" / "final_metrics.json").is_file():
        return [root]
    return sorted([p for p in root.glob("exp_*") if (p / "metrics" / "final_metrics.json").is_file()])

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlruns-root", required=True, help="Path to experiment folder or single run folder")
    ap.add_argument("--submit", action="store_true", help="Submit SLURM jobs after generating YAMLs")
    ap.add_argument("--only-one-shard", action="store_true", help="Smoke eval on a single shard (env flag)")
    ap.add_argument("--sbatch-path", default=None, help="Absolute path to eval_models.sbatch (optional)")
    # Optional overrides (normally you can omit these)
    ap.add_argument("--wds-test-dir", default=None)
    ap.add_argument("--wds-pattern", default=None)
    ap.add_argument("--image-key", default=None)
    ap.add_argument("--meta-key", default=None)
    ap.add_argument("--labels", nargs="+", default=["ccRCC","pRCC","CHROMO","ONCO","NOT_TUMOR"])
    args = ap.parse_args()

    root = Path(args.mlruns_root)
    runs = _discover_runs(root)
    if not runs:
        raise SystemExit(f"No runs found under: {root}")

    out_cfgs = []
    for r in runs:
        test_dir, pattern, image_key, meta_key = _resolve_wds_params(r, args)
        cfgp = build_eval_cfg(
            r, test_root=test_dir, pattern=pattern,
            image_key=image_key, meta_key=meta_key, labels=args.labels
        )
        print(f"[OK] Config: {cfgp}")
        print(f"[INFO]  -> test_dir={test_dir}  pattern={pattern}  image_key={image_key}  meta_key={meta_key}")
        out_cfgs.append(cfgp)

    if args.submit:
        sb = Path(args.sbatch_path).resolve() if args.sbatch_path else SBATCH_DEFAULT
        if not sb.is_file():
            raise SystemExit(
                f"[ERROR] SBATCH file not found: {sb}\n"
                f"Pass --sbatch-path /home/mla_group_01/rcc-ssrl/src/evaluation/eval_models.sbatch if needed."
            )
        for cfg in out_cfgs:
            env = os.environ.copy()
            env["CFG_PATH"] = str(cfg)
            if args.only_one_shard:
                env["ONLY_ONE_SHARD"] = "1"
                env["SHARD_EXAMPLE"] = "shard-000000.tar"
            print(f"[SUBMIT] sbatch {sb}  CFG_PATH={cfg}")
            subprocess.run(["sbatch", str(sb)], check=True, env=env)

if __name__ == "__main__":
    main()
