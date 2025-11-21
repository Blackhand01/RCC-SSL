#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate SSL ablation experiment YAMLs from a shared template and per-model JSON definitions.

Usage examples:
  # Generate all models found in .../ablations/*_ablations.json
  python src/training/configs/tools/generate_ssl_ablation_configs.py

  # Only for one model (e.g., moco_v3)
  python src/training/configs/tools/generate_ssl_ablation_configs.py --model moco_v3

  # Also emit sbatch launchers
  python src/training/configs/tools/generate_ssl_ablation_configs.py --emit-launchers

This script:
  1) Loads a shared template YAML (exp_template_ssl.yaml).
  2) Optionally merges a per-suite base_config YAML (if provided in the JSON).
  3) Applies each experiment "override" (deep-merge) and enriches metadata (name, tags, date).
  4) Writes each final YAML into configs/ablations/{model_name}/exp_{model_name}_ablXX.yaml.
  5) Creates a single exp_debug_pipeline_smoke.yaml to quickly smoke-test the pipeline.

All comments/docstrings are in English as requested.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

try:
    import yaml  # PyYAML
except Exception as e:  # pragma: no cover
    print("ERROR: PyYAML is required. Try: pip install pyyaml", file=sys.stderr)
    raise

# --- Repository-absolute defaults (adapt to your layout if needed) -----------------

REPO_ROOT = Path("/home/mla_group_01/rcc-ssrl").resolve()
CFG_ROOT = REPO_ROOT / "src" / "training" / "configs"
TEMPLATE_PATH = CFG_ROOT / "templates" / "exp_template_ssl.yaml"
ABLATIONS_ROOT = CFG_ROOT / "ablations"
ABLATIONS_JSON = CFG_ROOT / "ablations" / "json_abl_exp"

SMOKE_CFG_PATH = ABLATIONS_ROOT / "exp_debug_pipeline_smoke.yaml"

# Pattern for per-model JSON files:
#   /.../configs/ablations/json_abl_exp/{model_name}_ablations.json
JSON_GLOB = (ABLATIONS_ROOT / "json_abl_exp").glob("*_ablations.json")

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def read_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file as a Python dict (empty dict if missing)."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(data: Mapping[str, Any], path: Path) -> None:
    """Write dict to YAML file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def deep_update(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """
    Recursively merge src into dst.
    - Dicts are merged key-by-key (src values override).
    - Lists and scalars are replaced by src.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def today_iso() -> str:
    """Return today's date in ISO format."""
    return dt.date.today().isoformat()


def ensure_list(x: Optional[Any]) -> List[Any]:
    """Return x as list; scalar -> [scalar], None -> []."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def enrich_experiment_meta(cfg: MutableMapping[str, Any],
                           suite: Mapping[str, Any],
                           exp: Mapping[str, Any],
                           model_name: str) -> None:
    """
    Set/augment experiment metadata:
    - experiment.name
    - experiment.date
    - experiment.paper_tags (union of existing + suite/model/exp tags)
    """
    # Canonical experiment name, independent of any "pretty" name in JSON:
    abl_id = int(exp.get("id", 0))
    exp_name = f"exp_{model_name}_abl{abl_id:02d}"
    exp_tags = ensure_list(exp.get("tags"))
    suite_name = suite.get("suite_name", "")
    extra_tags = [t for t in [suite_name, model_name, "ablation"] if t]

    # Ensure 'experiment' section exists
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = exp_name
    cfg["experiment"]["date"] = today_iso()

    # Merge/unique-ify paper_tags
    existing = ensure_list(cfg["experiment"].get("paper_tags"))
    merged = list(dict.fromkeys(existing + extra_tags + exp_tags))  # order-preserving unique
    cfg["experiment"]["paper_tags"] = merged


def apply_override(base_cfg: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a new dict with override deep-merged on top of base."""
    final = json.loads(json.dumps(base_cfg))  # deep copy via roundtrip
    return deep_update(final, override or {})


def load_suite_json(json_path: Path) -> Dict[str, Any]:
    """Load and minimally validate a suite JSON."""
    with json_path.open("r", encoding="utf-8") as f:
        suite = json.load(f)
    if "model" not in suite:
        raise ValueError(f"Missing 'model' in {json_path}")
    if "experiments" not in suite or not isinstance(suite["experiments"], list):
        raise ValueError(f"Missing/invalid 'experiments' in {json_path}")
    return suite


def build_full_smoke_config_from_template(template_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a full smoke config with shared fields and a runs section for each SSL family.
    """
    # Shared config fields (copied and pruned as needed)
    base = json.loads(json.dumps(template_cfg))

    # Overwrite/force shared fields to match requested output
    base["experiment"] = {
        "name": "exp_debug_pipeline_smoke",
        "date": today_iso(),
        "seed": 1337,
        "default_run_index": -1,
        "paper_tags": ["ssl", "histopathology", "smoke", "pipeline"],
        "mlflow_experiment": "RCC_SSL",
        "outputs_layout": "v2",
        "validate": {"paths": True, "steps_per_epoch": True, "batch_sizes": True},
        "allow_cpu": True,
    }

    # Data section
    base["data"] = {
        "sampler": {"limit_per_epoch": 128},
        "img_size": 224,
        "webdataset": {
            "dataset_key": "rcc_final_ablation",
            "shuffle_shards": 64,
            "shuffle_samples": 2000,
            "prefetch_factor": 2,
            "batch_size_ssl": 64,
            "batch_size_sl": 64,
            "num_workers": 4,
            "class_to_id": {"ccRCC": 0, "pRCC": 1, "CHROMO": 2, "ONCO": 3, "NOT_TUMOR": 4},
        },
    }

    # Model section
    base["model"] = {
        "backbone": {"name": "vit_small_patch16_224", "patch_size": 16},
        "ssl": {
            "name": "dino_v3",
            "temperature": 0.2,
            "hidden_dim": 4096,
            "proj_dim": 256,
            "use_multicrop": False,
            "temp_teacher_schedule": None,
            "ema_to_one": True,
            "clip_qk": None,
            "sync_bn": False,
            "aug": {
                "jitter": 0.4,
                "blur_prob": 0.1,
                "gray_prob": 0.2,
                "solarize_prob": 0.0,
            },
        },
    }

    # Train section
    base["train"] = {
        "optim": {
            "name": "adamw",
            "lr": 0.0003,
            "weight_decay": 0.05,
            "betas": [0.9, 0.999],
        },
        "scheduler": {"name": "cosine", "T_max": None},
        "ssl": {
            "epochs": 2,
            "steps_per_epoch": 2,
            "batch_size": 64,
            "accumulate_steps": 1,
            "num_workers": 4,
            "ema_momentum": 0.996,
            "amp": True,
            "grad_clip_max": 1.0,
            "probe": {
                "enabled": True,
                "epochs": 2,
                "lr": 0.05,
                "weight_decay": 0.0,
                "batch_size": 512,
            },
        },
    }

    # Aug section
    base["aug"] = {
        "base": {
            "rotate90": True,
            "hflip": True,
            "vflip": True,
            "random_resized_crop": {"scale": [0.6, 1.0], "ratio": [0.75, 1.33]},
            "gaussian_blur_p": 0.2,
            "sharpen_p": 0.2,
            "jpeg_artifacts_p": 0.1,
        },
        "stain": {
            "normalize": {"enable": False, "method": "macenko"},
            "jitter": {"enable": True, "mode": "HED", "delta": 0.02},
        },
        "mixing": {
            "mixup": {"enable": True, "alpha": 0.3},
            "cutmix": {"enable": True, "beta": 1.0, "p": 0.5},
        },
    }

    # Multiscale section
    base["multiscale"] = {
        "enable": True,
        "scales": [1.0, 1.5, 2.0],
        "n_per_scale": 1,
    }

    # Sampler section
    base["sampler"] = {
        "tissue_aware": {"enable": True, "min_tissue_frac": 0.6},
    }

    # Logging section
    base["logging"] = {
        "log_every_steps": 10,
        "metrics_csv_name": "ssl_timeseries.csv",
        "smoothing_window": 100,
    }

    # Artifacts section
    base["artifacts"] = {
        "save_best_model": True,
        "save_ckpt_every": 1,
        "export_csv": ["metrics"],
        "report_md": True,
    }

    # Runs section: one per SSL family
    base["runs"] = [
        {
            "name": "smoke_dino_v3_vit_s16",
            "mode": "ssl",
            "override": {
                "model": {"ssl": {"name": "dino_v3"}},
                "train": {"ssl": {"epochs": 2, "steps_per_epoch": 2, "accumulate_steps": 1, "probe": {"epochs": 2}}},
            },
        },
        {
            "name": "smoke_moco_v3_vit_s16",
            "mode": "ssl",
            "override": {
                "model": {"ssl": {"name": "moco_v3", "clip_qk": 50.0}},
                "train": {"ssl": {"epochs": 2, "steps_per_epoch": 2, "accumulate_steps": 1, "probe": {"epochs": 2}}},
            },
        },
        {
            "name": "smoke_ibot_vit_s16",
            "mode": "ssl",
            "override": {
                "model": {"ssl": {"name": "ibot", "num_prototypes": 8192, "temp_student": 0.10, "temp_teacher": 0.07, "use_multicrop": False}},
                "train": {"ssl": {"epochs": 2, "steps_per_epoch": 2, "accumulate_steps": 1, "probe": {"epochs": 2}}},
            },
        },
        {
            "name": "smoke_i_jepa_vit_s16",
            "mode": "ssl",
            "override": {
                "model": {"ssl": {"name": "i_jepa", "prediction_space": "global_mean", "ema_to_one": True, "jepa": {"context": {"scale": [0.85, 1.0], "aspect_ratio": [0.9, 1.1]}, "target": {"n": 2, "scale": [0.15, 0.20], "aspect_ratio": [0.75, 1.5], "no_overlap": True}}}},
                "train": {"ssl": {"epochs": 2, "steps_per_epoch": 2, "accumulate_steps": 1, "probe": {"epochs": 2}}},
            },
        },
    ]

    return base

# ------------------------------------------------------------------------------
# Main generation logic
# ------------------------------------------------------------------------------

def generate_for_suite(json_path: Path,
                       template_cfg: Dict[str, Any],
                       emit_launchers: bool = False) -> Tuple[str, List[Path]]:
    """
    Generate YAMLs for one suite JSON.
    Returns (model_name, [written_paths])
    """
    suite = load_suite_json(json_path)
    model_name: str = suite["model"]
    out_dir = ABLATIONS_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base config: merge (template <- base_config <- experiment override)
    base_cfg = json.loads(json.dumps(template_cfg))  # deep copy of template
    # Set the actual model name from suite
    base_cfg["model"]["ssl"]["name"] = model_name
    base_cfg_path = suite.get("base_config")
    if base_cfg_path:
        # Allow relative path from repo root
        base_path = (REPO_ROOT / base_cfg_path).resolve()
        if base_path.exists():
            deep_update(base_cfg, read_yaml(base_path))
        else:
            print(f"[WARN] base_config not found: {base_path}", file=sys.stderr)

    written: List[Path] = []

    # Create YAMLs per experiment
    for exp in suite["experiments"]:
        cfg_filename = exp.get("config_filename")
        if not cfg_filename:
            # Fallback to a conventional name
            cfg_filename = f"exp_{model_name}_abl{int(exp.get('id', 0)):02d}.yaml"

        override = exp.get("override", {}) or {}
        final_cfg = apply_override(base_cfg, override)
        enrich_experiment_meta(final_cfg, suite, exp, model_name)

        out_path = out_dir / cfg_filename
        write_yaml(final_cfg, out_path)
        written.append(out_path)

    # Optionally emit a launcher script using sbatch_script
    if emit_launchers:
        sbatch_script = suite.get("sbatch_script", "slurm/train_single_node.sbatch")
        launch_path = out_dir / f"launch_all_{model_name}.sh"
        with launch_path.open("w", encoding="utf-8") as sh:
            sh.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
            sh.write(f'echo "Launching {model_name} ablations..." \n\n')
            for p in written:
                rel = p.relative_to(REPO_ROOT)
                sh.write(f"sbatch {sbatch_script} {rel}\n")
            sh.write('\necho "Done."\n')
        os.chmod(launch_path, 0o755)

    return model_name, written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SSL ablation experiment configs.")
    parser.add_argument("--model", type=str, default=None,
                        help="Only generate for this model name (e.g., moco_v3).")
    parser.add_argument("--emit-launchers", action="store_true",
                        help="Also create per-model launch_all_{model}.sh with sbatch commands.")
    parser.add_argument("--no-smoke", action="store_true",
                        help="Do not (re)generate the pipeline smoke config.")
    args = parser.parse_args(argv)

    # Load shared template
    if not TEMPLATE_PATH.exists():
        print(f"ERROR: Template not found: {TEMPLATE_PATH}", file=sys.stderr)
        return 2
    template_cfg = read_yaml(TEMPLATE_PATH)

    # Discover JSONs
    json_files = sorted(JSON_GLOB)
    if args.model:
        json_files = [p for p in json_files if p.name == f"{args.model}_ablations.json"]

    if not json_files:
        print("ERROR: No *_ablations.json found matching the selection.", file=sys.stderr)
        return 3

    print(f"[INFO] Using template: {TEMPLATE_PATH}")
    for j in json_files:
        print(f"[INFO] Suite: {j}")

    # Generate per-suite
    summary: List[Tuple[str, int]] = []
    for json_path in json_files:
        model, written = generate_for_suite(json_path, template_cfg, emit_launchers=args.emit_launchers)
        summary.append((model, len(written)))

    # Generate/update smoke config unless disabled
    if not args.no_smoke:
        smoke_cfg = build_full_smoke_config_from_template(template_cfg)
        write_yaml(smoke_cfg, SMOKE_CFG_PATH)
        print(f"[INFO] Smoke config written: {SMOKE_CFG_PATH}")

    # Report
    print("\n[SUMMARY]")
    for model, count in summary:
        print(f"  {model}: {count} YAMLs written in {ABLATIONS_ROOT / model}")
    if not args.no_smoke:
        print(f"  + smoke: {SMOKE_CFG_PATH}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
