#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------
MODULE_ROOT = Path(__file__).resolve().parent          # .../src/training
REPO_ROOT = MODULE_ROOT.parent.parent                  # .../
SRC_ROOT = REPO_ROOT / "src"

for p in (REPO_ROOT, SRC_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from src.training.orchestrator import Orchestrator
from src.training.utils.io import append_row_csv, make_exp_id
from src.training.utils.reproducibility import set_global_seed, copy_code_snapshot
from src.training.utils.paths import CONFIG_PATH as DEFAULT_CONFIG_PATH, RUN_INDEX as DEFAULT_RUN_INDEX, _as_abs

# ---------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------
def read_yaml(path: Path | str) -> Dict[str, Any]:
    with _as_abs(path).open("r") as handle:
        return yaml.safe_load(handle) or {}

# ---------------------------------------------------------------------
# paths(): centralization of resolved project paths
# ---------------------------------------------------------------------
def _is_rank_zero() -> bool:
    return os.environ.get("RANK", "0") == "0"

def _stringify_paths(tree: Dict[str, Any]) -> Dict[str, Any]:
    def _convert(val: Any) -> Any:
        if isinstance(val, Path):
            return str(val)
        if isinstance(val, dict):
            return {k: _convert(v) for k, v in val.items()}
        return val
    return {key: _convert(val) for key, val in tree.items()}

def paths() -> Dict[str, Any]:
    from src.training.utils import paths as pathmod

    resolved = pathmod.get_all()
    project_root = resolved["project_root"]
    outputs_root = resolved["outputs_root"]
    if _is_rank_zero():
        print(f"[paths] project_root={project_root} outputs_root={outputs_root}")

    webdataset = resolved.get("webdataset", {})
    if not webdataset:
        raise KeyError("[paths] missing or empty 'webdataset' section")

    for key, section in webdataset.items():
        for field in ("train_dir", "val_dir", "test_dir"):
            candidate = section.get(field)
            if candidate is None:
                raise KeyError(f"[paths] webdataset.{key}.{field} missing")
            if not Path(candidate).exists():
                raise FileNotFoundError(f"[paths] missing {key}.{field}: {candidate}")
        if _is_rank_zero():
            print(f"[paths] webdataset.{key}: train={section['train_dir']} "
                  f"val={section['val_dir']} test={section['test_dir']}")

    return resolved

# ---------------------------------------------------------------------
# Merge utils
# ---------------------------------------------------------------------
def deep_update(base: Dict, override: Optional[Dict]) -> Dict:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged

def compose_run_config(base_cfg: Dict[str, Any], run_block: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    """Compose a single run config from a base config + a 'runs' entry override."""
    cfg = deep_update(base_cfg, run_block.get("override", {}))
    experiment = cfg.setdefault("experiment", {})
    experiment["name"] = base_cfg["experiment"]["name"]  # keep main name
    runtime = cfg.setdefault("_runtime", {})
    runtime["mode"] = run_block.get("mode", base_cfg.get("model", {}).get("type", "ssl"))
    runtime["config_path"] = str(cfg_path)

    # Prefer RUN_NAME from environment if provided, otherwise runs[].name
    env_run_name = os.environ.get("RUN_NAME", "").strip()
    runtime["run_name"] = env_run_name if env_run_name else run_block["name"]
    # Thread-through of ablation id / explicit run subdir from env (set by sbatch launcher)
    abl_id = os.environ.get("ABLATION_ID", "").strip()
    if abl_id:
        runtime["ablation_id"] = abl_id
    subdir = os.environ.get("EXP_SUBDIR", "").strip()
    if subdir:
        runtime["run_subdir"] = subdir
        # Force experiment.name to the canonical leaf dir (no "pretty" names)
        experiment["name"] = subdir

    # Drop top-level 'runs' to freeze config
    cfg.pop("runs", None)
    return cfg

def expand_runs(base_cfg: Dict[str, Any], cfg_path: Path, run_index: int) -> List[Dict[str, Any]]:
    """Expand 'runs' block to a list of concrete run configs respecting RUN_INDEX."""
    runs = base_cfg.get("runs", [])
    if run_index >= 0:
        if not runs:
            if _is_rank_zero():
                print(f"[runs] Ignoring RUN_INDEX={run_index}: config '{cfg_path}' has no runs.")
        elif run_index >= len(runs):
            raise IndexError(
                f"RUN_INDEX={run_index} out of range for config '{cfg_path}' "
                f"(available runs: {len(runs)})"
            )
        else:
            runs = [runs[run_index]]
    if runs:
        return [compose_run_config(base_cfg, block, cfg_path) for block in runs]

    # Single-run config (no 'runs' block)
    single = copy.deepcopy(base_cfg)
    runtime = single.setdefault("_runtime", {})
    runtime["mode"] = runtime.get("mode", single.get("model", {}).get("type", "ssl"))
    runtime["config_path"] = str(cfg_path)
    env_run_name = os.environ.get("RUN_NAME", "").strip()
    runtime["run_name"] = env_run_name if env_run_name else single.get("experiment", {}).get("name", "default")
    single.pop("runs", None)
    return [single]

# ---------------------------------------------------------------------
# Inject resolved paths into run config
# ---------------------------------------------------------------------
def inject_paths_into_cfg(cfg: Dict[str, Any], resolved_paths: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    experiment = cfg.setdefault("experiment", {})
    experiment["outputs_root"] = str(resolved_paths["outputs_root"])
    experiment["project_root"] = str(resolved_paths["project_root"])

    # Optional MLflow experiment override from env
    mlflow_env = os.environ.get("MLFLOW_EXPERIMENT_NAME", "").strip()
    if mlflow_env:
        experiment["mlflow_experiment"] = mlflow_env

    wds_cfg = cfg.setdefault("data", {}).setdefault("webdataset", {})
    dataset_key = wds_cfg.get("dataset_key")
    if not dataset_key:
        raise KeyError("data.webdataset.dataset_key missing: it must match a key in src/training/paths.py")
    if dataset_key not in resolved_paths["webdataset"]:
        raise KeyError(f"dataset_key='{dataset_key}' not present in src/training/paths.py")

    selected = resolved_paths["webdataset"][dataset_key]
    wds_cfg["train_dir"] = str(selected["train_dir"])
    wds_cfg["val_dir"] = str(selected["val_dir"])
    wds_cfg["test_dir"] = str(selected["test_dir"])
    print(f"[paths] dataset_key={dataset_key} -> train={selected['train_dir']} "
          f"val={selected['val_dir']} test={selected['test_dir']}")

    runtime = cfg.setdefault("_runtime", {})
    runtime["paths"] = _stringify_paths(resolved_paths)
    return cfg

# ---------------------------------------------------------------------
# CSV Summary utilities
# ---------------------------------------------------------------------
def _summary_csv_path(run_root: Path, mode: str) -> Path:
    exp_folder = run_root.parents[1]
    exp_folder.mkdir(parents=True, exist_ok=True)
    return exp_folder / f"runs_summary_{mode}.csv"

def _record_summary(orch: Orchestrator, metrics: Dict[str, Any], elapsed_s: float) -> Path:
    run_label = orch.cfg.get("_runtime", {}).get("run_name", orch.cfg["experiment"]["name"])
    row = {
        "exp_id": orch.exp_id,
        "run_name": run_label,
        "mode": orch.mode,
        "model": orch.model_key,
        "elapsed_s": round(elapsed_s, 2),
        **metrics,
    }
    return append_row_csv(_summary_csv_path(orch.run_dirs["root"], orch.mode), row)

# ---------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------
def _env_exp_datetime() -> str | None:
    """Return EXP_DATETIME if valid (YYYYMMDD-HHMMSS)."""
    v = os.environ.get("EXP_DATETIME", "").strip()
    return v if re.match(r"^\d{8}-\d{6}$", v) else None

def _env_exp_group() -> str | None:
    """Return EXP_GROUP if non-empty (used as shared exp_id)."""
    v = os.environ.get("EXP_GROUP", "").strip()
    return v or None

def _resolve_config_path(cli_cfg: Optional[str]) -> Path:
    """
    Resolve the configuration file path with this precedence:
    1) --config CLI
    2) $CONFIG_PATH
    3) $RUN_CFG
    4) $TRAIN_CONFIG
    5) DEFAULT_CONFIG_PATH (from utils.paths)
    """
    candidates = [
        cli_cfg,
        os.environ.get("CONFIG_PATH"),
        os.environ.get("RUN_CFG"),
        os.environ.get("TRAIN_CONFIG"),
        os.environ.get("EXPERIMENT_CONFIG_PATH"),
        str(DEFAULT_CONFIG_PATH),
    ]
    for c in candidates:
        if c:
            p = Path(c).expanduser()
            if p.is_file():
                return p
    # Last resort: show where we looked
    raise FileNotFoundError(
        "No valid config file found. Tried (in order): "
        f"--config, $CONFIG_PATH, $RUN_CFG, $TRAIN_CONFIG, $EXPERIMENT_CONFIG_PATH, DEFAULT_CONFIG_PATH={DEFAULT_CONFIG_PATH}"
    )

def _resolve_run_index(cli_idx: Optional[int]) -> int:
    """
    Resolve RUN_INDEX with precedence:
    1) --run-index CLI
    2) $RUN_INDEX env
    3) DEFAULT_RUN_INDEX (from utils.paths)
    """
    if cli_idx is not None:
        return cli_idx
    env_val = os.environ.get("RUN_INDEX", "").strip()
    if env_val != "":
        try:
            return int(env_val)
        except ValueError:
            raise ValueError(f"Invalid RUN_INDEX env value: '{env_val}'")
    return int(DEFAULT_RUN_INDEX)

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SSL/SL training launcher")
    parser.add_argument("--config", type=str, help="Path to YAML config", default=None)
    parser.add_argument("--run-index", type=int, help="Index into 'runs' list (>=0) or -1 for all/single", default=None)
    args = parser.parse_args(argv)

    # Resolve config path and run index with clear precedence
    cfg_path = _resolve_config_path(args.config)
    run_index = _resolve_run_index(args.run_index)

    if _is_rank_zero():
        print(f"[config] Using config: {cfg_path}")
        print(f"[config] RUN_INDEX={run_index}")

    base_cfg = read_yaml(cfg_path)
    resolved_paths = paths()
    all_runs = expand_runs(base_cfg, cfg_path, run_index)

    # Decide shared experiment id (directory name under outputs/experiments/<exp_id>)
    shared_exp_id: Optional[str] = None
    exp_dt = _env_exp_datetime()
    exp_group = _env_exp_group()

    for cfg_run in all_runs:
        cfg_run = inject_paths_into_cfg(cfg_run, resolved_paths)

        runtime = cfg_run.setdefault("_runtime", {})

        # Prefer EXP_GROUP (exact name), then EXP_DATETIME (prefixed), else autogenerated
        if shared_exp_id is None:
            if exp_group:
                shared_exp_id = exp_group
            elif exp_dt:
                shared_exp_id = f"exp_{exp_dt}"
            else:
                shared_exp_id = make_exp_id(cfg_run["experiment"]["outputs_root"], exp_dt)

        runtime["exp_id"] = shared_exp_id
        cfg_run.setdefault("experiment", {})["id"] = shared_exp_id

        # Also surface the canonical run_subdir on the experiment section (for downstream tools)
        rsd = cfg_run.get("_runtime", {}).get("run_subdir")
        if rsd:
            cfg_run["experiment"]["name"] = rsd

        # Expose for downstream tooling
        os.environ["EXP_ID"] = shared_exp_id

        # Seed and snapshot
        set_global_seed(cfg_run["experiment"].get("seed", 1337))
        orchestrator = Orchestrator(cfg_run)

        # Best-effort code snapshot (non-fatal)
        try:
            # Save code under .../records/code_snapshot (no virtualenv / large artifacts).
            snap_dst = os.path.join(str(orchestrator.run_dirs["records"]), "code_snapshot")
            copy_code_snapshot(
                str(SRC_ROOT / "training"),
                snap_dst,
                excludes=(),  # default excludes already skip .venv/site-packages/etc.
            )
        except Exception:
            pass

        try:
            start_time = time.time()
            metrics = orchestrator.fit()
            _record_summary(orchestrator, metrics, time.time() - start_time)
        except Exception as e:
            # Always emit a reporting stub so downstream tooling finds the folder.
            try:
                rep_dir = orchestrator.run_dirs["root"] / "reporting"
                rep_dir.mkdir(parents=True, exist_ok=True)
                with (rep_dir / "FAILED.txt").open("w") as fh:
                    fh.write(f"{type(e).__name__}: {e}\n")
            except Exception:
                pass
            raise

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
