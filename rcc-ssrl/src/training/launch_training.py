from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------
# Layout progetto
# ---------------------------------------------------------------------
MODULE_ROOT = Path(__file__).resolve().parent          # .../src/training
REPO_ROOT = MODULE_ROOT.parent.parent                  # .../
SRC_ROOT = REPO_ROOT / "src"

for p in (REPO_ROOT, SRC_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from src.training.orchestrator import Orchestrator
from src.training.utils.io import append_row_csv
from src.training.utils.reproducibility import set_global_seed

# ---------------------------------------------------------------------
# Config: file esperimento + file paths
# ---------------------------------------------------------------------
ENV_CFG = "EXPERIMENT_CONFIG_PATH"
ENV_RUN = "RUN_INDEX"


def _as_abs(path_like: str | Path) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return (MODULE_ROOT / candidate).resolve()


DEFAULT_CONFIG_PATH = _as_abs("configs/experiment_debug.yaml")
DEFAULT_RUN_INDEX = -1

CONFIG_PATH = _as_abs(os.environ.get(ENV_CFG, str(DEFAULT_CONFIG_PATH)))
RUN_INDEX = int(os.environ.get(ENV_RUN, str(DEFAULT_RUN_INDEX)))


def read_yaml(path: Path | str) -> Dict[str, Any]:
    with _as_abs(path).open("r") as handle:
        return yaml.safe_load(handle)


# ---------------------------------------------------------------------
# paths(): centralizzazione dei percorsi del progetto
# ---------------------------------------------------------------------
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
    print(f"[paths] project_root={project_root} outputs_root={outputs_root}")

    webdataset = resolved.get("webdataset", {})
    if not webdataset:
        raise KeyError("[paths] sezione webdataset mancante o vuota")

    for key, section in webdataset.items():
        for field in ("train_dir", "val_dir", "test_dir"):
            candidate = section.get(field)
            if candidate is None:
                raise KeyError(f"[paths] webdataset.{key}.{field} mancante")
            if not Path(candidate).exists():
                raise FileNotFoundError(f"[paths] missing {key}.{field}: {candidate}")
        print(f"[paths] webdataset.{key}: train={section['train_dir']} "
              f"val={section['val_dir']} test={section['test_dir']}")

    return resolved


# ---------------------------------------------------------------------
# Merge util
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
    cfg = deep_update(base_cfg, run_block.get("override", {}))
    cfg.setdefault("experiment", {})["name"] = f"{base_cfg['experiment']['name']}__{run_block['name']}"
    runtime = cfg.setdefault("_runtime", {})
    runtime["mode"] = run_block.get("mode", base_cfg.get("model", {}).get("type", "ssl"))
    runtime["config_path"] = str(cfg_path)
    cfg.pop("runs", None)
    return cfg


def expand_runs(base_cfg: Dict[str, Any], cfg_path: Path) -> List[Dict[str, Any]]:
    runs = base_cfg.get("runs", [])
    if RUN_INDEX >= 0:
        runs = [runs[RUN_INDEX]]
    if runs:
        return [compose_run_config(base_cfg, block, cfg_path) for block in runs]

    single = copy.deepcopy(base_cfg)
    runtime = single.setdefault("_runtime", {})
    runtime["mode"] = runtime.get("mode", single.get("model", {}).get("type", "ssl"))
    runtime["config_path"] = str(cfg_path)
    single.pop("runs", None)
    return [single]


# ---------------------------------------------------------------------
# Iniezione path risolti (da src/training/paths.py) nella config del run
# ---------------------------------------------------------------------
def inject_paths_into_cfg(cfg: Dict[str, Any], resolved_paths: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    experiment = cfg.setdefault("experiment", {})
    experiment["outputs_root"] = str(resolved_paths["outputs_root"])
    experiment["project_root"] = str(resolved_paths["project_root"])

    wds_cfg = cfg.setdefault("data", {}).setdefault("webdataset", {})
    dataset_key = wds_cfg.get("dataset_key")
    if not dataset_key:
        raise KeyError("data.webdataset.dataset_key mancante: deve puntare a webdataset.<key> in src/training/paths.py")
    if dataset_key not in resolved_paths["webdataset"]:
        raise KeyError(f"dataset_key='{dataset_key}' non presente in src/training/paths.py")

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
# Riepilogo CSV per tutti i run
# ---------------------------------------------------------------------
def _summary_csv_path(run_root: Path) -> Path:
    exp_folder = run_root.parents[1]
    exp_folder.mkdir(parents=True, exist_ok=True)
    return exp_folder / "runs_summary.csv"


def _record_summary(orch: Orchestrator, metrics: Dict[str, Any], elapsed_s: float) -> Path:
    row = {
        "exp_id": orch.exp_id,
        "run_name": orch.cfg["experiment"]["name"],
        "mode": orch.mode,
        "model": orch.model_key,
        "elapsed_s": round(elapsed_s, 2),
        **metrics,
    }
    return append_row_csv(_summary_csv_path(orch.run_dirs["root"]), row)


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> int:
    base_cfg = read_yaml(CONFIG_PATH)
    resolved_paths = paths()
    all_runs = expand_runs(base_cfg, CONFIG_PATH)

    for cfg_run in all_runs:
        cfg_run = inject_paths_into_cfg(cfg_run, resolved_paths)
        set_global_seed(cfg_run["experiment"].get("seed", 1337))
        orchestrator = Orchestrator(cfg_run)
        start_time = time.time()
        metrics = orchestrator.fit()
        _record_summary(orchestrator, metrics, time.time() - start_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
