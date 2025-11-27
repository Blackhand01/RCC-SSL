"""Centralized path resolution utilities for the training pipeline."""
from pathlib import Path
import os
from typing import Dict, Optional, Union

import yaml

# ---------------------------------------------------------------------
# Config path & RUN_INDEX (leggibili da ENV, con default sensati)
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CFG = REPO_ROOT / "src" / "training" / "configs" / "ablations" / "exp_debug_pipeline.yaml"
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(DEFAULT_CFG))).resolve()
RUN_INDEX = int(os.environ.get("RUN_INDEX", "-1"))

# Roots can be overridden from the environment for HPC/SLURM consistency
HOME_ROOT = Path(os.environ.get("HOME_ROOT", str(Path.home())))
SCRATCH_ROOT = Path(
    os.environ.get("SCRATCH_ROOT", "/beegfs-scratch/mla_group_01/workspace/mla_group_01")
)

def _as_abs(p: Union[Path, str]) -> Path:
    return Path(p).resolve() if not isinstance(p, Path) else p.resolve()

def _first_existing(*candidates: Path) -> Optional[Path]:
    for c in candidates:
        if c and c.exists():
            return c
    return None

def _env_path(name: str) -> Optional[Path]:
    val = os.environ.get(name, "").strip()
    return Path(val).expanduser() if val else None


def _expand_env_in_tree(tree: Dict) -> Dict:
    """Recursively expand environment variables in a mapping."""
    expanded: Dict = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            expanded[key] = _expand_env_in_tree(value)
        elif isinstance(value, str):
            expanded[key] = os.path.expandvars(value)
        else:
            expanded[key] = value
    return expanded


def _load_wds_from_yaml(include_path: Path) -> Optional[Dict[str, Dict[str, str]]]:
    """
    Optional override: load WebDataset paths from a YAML include file.
    The YAML is expected to contain:
    data:
      <dataset_key>:
        train_dir: ...
        val_dir: ...
        test_dir: ...
    """
    if not include_path.exists():
        return None
    raw = yaml.safe_load(include_path.read_text()) or {}
    data = raw.get("data") or {}
    data = _expand_env_in_tree(data)
    result: Dict[str, Dict[str, str]] = {}
    for key, section in data.items():
        if not isinstance(section, dict):
            continue
        result[key] = {
            "train_dir": str(Path(section.get("train_dir", "")).expanduser()),
            "val_dir": str(Path(section.get("val_dir", "")).expanduser()),
            "test_dir": str(Path(section.get("test_dir", "")).expanduser()),
        }
    return result or None

def _infer_from_outputs_group() -> tuple[Optional[Path], Optional[Path]]:
    """
    Se OUTPUTS_GROUP_DIR è settata (da launch_ssl_ablations.sh),
    prova a risalire a outputs_root e project_root.
    Atteso: .../outputs/mlruns/experiments/<group_name>
    """
    og = _env_path("OUTPUTS_GROUP_DIR")
    if not og:
        return None, None
    try:
        outputs_root = og.parents[1]   # .../outputs/mlruns
        project_root = og.parents[3]   # .../wsi-ssrl-rcc_project
    except IndexError:
        return None, None
    if outputs_root.name != "mlruns" or outputs_root.parent.name != "outputs":
        return None, None
    return outputs_root, project_root


def get_all():
    """
    Ritorna tutti i path risolti, scegliendo prima BeeGFS (veloce, condiviso),
    con fallback alla copia locale nella repo solo se necessario.
    """
    env_project_root, env_outputs_root = _env_path("PROJECT_ROOT"), _env_path("OUTPUTS_ROOT")
    og_outputs_root, og_project_root   = _infer_from_outputs_group()

    # Roots preferite su BeeGFS, con fallback locale
    beegfs_project = SCRATCH_ROOT / "wsi-ssrl-rcc_project"
    home_project   = HOME_ROOT / "rcc-ssrl"  # fallback minimale

    project_root = _first_existing(
        env_project_root,
        og_project_root,
        beegfs_project,
        home_project,
    ) or home_project
    outputs_root = _first_existing(
        env_outputs_root,
        og_outputs_root,
        project_root / "outputs" / "mlruns",
    ) or (project_root / "outputs" / "mlruns")

    # WebDataset shards: prima BeeGFS, poi fallback locale nella repo
    wds_env_root = (
        _env_path("RCC_DATASET_ROOT")
        or _env_path("RCC_WDS_ROOT")
        or _env_path("WDS_ROOT")
        or _env_path("WEB_DATASET_ROOT")
    )
    if not wds_env_root and (env_project_root or og_project_root):
        base = env_project_root or og_project_root
        wds_env_root = base / "data" / "processed"

    wds_beegfs_root = beegfs_project / "data" / "processed"
    wds_home_root   = REPO_ROOT / "src" / "data" / "processed"

    def _dataset_root(base: Path) -> Path:
        base = Path(base)
        return base if base.name == "rcc_webdataset_final" else base / "rcc_webdataset_final"

    wds_env_root = _dataset_root(wds_env_root) if wds_env_root else _dataset_root(wds_beegfs_root)

    def _wds_dir(name: str) -> Optional[Path]:
        return _first_existing(
            wds_env_root and Path(wds_env_root) / name,
            _dataset_root(wds_beegfs_root) / name,
            _dataset_root(wds_home_root) / name,
        )

    # Usata come hint se tutte le dir mancano: preferiamo rispettare eventuali override da ENV
    wds_root_hint = wds_env_root or _dataset_root(wds_beegfs_root) or _dataset_root(wds_home_root)

    wds_train = _wds_dir("train")
    wds_val   = _wds_dir("val")
    wds_test  = _wds_dir("test")

    # Se mancano gli shard ovunque, lasciamo che l'alto livello tiri un errore chiaro
    wds_map = {
        # "rcc_v2": {
        #     "train_dir": str(wds_train or wds_beegfs / "train"),
        #     "val_dir":   str(wds_val   or wds_beegfs / "val"),
        #     "test_dir":  str(wds_test  or wds_beegfs / "test"),
        # },
        "rcc_final_ablation": {
            "train_dir": str(wds_train or wds_root_hint / "train"),
            "val_dir":   str(wds_val   or wds_root_hint / "val"),
            "test_dir":  str(wds_test  or wds_root_hint / "test"),
        },
    }

    # Optional override from YAML include (centralized dataset mapping)
    include_path = REPO_ROOT / "src" / "training" / "configs" / "includes" / "data_paths.yaml"
    yaml_map = _load_wds_from_yaml(include_path) or {}
    wds_map.update(yaml_map)

    return {
        "project_root": str(project_root),
        "outputs_root": str(outputs_root),
        "webdataset": wds_map,
    }
