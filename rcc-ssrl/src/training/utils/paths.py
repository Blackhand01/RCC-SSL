"""Centralized path resolution utilities for the training pipeline."""
from pathlib import Path
import os
from typing import Optional, Union

# ---------------------------------------------------------------------
# Config path & RUN_INDEX (leggibili da ENV, con default sensati)
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CFG = REPO_ROOT / "src" / "training" / "configs" / "ablations" / "exp_debug_pipeline.yaml"
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(DEFAULT_CFG))).resolve()
RUN_INDEX = int(os.environ.get("RUN_INDEX", "-1"))

def _as_abs(p: Union[Path, str]) -> Path:
    return Path(p).resolve() if not isinstance(p, Path) else p.resolve()

def _first_existing(*candidates: Path) -> Optional[Path]:
    for c in candidates:
        if c and c.exists():
            return c
    return None


def get_all():
    """
    Ritorna tutti i path risolti, scegliendo prima BeeGFS (veloce, condiviso),
    con fallback alla copia locale nella repo solo se necessario.
    """
    # Roots preferite su BeeGFS
    beegfs_project = Path("/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project")
    home_project   = REPO_ROOT  # fallback minimale

    project_root = _first_existing(beegfs_project, home_project) or home_project
    outputs_root = project_root / "outputs" / "mlruns"

    # WebDataset shards: prima BeeGFS, poi fallback locale nella repo
    wds_beegfs = beegfs_project / "data" / "processed" / "rcc_webdataset_final"
    wds_home   = REPO_ROOT / "src" / "data" / "processed" / "rcc_webdataset_final"

    wds_train = _first_existing(wds_beegfs / "train", wds_home / "train")
    wds_val   = _first_existing(wds_beegfs / "val",   wds_home / "val")
    wds_test  = _first_existing(wds_beegfs / "test",  wds_home / "test")

    # Se mancano gli shard ovunque, lasciamo che l'alto livello tiri un errore chiaro
    wds_map = {
        # "rcc_v2": {
        #     "train_dir": str(wds_train or wds_beegfs / "train"),
        #     "val_dir":   str(wds_val   or wds_beegfs / "val"),
        #     "test_dir":  str(wds_test  or wds_beegfs / "test"),
        # },
        "rcc_final_ablation": {
            "train_dir": str(wds_train or wds_beegfs / "train"),
            "val_dir":   str(wds_val   or wds_beegfs / "val"),
            "test_dir":  str(wds_test  or wds_beegfs / "test"),
        },
    }

    return {
        "project_root": str(project_root),
        "outputs_root": str(outputs_root),
        "webdataset": wds_map,
    }
