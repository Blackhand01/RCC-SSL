# utils/io.py
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict

__all__ = [
    "ensure_dir",
    "make_exp_id",
    "make_run_dirs",
    "prefixed",
    "append_row_csv",
    "dump_json",
    "load_json",
    "copy_yaml_config",
]


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (recursively) if missing and return it as a Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def make_exp_id(outputs_root: str) -> str:
    root = ensure_dir(outputs_root)
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"exp_{ts}"


def make_run_dirs(outputs_root: str, exp_id: str, exp_name: str, model_key: str) -> Dict[str, Path]:
    base = Path(outputs_root) / "experiments" / exp_id / exp_name / model_key
    sub = {
        "configuration": base / "configuration" / model_key,
        "checkpoints": base / "checkpoints",
        "metrics": base / "metrics",
        "plots": base / "plots",
    }
    for path in sub.values():
        ensure_dir(path)
    return {
        "root": base,
        **sub,
        "artifacts": sub["checkpoints"],
        "figures": sub["plots"],
        "records": sub["configuration"],
    }


def prefixed(path_dir: Path | str, model_key: str, stem: str, ext: str) -> Path:
    directory = ensure_dir(path_dir)
    extension = ext.lstrip(".")
    return directory / f"{model_key}__{stem}.{extension}"


def append_row_csv(path: Path | str, row: Dict[str, Any]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    write_header = not target.exists()
    with target.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return target


def dump_json(path: Path | str, payload: Dict[str, Any]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return target


def load_json(path: Path | str) -> Dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"JSON file not found: {target}")
    return json.loads(target.read_text())


def copy_yaml_config(src: str | None, dst_dir: Path | str) -> Path | None:
    if not src:
        return None
    source = Path(src)
    if not source.exists():
        return None
    destination = ensure_dir(dst_dir) / "experiment_snapshot.yaml"
    shutil.copy2(source, destination)
    return destination
