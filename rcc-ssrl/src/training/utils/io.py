# utils/io.py
from __future__ import annotations

import csv
import json
import platform
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

__all__ = [
    "ensure_dir",
    "make_exp_id",
    "make_run_dirs",
    "prefixed",
    "append_row_csv",
    "dump_json",
    "load_json",
    "copy_yaml_config",
    "save_env_info",
    "save_state_dict",
    "write_artifacts_manifest",
    "write_run_readme",
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
    base = Path(outputs_root) / exp_id / exp_name / model_key
    subdirs = {
        key: base / key
        for key in ("configuration", "metrics", "plots", "figures", "records", "artifacts")
    }
    for path in subdirs.values():
        ensure_dir(path)
    return {"root": base, **subdirs}


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
    destination = ensure_dir(dst_dir) / "experiment_config.yaml"
    shutil.copy2(source, destination)
    return destination


def save_env_info(dst_dir: Path | str, seed: int) -> Path:
    payload = {"python": platform.python_version(), "seed": int(seed)}
    try:
        payload["torch"] = torch.__version__
        payload["cuda_available"] = torch.cuda.is_available()
        payload["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # pragma: no cover - torch missing
        payload.update({"torch": "n/a", "cuda_available": False, "device_count": 0})
    return dump_json(Path(dst_dir) / "env_info.json", payload)


def save_state_dict(state: Mapping[str, Any], out_dir: Path | str, model_key: str, stem: str, ext: str = "pt") -> Path:
    """
    Centralised helper to persist model checkpoints with the `<model>__<stem>.<ext>` convention.
    """
    path = prefixed(out_dir, model_key, stem, ext)
    torch.save(state, path)
    return path


def _artifact_list(model_key: str, mode: str) -> list[str]:
    if mode == "ssl":
        return [
            f"{model_key}__ssl_best.pt",
            f"{model_key}__best_classifier.pt",
            f"{model_key}__best_classifier_meta.json",
            f"{model_key}__best_classifier_sklearn.joblib",
        ]
    return [f"{model_key}__sl_best_classifier.pt"]


def write_artifacts_manifest(run_dirs: Dict[str, Path], model_key: str, mode: str) -> Path:
    manifest_path = prefixed(run_dirs["records"], model_key, "artifacts_manifest", "json")
    payload = {"mode": mode, "model": model_key, "artifacts": _artifact_list(model_key, mode)}
    return dump_json(manifest_path, payload)


def write_run_readme(run_dirs: Dict[str, Path], model_key: str, mode: str,
                     cfg: Dict[str, Any], metrics: Dict[str, float], exp_id: str) -> Path:
    meta = {
        "exp_name": cfg["experiment"]["name"],
        "exp_id": exp_id,
        "mode": mode,
        "model": model_key,
        "seed": cfg["experiment"].get("seed"),
    }
    manifest_path = write_artifacts_manifest(run_dirs, model_key, mode)
    readme_path = prefixed(run_dirs["records"], model_key, "README", "md")
    sections = [
        f"# {meta['exp_name']} — {model_key}",
        "",
        "## meta",
        "```json",
        json.dumps(meta, indent=2, sort_keys=True),
        "```",
        "",
        "## key metrics",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
        "",
        "## artifacts",
        *[f"- artifacts/{item}" for item in _artifact_list(model_key, mode)],
        "",
        f"_Autogenerated manifest_: `records/{manifest_path.name}`",
    ]
    readme_path.write_text("\n".join(sections))
    return readme_path
