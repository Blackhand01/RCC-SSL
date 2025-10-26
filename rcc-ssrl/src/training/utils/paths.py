"""Centralized path resolution utilities for the training pipeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from .io import ensure_dir


def _resolve(path_like: str | Path, base: Path | None = None) -> Path:
    """Resolve a path, optionally relative to the provided base directory."""
    candidate = Path(path_like).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if base is None:
        raise ValueError("Relative path requires a base directory.")
    return (base / candidate).resolve()


# Project roots default to the repo layout but can be overridden at runtime.
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_PROJECT_ROOT_RAW = os.environ.get("PROJECT_ROOT")
if _PROJECT_ROOT_RAW:
    PROJECT_ROOT = _resolve(_PROJECT_ROOT_RAW, _DEFAULT_PROJECT_ROOT)
else:
    PROJECT_ROOT = _DEFAULT_PROJECT_ROOT

_OUTPUT_ROOT_RAW = os.environ.get("OUTPUTS_ROOT")
if _OUTPUT_ROOT_RAW:
    OUTPUTS_ROOT = _resolve(_OUTPUT_ROOT_RAW, PROJECT_ROOT)
else:
    OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "mlruns"

# Mapping of WebDataset aliases to their relative directory layout.
WEB_DATASETS: Dict[str, Dict[str, str]] = {
    "rcc_v2": {
        "train_dir": "data/processed/rcc_webdataset_v2/train",
        "val_dir": "data/processed/rcc_webdataset_v2/val",
        "test_dir": "data/processed/rcc_webdataset_v2/test",
    },
}


def _dataset_base(alias: str) -> Path:
    """Determine the root directory used to resolve dataset paths for an alias."""
    alias_key = f"WEB_DATASET_{alias.upper()}_ROOT"
    global_key = "WEB_DATASET_ROOT"
    if alias_key in os.environ:
        return _resolve(os.environ[alias_key], PROJECT_ROOT)
    if global_key in os.environ:
        return _resolve(os.environ[global_key], PROJECT_ROOT)
    return PROJECT_ROOT


def _dataset_field_override(alias: str, field: str) -> Path | None:
    key = f"WEB_DATASET_{alias.upper()}_{field.upper()}"
    if key in os.environ:
        return _resolve(os.environ[key], PROJECT_ROOT)
    return None


def get_all() -> Dict[str, Any]:
    """
    Return the canonical dictionary of resolved paths used during training.
    The outputs root is created eagerly so downstream code can assume it exists.
    """
    resolved: Dict[str, Any] = {
        "project_root": PROJECT_ROOT.resolve(),
        "outputs_root": ensure_dir(OUTPUTS_ROOT).resolve(),
        "webdataset": {},
    }

    for alias, section in WEB_DATASETS.items():
        base = _dataset_base(alias)
        resolved_section: Dict[str, Path] = {}
        for key, path in section.items():
            override = _dataset_field_override(alias, key)
            resolved_section[key] = override or _resolve(path, base)
        resolved["webdataset"][alias] = resolved_section

    return resolved
