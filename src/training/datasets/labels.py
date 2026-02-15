"""Label normalisation and metadata helpers for WebDataset-backed loaders."""
from __future__ import annotations

from typing import Any, Dict, List

__all__ = ["make_class_to_id_norm", "class_labels_from_cfg", "normalise_label"]

_ALIAS_TABLE = {
    "CCRCC": ["CCRCC", "CC_RCC", "CLEAR_CELL_RCC", "CLEARCELL"],
    "PRCC": ["PRCC", "P_RCC", "PAPILLARY_RCC"],
    "CHROMO": ["CHROMO", "CHROMOPHOBE", "CHR", "CHRCC"],
    "ONCO": ["ONCO", "ONCOCYTOMA"],
    "NOT_TUMOR": ["NOT_TUMOR", "NON_TUMOR", "NONTUMOR", "NORMAL", "BACKGROUND"],
}


def normalise_label(value: Any) -> str:
    """Normalise class labels by uppercasing and removing separators."""
    return str(value).strip().upper().replace(" ", "_").replace("-", "_")


def make_class_to_id_norm(user_map: Dict[str, int]) -> Dict[str, int]:
    """
    Build a normalized class mapping, appending canonical aliases when present
    in the user-provided YAML.
    """
    base = {normalise_label(k): v for k, v in user_map.items()}
    out = dict(base)
    for canon, synonyms in _ALIAS_TABLE.items():
        if canon in base:
            for synonym in synonyms:
                out[normalise_label(synonym)] = base[canon]
    return out


def class_labels_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    mapping = (cfg.get("data", {}).get("webdataset", {}) or {}).get("class_to_id", {})
    if not mapping:
        return []
    inverse = {idx: name for name, idx in mapping.items()}
    return [inverse[idx] for idx in sorted(inverse)]
