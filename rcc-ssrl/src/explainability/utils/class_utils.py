from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import yaml


# Canonicalize class names across configs (handles common drift).
CLASS_ALIASES: Dict[str, str] = {
    "Oncocytoma": "ONCO",
    "onco": "ONCO",
    "ONCOCYTOMA": "ONCO",
    "chRCC": "CHROMO",
    "chrcc": "CHROMO",
    "Chromophobe": "CHROMO",
    "CHROMOPHOBE": "CHROMO",
    "Other": "NOT_TUMOR",
    "OTHER": "NOT_TUMOR",
    "Normal": "NOT_TUMOR",
    "NORMAL": "NOT_TUMOR",
}


def canon_class(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    return CLASS_ALIASES.get(s, s)


def idx_to_class(idx: Optional[int], class_names: Optional[Sequence[str]]) -> Optional[str]:
    if idx is None:
        return None
    if class_names and 0 <= idx < len(class_names):
        return str(class_names[idx])
    return str(idx)


def load_class_names(eval_run_dir: Path) -> Optional[List[str]]:
    """
    Best-effort load of class names from eval config (if present).
    """
    for name in ("config_eval.yaml", "config_resolved.yaml", "config.yaml"):
        cfg_path = eval_run_dir / name
        if not cfg_path.exists():
            continue
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
        except Exception:
            continue
        for key_path in [
            ("data", "class_names"),
            ("dataset", "class_names"),
            ("data", "classes"),
            ("dataset", "classes"),
        ]:
            cur: Any = cfg
            ok = True
            for k in key_path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok and isinstance(cur, list) and all(isinstance(x, str) for x in cur):
                return list(cur)
    return None


def load_shortlist_idx(path: Path, concept_to_idx: Dict[str, int], log: Any = None) -> Dict[str, Dict[str, List[int]]]:
    # Support both JSON (legacy) and YAML (canonical required file).
    if path.suffix.lower() in (".yaml", ".yml"):
        raw = yaml.safe_load(path.read_text())
    else:
        raw = json.loads(path.read_text())
    classes = raw.get("classes", {})
    if not isinstance(classes, dict) or not classes:
        raise RuntimeError(f"Invalid shortlist JSON (classes missing/empty): {path}")
    out: Dict[str, Dict[str, List[int]]] = {}
    for cls, items in classes.items():
        cls_norm = canon_class(str(cls)) or str(cls)
        if not isinstance(items, dict):
            continue
        prim = [concept_to_idx[c] for c in items.get("primary", []) if c in concept_to_idx]
        conf = [concept_to_idx[c] for c in items.get("confounds", []) if c in concept_to_idx]
        missing = [c for c in items.get("primary", []) + items.get("confounds", []) if c not in concept_to_idx]
        if missing and log is not None:
            try:
                log.warning("[SHORTLIST] Concepts missing in ontology (ignored) for %s: %s", cls_norm, missing)
            except Exception:
                pass
        out[cls_norm] = {"primary": prim, "confounds": conf}
    return out


def concept_indices_for_patch(shortlist: Dict[str, Dict[str, List[int]]], true_cls: Optional[str], pred_cls: Optional[str]) -> List[int]:
    idxs: Set[int] = set()
    if pred_cls and pred_cls in shortlist:
        idxs.update(shortlist[pred_cls].get("primary", []))
        idxs.update(shortlist[pred_cls].get("confounds", []))
    if true_cls and true_cls in shortlist:
        idxs.update(shortlist[true_cls].get("primary", []))
    return sorted(idxs)
