from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import yaml


# Canonicalize class names across configs (handles common drift).
CLASS_ALIASES: Dict[str, str] = {
    # ONCO
    "Oncocytoma": "ONCO",
    "onco": "ONCO",
    "ONCOCYTOMA": "ONCO",
    "oncocytoma": "ONCO",
    "ONCO": "ONCO",

    # CHROMO
    "chRCC": "CHROMO",
    "chrcc": "CHROMO",
    "Chromophobe": "CHROMO",
    "CHROMOPHOBE": "CHROMO",
    "chromophobe": "CHROMO",
    "CHROMO": "CHROMO",

    # ccRCC / pRCC (common casing drift)
    "ccrcc": "ccRCC",
    "CCRCC": "ccRCC",
    "ccRCC": "ccRCC",
    "clearcell": "ccRCC",
    "clear_cell": "ccRCC",
    "clear-cell": "ccRCC",

    "prcc": "pRCC",
    "PRCC": "pRCC",
    "pRCC": "pRCC",
    "papillary": "pRCC",
    "papillaryrcc": "pRCC",
    "papillary_rcc": "pRCC",
    "papillary-rcc": "pRCC",

    # NOT_TUMOR (common synonyms)
    "Other": "NOT_TUMOR",
    "OTHER": "NOT_TUMOR",
    "Normal": "NOT_TUMOR",
    "NORMAL": "NOT_TUMOR",
    "not_tumor": "NOT_TUMOR",
    "not-tumor": "NOT_TUMOR",
    "not tumor": "NOT_TUMOR",
    "nontumor": "NOT_TUMOR",
    "benign": "NOT_TUMOR",
}

def _norm_key(s: str) -> str:
    """
    Normalization used for robust alias lookup (case-insensitive, ignores separators).
    """
    t = str(s).strip()
    t = t.casefold()
    for ch in (" ", "\t", "\n", "\r", "_", "-", "/"):
        t = t.replace(ch, "")
    return t


_CLASS_ALIASES_NORM: Dict[str, str] = {_norm_key(k): v for k, v in CLASS_ALIASES.items()}


def canon_class(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    # exact match first (preserve any intentionally-cased keys)
    if s in CLASS_ALIASES:
        return CLASS_ALIASES[s]
    # then normalized match (robust to casing / separators)
    nn = _norm_key(s)
    return _CLASS_ALIASES_NORM.get(nn, s)


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
    if not isinstance(raw, dict):
        raise RuntimeError(f"Invalid shortlist file (expected mapping at root): {path}")
    # Accept both:
    #   {classes: {CLS: {primary: [...], confounds: [...]}}}
    # and legacy/simple:
    #   {CLS: {primary: [...], confounds: [...]}}
    classes: Any = raw.get("classes", None)
    if classes is None:
        # heuristic: if root keys look like class names and values are dicts with primary/confounds -> treat as classes map
        looks_like_classes = True
        for _k, _v in raw.items():
            if not isinstance(_v, dict):
                looks_like_classes = False
                break
            if not (("primary" in _v) or ("confounds" in _v)):
                looks_like_classes = False
                break
        classes = raw if looks_like_classes else {}

    if not isinstance(classes, dict) or not classes:
        raise RuntimeError(f"Invalid shortlist file (classes missing/empty): {path}")

    def _as_str_list(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, (tuple, list)):
            return [str(x) for x in v if str(x).strip()]
        # tolerate single string
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []
    out: Dict[str, Dict[str, List[int]]] = {}
    for cls, items in classes.items():
        cls_norm = canon_class(str(cls)) or str(cls)
        if not isinstance(items, dict):
            continue
        primary_names = _as_str_list(items.get("primary", []))
        confound_names = _as_str_list(items.get("confounds", []))
        prim = [concept_to_idx[c] for c in primary_names if c in concept_to_idx]
        conf = [concept_to_idx[c] for c in confound_names if c in concept_to_idx]
        missing = [c for c in (primary_names + confound_names) if c not in concept_to_idx]
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
