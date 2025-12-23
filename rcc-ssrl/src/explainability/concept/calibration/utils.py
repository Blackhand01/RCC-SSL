#!/usr/bin/env python3
from __future__ import annotations

"""
Shared utilities for concept calibration/deep-validation.
Includes plotting, metrics, report/LaTeX helpers, and artifact validation.
"""

import copy
import json
import logging
import math
import shutil
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from explainability.utils.class_utils import canon_class
from ...paths import CALIBRATION_PATHS

# Canonical defaults (used by the unified CLI)
DEFAULT_CALIB_DIR = CALIBRATION_PATHS.metadata_dir
DEFAULT_ANALYSIS_DIR = CALIBRATION_PATHS.analysis_dir
DEFAULT_REPORT_DIR = CALIBRATION_PATHS.report_dir
DEFAULT_SHORTLIST_YAML = CALIBRATION_PATHS.shortlist_yaml


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

_PLT = None


def _get_plt():
    global _PLT
    if _PLT is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _PLT = plt
    return _PLT


def get_plt():
    return _get_plt()


def _save_fig(fig, out_base: Path, formats: Sequence[str], dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")


def plot_heatmap(
    mat: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    out_base: Path,
    title: str,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    plt = _get_plt()
    fig = plt.figure(figsize=(max(8, 0.35 * len(col_labels)), max(4, 0.35 * len(row_labels))))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(list(row_labels))
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(list(col_labels), rotation=90)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save_fig(fig, out_base, formats, dpi)
    plt.close(fig)


def plot_bar(
    labels: Sequence[str],
    values: np.ndarray,
    out_base: Path,
    title: str,
    *,
    xlabel: str = "",
    ylabel: str = "",
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
    rotate: int = 90,
) -> None:
    plt = _get_plt()
    fig = plt.figure(figsize=(max(8, 0.35 * len(labels)), 4.5))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(labels)), values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(list(labels), rotation=rotate, ha="right")
    fig.tight_layout()
    _save_fig(fig, out_base, formats, dpi)
    plt.close(fig)


def plot_barh(
    values: np.ndarray,
    labels: Sequence[str],
    out_base: Path,
    title: str,
    *,
    xlabel: str,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 200,
) -> None:
    plt = _get_plt()
    order = np.argsort(values)
    vals = values[order]
    labs = [labels[i] for i in order]
    fig = plt.figure(figsize=(10, max(4, 0.35 * len(labs))))
    ax = fig.add_subplot(111)
    ax.barh(np.arange(len(vals)), vals)
    ax.set_yticks(np.arange(len(vals)))
    ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    _save_fig(fig, out_base, formats, dpi)
    plt.close(fig)


def copy_plot_files(
    src_dir: Path,
    dst_dir: Path,
    *,
    exts: Sequence[str] = ("png", "pdf"),
) -> List[Path]:
    """
    Copy plot artifacts from a source directory into a report-ready figures dir.
    Preserves relative paths under src_dir.
    """
    if not src_dir.exists() or not src_dir.is_dir():
        return []
    dst_dir.mkdir(parents=True, exist_ok=True)
    ext_set = {f".{e.lstrip('.')}".lower() for e in exts}
    copied: List[Path] = []
    for p in src_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ext_set:
            continue
        rel = p.relative_to(src_dir)
        out = dst_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        copied.append(out)
    return copied


# ---------------------------------------------------------------------
# Concept + metric helpers
# ---------------------------------------------------------------------


def is_constant(x: np.ndarray, eps: float = 1e-12) -> bool:
    return bool(np.nanmax(x) - np.nanmin(x) <= eps)


def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


@dataclass(frozen=True)
class Concept:
    idx: int
    id: Optional[int]
    short_name: str
    name: str
    group: Optional[str]
    primary_class: Optional[str]
    prompt: Optional[str]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_yaml(path: Path) -> Any:
    obj = yaml.safe_load(path.read_text())
    return {} if obj is None else obj


def decode_np_str_array(x: np.ndarray) -> np.ndarray:
    # Handles bytes/object arrays coming from np.save
    if x.dtype == object:
        out = []
        for v in x.tolist():
            if isinstance(v, bytes):
                out.append(v.decode("utf-8", errors="replace"))
            else:
                out.append(str(v))
        return np.asarray(out, dtype=object)
    if np.issubdtype(x.dtype, np.bytes_):
        return np.asarray([v.decode("utf-8", errors="replace") for v in x], dtype=object)
    return x


def load_concepts(concepts_json: Path) -> List[Concept]:
    raw = _load_json(concepts_json)
    if isinstance(raw, dict) and "concepts" in raw:
        raw_list = raw["concepts"]
    else:
        raw_list = raw
    if not isinstance(raw_list, list) or not raw_list:
        raise RuntimeError(f"Invalid concepts.json format: {concepts_json}")

    concepts: List[Concept] = []
    for i, c in enumerate(raw_list):
        if not isinstance(c, dict):
            raise RuntimeError(f"Invalid concept entry at index {i}: {type(c)}")
        concepts.append(
            Concept(
                idx=i,
                id=c.get("id", None),
                short_name=str(c.get("short_name") or c.get("concept_short_name") or f"concept_{i}"),
                name=str(c.get("name") or c.get("concept_name") or f"Concept {i}"),
                group=(None if c.get("group") is None else str(c.get("group"))),
                primary_class=(None if c.get("primary_class") is None else str(c.get("primary_class"))),
                prompt=(None if c.get("prompt") is None else str(c.get("prompt"))),
            )
        )
    return concepts


def guess_class_names(cfg: Dict[str, Any], labels_raw: np.ndarray) -> Optional[List[str]]:
    # Best-effort: look for common patterns without assuming a schema.
    # If labels are strings already, prefer their sorted unique values.
    labels_raw = decode_np_str_array(labels_raw)
    if labels_raw.dtype == object:
        uniq = sorted({str(x) for x in labels_raw.tolist()})
        if len(uniq) >= 2:
            return uniq

    # Otherwise, try config keys.
    for key_path in [
        ("data", "class_names"),
        ("data", "classes"),
        ("dataset", "class_names"),
        ("dataset", "classes"),
        ("labels", "class_names"),
        ("labels", "classes"),
    ]:
        cur: Any = cfg
        ok = True
        for k in key_path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, list) and all(isinstance(x, str) for x in cur) and len(cur) >= 2:
            return cur
    return None


def normalize_labels(labels_raw: np.ndarray, class_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    labels_raw = decode_np_str_array(labels_raw)

    # String labels
    if labels_raw.dtype == object:
        uniq = sorted({canon_class(str(x)) or str(x) for x in labels_raw.tolist()})
        name_to_idx = {n: i for i, n in enumerate(uniq)}
        labels = np.asarray([name_to_idx[canon_class(str(x)) or str(x)] for x in labels_raw.tolist()], dtype=np.int64)
        return labels, uniq

    # Integer labels
    labels_int = labels_raw.astype(np.int64)
    uniq_int = sorted(set(labels_int.tolist()))
    remap = {v: i for i, v in enumerate(uniq_int)}
    labels = np.asarray([remap[v] for v in labels_int.tolist()], dtype=np.int64)

    if class_names is not None and len(class_names) == len(uniq_int):
        return labels, list(class_names)
    # Fallback names = stringified ints
    return labels, [str(v) for v in uniq_int]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _chunk_slices(n: int, chunk: int) -> Iterable[Tuple[int, int]]:
    i = 0
    while i < n:
        j = min(n, i + chunk)
        yield i, j
        i = j


def compute_fast_stats(
    scores: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    topk: int,
    chunk_size: int,
    *,
    allow_nonfinite: bool,
    log: logging.Logger,
) -> Dict[str, np.ndarray]:
    """
    One pass over scores (memmap-friendly) to compute:
      - sums, sums_sq per class
      - counts per class
      - top1 counts per class x concept
      - topk counts per class x concept
    """
    n, n_concepts = scores.shape
    counts = np.zeros((n_classes,), dtype=np.int64)
    sums = np.zeros((n_classes, n_concepts), dtype=np.float64)
    sums_sq = np.zeros((n_classes, n_concepts), dtype=np.float64)
    top1_counts = np.zeros((n_classes, n_concepts), dtype=np.int64)
    topk_counts = np.zeros((n_classes, n_concepts), dtype=np.int64)
    nonfinite_total = 0

    if topk < 1:
        raise ValueError("topk must be >= 1")
    k = min(topk, n_concepts)

    for a, b in _chunk_slices(n, chunk_size):
        y = labels[a:b]
        X = np.asarray(scores[a:b], dtype=np.float32)  # small chunk in RAM
        if not np.isfinite(X).all():
            bad = int((~np.isfinite(X)).sum())
            nonfinite_total += bad
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # counts
        counts += np.bincount(y, minlength=n_classes)

        # sums and sums_sq via indexed accumulation (no per-class loops)
        np.add.at(sums, y, X.astype(np.float64))
        np.add.at(sums_sq, y, (X.astype(np.float64) ** 2))

        # top-1
        t1 = np.argmax(X, axis=1).astype(np.int64)
        np.add.at(top1_counts, (y, t1), 1)

        # top-k
        if k == 1:
            tk = t1[:, None]
        else:
            tk = np.argpartition(X, -k, axis=1)[:, -k:]
        y_rep = np.repeat(y, tk.shape[1])
        c_rep = tk.reshape(-1).astype(np.int64)
        np.add.at(topk_counts, (y_rep, c_rep), 1)

    if nonfinite_total > 0:
        msg = f"Found non-finite scores in calibration matrix (replaced with 0.0): n={nonfinite_total}"
        if allow_nonfinite:
            log.warning(msg)
        else:
            raise RuntimeError("[ERROR] " + msg + " (use --allow-nonfinite to override)")

    return {
        "counts": counts,
        "sums": sums,
        "sums_sq": sums_sq,
        "top1_counts": top1_counts,
        "topk_counts": topk_counts,
    }


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Safe element-wise division with broadcasting.
    Uses np.divide(where=...) instead of boolean indexing, because boolean masks
    must match the indexed array shape (common failure case: b is (K,1) but a is (K,C)).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.zeros_like(a, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(a, b, out=out, where=(b != 0))
    return out


def build_metrics_tables(
    concepts: List[Concept],
    class_names: List[str],
    scores: np.ndarray,
    labels: np.ndarray,
    stats: Dict[str, np.ndarray],
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    n_classes = len(class_names)
    n_concepts = len(concepts)

    counts = stats["counts"].astype(np.float64)
    sums = stats["sums"]
    sums_sq = stats["sums_sq"]

    # mean/std per class x concept
    means = _safe_div(sums, counts[:, None])
    ex2 = _safe_div(sums_sq, counts[:, None])
    var = np.maximum(ex2 - means ** 2, 0.0)
    std = np.sqrt(var)

    # rest-of-classes stats (for delta and Cohen's d)
    total_count = float(np.sum(counts))
    total_sum = np.sum(sums, axis=0)
    total_sum_sq = np.sum(sums_sq, axis=0)

    rest_count = (total_count - counts)  # (K,)
    rest_sum = (total_sum[None, :] - sums)  # (K,C)
    rest_sum_sq = (total_sum_sq[None, :] - sums_sq)
    rest_mean = _safe_div(rest_sum, rest_count[:, None])
    rest_ex2 = _safe_div(rest_sum_sq, rest_count[:, None])
    rest_var = np.maximum(rest_ex2 - rest_mean ** 2, 0.0)
    rest_std = np.sqrt(rest_var)

    delta = means - rest_mean  # (K,C)

    # Cohen's d
    # pooled std with Bessel correction (guard tiny counts)
    cohen_d = np.zeros_like(delta, dtype=np.float64)
    for k in range(n_classes):
        n1 = counts[k]
        n0 = rest_count[k]
        if n1 < 2 or n0 < 2:
            continue
        s1 = std[k] ** 2
        s0 = rest_std[k] ** 2
        pooled = ((n1 - 1.0) * s1 + (n0 - 1.0) * s0) / (n1 + n0 - 2.0)
        pooled = np.maximum(pooled, 1e-12)
        cohen_d[k] = delta[k] / np.sqrt(pooled)

    # top-1/top-k frequency per class
    top1_freq = _safe_div(stats["top1_counts"].astype(np.float64), counts[:, None])
    # topk_counts counts occurrences across patches; each patch contributes 0/1 per concept.
    topk_freq = _safe_div(stats["topk_counts"].astype(np.float64), counts[:, None])

    rows: List[Dict[str, Any]] = []
    for k, cls in enumerate(class_names):
        for c in concepts:
            j = c.idx
            rows.append(
                {
                    "class": cls,
                    "concept_idx": j,
                    "concept_short_name": c.short_name,
                    "concept_name": c.name,
                    "group": c.group,
                    "primary_class": c.primary_class,
                    "n_pos": int(counts[k]),
                    "mean_pos": float(means[k, j]),
                    "std_pos": float(std[k, j]),
                    "mean_rest": float(rest_mean[k, j]),
                    "std_rest": float(rest_std[k, j]),
                    "delta_mean": float(delta[k, j]),
                    "cohen_d": float(cohen_d[k, j]),
                    "top1_freq": float(top1_freq[k, j]),
                    "topk_freq": float(topk_freq[k, j]),
                }
            )

    mats = {
        "means": means,
        "std": std,
        "delta": delta,
        "cohen_d": cohen_d,
        "top1_freq": top1_freq,
        "topk_freq": topk_freq,
        "counts": stats["counts"],
    }
    return rows, mats


def compute_auc_ap_for_selected(
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    concept_indices_by_class: Dict[int, List[int]],
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Compute AUC/AP only for selected (class, concept) pairs to keep runtime bounded.
    """
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError("scikit-learn is required for AUC/AP computation.") from e

    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    n = labels.shape[0]
    for k, _cls in enumerate(class_names):
        idxs = concept_indices_by_class.get(k, [])
        if not idxs:
            continue
        y = (labels == k).astype(np.int32)
        n_pos = int(y.sum())
        n_neg = int(n - n_pos)
        # Need both classes for OVR metrics
        if n_pos < 1 or n_neg < 1:
            for j in idxs:
                out[(k, j)] = {
                    "auc_ovr": float("nan"),
                    "ap_ovr": float("nan"),
                    "auc_valid": 0.0,
                    "ap_valid": 0.0,
                    "reason": 1.0,  # 1 = only one class present
                }
            continue
        for j in idxs:
            s = np.asarray(scores[:, j], dtype=np.float32)
            if not np.isfinite(s).all():
                s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            # Constant scores: AUC=0.5; AP=prevalence (baseline). Keep numeric (valid but uninformative).
            if is_constant(s):
                prev = float(n_pos) / float(max(1, n))
                out[(k, j)] = {
                    "auc_ovr": 0.5,
                    "ap_ovr": prev,
                    "auc_valid": 1.0,
                    "ap_valid": 1.0,
                    "reason": 2.0,  # 2 = constant scores
                }
                continue
            # Small sample sizes can make metrics noisy; still compute if sklearn allows.
            auc = float("nan")
            ap = float("nan")
            auc_valid = 0.0
            ap_valid = 0.0
            try:
                auc = float(roc_auc_score(y, s))
                auc_valid = 1.0 if math.isfinite(auc) else 0.0
            except Exception:
                auc = float("nan")
                auc_valid = 0.0
            try:
                ap = float(average_precision_score(y, s))
                ap_valid = 1.0 if math.isfinite(ap) else 0.0
            except Exception:
                ap = float("nan")
                ap_valid = 0.0
            out[(k, j)] = {
                "auc_ovr": auc,
                "ap_ovr": ap,
                "auc_valid": auc_valid,
                "ap_valid": ap_valid,
                "reason": 0.0,  # 0 = ok
            }
    return out


def resolve_calibration_dir(cal_run: Path) -> Path:
    """
    Backward-compatible resolver:
    - If cal_run directly contains calibration artifacts -> use it
    - Else if cal_run/runs/<id>/ contains them -> use the newest run dir
    """
    req = ["text_features.pt", "scores_fp32.npy", "labels.npy", "concepts.json"]
    if all((cal_run / r).exists() for r in req):
        return cal_run
    runs = cal_run / "runs"
    if runs.exists() and runs.is_dir():
        candidates = sorted([p for p in runs.iterdir() if p.is_dir()], key=lambda p: p.name)
        for p in reversed(candidates):
            if all((p / r).exists() for r in req):
                print(f"[WARN] Using legacy calibration run dir: {p}")
                return p
    return cal_run


def build_selection_from_delta(
    delta: np.ndarray,
    topm_per_class: int,
) -> Dict[int, List[int]]:
    """
    Pick top-M concepts per class using delta(mean_pos - mean_rest).
    """
    n_classes, n_concepts = delta.shape
    out: Dict[int, List[int]] = {}
    m = max(1, int(topm_per_class))
    for k in range(n_classes):
        order = np.argsort(delta[k])[::-1]
        out[k] = order[: min(m, n_concepts)].tolist()
    return out


def build_selection_union(
    mats: Dict[str, np.ndarray],
    *,
    topm_per_metric: int,
) -> Dict[int, List[int]]:
    """
    Build a robust selection for AUC/AP computation.
    Union of top-M by:
      - delta_mean
      - cohen_d
      - top1_freq
      - mean_pos (helps when delta is small but the concept is consistently high in-class)
    """
    delta = mats["delta"]
    cohen_d = mats["cohen_d"]
    top1 = mats["top1_freq"]
    means = mats.get("means", None)
    n_classes, n_concepts = delta.shape
    m = max(1, int(topm_per_metric))

    out: Dict[int, List[int]] = {}
    for k in range(n_classes):
        sel = set()
        sel.update(np.argsort(delta[k])[::-1][: min(m, n_concepts)].tolist())
        sel.update(np.argsort(cohen_d[k])[::-1][: min(m, n_concepts)].tolist())
        sel.update(np.argsort(top1[k])[::-1][: min(m, n_concepts)].tolist())
        if means is not None:
            sel.update(np.argsort(means[k])[::-1][: min(m, n_concepts)].tolist())
        out[k] = sorted(sel)
    return out


def augment_selection_with_primary_concepts(
    concepts: List[Concept], class_names: List[str], selected: Dict[int, List[int]]
) -> Dict[int, List[int]]:
    out = copy.deepcopy(selected)
    for k, cls in enumerate(class_names):
        cls_c = canon_class(cls) or str(cls)
        js = set(out.get(k, []))
        for c in concepts:
            pc = canon_class(c.primary_class) if c.primary_class is not None else None
            if pc is not None and pc == cls_c:
                js.add(int(c.idx))
        out[k] = sorted(js)
    return out


def write_exemplars(
    out_dir: Path,
    scores: np.ndarray,
    labels: np.ndarray,
    keys: Optional[np.ndarray],
    class_names: List[str],
    concepts: List[Concept],
    selected: Dict[int, List[int]],
    max_exemplars: int,
    chunk_size: int,
) -> None:
    """
    Save top exemplars (wds_key) per (class, concept) for qualitative audit.
    """
    if keys is None:
        return
    keys = decode_np_str_array(keys)

    n, n_concepts = scores.shape
    K = len(class_names)
    max_ex = max(1, int(max_exemplars))

    # Track top exemplars per pair with simple fixed-size buffers
    # store (score, idx)
    buffers: Dict[Tuple[int, int], List[Tuple[float, int]]] = {}
    for k, js in selected.items():
        for j in js:
            buffers[(k, j)] = []

    def _push(buf: List[Tuple[float, int]], val: float, idx: int) -> None:
        if math.isnan(val):
            return
        if len(buf) < max_ex:
            buf.append((val, idx))
            if len(buf) == max_ex:
                buf.sort(key=lambda t: t[0])  # ascending
            return
        # buf full and sorted asc
        if val <= buf[0][0]:
            return
        buf[0] = (val, idx)
        buf.sort(key=lambda t: t[0])

    for a, b in _chunk_slices(n, chunk_size):
        y = labels[a:b]
        X = np.asarray(scores[a:b], dtype=np.float32)
        for k in range(K):
            js = selected.get(k, [])
            if not js:
                continue
            mask = (y == k)
            if not np.any(mask):
                continue
            idx_local = np.where(mask)[0]
            idx_global = (idx_local + a).astype(np.int64)
            for j in js:
                vals = X[mask, j]
                buf = buffers[(k, j)]
                for vv, ii in zip(vals.tolist(), idx_global.tolist()):
                    _push(buf, float(vv), int(ii))

    ex_dir = out_dir / "exemplars"
    ensure_dir(ex_dir)

    for (k, j), buf in buffers.items():
        cls = class_names[k]
        c = concepts[j]
        buf_sorted = sorted(buf, key=lambda t: t[0], reverse=True)
        out_csv = ex_dir / f"top_{cls}__{c.short_name}.csv"
        with out_csv.open("w") as f:
            f.write("rank,score,wds_key,label\n")
            for r, (sc, idx) in enumerate(buf_sorted, start=1):
                f.write(f"{r},{sc:.6f},{keys[idx]},{cls}\n")


# ---------------------------------------------------------------------
# Report / LaTeX helper
# ---------------------------------------------------------------------


def format_tex_table(df: pd.DataFrame, cols: List[str], caption: str, label: str) -> str:
    # Minimal LaTeX table generator (no external deps).
    # Escapes underscores in strings for LaTeX.
    def esc(s: str) -> str:
        return s.replace("_", "\\_")

    header = " & ".join([esc(c) for c in cols]) + " \\\\"
    lines = [header, "\\hline"]
    for _, r in df.iterrows():
        row = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float) and np.isnan(v):
                row.append("")
            elif isinstance(v, (int, np.integer)):
                row.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                row.append(f"{float(v):.3f}")
            else:
                row.append(esc(str(v)))
        lines.append(" & ".join(row) + " \\\\")

    body = "\n".join(lines)
    align = "l" * len(cols)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{esc(caption)}}}\n"
        f"\\label{{{esc(label)}}}\n"
        f"\\begin{{tabular}}{{{align}}}\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def copy_plot_files(src_dir: Path, dst_dir: Path, *, exts: Sequence[str] = (".png", ".pdf")) -> int:
    if not src_dir.exists() or not src_dir.is_dir():
        return 0
    ensure_dir(dst_dir)
    n = 0
    for p in src_dir.glob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        shutil.copy2(p, dst_dir / p.name)
        n += 1
    return n


# ---------------------------------------------------------------------
# Artifact validation helpers
# ---------------------------------------------------------------------


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if v < 1024.0:
            return f"{v:.1f}{u}"
        v /= 1024.0
    return f"{v:.1f}PB"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    # robust against empty or weird encodings
    return pd.read_csv(path)


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")


def _has_any_parent_named(p: Path, root: Path, names: Tuple[str, ...]) -> bool:
    try:
        rel = p.relative_to(root)
    except Exception:
        rel = p
    parts = [x.lower() for x in rel.parts]
    return any(n.lower() in parts for n in names)


def _find_forbidden_images(root: Path, *, allow_dirs: Tuple[str, ...]) -> List[Path]:
    bad: List[Path] = []
    if not root.exists() or not root.is_dir():
        return bad
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if not _is_image_file(p):
            continue
        if _has_any_parent_named(p, root, allow_dirs):
            continue
        bad.append(p)
        if len(bad) >= 50:
            break
    return bad


def _is_nonempty_file(path: Path, min_bytes: int = 32) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


def _collect_leaf_strings(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            out.append(s)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect_leaf_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_collect_leaf_strings(v))
    return out


def _finite_sample_ok(arr: np.ndarray, max_rows: int = 2048, max_cols: int = 2048) -> Tuple[bool, str]:
    """
    Avoid scanning gigantic arrays: sample a top-left slice and verify finiteness.
    """
    if arr.ndim == 1:
        sl = arr[: min(arr.shape[0], max_rows)]
    elif arr.ndim == 2:
        sl = arr[: min(arr.shape[0], max_rows), : min(arr.shape[1], max_cols)]
    else:
        return False, f"Unsupported ndim={arr.ndim}"

    # Non-numeric arrays (e.g., labels/keys saved as strings/objects) are valid here.
    # Finiteness only matters for numeric types.
    try:
        if hasattr(sl, "dtype") and sl.dtype is not None and sl.dtype.kind not in ("b", "i", "u", "f"):
            return True, f"SKIP finiteness check for non-numeric dtype={sl.dtype}"
    except Exception:
        pass

    # Convert to float for finiteness check if needed
    try:
        slf = np.asarray(sl, dtype=np.float32)
    except Exception as e:
        return False, f"Cannot cast to float32: {e}"

    bad = ~np.isfinite(slf)
    n_bad = int(bad.sum())
    if n_bad > 0:
        return False, f"Found {n_bad} non-finite values in sampled slice {slf.shape}"
    return True, f"OK (finite) on sampled slice {slf.shape}"


def _require_exists(path: Path, errors: List[str], what: str) -> None:
    if not path.exists():
        errors.append(f"[MISSING] {what}: {path}")


def _require_dir(path: Path, errors: List[str], what: str) -> None:
    if not path.exists() or not path.is_dir():
        errors.append(f"[MISSING] {what} dir: {path}")


def _require_nonempty_file(path: Path, errors: List[str], what: str, min_bytes: int = 32) -> None:
    if not _is_nonempty_file(path, min_bytes=min_bytes):
        if not path.exists():
            errors.append(f"[MISSING] {what}: {path}")
        elif path.is_dir():
            errors.append(f"[INVALID] {what} is a directory, expected file: {path}")
        else:
            errors.append(
                f"[INVALID] {what} is empty/too small ({_human_bytes(path.stat().st_size)}): {path}"
            )


def check_calibration(calib_dir: Path, strict: bool) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warns: List[str] = []

    _require_dir(calib_dir, errors, "Calibration metadata")
    if errors:
        return errors, warns

    req_files = {
        "concepts.json": calib_dir / "concepts.json",
        "text_features.pt": calib_dir / "text_features.pt",
        "scores_fp32.npy": calib_dir / "scores_fp32.npy",
        "labels.npy": calib_dir / "labels.npy",
        "keys.npy": calib_dir / "keys.npy",
        "auc_primary_class.csv": calib_dir / "auc_primary_class.csv",
        "config_resolved.yaml": calib_dir / "config_resolved.yaml",
    }
    for k, p in req_files.items():
        _require_nonempty_file(p, errors, f"Calibration file {k}")

    plots_dir = calib_dir / "plots"
    if plots_dir.exists() and plots_dir.is_dir():
        pngs = sorted(plots_dir.glob("*.png"))
        pdfs = sorted(plots_dir.glob("*.pdf"))
        if len(pngs) + len(pdfs) == 0:
            warns.append(f"[WARN] Calibration plots/ exists but is empty: {plots_dir}")
    else:
        if strict:
            errors.append(f"[MISSING] Calibration plots dir (strict): {plots_dir}")
        else:
            warns.append(f"[WARN] Calibration plots dir missing: {plots_dir}")

    # Parse concepts.json
    cj = req_files["concepts.json"]
    if cj.exists():
        try:
            obj = json.loads(cj.read_text())
            # Accept both legacy list and current {"meta":..., "concepts":[...]}.
            if isinstance(obj, dict) and "concepts" in obj:
                concepts = obj.get("concepts", [])
            else:
                concepts = obj
            if not isinstance(concepts, list) or len(concepts) == 0:
                errors.append(f"[INVALID] concepts.json is not a non-empty list (or dict-with-concepts): {cj}")
            else:
                bad = 0
                for c in concepts[:50]:
                    if not isinstance(c, dict):
                        bad += 1
                        continue
                    # current schema: name, short_name, prompt/prompts
                    if not (c.get("name") and (c.get("short_name") or c.get("concept_short_name"))):
                        bad += 1
                    if not (c.get("prompt") or c.get("prompts")):
                        bad += 1
                if bad > 0:
                    warns.append(f"[WARN] concepts.json has {bad} suspicious entries (first 50 checked).")
        except Exception as e:
            errors.append(f"[INVALID] concepts.json parse error: {e}")

    # Load arrays and validate shape consistency (sampled checks to avoid huge RAM)
    scores_p = req_files["scores_fp32.npy"]
    labels_p = req_files["labels.npy"]
    keys_p = req_files["keys.npy"]

    n_concepts: int | None = None
    if (calib_dir / "concepts.json").exists():
        try:
            obj = json.loads((calib_dir / "concepts.json").read_text())
            if isinstance(obj, dict) and "concepts" in obj and isinstance(obj["concepts"], list):
                n_concepts = len(obj["concepts"])
            elif isinstance(obj, list):
                n_concepts = len(obj)
        except Exception:
            pass

    try:
        # scores are large numeric arrays -> memmap is fine
        scores = np.load(scores_p, mmap_mode="r")
        # labels/keys are often dtype=object (strings); mmap is not supported.
        labels = np.load(labels_p, allow_pickle=True)
        keys = np.load(keys_p, allow_pickle=True)
    except Exception as e:
        errors.append(f"[INVALID] Failed to np.load calibration arrays: {e}")
        return errors, warns

    if scores.ndim != 2:
        errors.append(f"[INVALID] scores_fp32.npy expected 2D, got shape={getattr(scores, 'shape', None)}")
    if labels.ndim != 1:
        errors.append(f"[INVALID] labels.npy expected 1D, got shape={getattr(labels, 'shape', None)}")
    if keys.ndim != 1:
        errors.append(f"[INVALID] keys.npy expected 1D, got shape={getattr(keys, 'shape', None)}")

    if scores.ndim == 2 and labels.ndim == 1 and keys.ndim == 1:
        n = scores.shape[0]
        if labels.shape[0] != n:
            errors.append(f"[INVALID] labels length {labels.shape[0]} != scores rows {n}")
        if keys.shape[0] != n:
            errors.append(f"[INVALID] keys length {keys.shape[0]} != scores rows {n}")
        if n_concepts is not None and scores.shape[1] != n_concepts:
            warns.append(
                f"[WARN] scores columns {scores.shape[1]} != len(concepts.json) {n_concepts} (check pipeline expectations)"
            )

        ok, msg = _finite_sample_ok(scores)
        if not ok:
            errors.append(f"[INVALID] scores_fp32.npy: {msg}")

        ok, msg = _finite_sample_ok(labels)
        if not ok:
            errors.append(f"[INVALID] labels.npy: {msg}")

    # Validate text_features.pt
    tfp = req_files["text_features.pt"]
    if tfp.exists():
        try:
            import torch

            obj = torch.load(tfp, map_location="cpu")
            if torch.is_tensor(obj):
                tf = obj
            elif isinstance(obj, dict) and any(k in obj for k in ("text_features", "features")):
                tf = obj.get("text_features", obj.get("features"))
            else:
                tf = None
            if tf is None or not torch.is_tensor(tf):
                errors.append(f"[INVALID] text_features.pt unexpected format (not tensor/dict-with-tensor): {tfp}")
            else:
                if tf.ndim != 2:
                    warns.append(f"[WARN] text_features tensor expected 2D, got shape={tuple(tf.shape)}")
                if n_concepts is not None and tf.shape[0] != n_concepts:
                    warns.append(
                        f"[WARN] text_features rows {tf.shape[0]} != len(concepts.json) {n_concepts}"
                    )
                if not torch.isfinite(tf).all().item():
                    errors.append("[INVALID] text_features contains non-finite values")
        except Exception as e:
            errors.append(f"[INVALID] torch.load(text_features.pt) failed: {e}")

    # Validate auc_primary_class.csv minimal
    aucp = req_files["auc_primary_class.csv"]
    if aucp.exists():
        try:
            df = _safe_read_csv(aucp)
            if df.shape[0] == 0:
                errors.append(f"[INVALID] auc_primary_class.csv is empty: {aucp}")
            # numeric column presence (best-effort)
            num_cols = [c for c in df.columns if "auc" in c.lower() or "ap" in c.lower()]
            if len(num_cols) == 0:
                warns.append(f"[WARN] auc_primary_class.csv has no obvious auc/ap columns: cols={list(df.columns)}")
        except Exception as e:
            errors.append(f"[INVALID] auc_primary_class.csv parse error: {e}")

    # config_resolved.yaml should parse
    cr = req_files["config_resolved.yaml"]
    if cr.exists():
        try:
            _ = load_yaml(cr)
        except Exception as e:
            errors.append(f"[INVALID] config_resolved.yaml parse error: {e}")

    return errors, warns


def check_deep_validation(analysis_dir: Path, strict: bool, min_valid_per_class: int) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warns: List[str] = []

    _require_dir(analysis_dir, errors, "Deep validation analysis")
    if errors:
        return errors, warns

    metrics_csv = analysis_dir / "metrics_per_class.csv"
    top_json = analysis_dir / "top_concepts_by_class.json"
    plots_dir = analysis_dir / "plots"

    _require_nonempty_file(metrics_csv, errors, "metrics_per_class.csv")
    _require_nonempty_file(top_json, errors, "top_concepts_by_class.json")

    if plots_dir.exists() and plots_dir.is_dir():
        imgs = list(plots_dir.rglob("*.png")) + list(plots_dir.rglob("*.pdf"))
        if len(imgs) == 0:
            warns.append(f"[WARN] analysis/plots exists but no png/pdf found: {plots_dir}")
    else:
        if strict:
            errors.append(f"[MISSING] analysis/plots dir (strict): {plots_dir}")
        else:
            warns.append(f"[WARN] analysis/plots dir missing: {plots_dir}")

    # metrics sanity: auc_ovr/ap_ovr should be numeric where defined
    if metrics_csv.exists():
        try:
            df = _safe_read_csv(metrics_csv)
            if df.shape[0] == 0:
                errors.append(f"[INVALID] metrics_per_class.csv is empty: {metrics_csv}")
                return errors, warns

            # Backward/forward compatible concept column naming:
            if "concept_short_name" not in df.columns and "concept" in df.columns:
                df = df.rename(columns={"concept": "concept_short_name"})
            required_cols = {"class", "concept_short_name"}
            missing = sorted(list(required_cols - set(df.columns)))
            if missing:
                errors.append(f"[INVALID] metrics_per_class.csv missing required columns: {missing}")

            # These are used for ranking/gating in your pipeline; they must be numeric when present.
            for col in ("auc_ovr", "ap_ovr"):
                if col not in df.columns:
                    warns.append(f"[WARN] metrics_per_class.csv missing column {col} (expected for ranking)")
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "auc_ovr" in df.columns or "ap_ovr" in df.columns:
                valid_auc = df["auc_ovr"].notna() if "auc_ovr" in df.columns else False
                valid_ap = df["ap_ovr"].notna() if "ap_ovr" in df.columns else False
                valid = (valid_auc | valid_ap)
                n_valid = int(valid.sum())
                if n_valid == 0:
                    errors.append("[INVALID] No rows have auc_ovr or ap_ovr numeric (ranking would be broken).")

                # Per-class coverage check: metrics like AP are defined for binary tasks (needs pos/neg).
                if "class" in df.columns:
                    per = df.assign(valid=valid).groupby("class")["valid"].sum().sort_values()
                    low = per[per < min_valid_per_class]
                    if len(low) > 0:
                        msg = ", ".join([f"{k}={int(v)}" for k, v in low.items()])
                        if strict:
                            errors.append(f"[INVALID] Too few valid (auc_ovr/ap_ovr) rows per class: {msg}")
                        else:
                            warns.append(f"[WARN] Low valid (auc_ovr/ap_ovr) rows per class: {msg}")

            # quick NaN bomb detection
            if "auc_ovr" in df.columns:
                frac_nan_auc = float(df["auc_ovr"].isna().mean())
                if frac_nan_auc > 0.95:
                    warns.append(f"[WARN] auc_ovr is NaN for {frac_nan_auc*100:.1f}% rows (check gating inputs).")
            if "ap_ovr" in df.columns:
                frac_nan_ap = float(df["ap_ovr"].isna().mean())
                if frac_nan_ap > 0.95:
                    warns.append(f"[WARN] ap_ovr is NaN for {frac_nan_ap*100:.1f}% rows (check gating inputs).")

        except Exception as e:
            errors.append(f"[INVALID] Failed reading/parsing metrics_per_class.csv: {e}")

    # top_concepts_by_class.json should parse
    if top_json.exists():
        try:
            obj = json.loads(top_json.read_text())
            if not isinstance(obj, dict) or len(obj) == 0:
                errors.append(f"[INVALID] top_concepts_by_class.json is not a non-empty dict: {top_json}")
        except Exception as e:
            errors.append(f"[INVALID] top_concepts_by_class.json parse error: {e}")

    # exemplars is optional, but warn if missing (useful for debugging)
    ex_dir = analysis_dir / "exemplars"
    if not ex_dir.exists():
        warns.append(f"[WARN] exemplars/ not found (optional but useful): {ex_dir}")

    return errors, warns


def check_report(report_dir: Path, strict: bool) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warns: List[str] = []

    _require_dir(report_dir, errors, "Report")
    if errors:
        return errors, warns

    report_md = report_dir / "report.md"
    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "paper_tables"

    _require_nonempty_file(report_md, errors, "report.md", min_bytes=32)

    if figures_dir.exists() and figures_dir.is_dir():
        imgs = list(figures_dir.rglob("*.png")) + list(figures_dir.rglob("*.pdf"))
        if len(imgs) == 0:
            warns.append(f"[WARN] report/figures exists but empty: {figures_dir}")
    else:
        if strict:
            errors.append(f"[MISSING] report/figures dir (strict): {figures_dir}")
        else:
            warns.append(f"[WARN] report/figures dir missing: {figures_dir}")

    if tables_dir.exists() and tables_dir.is_dir():
        tex = list(tables_dir.rglob("*.tex"))
        if len(tex) == 0:
            warns.append(f"[WARN] report/paper_tables exists but no .tex found: {tables_dir}")
    else:
        if strict:
            errors.append(f"[MISSING] report/paper_tables dir (strict): {tables_dir}")
        else:
            warns.append(f"[WARN] report/paper_tables dir missing: {tables_dir}")

    return errors, warns


def check_shortlist(shortlist_yaml: Path, strict: bool) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warns: List[str] = []

    _require_nonempty_file(shortlist_yaml, errors, "Shortlist YAML", min_bytes=32)
    if errors:
        return errors, warns

    # Also require the sibling JSON/CSV produced by the pipeline.
    shortlist_json = shortlist_yaml.with_suffix(".json")
    flat_csv = shortlist_yaml.with_name("concepts_shortlist_flat.csv")
    _require_nonempty_file(shortlist_json, errors, "Shortlist JSON", min_bytes=32)
    _require_nonempty_file(flat_csv, errors, "Shortlist flat CSV", min_bytes=32)

    try:
        obj = load_yaml(shortlist_yaml)
    except Exception as e:
        errors.append(f"[INVALID] shortlist YAML parse error: {e}")
        return errors, warns

    # Generic schema sanity: ensure we actually have some concepts as leaf strings.
    leaf = _collect_leaf_strings(obj)
    # Heuristic: filter out obvious non-concept strings (very short / numbers)
    concepts = [s for s in leaf if len(s) >= 3 and not s.strip().isdigit()]
    uniq = sorted(set(concepts))
    if len(uniq) < 10:
        msg = f"[WARN] shortlist seems too small (unique leaf strings={len(uniq)})."
        if strict:
            errors.append(msg.replace("[WARN]", "[INVALID]"))
        else:
            warns.append(msg)

    # Validate JSON parses
    try:
        _ = json.loads(shortlist_json.read_text())
    except Exception as e:
        errors.append(f"[INVALID] shortlist JSON parse error: {e}")

    # Validate CSV parse
    try:
        df = _safe_read_csv(flat_csv)
        if df.shape[0] == 0:
            errors.append(f"[INVALID] shortlist flat CSV empty: {flat_csv}")
    except Exception as e:
        errors.append(f"[INVALID] shortlist flat CSV parse error: {e}")

    return errors, warns


def check_spatial_concept_light_outputs(
    output_root: Path,
    strict: bool,
) -> Tuple[List[str], List[str]]:
    """
    Validate that light outputs under:
      - output/spatial/<MODEL_ID>/
      - output/roi/<MODEL_ID>/
    do NOT contain per-patch image dumps, and that summaries point to existing heavy artifacts
    under <MODEL_ROOT>/attention_rollout_concept/run_<...>/.
    """
    errors: List[str] = []
    warns: List[str] = []

    allow_dirs = ("plots", "figures", "paper_tables", "tables")

    def _check_one_model_dir(kind: str, model_dir: Path) -> None:
        # 1) forbid per-patch images outside allowed plot/figure dirs
        bad = _find_forbidden_images(model_dir, allow_dirs=allow_dirs)
        if bad:
            ex = ", ".join([str(p) for p in bad[:8]])
            errors.append(
                f"[INVALID] {kind} light dir contains image files outside {allow_dirs}: {model_dir} (e.g. {ex})"
            )

        # 2) also forbid common per-patch directories by name (idx_*/items/)
        idx_dirs = [p for p in model_dir.rglob("idx_*") if p.is_dir()]
        if idx_dirs:
            errors.append(f"[INVALID] {kind} light dir contains idx_* subdirs (per-patch dump): {model_dir}")
        items_dirs = [p for p in model_dir.rglob("items") if p.is_dir()]
        if items_dirs and any((p / "idx_00000001").parent == p for p in items_dirs):
            errors.append(f"[INVALID] {kind} light dir contains items/ (per-patch dump): {model_dir}")

        # 3) validate latest_run.json (if present)
        latest = model_dir / "spatial_concept" / "latest_run.json"
        if not latest.exists():
            warns.append(f"[WARN] Missing spatial_concept/latest_run.json (skip heavy-path checks): {latest}")
            return

        try:
            obj = json.loads(latest.read_text())
        except Exception as e:
            errors.append(f"[INVALID] Failed to parse {latest}: {e}")
            return

        heavy_dir = Path(str(obj.get('heavy_run_dir') or '')).expanduser()
        if not heavy_dir.exists():
            errors.append(f"[INVALID] heavy_run_dir does not exist (from {latest}): {heavy_dir}")
            return

        # Expected heavy structure
        if not (heavy_dir.name.startswith("run_") and heavy_dir.parent.name == "attention_rollout_concept"):
            warns.append(
                f"[WARN] heavy_run_dir does not look canonical (expected .../attention_rollout_concept/run_*): {heavy_dir}"
            )

        req = [
            heavy_dir / "selection" / "xai_selection.json",
            heavy_dir / "xai_summary.csv",
            heavy_dir / "xai_summary.json",
        ]
        for p in req:
            _require_nonempty_file(p, errors, f"Heavy artifact {p.name}")

        # 4) validate indexed file paths in heavy summary json
        sj = heavy_dir / "xai_summary.json"
        if not sj.exists():
            return
        try:
            payload = json.loads(sj.read_text())
        except Exception as e:
            errors.append(f"[INVALID] Failed to parse heavy summary json: {sj} ({e})")
            return

        items = payload.get("items", [])
        if not isinstance(items, list):
            errors.append(f"[INVALID] heavy summary items is not a list: {sj}")
            return

        keys_to_check = [
            "input_png",
            "attn_rollout_npy",
            "attn_rollout_png",
            "roi_png",
            "roi_bbox_json",
            "concept_scores_json",
        ]
        n_missing = 0
        for it in items[:200]:  # bound checks
            if not isinstance(it, dict):
                continue
            for k in keys_to_check:
                v = it.get(k, None)
                if not v:
                    continue
                p = heavy_dir / str(v)
                if not p.exists():
                    n_missing += 1
                    if n_missing <= 10:
                        errors.append(f"[INVALID] Indexed path missing: {p}")

        if n_missing > 10:
            errors.append(f"[INVALID] Indexed path missing count is high: {n_missing} (first 10 shown)")

    for kind in ("spatial", "roi"):
        root = output_root / kind
        if not root.exists():
            warns.append(f"[WARN] Light output root missing (skip): {root}")
            continue
        for model_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
            _check_one_model_dir(kind, model_dir)

    return errors, warns
