build_shortlist.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

"""
Post-process deep validation metrics to:
  - separate diagnostic vs confounding concepts per class
  - build FINAL concept shortlist (YAML canonical + JSON mirror) for downstream concept-XAI (test/no-ROI)
  - write a paper-ready report (markdown + figures + optional LaTeX tables)

Inputs:
  canonical: output/calibration/analysis/metrics_per_class.csv

Outputs (canonical analysis/):
  - concepts_shortlist.yaml   (OFFICIAL)
  - concepts_shortlist.json   (mirror for backward compat)
  - (optional) concepts_shortlist_flat.csv

Outputs (canonical analysis/report/):
  - report.md
  - figures/ (copied from deep_validation/plots when available)
  - paper_tables/ (LaTeX tables)
"""

import argparse
import json
import shutil
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from explainability.utils.bootstrap import bootstrap_package

bootstrap_package(__file__, globals())

from . import ensure_layout, SHORTLIST_DIR, REPORT_DIR, ANALYSIS_DIR
from ..utils.class_utils  import canon_class


# ---------------------------------------------------------------------
# Canonical mappings
# ---------------------------------------------------------------------
# Your dataset labels (from labels.npy / config) vs ontology group naming.
# This is ONLY for "group match" sanity; the diagnostic match should use primary_class.
GROUP_BY_CLASS = {
    "ccRCC": "ccRCC",
    "pRCC": "pRCC",
    "CHROMO": "chRCC",
    "ONCO": "Oncocytoma",
    "NOT_TUMOR": "NOT_TUMOR",
}


def _norm_str(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    return s if s else None


def _canon_class(x: Optional[str]) -> Optional[str]:
    s = _norm_str(x)
    if s is None:
        return None
    return canon_class(s) or s


def _choose_rank_col(df: pd.DataFrame, preferred: List[str]) -> str:
    for c in preferred:
        if c in df.columns:
            return c
    raise ValueError(f"No ranking column found among {preferred}. Available={list(df.columns)}")


def _is_primary_match(row: pd.Series) -> bool:
    cls = _canon_class(row.get("class"))
    pc = _canon_class(row.get("primary_class"))
    if cls is None:
        return False
    # Backward compatible NOT_TUMOR handling:
    # - new ontology: primary_class == NOT_TUMOR (group may be NOT_TUMOR)
    # - legacy concepts: primary_class null, group == Other
    if cls == "NOT_TUMOR":
        grp = _norm_str(row.get("group"))
        return (pc == "NOT_TUMOR") or ((pc is None) and (grp == "Other"))
    return (pc is not None) and (pc == cls)


def _is_group_match(row: pd.Series) -> bool:
    cls = _canon_class(row.get("class"))
    grp = _norm_str(row.get("group"))
    if cls is None or grp is None:
        return False
    if cls == "NOT_TUMOR":
        return grp in ("NOT_TUMOR", "Other")  # tolerate legacy
    return GROUP_BY_CLASS.get(cls) == grp


def _format_tex_table(df: pd.DataFrame, cols: List[str], caption: str, label: str) -> str:
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


def _fail_if_missing_rank(df: pd.DataFrame, rank_col: str, metrics_csv: Path) -> None:
    if rank_col not in df.columns:
        raise ValueError(f"rank_by='{rank_col}' not found in columns. Available={list(df.columns)}")
    nn = int(df[rank_col].notna().sum())
    if nn == 0:
        raise RuntimeError(
            f"[ERROR] rank_by='{rank_col}' has 0 non-NaN rows in {metrics_csv}. "
            "This usually means deep validation did not compute AUC/AP. "
            "Re-run deep validation with --compute-auc."
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("plip_build_shortlist")
    ensure_layout()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=(ANALYSIS_DIR / "metrics_per_class.csv"),
        help="Path to metrics_per_class.csv (default: canonical analysis/metrics_per_class.csv)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=SHORTLIST_DIR,
        help="Output directory for FINAL shortlist artifacts (default: canonical analysis/)",
    )
    ap.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR,
        help="Output directory for paper-ready report (default: canonical analysis/report/)",
    )
    ap.add_argument(
        "--rank-by",
        type=str,
        default="auc_ovr",
        help="Ranking column. If omitted: prefers auc_ovr then cohen_d then delta_mean.",
    )
    ap.add_argument("--k-primary", type=int, default=8, help="Top-K diagnostic concepts per class.")
    ap.add_argument("--k-confounds", type=int, default=5, help="Top-K confounder concepts per class.")
    ap.add_argument("--min-auc", type=float, default=0.60, help="Min auc_ovr to keep (if auc exists).")
    ap.add_argument("--min-ap", type=float, default=0.00, help="Min ap_ovr to keep (if ap exists).")
    ap.add_argument("--min-cohen-d", type=float, default=0.30, help="Min cohen_d to keep (if used).")
    ap.add_argument("--write-tex", action="store_true", help="Also emit LaTeX tables under paper_tables/")
    args = ap.parse_args()

    if not args.metrics_csv.exists():
        raise FileNotFoundError(f"metrics_per_class.csv not found: {args.metrics_csv}")

    df = pd.read_csv(args.metrics_csv)
    if "class" not in df.columns:
        raise ValueError(f"Expected column 'class' in {args.metrics_csv}")

    # Normalize
    df["primary_class"] = df["primary_class"].apply(_canon_class)
    df["group"] = df["group"].apply(_norm_str)
    df["class"] = df["class"].apply(_canon_class)

    # Ensure numeric rank columns (fix blank/str issues)
    for c in ["auc_ovr", "ap_ovr", "cohen_d", "delta_mean", "top1_freq", "topk_freq"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Derived flags
    df["is_primary_match"] = df.apply(_is_primary_match, axis=1)
    df["is_group_match"] = df.apply(_is_group_match, axis=1)

    preferred = ["auc_ovr", "cohen_d", "delta_mean", "topk_freq", "top1_freq"]
    rank_col = args.rank_by or _choose_rank_col(df, preferred)
    _fail_if_missing_rank(df, rank_col, args.metrics_csv)

    shortlist_dir = args.out_dir
    shortlist_dir.mkdir(parents=True, exist_ok=True)

    report_dir = args.report_dir if args.report_dir is not None else REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.write_tex:
        tex_dir_report = report_dir / "paper_tables"
        tex_dir_report.mkdir(parents=True, exist_ok=True)

    # Filtering thresholds (only if columns exist).
    # IMPORTANT FIX:
    #   Do not blindly apply cohen_d threshold when ranking by AUC/AP.
    #   That was a silent “kill switch” that can zero-out ONCO (and small classes).
    filt = df.copy()
    if rank_col == "auc_ovr" and "auc_ovr" in filt.columns:
        if "auc_valid" in filt.columns:
            filt = filt[filt["auc_valid"].fillna(0.0) > 0.0]
        filt = filt[~filt["auc_ovr"].isna()]
        filt = filt[filt["auc_ovr"] >= float(args.min_auc)]
        if float(args.min_ap) > 0.0 and "ap_ovr" in filt.columns:
            if "ap_valid" in filt.columns:
                filt = filt[filt["ap_valid"].fillna(0.0) > 0.0]
            filt = filt[~filt["ap_ovr"].isna()]
            filt = filt[filt["ap_ovr"] >= float(args.min_ap)]
    elif rank_col == "ap_ovr" and "ap_ovr" in filt.columns:
        if "ap_valid" in filt.columns:
            filt = filt[filt["ap_valid"].fillna(0.0) > 0.0]
        filt = filt[~filt["ap_ovr"].isna()]
        filt = filt[filt["ap_ovr"] >= float(args.min_ap)]
    elif rank_col == "cohen_d" and "cohen_d" in filt.columns:
        filt = filt[~filt["cohen_d"].isna()]
        filt = filt[filt["cohen_d"] >= float(args.min_cohen_d)]

    classes = sorted([c for c in df["class"].dropna().unique().tolist()])

    shortlist: Dict[str, Dict[str, List[str]]] = {}
    relaxed: Dict[str, List[str]] = {}
    report_lines: List[str] = []

    report_lines.append("# PLIP concept deep validation — paper-ready shortlist and audit\n")
    report_lines.append(f"- metrics: `{args.metrics_csv}`")
    report_lines.append(f"- rank_by: `{rank_col}`")
    report_lines.append(f"- k_primary={args.k_primary}, k_confounds={args.k_confounds}\n")
    report_lines.append(f"- filters: min_auc={args.min_auc}, min_ap={args.min_ap}, min_cohen_d={args.min_cohen_d}\n")
    report_lines.append(f"- shortlist_dir: `{shortlist_dir}`")
    report_lines.append(f"- report_dir: `{report_dir}`\n")

    for cls in classes:
        sub_raw = df[df["class"] == cls].copy()
        sub_filt = filt[filt["class"] == cls].copy()
        if sub_raw.empty:
            shortlist[cls] = {"primary": [], "confounds": []}
            report_lines.append(f"## {cls}\nNo rows for this class in metrics.\n")
            continue

        # Default: use filtered rows; if empty, fall back to raw (relax thresholds) but record it.
        sub_rank = sub_filt if not sub_filt.empty else sub_raw
        if sub_filt.empty:
            relaxed.setdefault(cls, []).append("dropped_threshold_filters_for_empty_class")

        sub_rank = sub_rank.sort_values(rank_col, ascending=False, na_position="last")

        primary = sub_rank[sub_rank["is_primary_match"]].head(args.k_primary)
        confounds = sub_rank[~sub_rank["is_primary_match"]].head(args.k_confounds)

        # If primary is still empty, this is almost always overly strict ranking/NaNs.
        # Backoff deterministically: rank primary matches by rank_col on raw; then by delta_mean if needed.
        if primary.empty and sub_raw["is_primary_match"].any():
            relaxed.setdefault(cls, []).append("primary_empty_after_filter_backoff_to_raw_rank")
            sub_raw_rank = sub_raw.sort_values(rank_col, ascending=False, na_position="last")
            primary = sub_raw_rank[sub_raw_rank["is_primary_match"]].head(args.k_primary)
            confounds = sub_raw_rank[~sub_raw_rank["is_primary_match"]].head(args.k_confounds)
            if primary.empty and "delta_mean" in sub_raw.columns:
                relaxed.setdefault(cls, []).append("primary_still_empty_backoff_to_delta_mean")
                sub_dm = sub_raw.sort_values("delta_mean", ascending=False, na_position="last")
                primary = sub_dm[sub_dm["is_primary_match"]].head(args.k_primary)
                confounds = sub_dm[~sub_dm["is_primary_match"]].head(args.k_confounds)

        shortlist[cls] = {
            "primary": primary["concept_short_name"].astype(str).tolist(),
            "confounds": confounds["concept_short_name"].astype(str).tolist(),
        }

        # Hard visibility for ONCO if filtering eliminates everything
        if cls == "ONCO" and (len(shortlist[cls]["primary"]) == 0):
            report_lines.append("**WARNING:** ONCO has 0 diagnostic concepts after filtering.\n")
            report_lines.append(
                "Check calibration/analysis/onco_audit/ and consider lowering min-auc/min-cohen-d or fixing ontology mapping.\n"
            )

        if cls in relaxed:
            report_lines.append(f"- **relaxed_filters**: `{relaxed[cls]}`\n")

        # Diagnostics for paper/report
        report_lines.append(f"## {cls}\n")
        raw_top10 = df[df["class"] == cls].sort_values(rank_col, ascending=False).head(10)
        pm = float(raw_top10["is_primary_match"].mean()) if len(raw_top10) else 0.0
        gm = float(raw_top10["is_group_match"].mean()) if len(raw_top10) else 0.0
        report_lines.append(f"- top10 primary_match ratio: **{pm:.2f}**")
        report_lines.append(f"- top10 group_match ratio: **{gm:.2f}** (sanity only)\n")

        cols = [
            c
            for c in [
                "concept_short_name",
                "concept_name",
                "group",
                "primary_class",
                rank_col,
                "cohen_d",
                "auc_ovr",
                "ap_ovr",
                "top1_freq",
                "topk_freq",
            ]
            if c in primary.columns
        ]

        report_lines.append("### Diagnostic concepts (primary_class match)\n")
        report_lines.append(primary[cols].to_markdown(index=False) if not primary.empty else "_None after filtering._")
        report_lines.append("\n### Confounders (high rank but primary_class mismatch)\n")
        report_lines.append(confounds[cols].to_markdown(index=False) if not confounds.empty else "_None after filtering._")
        report_lines.append("")

        if args.write_tex:
            tex_cols = [c for c in ["concept_short_name", "concept_name", rank_col, "cohen_d", "auc_ovr"] if c in primary.columns]
            tex = _format_tex_table(
                primary.head(min(args.k_primary, 8))[tex_cols],
                cols=tex_cols,
                caption=f"Top diagnostic concepts for {cls} (ranked by {rank_col}).",
                label=f"tab:concepts_{cls}",
            )
            # Write to report dir (paper-ready)
            (tex_dir_report / f"top_concepts_{cls}.tex").write_text(tex)

    # Canonical names for downstream
    out_json = shortlist_dir / "concepts_shortlist.json"
    out_yaml = shortlist_dir / "concepts_shortlist.yaml"
    payload = {
        "version": 1,
        "rank_by": rank_col,
        "k_primary": args.k_primary,
        "k_confounds": args.k_confounds,
        "classes": shortlist,
        "relaxed_filters": relaxed,
        "notes": [
            "primary concepts: primary_class matches class (NOT_TUMOR: primary_class==NOT_TUMOR OR legacy primary_class null AND group==Other)",
            "confounds: high-ranked concepts where primary_class mismatches the class (possible leakage / dataset bias / prompt ambiguity)",
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2))

    # YAML is the OFFICIAL shortlist (required by project layout).
    out_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))

    out_md = report_dir / "report.md"
    out_md.write_text("\n".join(report_lines) + "\n")

    flat_rows = []
    for cls, d in shortlist.items():
        for sn in d["primary"]:
            flat_rows.append({"class": cls, "kind": "primary", "concept_short_name": sn})
        for sn in d["confounds"]:
            flat_rows.append({"class": cls, "kind": "confound", "concept_short_name": sn})
    flat_csv = shortlist_dir / "concepts_shortlist_flat.csv"
    pd.DataFrame(flat_rows).to_csv(flat_csv, index=False)

    # Copy deep validation plots into report/figures (paper-ready).
    # metrics_csv is typically: <deep_validation>/metrics_per_class.csv
    deepval_dir = args.metrics_csv.parent
    plots_dir = deepval_dir / "plots"
    fig_dir = report_dir / "figures"
    if plots_dir.exists() and plots_dir.is_dir():
        fig_dir.mkdir(parents=True, exist_ok=True)
        for p in plots_dir.glob("*"):
            if p.is_file() and p.suffix.lower() in (".png", ".pdf"):
                shutil.copy2(p, fig_dir / p.name)

    log.info("Wrote shortlist JSON: %s", out_json)
    log.info("Wrote shortlist YAML (OFFICIAL): %s", out_yaml)
    log.info("Wrote report.md: %s", out_md)
    log.info("Wrote flat CSV: %s", flat_csv)
    if args.write_tex:
        log.info("Wrote LaTeX tables under: %s", tex_dir_report)


if __name__ == "__main__":
    main()
>>

calibration_validation.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

import copy
import argparse
import csv
import json
import math
import os
import logging
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# Optional deps (present in requirements_xai.txt in your stack)
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt

from explainability.utils.class_utils  import canon_class
from . import ensure_layout, METADATA_DIR, ANALYSIS_DIR

from explainability.utils.bootstrap import bootstrap_package

bootstrap_package(__file__, globals())


def _truthy(x: str) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")


def _is_constant(x: np.ndarray, eps: float = 1e-12) -> bool:
    return bool(np.nanmax(x) - np.nanmin(x) <= eps)


def _as_float(x: Any, default: float = float("nan")) -> float:
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


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _decode_np_str_array(x: np.ndarray) -> np.ndarray:
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


def _guess_class_names(cfg: Dict[str, Any], labels_raw: np.ndarray) -> Optional[List[str]]:
    # Best-effort: look for common patterns without assuming a schema.
    # If labels are strings already, prefer their sorted unique values.
    labels_raw = _decode_np_str_array(labels_raw)
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
    labels_raw = _decode_np_str_array(labels_raw)

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


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_heatmap(
    mat: np.ndarray,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    title: str,
    out_png: Path,
    out_pdf: Path,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    plt.figure(figsize=(max(10, 0.35 * len(xlabels)), max(4, 0.5 * len(ylabels))))
    plt.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(ylabels)), ylabels, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()


def _save_barh(
    values: np.ndarray,
    labels: Sequence[str],
    title: str,
    out_png: Path,
    out_pdf: Path,
    *,
    xlabel: str,
) -> None:
    order = np.argsort(values)
    vals = values[order]
    labs = [labels[i] for i in order]
    plt.figure(figsize=(10, max(4, 0.35 * len(labs))))
    plt.barh(np.arange(len(vals)), vals)
    plt.yticks(np.arange(len(vals)), labs, fontsize=9)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()


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
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    n = labels.shape[0]
    for k, cls in enumerate(class_names):
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
            if _is_constant(s):
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


def _resolve_calibration_dir(cal_run: Path) -> Path:
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


def _augment_selection_with_primary_concepts(
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
    keys = _decode_np_str_array(keys)

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
    _ensure_dir(ex_dir)

    for (k, j), buf in buffers.items():
        cls = class_names[k]
        c = concepts[j]
        buf_sorted = sorted(buf, key=lambda t: t[0], reverse=True)
        out_csv = ex_dir / f"top_{cls}__{c.short_name}.csv"
        with out_csv.open("w") as f:
            f.write("rank,score,wds_key,label\n")
            for r, (sc, idx) in enumerate(buf_sorted, start=1):
                f.write(f"{r},{sc:.6f},{keys[idx]},{cls}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep per-class validation for PLIP concept prompts (canonical layout).")
    ap.add_argument(
        "--cal-run",
        default=None,
        type=Path,
        help="(Backward compat) Calibration dir. Default: canonical calibration/metadata/.",
    )
    ap.add_argument(
        "--metadata-dir",
        default=None,
        type=Path,
        help="Canonical metadata dir (default: output/calibration/metadata).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="Canonical analysis dir (default: output/calibration/analysis).",
    )
    ap.add_argument("--topk", type=int, default=5, help="Top-k for topk_freq (default: 5)")
    ap.add_argument("--chunk-size", type=int, default=16384, help="Chunk size for memmap-friendly passes")
    ap.add_argument(
        "--compute-auc",
        dest="compute_auc",
        action="store_true",
        default=True,
        help="Compute AUC/AP (bounded) for selected concepts per class (default: ON).",
    )
    ap.add_argument(
        "--no-compute-auc",
        dest="compute_auc",
        action="store_false",
        help="Disable AUC/AP computation (debug only).",
    )
    ap.add_argument("--require-auc", action="store_true", help="Fail if AUC/AP are missing/invalid for most selected pairs.")
    ap.add_argument(
        "--auc-topm-per-class",
        type=int,
        default=25,
        help="Compute AUC/AP for union of top-M per class (delta/cohen_d/top1_freq). (default: 25)",
    )
    ap.add_argument("--plots-topn", type=int, default=12, help="Top-N concepts to plot per class (default: 12)")
    ap.add_argument("--max-exemplars", type=int, default=40, help="Top exemplars (keys) per (class, concept) (default: 40)")
    ap.add_argument("--quiet-tokenizers", action="store_true", help="Set TOKENIZERS_PARALLELISM=false inside the process")
    ap.add_argument("--allow-nonfinite", action="store_true", help="Allow non-finite scores (they will be replaced with 0.0). Default: fail-fast.")
    ap.add_argument("--require-onco", action="store_true", default=True, help="Fail if ONCO has 0 diagnostic concepts in top list (default: ON).")
    ap.add_argument("--no-require-onco", dest="require_onco", action="store_false", help="Disable ONCO gating.")
    ap.add_argument("--log-level", default="INFO", type=str, help="Logging level (DEBUG, INFO, WARNING).")
    args = ap.parse_args()

    if args.quiet_tokenizers:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("plip_deep_validation")

    ensure_layout()

    # Resolve metadata dir (canonical by default)
    if args.metadata_dir is not None:
        cal = args.metadata_dir
    elif args.cal_run is not None:
        cal = _resolve_calibration_dir(args.cal_run)
    else:
        cal = METADATA_DIR

    if not cal.is_dir():
        raise FileNotFoundError(f"cal-run not found: {cal}")

    cfg_path = cal / "config_resolved.yaml"
    concepts_path = cal / "concepts.json"
    scores_path = cal / "scores_fp32.npy"
    labels_path = cal / "labels.npy"
    keys_path = cal / "keys.npy"

    for p in [concepts_path, scores_path, labels_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}
    concepts = load_concepts(concepts_path)
    scores = np.load(scores_path, mmap_mode="r")  # (N,C)
    labels_raw = np.load(labels_path, allow_pickle=True)
    keys = np.load(keys_path, allow_pickle=True) if keys_path.exists() else None

    class_names_guess = _guess_class_names(cfg, labels_raw)
    labels, class_names = normalize_labels(labels_raw, class_names_guess)

    n = labels.shape[0]
    n_concepts = len(concepts)
    if scores.ndim != 2 or scores.shape[0] != n or scores.shape[1] != n_concepts:
        raise RuntimeError(f"Shape mismatch: scores {scores.shape} labels {labels.shape} concepts {n_concepts}")

    out_dir = args.out_dir if args.out_dir is not None else ANALYSIS_DIR
    _ensure_dir(out_dir)
    plot_dir = out_dir / "plots"
    _ensure_dir(plot_dir)

    # Pass 1: fast stats
    stats = compute_fast_stats(
        scores,
        labels,
        len(class_names),
        args.topk,
        args.chunk_size,
        allow_nonfinite=bool(args.allow_nonfinite),
        log=log,
    )
    rows, mats = build_metrics_tables(concepts, class_names, scores, labels, stats)

    # Selection for bounded AUC/AP
    selected = build_selection_union(mats, topm_per_metric=args.auc_topm_per_class)
    # Critical: guarantee AUC/AP for *all* primary_class-matching concepts (prevents empty ONCO shortlist).
    selected = _augment_selection_with_primary_concepts(concepts, class_names, selected)
    auc_ap: Dict[Tuple[int, int], Dict[str, float]] = {}
    if args.compute_auc:
        auc_ap = compute_auc_ap_for_selected(scores, labels, class_names, selected)

    if args.compute_auc:
        # Per-class validity logging (how many selected pairs have finite AUC/AP).
        for k, cls in enumerate(class_names):
            js = selected.get(k, [])
            if not js:
                continue
            n_sel = len(js)
            n_auc = 0
            n_ap = 0
            for j in js:
                m = auc_ap.get((k, j))
                if not m:
                    continue
                if math.isfinite(float(m.get("auc_ovr", float("nan")))):
                    n_auc += 1
                if math.isfinite(float(m.get("ap_ovr", float("nan")))):
                    n_ap += 1
            log.info("[AUC/AP] class=%s selected=%d valid_auc=%d valid_ap=%d", cls, n_sel, n_auc, n_ap)
            if args.require_auc and (n_auc < max(1, int(0.5 * n_sel))):
                raise RuntimeError(f"[ERROR] Too few valid AUC values for class={cls}: {n_auc}/{n_sel}.")

    # Merge auc/ap into rows
    cls_to_k = {c: i for i, c in enumerate(class_names)}
    for r in rows:
        k = cls_to_k[str(r["class"])]
        j = int(r["concept_idx"])
        m = auc_ap.get((k, j))
        r["auc_ovr"] = (float("nan") if m is None else _as_float(m.get("auc_ovr", float("nan"))))
        r["ap_ovr"] = (float("nan") if m is None else _as_float(m.get("ap_ovr", float("nan"))))
        r["auc_valid"] = (0.0 if m is None else _as_float(m.get("auc_valid", 1.0 if math.isfinite(r["auc_ovr"]) else 0.0), 0.0))
        r["ap_valid"] = (0.0 if m is None else _as_float(m.get("ap_valid", 1.0 if math.isfinite(r["ap_ovr"]) else 0.0), 0.0))
        r["auc_ap_reason"] = (float("nan") if m is None else _as_float(m.get("reason", float("nan"))))

    # Write metrics CSV
    metrics_csv = out_dir / "metrics_per_class.csv"
    cols = [
        "class",
        "concept_idx",
        "concept_short_name",
        "concept_name",
        "group",
        "primary_class",
        "n_pos",
        "mean_pos",
        "std_pos",
        "mean_rest",
        "std_rest",
        "delta_mean",
        "cohen_d",
        "top1_freq",
        "topk_freq",
        "auc_ovr",
        "ap_ovr",
        "auc_valid",
        "ap_valid",
        "auc_ap_reason",
    ]
    with metrics_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # Summary ranking per class (paper-ready)
    topn = max(3, int(args.plots_topn))
    summary: Dict[str, Any] = {"class_names": class_names, "top_by_class": {}}
    for k, cls in enumerate(class_names):
        # rank by delta_mean (primary), then top1_freq
        order = np.argsort(mats["delta"][k])[::-1]
        top = []
        for j in order[:topn]:
            c = concepts[j]
            m = auc_ap.get((k, j), {})
            top.append(
                {
                    "concept_short_name": c.short_name,
                    "concept_name": c.name,
                    "group": c.group,
                    "primary_class": c.primary_class,
                    "delta_mean": float(mats["delta"][k, j]),
                    "cohen_d": float(mats["cohen_d"][k, j]),
                    "top1_freq": float(mats["top1_freq"][k, j]),
                    "topk_freq": float(mats["topk_freq"][k, j]),
                    "auc_ovr": float(m.get("auc_ovr", float("nan"))),
                    "ap_ovr": float(m.get("ap_ovr", float("nan"))),
                }
            )
        summary["top_by_class"][cls] = top

    (out_dir / "top_concepts_by_class.json").write_text(json.dumps(summary, indent=2))

    # ONCO audit & gating (fail clearly if ONCO has no diagnostic concepts).
    # Diagnostic concept: concept.primary_class matches class (canonicalized).
    onco_name = "ONCO"
    if args.require_onco and onco_name in class_names:
        k_onco = class_names.index(onco_name)
        # Prefer AUC if computed, else delta
        rank = np.asarray([r["auc_ovr"] for r in rows if r["class"] == onco_name], dtype=np.float64)
        use_auc = bool(args.compute_auc) and np.isfinite(rank).any()
        if use_auc:
            # build list of (concept_idx, auc)
            items = []
            for r in rows:
                if r["class"] != onco_name:
                    continue
                items.append((int(r["concept_idx"]), float(r["auc_ovr"])))
            items = sorted(items, key=lambda t: (float("-inf") if math.isnan(t[1]) else t[1]), reverse=True)
            top_idx = [j for j, _ in items[:20]]
        else:
            top_idx = np.argsort(mats["delta"][k_onco])[::-1][:20].tolist()

        diag = 0
        audit_rows = []
        for j in top_idx:
            c = concepts[j]
            pc = canon_class(c.primary_class) if c.primary_class is not None else None
            if pc == onco_name:
                diag += 1
            audit_rows.append(
                {
                    "concept_short_name": c.short_name,
                    "concept_name": c.name,
                    "group": c.group,
                    "primary_class": c.primary_class,
                    "delta_mean": float(mats["delta"][k_onco, j]),
                    "cohen_d": float(mats["cohen_d"][k_onco, j]),
                    "top1_freq": float(mats["top1_freq"][k_onco, j]),
                    "auc_ovr": float(auc_ap.get((k_onco, j), {}).get("auc_ovr", float("nan"))),
                    "ap_ovr": float(auc_ap.get((k_onco, j), {}).get("ap_ovr", float("nan"))),
                }
            )

        onco_audit_dir = out_dir / "onco_audit"
        _ensure_dir(onco_audit_dir)
        (onco_audit_dir / "onco_top_concepts.json").write_text(json.dumps(audit_rows, indent=2))
        with (onco_audit_dir / "onco_top_concepts.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()) if audit_rows else ["concept_short_name"])
            w.writeheader()
            for rr in audit_rows:
                w.writerow(rr)

        if diag == 0:
            raise RuntimeError(
                "[ERROR] ONCO gating failed: 0 diagnostic concepts found in top ONCO list. "
                f"See audit: {onco_audit_dir}"
            )

    # Plots (heatmaps)
    xlabels = [c.short_name for c in concepts]
    ylabels = class_names
    _save_heatmap(
        mats["means"],
        xlabels,
        ylabels,
        "Mean score (class x concept)",
        plot_dir / "heatmap_mean_score.png",
        plot_dir / "heatmap_mean_score.pdf",
    )
    _save_heatmap(
        mats["delta"],
        xlabels,
        ylabels,
        "Delta mean vs rest (class x concept)",
        plot_dir / "heatmap_delta_mean.png",
        plot_dir / "heatmap_delta_mean.pdf",
    )
    _save_heatmap(
        mats["top1_freq"],
        xlabels,
        ylabels,
        "Top-1 freq (class x concept)",
        plot_dir / "heatmap_top1_freq.png",
        plot_dir / "heatmap_top1_freq.pdf",
        vmin=0.0,
        vmax=1.0,
    )
    _save_heatmap(
        mats["topk_freq"],
        xlabels,
        ylabels,
        f"Top-{args.topk} freq (class x concept)",
        plot_dir / f"heatmap_top{args.topk}_freq.png",
        plot_dir / f"heatmap_top{args.topk}_freq.pdf",
        vmin=0.0,
        vmax=1.0,
    )

    # Per-class bar plots (top delta and top auc if available)
    for k, cls in enumerate(class_names):
        order = np.argsort(mats["delta"][k])[::-1][:topn]
        vals = mats["delta"][k, order]
        labs = [concepts[j].short_name for j in order]
        _save_barh(
            vals,
            labs,
            f"{cls}: top-{topn} concepts by delta(mean)",
            plot_dir / f"bar_{cls}_top_delta.png",
            plot_dir / f"bar_{cls}_top_delta.pdf",
            xlabel="delta(mean_pos - mean_rest)",
        )

        if args.compute_auc:
            auc_vals = []
            auc_labs = []
            for j in selected.get(k, []):
                m = auc_ap.get((k, j))
                if m is None:
                    continue
                v = float(m.get("auc_ovr", float("nan")))
                if not math.isnan(v):
                    auc_vals.append(v)
                    auc_labs.append(concepts[j].short_name)
            if auc_vals:
                auc_vals_np = np.asarray(auc_vals, dtype=np.float64)
                ord2 = np.argsort(auc_vals_np)[::-1][:topn]
                _save_barh(
                    auc_vals_np[ord2],
                    [auc_labs[i] for i in ord2],
                    f"{cls}: top-{min(topn, len(ord2))} concepts by AUC(OVR)",
                    plot_dir / f"bar_{cls}_top_auc.png",
                    plot_dir / f"bar_{cls}_top_auc.pdf",
                    xlabel="AUC one-vs-rest",
                )

    # Exemplars (keys)
    write_exemplars(
        out_dir=out_dir,
        scores=scores,
        labels=labels,
        keys=keys,
        class_names=class_names,
        concepts=concepts,
        selected=build_selection_from_delta(mats["delta"], min(10, args.auc_topm_per_class)),
        max_exemplars=args.max_exemplars,
        chunk_size=args.chunk_size,
    )

    # Optional: quick ROC/PR curves for top-5 delta concepts per class (cheap enough)
    for k, cls in enumerate(class_names):
        y = (labels == k).astype(np.int32)
        n_pos = int(y.sum())
        n_neg = int(y.shape[0] - n_pos)
        if n_pos < 2 or n_neg < 2:
            continue
        order = np.argsort(mats["delta"][k])[::-1][: min(5, n_concepts)]
        # ROC
        plt.figure(figsize=(6, 5))
        for j in order:
            s = np.asarray(scores[:, j], dtype=np.float32)
            fpr, tpr, _ = roc_curve(y, s)
            plt.plot(fpr, tpr, label=concepts[j].short_name)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"{cls}: ROC curves (top delta concepts)")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / f"roc_{cls}_top_delta.png", dpi=200)
        plt.savefig(plot_dir / f"roc_{cls}_top_delta.pdf")
        plt.close()

        # PR
        plt.figure(figsize=(6, 5))
        for j in order:
            s = np.asarray(scores[:, j], dtype=np.float32)
            p, r, _ = precision_recall_curve(y, s)
            plt.plot(r, p, label=concepts[j].short_name)
        plt.title(f"{cls}: PR curves (top delta concepts)")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / f"pr_{cls}_top_delta.png", dpi=200)
        plt.savefig(plot_dir / f"pr_{cls}_top_delta.pdf")
        plt.close()

    log.info("Wrote deep validation (canonical): %s", out_dir)
    log.info("  - metrics: %s", metrics_csv)
    log.info("  - plots:   %s", plot_dir)


if __name__ == "__main__":
    main()
>>

__init__.py codice <<
"""
Calibration + deep validation for PLIP concept prompts.

Fixed layout (no runs/):
  - output/calibration/metadata/: unified calibration artifacts (TRAIN+VAL)
  - output/calibration/analysis/: deep validation outputs + audits
  - output/calibration/analysis/report/: paper-ready report (md/tables/figures)
  - output/calibration/analysis/concepts_shortlist.yaml: final shortlist for test
"""

from typing import Optional

from ...paths import CALIBRATION_PATHS, CalibrationLayout, ensure_calibration_layout

# Canonical layout (pulled directly from explainability.paths)
METADATA_DIR = CALIBRATION_PATHS.metadata_dir
ANALYSIS_DIR = CALIBRATION_PATHS.analysis_dir
REPORT_DIR = CALIBRATION_PATHS.report_dir
SHORTLIST_DIR = CALIBRATION_PATHS.shortlist_dir
SHORTLIST_JSON = CALIBRATION_PATHS.shortlist_json
SHORTLIST_YAML = CALIBRATION_PATHS.shortlist_yaml


def ensure_layout(layout: Optional[CalibrationLayout] = None) -> CalibrationLayout:
    """
    Ensure canonical calibration directories exist and return the resolved layout.
    """
    return ensure_calibration_layout(layout or CALIBRATION_PATHS)


__all__ = [
    "ANALYSIS_DIR",
    "SHORTLIST_DIR",
    "METADATA_DIR",
    "REPORT_DIR",
    "SHORTLIST_JSON",
    "SHORTLIST_YAML",
    "CALIBRATION_PATHS",
    "ensure_layout",
]
>>

plots.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, out_base: Path, formats: Sequence[str], dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")


def plot_heatmap(
    mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    out_base: Path,
    title: str,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
) -> None:
    fig = plt.figure(figsize=(max(8, 0.35 * len(col_labels)), max(4, 0.35 * len(row_labels))))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, out_base, formats, dpi)
    plt.close(fig)


def plot_bar(
    labels: List[str],
    values: np.ndarray,
    out_base: Path,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
    rotate: int = 90,
) -> None:
    fig = plt.figure(figsize=(max(8, 0.35 * len(labels)), 4.5))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(labels)), values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right")
    fig.tight_layout()
    _save(fig, out_base, formats, dpi)
    plt.close(fig)
>>

run_calibration.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import shutil
import yaml

import torch

from explainability.utils.bootstrap import bootstrap_package

bootstrap_package(__file__, globals())

from ..plip.ontology_io import load_ontology, concepts_to_prompt_lists, concepts_to_dicts
from ..plip.plip_model import load_plip, encode_text, encode_images, score
from ..plip.wds_loader import build_wds_loader
from .plots import plot_heatmap, plot_bar
from . import ensure_layout, METADATA_DIR
from ...paths import resolve_config, CALIBRATION_CONFIG_YAML


def _truthy(x: str) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")


def _clean_metadata_dir(d: Path, log: logging.Logger) -> None:
    """
    Ensure canonical metadata dir does not keep stale artifacts from previous calibrations.
    Removes only known artifacts produced by this script.
    """
    known_files = [
        "concepts.json",
        "config_resolved.yaml",
        "keys.npy",
        "labels.npy",
        "scores_fp32.npy",
        "text_features.pt",
        "auc_primary_class.csv",
        "split_stats.json",
    ]
    for fn in known_files:
        p = d / fn
        if p.exists():
            p.unlink()
    for dn in ["plots", "cache"]:
        p = d / dn
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    log.info("Cleaned metadata dir (known artifacts only): %s", d)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Calibrate PLIP concepts on TRAIN and VAL separately (WebDataset stays separate), "
            "then unify the OUTPUT under the fixed canonical metadata dir."
        )
    )
    p.add_argument(
        "--config",
        required=False,
        type=Path,
        default=CALIBRATION_CONFIG_YAML,
        help="YAML config path (default: central configs/calibration.yaml).",
    )
    p.add_argument("--log-level", default="INFO", type=str, help="Logging level (DEBUG, INFO, WARNING).")
    return p.parse_args()


def _require(d: Dict[str, Any], path: str) -> Any:
    cur = d
    for part in path.split("."):
        if part not in cur:
            raise KeyError(f"Missing config key: {path}")
        cur = cur[part]
    return cur


def _resolve_split_dir(
    wds_cfg: Dict[str, Any], *, cfg_key: str, env_keys: List[str], fallback_cfg_key: Optional[str] = None
) -> str:
    """
    Resolve a split directory without ever concatenating/merging WebDataset shards.
    Priority:
      1) data.webdataset.<cfg_key>
      2) env vars in env_keys
      3) optional fallback_cfg_key (for backward compat, e.g. split_dir)
    """
    v = str(wds_cfg.get(cfg_key) or "").strip()
    if v:
        return v
    for k in env_keys:
        ev = os.getenv(k, "").strip()
        if ev:
            return ev
    if fallback_cfg_key:
        v2 = str(wds_cfg.get(fallback_cfg_key) or "").strip()
        if v2:
            return v2
    return ""


def _aggregate_text_features_from_prompt_lists(
    plip,
    prompt_lists: List[List[str]],
) -> Tuple[torch.Tensor, List[str]]:
    """
    Encode all prompt variants and aggregate to one embedding per concept.
    Returns:
      - text_features: [n_concepts, dim] (L2-normalized)
      - flat_prompts: flattened prompt list (for provenance/logging)
    """
    flat_prompts: List[str] = []
    ranges: List[Tuple[int, int]] = []
    for ps in prompt_lists:
        start = len(flat_prompts)
        flat_prompts.extend([str(p) for p in ps])
        end = len(flat_prompts)
        ranges.append((start, end))

    if not flat_prompts:
        raise RuntimeError("No prompts found after flattening ontology prompt lists.")

    feats = encode_text(plip, flat_prompts)  # [M,D], already normalized per-prompt
    agg: List[torch.Tensor] = []
    for (a, b) in ranges:
        if b <= a:
            raise RuntimeError("Empty prompt list for a concept (ontology parsing bug).")
        v = feats[a:b].mean(dim=0)
        v = v / (v.norm(dim=-1, keepdim=False) + 1e-8)
        agg.append(v)

    text_feats = torch.stack(agg, dim=0)
    return text_feats, flat_prompts


def _score_one_split(
    *,
    split_name: str,
    split_dir: Path,
    plip: Any,
    text_feats: torch.Tensor,
    wds_cfg: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    max_patches: int,
    class_field: str,
    log: logging.Logger,
) -> Tuple[List[str], List[str], List[np.ndarray], Dict[str, Any]]:
    """
    Score PLIP on a single split. Returns:
      keys, labels, list_of_score_chunks, split_stats
    """
    pattern = str(wds_cfg.get("pattern", "shard-*.tar"))
    image_key = str(wds_cfg.get("image_key", "img.jpg;jpg;jpeg;png"))
    meta_key = str(wds_cfg.get("meta_key", "meta.json;json"))

    loader = build_wds_loader(
        split_dir=split_dir,
        pattern=pattern,
        image_key=image_key,
        meta_key=meta_key,
        preprocess=plip.preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    keys: List[str] = []
    labels: List[str] = []
    chunks: List[np.ndarray] = []
    n_seen = 0
    n_nonfinite = 0

    for batch in loader:
        if batch is None:
            continue
        imgs, metas, bkeys, _ = batch
        if imgs is None:
            continue

        img_feats = encode_images(plip, imgs)
        logits = score(plip, img_feats, text_feats)  # [B,C]
        logits_np = logits.detach().cpu().float().numpy()

        if not np.isfinite(logits_np).all():
            bad = int((~np.isfinite(logits_np)).sum())
            n_nonfinite += bad
            logits_np = np.nan_to_num(logits_np, nan=0.0, posinf=0.0, neginf=0.0)
            log.warning("[%s] Found %d non-finite logits; replaced with 0.0", split_name, bad)

        for m, k in zip(metas, bkeys):
            # Prefix keys so TRAIN/VAL provenance is preserved without extra files
            keys.append(f"{split_name}::{str(k)}")
            labels.append(str(m.get(class_field, "UNKNOWN")))

        chunks.append(logits_np.astype(np.float32, copy=False))
        n_seen += int(logits_np.shape[0])

        if max_patches > 0 and n_seen >= max_patches:
            break

    stats = {
        "split": split_name,
        "split_dir": str(split_dir),
        "n_samples": int(n_seen),
        "n_nonfinite_replaced": int(n_nonfinite),
    }
    return keys, labels, chunks, stats


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("plip_calibration")

    layout = ensure_layout()
    out_dir = METADATA_DIR

    cfg_path = resolve_config(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Calibration config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())

    # Enforce canonical (fixed) layout (no runs/, no timestamps).
    # Any experiment.outputs_root is ignored on purpose.
    _clean_metadata_dir(out_dir, log)

    # Ontology
    ontology_yaml_cfg = cfg.get("concepts", {}).get("ontology_yaml")
    ontology_yaml_env = os.environ.get("ONTOLOGY_YAML") or os.environ.get("CALIB_ONTOLOGY_YAML")
    if not (ontology_yaml_cfg or ontology_yaml_env):
        raise RuntimeError(
            "Missing concepts.ontology_yaml. Set it in the config or export ONTOLOGY_YAML/CALIB_ONTOLOGY_YAML."
        )
    ontology_yaml = Path(ontology_yaml_cfg or ontology_yaml_env)
    ont_meta, concepts = load_ontology(ontology_yaml)
    prompt_lists = concepts_to_prompt_lists(concepts)

    (out_dir / "concepts.json").write_text(
        json.dumps({"meta": ont_meta, "concepts": concepts_to_dicts(concepts)}, indent=2)
    )

    # PLIP
    plip_cfg = cfg.get("plip", {})
    plip = load_plip(
        model_id=str(plip_cfg.get("model_id", "vinid/plip")),
        model_local_dir=plip_cfg.get("model_local_dir", None),
        device=str(plip_cfg.get("device", "cuda")),
        precision=str(plip_cfg.get("precision", "fp16")),
        score_scale=float(cfg.get("concepts", {}).get("score_scale", 100.0)),
        hf_cache_dir=plip_cfg.get("hf_cache_dir", None),
    )

    # Text features
    text_feats, flat_prompts = _aggregate_text_features_from_prompt_lists(plip, prompt_lists)
    torch.save(
        {
            "text_features": text_feats.detach().cpu(),
            "concepts": concepts_to_dicts(concepts),
            "flat_prompts": flat_prompts,
            "model_id": plip.model_id,
            "model_path": plip.model_path,
            "max_text_len": plip.max_text_len,
            "score_scale": plip.score_scale,
        },
        out_dir / "text_features.pt",
    )

    # Data
    data = cfg.get("data", {})
    wds = data.get("webdataset", {})
    train_dir_str = _resolve_split_dir(
        wds,
        cfg_key="train_dir",
        env_keys=["WDS_CALIB_TRAIN_DIR", "WDS_TRAIN_DIR"],
        fallback_cfg_key="split_dir",  # backward compat: old configs only had split_dir
    )
    val_dir_str = _resolve_split_dir(
        wds,
        cfg_key="val_dir",
        env_keys=["WDS_CALIB_VAL_DIR", "WDS_VAL_DIR"],
        fallback_cfg_key=None,
    )

    if not train_dir_str:
        raise RuntimeError(
            "Missing TRAIN WebDataset dir. Set data.webdataset.train_dir (preferred) "
            "or export WDS_TRAIN_DIR / WDS_CALIB_TRAIN_DIR. "
            "NOTE: TRAIN+VAL calibration does NOT mean merging shards; splits remain separate."
        )

    train_dir = Path(train_dir_str)
    if not train_dir.exists():
        raise FileNotFoundError(f"TRAIN WebDataset dir not found: {train_dir}")

    val_dir: Optional[Path] = None
    if val_dir_str:
        val_dir = Path(val_dir_str)
        if not val_dir.exists():
            raise FileNotFoundError(f"VAL WebDataset dir not found: {val_dir}")

    class_field = str(wds.get("class_field", "class_label"))

    # Save resolved config (for reproducibility) AFTER resolving split dirs
    cfg_resolved = dict(cfg)
    cfg_resolved.setdefault("data", {})
    cfg_resolved["data"].setdefault("webdataset", {})
    cfg_resolved["data"]["webdataset"]["train_dir"] = str(train_dir)
    if val_dir is not None:
        cfg_resolved["data"]["webdataset"]["val_dir"] = str(val_dir)
    # Explicitly mark canonical layout
    cfg_resolved.setdefault("experiment", {})
    cfg_resolved["experiment"]["outputs_root"] = str(out_dir)
    cfg_resolved["experiment"]["use_runs"] = False
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg_resolved, sort_keys=False))

    batch_size = int(data.get("batch_size", 256))
    num_workers = int(data.get("num_workers", 8))
    max_patches = int(data.get("max_patches", 0))

    # Score TRAIN and VAL separately, then unify OUTPUT only.
    keys_all: List[str] = []
    labels_all: List[str] = []
    score_chunks_all: List[np.ndarray] = []
    split_stats: List[Dict[str, Any]] = []

    k_tr, y_tr, chunks_tr, st_tr = _score_one_split(
        split_name="train",
        split_dir=train_dir,
        plip=plip,
        text_feats=text_feats,
        wds_cfg=wds,
        batch_size=batch_size,
        num_workers=num_workers,
        max_patches=max_patches,
        class_field=class_field,
        log=log,
    )
    keys_all.extend(k_tr)
    labels_all.extend(y_tr)
    score_chunks_all.extend(chunks_tr)
    split_stats.append(st_tr)

    if val_dir is not None:
        k_va, y_va, chunks_va, st_va = _score_one_split(
            split_name="val",
            split_dir=val_dir,
            plip=plip,
            text_feats=text_feats,
            wds_cfg=wds,
            batch_size=batch_size,
            num_workers=num_workers,
            max_patches=max_patches,
            class_field=class_field,
            log=log,
        )
        keys_all.extend(k_va)
        labels_all.extend(y_va)
        score_chunks_all.extend(chunks_va)
        split_stats.append(st_va)

    scores_mat = (
        np.concatenate(score_chunks_all, axis=0) if score_chunks_all else np.zeros((0, len(concepts)), dtype=np.float32)
    )
    labels_arr = np.array(labels_all, dtype=object)
    keys_arr = np.array(keys_all, dtype=object)

    out_cfg = cfg.get("output", {})
    save_arrays = bool(out_cfg.get("save_arrays", True))
    if save_arrays:
        np.save(out_dir / "scores_fp32.npy", scores_mat.astype(np.float32))
        np.save(out_dir / "labels.npy", labels_arr)
        np.save(out_dir / "keys.npy", keys_arr)

    (out_dir / "split_stats.json").write_text(json.dumps({"splits": split_stats}, indent=2))

    # Aggregations for plots
    classes = sorted(list(set(labels_arr.tolist())))
    concept_short = [c.short_name for c in concepts]

    # Mean score per class
    mean_by_class = np.zeros((len(classes), len(concepts)), dtype=np.float32)
    for i, cls in enumerate(classes):
        idx = np.where(labels_arr == cls)[0]
        if idx.size > 0:
            mean_by_class[i] = scores_mat[idx].mean(axis=0)

    # Top-1 frequency per class
    top1 = scores_mat.argmax(axis=1) if scores_mat.shape[0] else np.zeros((0,), dtype=int)
    top1_freq = np.zeros((len(classes), len(concepts)), dtype=np.float32)
    for i, cls in enumerate(classes):
        idx = np.where(labels_arr == cls)[0]
        if idx.size == 0:
            continue
        counts = np.bincount(top1[idx], minlength=len(concepts)).astype(np.float32)
        top1_freq[i] = counts / max(1.0, counts.sum())

    # AUC per concept vs its primary_class (if defined)
    try:
        from sklearn.metrics import roc_auc_score
        auc_rows = []
        for j, c in enumerate(concepts):
            if not c.primary_class:
                continue
            y = (labels_arr == c.primary_class).astype(np.int32)
            # need both classes present
            if y.min() == y.max():
                continue
            s = np.asarray(scores_mat[:, j], dtype=np.float32)
            if not np.isfinite(s).all():
                s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            # If constant scores, AUC is 0.5 (uninformative) but numeric
            if float(np.nanmax(s) - np.nanmin(s)) <= 1e-12:
                auc = 0.5
            else:
                auc = float(roc_auc_score(y, s))
            auc_rows.append((c.short_name, c.primary_class, auc))
        auc_rows = sorted(auc_rows, key=lambda x: x[2], reverse=True)
        (out_dir / "auc_primary_class.csv").write_text(
            "concept_short_name,primary_class,auc\n"
            + "\n".join([f"{a},{b},{c:.6f}" for a, b, c in auc_rows])
            + "\n"
        )
    except Exception as e:
        log.warning("Failed to compute auc_primary_class.csv (%s). Writing empty CSV.", str(e))
        (out_dir / "auc_primary_class.csv").write_text("concept_short_name,primary_class,auc\n")

    # Plots
    if bool(out_cfg.get("plots", True)):
        dpi = int(out_cfg.get("plots_dpi", 300))
        formats = out_cfg.get("formats", ["pdf", "png"])
        formats = tuple(formats)

        plot_heatmap(
            mean_by_class,
            row_labels=classes,
            col_labels=concept_short,
            out_base=out_dir / "plots" / "heatmap_mean_score_class_x_concept",
            title="PLIP mean concept score by class (TRAIN+VAL unified output)",
            formats=formats,
            dpi=dpi,
        )
        plot_heatmap(
            top1_freq,
            row_labels=classes,
            col_labels=concept_short,
            out_base=out_dir / "plots" / "heatmap_top1_freq_class_x_concept",
            title="PLIP top-1 concept frequency by class (TRAIN+VAL unified output)",
            formats=formats,
            dpi=dpi,
        )

        # If AUC computed, plot bar for AUC
        try:
            import csv
            auc_path = out_dir / "auc_primary_class.csv"
            rows = []
            with auc_path.open() as f:
                r = csv.DictReader(f)
                for row in r:
                    if not row.get("concept_short_name"):
                        continue
                    rows.append((row["concept_short_name"], float(row["auc"])))
            if rows:
                labels_auc = [x[0] for x in rows]
                vals_auc = np.array([x[1] for x in rows], dtype=np.float32)
                plot_bar(
                    labels=labels_auc,
                    values=vals_auc,
                    out_base=out_dir / "plots" / "bar_auc_primary_class",
                    title="AUC of concept score vs its primary class (TRAIN+VAL unified output)",
                    ylabel="ROC-AUC",
                    formats=formats,
                    dpi=dpi,
                    rotate=90,
                )
        except Exception:
            pass

    log.info("Calibration done (canonical): %s", out_dir)
    log.info("  - concepts.json, text_features.pt, scores_fp32.npy, labels.npy, keys.npy, auc_primary_class.csv")
    log.info("  - plots/ and split_stats.json")


if __name__ == "__main__":
    main()
>>

utils.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import logging
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from explainability.utils.class_utils import canon_class


# ---------------------------------------------------------------------
# Generic IO helpers
# ---------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


# ---------------------------------------------------------------------
# Concept + label helpers
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Concept:
    idx: int
    id: Optional[int]
    short_name: str
    name: str
    group: Optional[str]
    primary_class: Optional[str]
    prompt: Optional[str]


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
    raw = load_json(concepts_json)
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


# ---------------------------------------------------------------------
# Simple numeric helpers
# ---------------------------------------------------------------------

def is_constant(x: np.ndarray, eps: float = 1e-12) -> bool:
    return bool(np.nanmax(x) - np.nanmin(x) <= eps)


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def chunk_slices(n: int, chunk: int) -> Iterable[Tuple[int, int]]:
    i = 0
    while i < n:
        j = min(n, i + chunk)
        yield i, j
        i = j


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_figure(fig, out_base: Path, formats: Sequence[str], dpi: int) -> None:
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
    plt = _import_matplotlib()
    fig = plt.figure(figsize=(max(8, 0.35 * len(col_labels)), max(4, 0.45 * len(row_labels))))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure(fig, out_base, formats, dpi)
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
    plt = _import_matplotlib()
    fig = plt.figure(figsize=(max(8, 0.35 * len(labels)), 4.5))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(labels)), values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right")
    fig.tight_layout()
    save_figure(fig, out_base, formats, dpi)
    plt.close(fig)


def plot_barh(
    values: np.ndarray,
    labels: Sequence[str],
    out_base: Path,
    title: str,
    *,
    xlabel: str,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
) -> None:
    plt = _import_matplotlib()
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
    save_figure(fig, out_base, formats, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------
# Metrics + selection helpers
# ---------------------------------------------------------------------

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

    for a, b in chunk_slices(n, chunk_size):
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
    means = safe_div(sums, counts[:, None])
    ex2 = safe_div(sums_sq, counts[:, None])
    var = np.maximum(ex2 - means ** 2, 0.0)
    std = np.sqrt(var)

    # rest-of-classes stats (for delta and Cohen's d)
    total_count = float(np.sum(counts))
    total_sum = np.sum(sums, axis=0)
    total_sum_sq = np.sum(sums_sq, axis=0)

    rest_count = (total_count - counts)  # (K,)
    rest_sum = (total_sum[None, :] - sums)  # (K,C)
    rest_sum_sq = (total_sum_sq[None, :] - sums_sq)
    rest_mean = safe_div(rest_sum, rest_count[:, None])
    rest_ex2 = safe_div(rest_sum_sq, rest_count[:, None])
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
    top1_freq = safe_div(stats["top1_counts"].astype(np.float64), counts[:, None])
    # topk_counts counts occurrences across patches; each patch contributes 0/1 per concept.
    topk_freq = safe_div(stats["topk_counts"].astype(np.float64), counts[:, None])

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
    from sklearn.metrics import average_precision_score, roc_auc_score

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

    n, _n_concepts = scores.shape
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

    for a, b in chunk_slices(n, chunk_size):
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
# Report + LaTeX helpers
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


# ---------------------------------------------------------------------
# Artifact validation helpers
# ---------------------------------------------------------------------

def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if v < 1024.0:
            return f"{v:.1f}{u}"
        v /= 1024.0
    return f"{v:.1f}PB"


def safe_read_csv(path: Path) -> pd.DataFrame:
    # robust against empty or weird encodings
    return pd.read_csv(path)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")


def has_any_parent_named(p: Path, root: Path, names: Tuple[str, ...]) -> bool:
    try:
        rel = p.relative_to(root)
    except Exception:
        rel = p
    parts = [x.lower() for x in rel.parts]
    return any(n.lower() in parts for n in names)


def find_forbidden_images(root: Path, *, allow_dirs: Tuple[str, ...]) -> List[Path]:
    bad: List[Path] = []
    if not root.exists() or not root.is_dir():
        return bad
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if not is_image_file(p):
            continue
        if has_any_parent_named(p, root, allow_dirs):
            continue
        bad.append(p)
        if len(bad) >= 50:
            break
    return bad


def is_nonempty_file(path: Path, min_bytes: int = 32) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


def collect_leaf_strings(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            out.append(s)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(collect_leaf_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(collect_leaf_strings(v))
    return out


def finite_sample_ok(arr: np.ndarray, max_rows: int = 2048, max_cols: int = 2048) -> Tuple[bool, str]:
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


def require_exists(path: Path, errors: List[str], what: str) -> None:
    if not path.exists():
        errors.append(f"[MISSING] {what}: {path}")


def require_dir(path: Path, errors: List[str], what: str) -> None:
    if not path.exists() or not path.is_dir():
        errors.append(f"[MISSING] {what} dir: {path}")


def require_nonempty_file(path: Path, errors: List[str], what: str, min_bytes: int = 32) -> None:
    if not is_nonempty_file(path, min_bytes=min_bytes):
        if not path.exists():
            errors.append(f"[MISSING] {what}: {path}")
        elif path.is_dir():
            errors.append(f"[INVALID] {what} is a directory, expected file: {path}")
        else:
            errors.append(f"[INVALID] {what} is empty/too small ({human_bytes(path.stat().st_size)}): {path}")


def check_calibration(calib_dir: Path, strict: bool) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warns: List[str] = []

    require_dir(calib_dir, errors, "Calibration metadata")
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
        require_nonempty_file(p, errors, f"Calibration file {k}")

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

        ok, msg = finite_sample_ok(scores)
        if not ok:
            errors.append(f"[INVALID] scores_fp32.npy: {msg}")

        ok, msg = finite_sample_ok(labels)
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
                    warns.append(f"[WARN] text_features rows {tf.shape[0]} != len(concepts.json) {n_concepts}")
                if not torch.isfinite(tf).all().item():
                    errors.append("[INVALID] text_features contains non-finite values")
        except Exception as e:
            errors.append(f"[INVALID] torch.load(text_features.pt) failed: {e}")

    # Validate auc_primary_class.csv minimal
    aucp = req_files["auc_primary_class.csv"]
    if aucp.exists():
        try:
            df = safe_read_csv(aucp)
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

    require_dir(analysis_dir, errors, "Deep validation analysis")
    if errors:
        return errors, warns

    metrics_csv = analysis_dir / "metrics_per_class.csv"
    top_json = analysis_dir / "top_concepts_by_class.json"
    plots_dir = analysis_dir / "plots"

    require_nonempty_file(metrics_csv, errors, "metrics_per_class.csv")
    require_nonempty_file(top_json, errors, "top_concepts_by_class.json")

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
            df = safe_read_csv(metrics_csv)
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

    require_dir(report_dir, errors, "Report")
    if errors:
        return errors, warns

    report_md = report_dir / "report.md"
    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "paper_tables"

    require_nonempty_file(report_md, errors, "report.md", min_bytes=32)

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

    require_nonempty_file(shortlist_yaml, errors, "Shortlist YAML", min_bytes=32)
    if errors:
        return errors, warns

    # Also require the sibling JSON/CSV produced by the pipeline.
    shortlist_json = shortlist_yaml.with_suffix(".json")
    flat_csv = shortlist_yaml.with_name("concepts_shortlist_flat.csv")
    require_nonempty_file(shortlist_json, errors, "Shortlist JSON", min_bytes=32)
    require_nonempty_file(flat_csv, errors, "Shortlist flat CSV", min_bytes=32)

    try:
        obj = load_yaml(shortlist_yaml)
    except Exception as e:
        errors.append(f"[INVALID] shortlist YAML parse error: {e}")
        return errors, warns

    # Generic schema sanity: ensure we actually have some concepts as leaf strings.
    leaf = collect_leaf_strings(obj)
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
        df = safe_read_csv(flat_csv)
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
        bad = find_forbidden_images(model_dir, allow_dirs=allow_dirs)
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

        heavy_dir = Path(str(obj.get("heavy_run_dir") or "")).expanduser()
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
            require_nonempty_file(p, errors, f"Heavy artifact {p.name}")

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

>>

validate_artifacts.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from ...paths import CALIBRATION_PATHS, OUTPUT_DIR
# Canonical paths (centralised in explainability.paths)
DEFAULT_CALIB_DIR = CALIBRATION_PATHS.metadata_dir
DEFAULT_ANALYSIS_DIR = CALIBRATION_PATHS.analysis_dir
DEFAULT_REPORT_DIR = CALIBRATION_PATHS.report_dir
DEFAULT_SHORTLIST_YAML = CALIBRATION_PATHS.shortlist_yaml

from explainability.utils.bootstrap import bootstrap_package

bootstrap_package(__file__, globals())


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if v < 1024.0:
            return f"{v:.1f}{u}"
        v /= 1024.0
    return f"{v:.1f}PB"


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


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
            obj = torch.load(tfp, map_location="cpu")  # loads objects saved with torch.save :contentReference[oaicite:0]{index=0}
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
            _ = _load_yaml(cr)
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

                # Per-class coverage check: metrics like AP are defined for binary tasks (needs pos/neg). :contentReference[oaicite:1]{index=1}
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
        obj = _load_yaml(shortlist_yaml)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate canonical calibration/deep-validation/report/shortlist artifacts."
    )
    p.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    p.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--shortlist-yaml", type=Path, default=DEFAULT_SHORTLIST_YAML)
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing optional dirs (plots/figures/paper_tables) and low per-class coverage.",
    )
    p.add_argument(
        "--min-valid-per-class",
        type=int,
        default=10,
        help="Minimum rows per class with numeric auc_ovr+ap_ovr required (strict only).",
    )
    p.add_argument(
        "--check-roi",
        action="store_true",
        help="Also validate light spatial/roi outputs and heavy indexed paths for the unified spatial+concept pipeline.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_errors: List[str] = []
    all_warns: List[str] = []

    e, w = check_calibration(args.calib_dir, args.strict)
    all_errors += e
    all_warns += w

    e, w = check_deep_validation(args.analysis_dir, args.strict, args.min_valid_per_class)
    all_errors += e
    all_warns += w

    e, w = check_report(args.report_dir, args.strict)
    all_errors += e
    all_warns += w

    e, w = check_shortlist(args.shortlist_yaml, args.strict)
    all_errors += e
    all_warns += w

    if bool(args.check_spatial_concept):
        e, w = check_spatial_concept_light_outputs(OUTPUT_DIR, args.strict)
        all_errors += e
        all_warns += w

    print("\n========== ARTIFACT VALIDATION ==========")
    print(f"calib_dir   : {args.calib_dir}")
    print(f"analysis_dir : {args.analysis_dir}")
    print(f"report_dir  : {args.report_dir}")
    print(f"shortlist   : {args.shortlist_yaml}")
    print(f"strict      : {args.strict}")
    print(f"check_spatial_concept: {args.check_spatial_concept}")
    print("-----------------------------------------")

    if all_warns:
        print("WARNINGS:")
        for m in all_warns:
            print(f"  {m}")
        print("-----------------------------------------")

    if all_errors:
        print("ERRORS:")
        for m in all_errors:
            print(f"  {m}")
        print("=========================================")
        raise SystemExit(2)

    print("[OK] All required artifacts look consistent.")
    print("=========================================")


if __name__ == "__main__":
    # Keep tokenizer fork warnings from spamming if this is run in multi-worker contexts.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
>>

