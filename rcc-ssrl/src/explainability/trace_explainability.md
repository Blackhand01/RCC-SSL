concept/calibration/build_shortlist.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

"""
Post-process deep validation metrics to:
  - separate diagnostic vs confounding concepts per class
  - build FINAL concept shortlist (YAML canonical + JSON mirror) for downstream concept-XAI (test/no-ROI)
  - write a paper-ready report (markdown + figures + optional LaTeX tables)

Inputs:
  canonical: concept/calibration/analysis/metrics_per_class.csv

Outputs (canonical configs/):
  - concepts_shortlist.yaml   (OFFICIAL)
  - concepts_shortlist.json   (mirror for backward compat)
  - (optional) concepts_shortlist_flat.csv
  - (optional) paper_tables/ (LaTeX tables; also copied to report dir if different)

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

def _bootstrap_package() -> None:
  if __package__:
    return
  this = Path(__file__).resolve()
  src_dir = this
  while src_dir.name != "src" and src_dir.parent != src_dir:
    src_dir = src_dir.parent
  if src_dir.name != "src":
    return
  src_str = str(src_dir)
  if src_str not in sys.path:
    sys.path.insert(0, src_str)
  rel = this.relative_to(src_dir).with_suffix("")
  globals()["__package__"] = ".".join(rel.parts[:-1])

_bootstrap_package()

from .paths import ensure_layout, CONFIGS_DIR, REPORT_DIR, ANALYSIS_DIR, SHORTLIST_JSON, SHORTLIST_YAML
from ..class_utils import canon_class


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
        default=CONFIGS_DIR,
        help="Output directory for FINAL shortlist artifacts (default: canonical configs/)",
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

    tex_dir_short = shortlist_dir / "paper_tables"
    tex_dir_report = report_dir / "paper_tables"
    if args.write_tex:
        tex_dir_short.mkdir(parents=True, exist_ok=True)
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
            # Write to shortlist_dir and report_dir (paper-ready)
            (tex_dir_short / f"top_concepts_{cls}.tex").write_text(tex)
            if tex_dir_report != tex_dir_short:
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

    # Also mirror to canonical absolute targets (defensive)
    try:
        SHORTLIST_JSON.write_text(out_json.read_text())
        SHORTLIST_YAML.write_text(out_yaml.read_text())
    except Exception as e:
        log.warning("Failed to mirror shortlist to canonical constants: %s", str(e))

    log.info("Wrote shortlist JSON: %s", out_json)
    log.info("Wrote shortlist YAML (OFFICIAL): %s", out_yaml)
    log.info("Wrote report.md: %s", out_md)
    log.info("Wrote flat CSV: %s", flat_csv)
    if args.write_tex:
        log.info("Wrote LaTeX tables under: %s", tex_dir_report)


if __name__ == "__main__":
    main()
>>

concept/calibration/calibration_validation.py codice <<
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

from ..class_utils import canon_class
from .paths import ensure_layout, METADATA_DIR, ANALYSIS_DIR

def _bootstrap_package() -> None:
    if __package__:
        return
    this = Path(__file__).resolve()
    src_dir = this
    while src_dir.name != "src" and src_dir.parent != src_dir:
        src_dir = src_dir.parent
    if src_dir.name != "src":
        return
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    rel = this.relative_to(src_dir).with_suffix("")
    globals()["__package__"] = ".".join(rel.parts[:-1])

_bootstrap_package()


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
        help="Canonical metadata dir (default: concept/calibration/metadata).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="Canonical analysis dir (default: concept/calibration/analysis).",
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

concept/calibration/__init__.py codice <<
"""
Calibration + deep validation for PLIP concept prompts.

Fixed layout (no runs/):
  - metadata/: unified calibration artifacts (TRAIN+VAL)
  - analysis/: deep validation outputs + audits
  - analysis/report/: paper-ready report (md/tables/figures)
  - configs/: final shortlist for test
"""

from .paths import (  # noqa: F401
    ANALYSIS_DIR,
    CONFIGS_DIR,
    METADATA_DIR,
    REPORT_DIR,
)
>>

concept/calibration/paths.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

"""
Centralized canonical paths for concept calibration pipeline.

This module now re-exports the single shared layout from explainability.paths
to keep one source of truth for paths.
"""

from dataclasses import dataclass
from typing import Optional

from ...paths import CALIBRATION_PATHS, ensure_calibration_layout

# Backward-compatible aliases
METADATA_DIR = CALIBRATION_PATHS.metadata_dir
ANALYSIS_DIR = CALIBRATION_PATHS.analysis_dir
REPORT_DIR = CALIBRATION_PATHS.report_dir
CONFIGS_DIR = CALIBRATION_PATHS.configs_dir
SHORTLIST_YAML = CALIBRATION_PATHS.shortlist_yaml
SHORTLIST_JSON = CALIBRATION_PATHS.shortlist_json


@dataclass(frozen=True)
class CanonicalLayout:
    metadata_dir = CALIBRATION_PATHS.metadata_dir
    analysis_dir = CALIBRATION_PATHS.analysis_dir
    report_dir = CALIBRATION_PATHS.report_dir
    configs_dir = CALIBRATION_PATHS.configs_dir
    shortlist_yaml = CALIBRATION_PATHS.shortlist_yaml
    shortlist_json = CALIBRATION_PATHS.shortlist_json


def ensure_layout(layout: Optional[CanonicalLayout] = None) -> CanonicalLayout:
    ensure_calibration_layout(CALIBRATION_PATHS)
    return layout or CanonicalLayout()
>>

concept/calibration/plots.py codice <<
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

concept/calibration/run_calibration.py codice <<
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

def _bootstrap_package() -> None:
    """
    Allow running this file directly:
      python .../src/explainability/concept/calibration/run_calibration.py ...
    without breaking relative imports (..plip, .paths, etc.).
    """
    if __package__:
        return
    this = Path(__file__).resolve()
    src_dir = this
    # climb to ".../src"
    while src_dir.name != "src" and src_dir.parent != src_dir:
        src_dir = src_dir.parent
    if src_dir.name != "src":
        return
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    rel = this.relative_to(src_dir).with_suffix("")  # explainability/concept/calibration/run_calibration
    pkg = ".".join(rel.parts[:-1])                   # explainability.concept.calibration
    globals()["__package__"] = pkg

_bootstrap_package()

from ..plip.ontology_io import load_ontology, concepts_to_prompt_lists, concepts_to_dicts
from ..plip.plip_model import load_plip, encode_text, encode_images, score
from ..plip.wds_loader import build_wds_loader
from .plots import plot_heatmap, plot_bar
from .paths import ensure_layout, METADATA_DIR


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
    p.add_argument("--config", required=True, type=Path, help="YAML config path.")
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

    cfg = yaml.safe_load(args.config.read_text())

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

concept/calibration/validate_artifacts.py codice <<
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

from ...paths import CALIBRATION_PATHS
# Canonical paths (centralised in explainability.paths)
DEFAULT_CALIB_DIR = CALIBRATION_PATHS.metadata_dir
DEFAULT_ANALYSIS_DIR = CALIBRATION_PATHS.analysis_dir
DEFAULT_REPORT_DIR = CALIBRATION_PATHS.report_dir
DEFAULT_SHORTLIST_YAML = CALIBRATION_PATHS.shortlist_yaml

def _bootstrap_package() -> None:
    if __package__:
        return
    this = Path(__file__).resolve()
    src_dir = this
    while src_dir.name != "src" and src_dir.parent != src_dir:
        src_dir = src_dir.parent
    if src_dir.name != "src":
        return
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    rel = this.relative_to(src_dir).with_suffix("")
    globals()["__package__"] = ".".join(rel.parts[:-1])

_bootstrap_package()


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

    print("\n========== ARTIFACT VALIDATION ==========")
    print(f"calib_dir   : {args.calib_dir}")
    print(f"analysis_dir : {args.analysis_dir}")
    print(f"report_dir  : {args.report_dir}")
    print(f"shortlist   : {args.shortlist_yaml}")
    print(f"strict      : {args.strict}")
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

concept/class_utils.py codice <<
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import yaml


# Canonicalize class names across splits/configs (common drift: Oncocytoma vs ONCO, chRCC vs CHROMO).
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


def load_shortlist_idx(
    path: Path, concept_to_idx: Dict[str, int], log: Any = None
) -> Dict[str, Dict[str, List[int]]]:
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
                log.warning(f"[SHORTLIST] Concepts missing in ontology (ignored) for {cls_norm}: {missing}")
            except Exception:
                pass
        out[cls_norm] = {"primary": prim, "confounds": conf}
    return out


def concept_indices_for_patch(
    shortlist: Dict[str, Dict[str, List[int]]],
    true_cls: Optional[str],
    pred_cls: Optional[str],
) -> List[int]:
    idxs: Set[int] = set()
    if pred_cls and pred_cls in shortlist:
        idxs.update(shortlist[pred_cls].get("primary", []))
        idxs.update(shortlist[pred_cls].get("confounds", []))
    if true_cls and true_cls in shortlist:
        idxs.update(shortlist[true_cls].get("primary", []))
    return sorted(idxs)
>>

concept/__init__.py codice <<
"""
Canonical (fixed) filesystem layout for PLIP concept calibration + validation.

Paths (single source of truth, see explainability.paths):
  A) Calibration unified metadata (TRAIN+VAL scored separately, output merged):
     outputs/xai/concept/calibration/metadata/

  B) Deep validation (analysis of the calibration):
     outputs/xai/concept/calibration/analysis/

  C) Paper-ready report:
     outputs/xai/concept/calibration/analysis/report/

  D) Final shortlist for test (no-ROI and ROI):
     src/explainability/configs/concepts_shortlist.yaml

Additional canonical pipelines:
  E) Concept XAI on TEST (NO-ROI, model-independent, computed once):
     outputs/xai/concept/no_roi/

  F) Concept XAI on TEST (ROI, model-dependent, uses spatial masks from a model):
     <MODEL_ROOT>/xai/concept/roi/

  G) Comparison ROI vs NO-ROI (paper-ready, per model):
     outputs/xai/concept/comparision/<MODEL_ID>/

NOTE:
  Folder name is intentionally "comparision" (typo kept for backward compatibility with your request).
"""
>>

concept/plip/__init__.py codice <<
"""
PLIP-based concept explainability toolkit.

This package includes calibration, deep validation, and concept-XAI runners.
"""
>>

concept/plip/ontology_io.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


@dataclass(frozen=True)
class Concept:
    id: int
    name: str
    short_name: str
    group: str
    primary_class: Optional[str]
    prompts: List[str]

    @property
    def prompt(self) -> str:
        # Backward compatibility: single-prompt codepaths
        return self.prompts[0]


def _clean_prompt(s: str) -> str:
    # YAML block scalars may contain newlines; collapse to single line.
    return " ".join(str(s).split())


def _scan_for_bad_unicode(text: str, path: Path) -> None:
    # U+FFFC commonly appears via copy/paste and can break YAML indentation semantics.
    bad = "\uFFFC"
    if bad in text:
        # Provide approximate location(s) for fast debugging.
        lines = text.splitlines()
        hits = []
        for i, ln in enumerate(lines, start=1):
            if bad in ln:
                col = ln.index(bad) + 1
                hits.append(f"line={i},col={col}")
                if len(hits) >= 5:
                    break
        loc = ", ".join(hits) if hits else "unknown"
        raise RuntimeError(
            f"Invalid character U+FFFC found in ontology YAML: {path} ({loc}). "
            "Remove the invisible placeholder character (often from copy/paste) and re-run."
        )


def _parse_prompts(c: Dict[str, Any], *, concept_ref: str) -> List[str]:
    # Support both:
    # - prompt: "..."
    # - prompts: ["...", "..."]
    raw: Union[None, str, List[Any]] = None
    if "prompts" in c and c.get("prompts") is not None:
        raw = c.get("prompts")
    else:
        raw = c.get("prompt")

    prompts: List[str] = []
    if raw is None:
        prompts = []
    elif isinstance(raw, str):
        prompts = [_clean_prompt(raw)]
    elif isinstance(raw, list):
        for x in raw:
            if x is None:
                continue
            prompts.append(_clean_prompt(str(x)))
    else:
        raise RuntimeError(f"Invalid prompts type for {concept_ref}: {type(raw)}")

    prompts = [p for p in prompts if p]
    if not prompts:
        raise RuntimeError(f"Missing prompt(s) for {concept_ref}. Expected 'prompt' or 'prompts'.")
    return prompts


def load_ontology(path: Path) -> Tuple[Dict[str, Any], List[Concept]]:
    txt = path.read_text(encoding="utf-8")
    _scan_for_bad_unicode(txt, path)
    data = yaml.safe_load(txt)
    concepts_raw = data.get("concepts", [])
    if not concepts_raw:
        raise RuntimeError(f"No concepts found in {path}")

    concepts: List[Concept] = []
    seen_short = set()
    seen_id = set()

    for c in concepts_raw:
        cid = int(c.get("id"))
        if cid in seen_id:
            raise RuntimeError(f"Duplicate concept id={cid} in {path}")
        seen_id.add(cid)

        short = str(c.get("short_name", "")).strip()
        if not short:
            raise RuntimeError(f"Missing short_name for concept id={cid} in {path}")
        if short in seen_short:
            raise RuntimeError(f"Duplicate short_name={short} in {path}")
        seen_short.add(short)

        name = str(c.get("name", "")).strip()
        if not name:
            raise RuntimeError(f"Missing name for concept short_name={short} in {path}")

        group = str(c.get("group", "")).strip() or "Other"
        primary = c.get("primary_class", None)
        primary = str(primary).strip() if primary is not None else None
        if primary == "" or primary == "null":
            primary = None

        prompts = _parse_prompts(c, concept_ref=f"concept short_name={short}")

        concepts.append(
            Concept(
                id=cid,
                name=name,
                short_name=short,
                group=group,
                primary_class=primary,
                prompts=prompts,
            )
        )

    concepts = sorted(concepts, key=lambda x: x.id)
    meta = {
        "version": data.get("version", None),
        "name": data.get("name", None),
        "global_instructions": data.get("global_instructions", None),
        "source_path": str(path),
        "n_concepts": len(concepts),
    }
    return meta, concepts


def concepts_to_prompts(concepts: List[Concept]) -> List[str]:
    return [c.prompt for c in concepts]


def concepts_to_prompt_lists(concepts: List[Concept]) -> List[List[str]]:
    return [list(c.prompts) for c in concepts]


def concepts_to_dicts(concepts: List[Concept]) -> List[Dict[str, Any]]:
    return [
        {
            "id": c.id,
            "name": c.name,
            "short_name": c.short_name,
            "group": c.group,
            "primary_class": c.primary_class,
            # Backward compatible single prompt + full list
            "prompt": c.prompt,
            "prompts": list(c.prompts),
        }
        for c in concepts
    ]
>>

concept/plip/plip_model.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from transformers import CLIPModel, AutoProcessor

try:
    from huggingface_hub import snapshot_download
    HAVE_HF_HUB = True
except Exception:
    HAVE_HF_HUB = False

try:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
except Exception as e:
    raise RuntimeError("torchvision is required for PLIP preprocessing") from e


def _as_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _dtype_from_precision(precision: str) -> torch.dtype:
    precision = (precision or "fp16").lower()
    if precision in ("fp16", "float16"):
        return torch.float16
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def ensure_model_local(model_id: str, model_local_dir: Path, hf_cache_dir: Optional[Path] = None) -> Path:
    model_local_dir.mkdir(parents=True, exist_ok=True)
    # Heuristic: if config.json exists, assume it's a valid snapshot.
    if (model_local_dir / "config.json").exists():
        return model_local_dir
    if not HAVE_HF_HUB:
        raise RuntimeError("huggingface_hub not available; cannot snapshot_download")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_local_dir),
        cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
    )
    return model_local_dir


def build_clip_preprocess(proc) -> transforms.Compose:
    # Use processor image stats for correctness.
    ip = getattr(proc, "image_processor", None)
    if ip is None:
        # Fallback: simple 224 center crop, CLIP-like mean/std unknown.
        size = 224
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        size = ip.size["shortest_edge"] if isinstance(ip.size, dict) and "shortest_edge" in ip.size else ip.size
        crop_h = ip.crop_size["height"] if isinstance(ip.crop_size, dict) else ip.crop_size
        crop_w = ip.crop_size["width"] if isinstance(ip.crop_size, dict) else ip.crop_size
        size = int(size)
        crop_h = int(crop_h)
        crop_w = int(crop_w)
        mean = list(ip.image_mean)
        std = list(ip.image_std)

        # If crop differs, respect crop size
        if crop_h != size or crop_w != size:
            crop = (crop_h, crop_w)
        else:
            crop = size

    # default CLIP uses bicubic resize
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(crop if "crop" in locals() else size),
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


@dataclass
class PLIP:
    model: CLIPModel
    processor: any
    device: torch.device
    dtype: torch.dtype
    max_text_len: int
    score_scale: float
    preprocess: transforms.Compose
    model_id: str
    model_path: str


def load_plip(
    model_id: str,
    model_local_dir: Optional[Union[str, Path]] = None,
    device: str = "cuda",
    precision: str = "fp16",
    score_scale: Optional[float] = 100.0,
    hf_cache_dir: Optional[Union[str, Path]] = None,
) -> PLIP:
    dev = _as_device(device)
    dtype = _dtype_from_precision(precision)
    hf_cache_dir = Path(hf_cache_dir) if hf_cache_dir else None

    model_path: Union[str, Path] = model_id
    if model_local_dir is not None:
        model_local_dir = Path(model_local_dir)
        model_path = ensure_model_local(model_id, model_local_dir, hf_cache_dir=hf_cache_dir)

    model = CLIPModel.from_pretrained(model_path)
    proc = AutoProcessor.from_pretrained(model_path)

    model.eval()
    model.to(dev)

    max_text_len = int(getattr(model.config.text_config, "max_position_embeddings", 77))

    # If score_scale is None, use learned CLIP logit_scale.
    if score_scale is None:
        score_scale = float(model.logit_scale.exp().detach().cpu().item())

    preprocess = build_clip_preprocess(proc)

    return PLIP(
        model=model,
        processor=proc,
        device=dev,
        dtype=dtype,
        max_text_len=max_text_len,
        score_scale=float(score_scale),
        preprocess=preprocess,
        model_id=model_id,
        model_path=str(model_path),
    )


@torch.inference_mode()
def encode_text(plip: PLIP, prompts: List[str]) -> torch.Tensor:
    # Force max_length to avoid "tokenizer.model_max_length is huge" truncation warnings.
    inputs = plip.processor(
        text=prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=plip.max_text_len,
    )
    inputs = {k: v.to(plip.device) for k, v in inputs.items()}
    if plip.device.type == "cuda" and plip.dtype in (torch.float16, torch.bfloat16):
        with torch.autocast(device_type="cuda", dtype=plip.dtype):
            feats = plip.model.get_text_features(**inputs)
    else:
        feats = plip.model.get_text_features(**inputs)
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
    out_dtype = plip.dtype if plip.device.type == "cuda" else torch.float32
    return feats.to(dtype=out_dtype)


@torch.inference_mode()
def encode_images(plip: PLIP, images: torch.Tensor) -> torch.Tensor:
    # images: [B,3,H,W] already normalized
    images = images.to(plip.device, non_blocking=True)
    if plip.device.type == "cuda" and plip.dtype in (torch.float16, torch.bfloat16):
        with torch.autocast(device_type="cuda", dtype=plip.dtype):
            feats = plip.model.get_image_features(pixel_values=images)
    else:
        feats = plip.model.get_image_features(pixel_values=images)
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
    out_dtype = plip.dtype if plip.device.type == "cuda" else torch.float32
    return feats.to(dtype=out_dtype)


@torch.inference_mode()
def score(plip: PLIP, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    # logits: [B,C]
    if image_features.dtype != text_features.dtype:
        text_features = text_features.to(dtype=image_features.dtype)
    return plip.score_scale * (image_features @ text_features.T)
>>

concept/plip/roi.py codice <<
from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageColor, ImageDraw


def _as_numpy(arr: Union[np.ndarray, "torch.Tensor", Iterable]) -> np.ndarray:  # type: ignore[name-defined]
    try:
        import torch

        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(arr)


def _ensure_mask_shape(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    if mask.shape[1] == target_size[0] and mask.shape[0] == target_size[1]:
        return mask
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = mask_img.resize(target_size, resample=Image.NEAREST)
    return (np.asarray(resized) > 0).astype(bool)


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _expand_bbox(
    bbox: Tuple[int, int, int, int], min_size: int, max_w: int, max_h: int
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    if w >= min_size and h >= min_size:
        return x0, y0, x1, y1
    pad_w = max(0, (min_size - w) // 2)
    pad_h = max(0, (min_size - h) // 2)
    x0 = max(0, x0 - pad_w)
    y0 = max(0, y0 - pad_h)
    x1 = min(max_w - 1, x1 + pad_w + ((min_size - w) % 2))
    y1 = min(max_h - 1, y1 + pad_h + ((min_size - h) % 2))
    return x0, y0, x1, y1


def roi_from_rollout(
    rollout_map: Union[np.ndarray, "torch.Tensor"], *, quantile: float = 0.9, min_size_px: int = 64
) -> np.ndarray:
    """
    Build a binary ROI mask from an attention rollout map.
    """
    arr = _as_numpy(rollout_map).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    thr = np.quantile(arr, quantile)
    mask = arr >= thr
    if not mask.any():
        mask = arr == arr.max()
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        h, w = mask.shape
        cy, cx = h // 2, w // 2
        half = max(1, min_size_px // 2)
        y0, y1 = max(0, cy - half), min(h - 1, cy + half)
        x0, x1 = max(0, cx - half), min(w - 1, cx + half)
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
        return mask
    x0, y0, x1, y1 = _expand_bbox(bbox, min_size_px, mask.shape[1], mask.shape[0])
    out = np.zeros_like(mask, dtype=bool)
    out[y0 : y1 + 1, x0 : x1 + 1] = mask[y0 : y1 + 1, x0 : x1 + 1]
    return out


def crop_from_mask(image: Image.Image, mask: np.ndarray, min_size_px: int = 32) -> Image.Image:
    """
    Crop a PIL image to the bounding box of the mask. Falls back to the input image if
    the mask is empty.
    """
    mask_bool = mask.astype(bool)
    mask_bool = _ensure_mask_shape(mask_bool, image.size)
    bbox = _bbox_from_mask(mask_bool)
    if bbox is None:
        return image
    x0, y0, x1, y1 = _expand_bbox(bbox, min_size_px, image.size[0], image.size[1])
    return image.crop((x0, y0, x1 + 1, y1 + 1))


def overlay_contour(
    image: Image.Image, mask: np.ndarray, color: Union[str, Tuple[int, int, int]] = "blue"
) -> Image.Image:
    """
    Draw a 1px contour of the mask on top of the image (ablation / visualization only).
    """
    mask_bool = _ensure_mask_shape(mask.astype(bool), image.size)
    if isinstance(color, str):
        color_rgb = ImageColor.getrgb(color)
    else:
        color_rgb = tuple(int(c) for c in color)

    # Boundary = mask pixels with at least one non-mask neighbour
    up = np.roll(mask_bool, 1, axis=0)
    down = np.roll(mask_bool, -1, axis=0)
    left = np.roll(mask_bool, 1, axis=1)
    right = np.roll(mask_bool, -1, axis=1)
    boundary = mask_bool & ~(up & down & left & right)

    if not boundary.any():
        return image

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    ys, xs = np.where(boundary)
    for y, x in zip(ys.tolist(), xs.tolist()):
        draw.point((int(x), int(y)), fill=(*color_rgb, 255))

    base = image.convert("RGBA")
    composed = Image.alpha_composite(base, overlay)
    return composed.convert("RGB")
>>

concept/plip/scoring.py codice <<
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import torch

from .plip_model import PLIP, encode_images, encode_text, score


@torch.inference_mode()
def encode_text_cached(
    plip: PLIP,
    prompts: List[str],
    cache_dir: Optional[Union[str, Path]] = None,
    cache_key: Optional[str] = None,
) -> torch.Tensor:
    """
    Encode prompts with optional on-disk caching to avoid recomputation across runs.
    """
    cache_path: Optional[Path] = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fname = cache_key if cache_key else "text_features.pt"
        cache_path = cache_dir / fname
        if cache_path.exists():
            saved = torch.load(cache_path, map_location="cpu")
            feats = saved["text_features"] if isinstance(saved, dict) and "text_features" in saved else saved
            return feats.to(device=plip.device, dtype=plip.dtype)

    feats = encode_text(plip, prompts)
    if cache_path:
        torch.save(
            {
                "text_features": feats.detach().cpu(),
                "prompts": prompts,
                "model_id": plip.model_id,
                "max_text_len": plip.max_text_len,
            },
            cache_path,
        )
    return feats


@torch.inference_mode()
def score_batch(
    plip: PLIP, images: Union[torch.Tensor, Iterable[torch.Tensor]], text_features: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode images and return (logits, probs) against provided text features.
    """
    if isinstance(images, Iterable) and not isinstance(images, torch.Tensor):
        images = torch.stack(list(images), dim=0)
    img_feats = encode_images(plip, images)
    logits = score(plip, img_feats, text_features)
    probs = torch.softmax(logits, dim=1)
    return logits, probs
>>

concept/plip/wds_loader.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

try:
    import webdataset as wds
except Exception as e:
    raise RuntimeError("webdataset is required for PLIP pipeline (pip install webdataset)") from e


def parse_meta(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, (bytes, bytearray)):
        try:
            return json.loads(x.decode("utf-8"))
        except Exception:
            return {}
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def list_shards(split_dir: Path, pattern: str) -> List[str]:
    split_dir = Path(split_dir)

    # Path("") becomes "." -> a very common misconfiguration when YAML has split_dir: ""
    if str(split_dir).strip() in ("", "."):
        raise ValueError(
            "WebDataset split_dir is empty. "
            "Set data.webdataset.split_dir in the YAML, or export WDS_TRAIN_DIR/WDS_TEST_DIR (or WDS_DIR) "
            "and let the caller inject it."
        )

    # Support passing a single shard tar directly
    if split_dir.is_file():
        return [str(split_dir)]

    if not split_dir.is_dir():
        raise FileNotFoundError(f"WebDataset split_dir not found: {split_dir}")

    # If the pattern is a non-glob URL/template, forward as-is (glob won't expand it).
    # WebDataset can still handle such URL-style specs internally.
    if any(tok in pattern for tok in ("{", "}", "pipe:", "::")):
        return [str(split_dir / pattern)]

    shard_glob = str(split_dir / pattern)
    shards = sorted(glob.glob(shard_glob))
    if shards:
        return shards

    # Helpful error for pattern mismatch: show what tar files exist.
    any_tars = sorted(glob.glob(str(split_dir / "*.tar")))
    if any_tars:
        ex = ", ".join(Path(p).name for p in any_tars[:8])
        raise FileNotFoundError(
            f"No shards matched pattern '{pattern}' under {split_dir}. "
            f"Found .tar files e.g. {ex}. "
            "Fix data.webdataset.pattern in the config to match your shard naming."
        )
    raise FileNotFoundError(f"No .tar shards found under {split_dir} (pattern '{pattern}').")


def build_wds_loader(
    split_dir: Path,
    pattern: str,
    image_key: str,
    meta_key: str,
    preprocess: Optional[Callable],
    batch_size: int,
    num_workers: int,
    *,
    return_raw: bool = False,
) -> Iterable[Tuple[Optional[torch.Tensor], List[Dict[str, Any]], List[str], Optional[List[Any]]]]:
    """
    Build a WebDataset loader that can optionally return raw PIL images alongside
    preprocessed tensors. The preprocess callable is applied only if provided.

    Yields tuples:
      - images: Tensor [B,3,H,W] if preprocess is set, otherwise None
      - metas: list of dict
      - keys: list of shard keys (str)
      - raw_images: list of PIL images if return_raw=True else None
    """
    shards = list_shards(split_dir, pattern)

    def _map_img(img):
        pil_img = img
        proc_img = preprocess(pil_img) if preprocess is not None else pil_img
        if return_raw:
            return proc_img, pil_img
        return proc_img

    ds = (
        wds.WebDataset(
            shards,
            shardshuffle=False,
            handler=wds.warn_and_continue,
            empty_check=False,
        )
        .decode("pil")
        .to_tuple(image_key, meta_key, "__key__", handler=wds.warn_and_continue)
        .map_tuple(_map_img, parse_meta, lambda k: k)
    )

    # Let WebDataset do batching; then use WebLoader (DataLoader wrapper)
    ds = ds.batched(int(batch_size), partial=True)

    try:
        loader = wds.WebLoader(ds, batch_size=None, num_workers=int(num_workers))
    except Exception:
        loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=int(num_workers))

    def _iter():
        for batch in loader:
            if batch is None:
                continue
            imgs, metas, keys = batch
            raw_imgs = None

            if return_raw:
                # WebDataset batching may return either a list of (proc_img, pil_img)
                # or a tuple (proc_batch, raw_batch) depending on collation.
                if isinstance(imgs, tuple) and len(imgs) == 2:
                    proc_part, raw_part = imgs
                    imgs = proc_part
                    raw_imgs = list(raw_part) if isinstance(raw_part, (list, tuple)) else [raw_part]
                else:
                    raw_imgs = [im[1] for im in imgs]
                    imgs = [im[0] for im in imgs]

            imgs_tensor: Optional[torch.Tensor]
            if preprocess is None:
                imgs_tensor = None
            else:
                # WebDataset may already emit a Tensor batch (common case).
                if isinstance(imgs, torch.Tensor):
                    imgs_tensor = imgs
                else:
                    imgs_tensor = torch.stack(list(imgs), dim=0)

            yield imgs_tensor, list(metas), [str(k) for k in keys], raw_imgs

    return _iter()
>>

concept/run_no_roi.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

"""
Compute PLIP concept scores on TEST patches WITHOUT ROI (NO-ROI).

Design constraints (requested):
  - Must be MODEL-INDEPENDENT (run once for the test set).
  - Must not depend on spatial XAI, backbone, checkpoint.
  - Must write deterministically under the canonical no_roi/ layout
    (no timestamps, idempotent, overwrite-safe).
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

def _bootstrap_package() -> None:
    if __package__:
        return
    this = Path(__file__).resolve()
    src_dir = this
    while src_dir.name != "src" and src_dir.parent != src_dir:
        src_dir = src_dir.parent
    if src_dir.name != "src":
        return
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    rel = this.relative_to(src_dir).with_suffix("")
    globals()["__package__"] = ".".join(rel.parts[:-1])

_bootstrap_package()

from ..common.eval_utils import (
    ensure_dir,
    make_wds_loader_with_keys,
    atomic_write_json,
    atomic_write_text,
)
from ..class_utils import load_shortlist_idx
from ..plip.plip_model import load_plip, encode_images, score
from ...paths import CALIBRATION_PATHS, CONFIG_DIR, ensure_no_roi_layout


def _load_text_features(text_features_pt: Path) -> torch.Tensor:
    obj = torch.load(text_features_pt, map_location="cpu")
    if torch.is_tensor(obj):
        tf = obj
    elif isinstance(obj, dict):
        tf = obj.get("text_features", None)
        if tf is None:
            tf = obj.get("features", None)
    else:
        tf = None
    if tf is None or not torch.is_tensor(tf) or tf.ndim != 2:
        raise RuntimeError(f"Invalid text_features.pt format: {text_features_pt}")
    return tf


def _load_concepts(concepts_json: Path) -> List[Dict[str, Any]]:
    raw = json.loads(concepts_json.read_text())
    if isinstance(raw, dict) and "concepts" in raw:
        concepts = raw["concepts"]
    else:
        concepts = raw
    if not isinstance(concepts, list) or not concepts:
        raise RuntimeError(f"Invalid concepts.json: {concepts_json}")
    return concepts


def _resolve_test_dir(cfg: Dict[str, Any]) -> Path:
    wds = cfg.get("data", {}).get("webdataset", {})
    td = str(wds.get("test_dir") or "").strip()
    if not td:
        td = os.getenv("WDS_TEST_DIR", "").strip()
    if not td:
        raise RuntimeError(
            "Missing TEST WebDataset dir. Set data.webdataset.test_dir or export WDS_TEST_DIR."
        )
    p = Path(td)
    if not p.exists():
        raise FileNotFoundError(f"TEST WebDataset dir not found: {p}")
    return p


def _build_selected_indices(
    *,
    concepts: List[Dict[str, Any]],
    shortlist_yaml: Path,
    use_shortlist_only: bool,
    log: logging.Logger,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    concept_to_idx = {}
    for i, c in enumerate(concepts):
        sn = str(c.get("short_name") or c.get("concept_short_name") or "").strip()
        if not sn:
            continue
        concept_to_idx[sn] = i

    if not use_shortlist_only:
        idxs = list(range(len(concepts)))
        sel = []
        for i in idxs:
            cc = concepts[i]
            sel.append(
                {
                    "concept_idx": int(i),
                    "concept_short_name": str(cc.get("short_name") or cc.get("concept_short_name") or f"c{i}"),
                    "concept_name": str(cc.get("name") or cc.get("concept_name") or f"Concept {i}"),
                    "group": cc.get("group", None),
                    "primary_class": cc.get("primary_class", None),
                }
            )
        return idxs, sel

    if not shortlist_yaml.exists():
        raise FileNotFoundError(f"Shortlist YAML not found: {shortlist_yaml}")

    shortlist = load_shortlist_idx(shortlist_yaml, concept_to_idx, log=log)
    union: set[int] = set()
    for cls, d in shortlist.items():
        union.update(d.get("primary", []))
        union.update(d.get("confounds", []))
    idxs = sorted(union)
    if not idxs:
        raise RuntimeError(f"Shortlist produced 0 indices (ontology mismatch?): {shortlist_yaml}")

    sel = []
    for i in idxs:
        cc = concepts[i]
        sel.append(
            {
                "concept_idx": int(i),
                "concept_short_name": str(cc.get("short_name") or cc.get("concept_short_name") or f"c{i}"),
                "concept_name": str(cc.get("name") or cc.get("concept_name") or f"Concept {i}"),
                "group": cc.get("group", None),
                "primary_class": cc.get("primary_class", None),
            }
        )
    log.info("Selected concepts (union shortlist): %d / %d", len(idxs), len(concepts))
    return idxs, sel


def _save_heatmap_mean_by_class(
    scores: np.ndarray,
    labels: np.ndarray,
    concept_short: List[str],
    out_base: Path,
    *,
    formats: Tuple[str, ...],
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = sorted({str(x) for x in labels.tolist()})
    if not classes:
        return
    mean_by_class = np.zeros((len(classes), scores.shape[1]), dtype=np.float32)
    for i, cls in enumerate(classes):
        idx = np.where(labels == cls)[0]
        if idx.size > 0:
            mean_by_class[i] = scores[idx].mean(axis=0)

    fig = plt.figure(figsize=(max(8, 0.35 * len(concept_short)), max(4, 0.45 * len(classes))))
    ax = fig.add_subplot(111)
    im = ax.imshow(mean_by_class, aspect="auto")
    ax.set_title("NO-ROI: mean PLIP score by class (TEST)")
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xticks(np.arange(len(concept_short)))
    ax.set_xticklabels(concept_short, rotation=90, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Concept NO-ROI on TEST (model-independent, canonical output).")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional YAML config. If omitted or missing, the runner falls back to env/defaults.\n"
            "Common minimum env requirement: WDS_TEST_DIR."
        ),
    )
    ap.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Override TEST WebDataset dir (otherwise from config.data.webdataset.test_dir or env WDS_TEST_DIR).",
    )
    ap.add_argument(
        "--calibration-metadata-dir",
        type=Path,
        default=None,
        help="Override calibration metadata dir (otherwise from config.inputs.calibration_metadata_dir or canonical).",
    )
    ap.add_argument(
        "--shortlist-yaml",
        type=Path,
        default=None,
        help="Override shortlist YAML path (otherwise from config.inputs.shortlist_yaml or canonical).",
    )
    ap.add_argument(
        "--all-concepts",
        action="store_true",
        help="Score all ontology concepts (ignore shortlist).",
    )
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing NO-ROI artifacts.")
    return ap.parse_args()



def _safe_load_yaml(path: Path, log: logging.Logger) -> Dict[str, Any]:
    try:
        obj = yaml.safe_load(path.read_text())
        return obj if isinstance(obj, dict) else {}
    except FileNotFoundError:
        log.warning("NO-ROI config not found (continuing with env/defaults): %s", path)
        return {}
    except Exception as e:
        raise RuntimeError(f"Failed to parse NO-ROI config YAML: {path} ({e})") from e


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("concept_no_roi")

    layout = ensure_no_roi_layout()

    cfg: Dict[str, Any] = {}
    if args.config is not None:
        cfg = _safe_load_yaml(args.config, log=log)
    else:
        # Keep it explicit in logs: this run relies on env/defaults.
        log.info("NO-ROI config: <none> (env/defaults mode).")

    # Inject CLI overrides into the expected config schema (without requiring a file).
    cfg.setdefault("inputs", {})
    cfg.setdefault("data", {})
    cfg["data"].setdefault("webdataset", {})
    cfg.setdefault("plip", {})

    if args.calibration_metadata_dir is not None:
        cfg["inputs"]["calibration_metadata_dir"] = str(args.calibration_metadata_dir)
    else:
        # Allow env override even without YAML.
        if os.getenv("CALIBRATION_METADATA_DIR"):
            cfg["inputs"]["calibration_metadata_dir"] = os.getenv("CALIBRATION_METADATA_DIR")

    if args.shortlist_yaml is not None:
        cfg["inputs"]["shortlist_yaml"] = str(args.shortlist_yaml)
    else:
        if os.getenv("CONCEPT_SHORTLIST_YAML"):
            cfg["inputs"]["shortlist_yaml"] = os.getenv("CONCEPT_SHORTLIST_YAML")

    if args.test_dir is not None:
        cfg["data"]["webdataset"]["test_dir"] = str(args.test_dir)

    if args.all_concepts:
        cfg["inputs"]["use_shortlist_only"] = False

    # Canonical output enforced
    ARTIFACTS_DIR = layout.artifacts_dir
    PLOTS_DIR = layout.plots_dir
    LOGS_DIR = layout.logs_dir
    ensure_dir(ARTIFACTS_DIR)
    ensure_dir(PLOTS_DIR)
    ensure_dir(LOGS_DIR)

    # Optional wipe (idempotent)
    scores_path = ARTIFACTS_DIR / "scores_fp32.npy"
    keys_path = ARTIFACTS_DIR / "keys.npy"
    labels_path = ARTIFACTS_DIR / "labels.npy"
    if scores_path.exists() and not args.overwrite:
        log.info("NO-ROI already computed (scores exist). Use --overwrite to recompute: %s", scores_path)
        return

    # Inputs
    inp = cfg.get("inputs", {})
    cal_dir = Path(inp.get("calibration_metadata_dir") or CALIBRATION_PATHS.metadata_dir)
    shortlist_yaml = Path(inp.get("shortlist_yaml") or CALIBRATION_PATHS.shortlist_yaml)
    if not shortlist_yaml.exists():
        alt_shortlist = CONFIG_DIR / "concepts_shortlist.yaml"
        if alt_shortlist.exists():
            shortlist_yaml = alt_shortlist
    use_shortlist_only = bool(inp.get("use_shortlist_only", True))

    # Resolve TEST dir now that we injected CLI/env overrides into cfg
    # (this keeps the rest of the code unchanged).
    _ = _resolve_test_dir(cfg)  # will raise with a clear message if still missing

    concepts_json = cal_dir / "concepts.json"
    text_features_pt = cal_dir / "text_features.pt"
    if not concepts_json.exists():
        raise FileNotFoundError(f"Missing calibration concepts.json: {concepts_json}")
    if not text_features_pt.exists():
        raise FileNotFoundError(f"Missing calibration text_features.pt: {text_features_pt}")

    concepts = _load_concepts(concepts_json)
    tf_all = _load_text_features(text_features_pt)  # [C,D]

    sel_idxs, sel_concepts = _build_selected_indices(
        concepts=concepts,
        shortlist_yaml=shortlist_yaml,
        use_shortlist_only=use_shortlist_only,
        log=log,
    )
    tf = tf_all[torch.tensor(sel_idxs, dtype=torch.long)]

    # PLIP
    plip_cfg = cfg.get("plip", {})
    plip = load_plip(
        model_id=str(plip_cfg.get("model_id", "vinid/plip")),
        model_local_dir=plip_cfg.get("model_local_dir", None),
        device=str(plip_cfg.get("device", "cuda")),
        precision=str(plip_cfg.get("precision", "fp16")),
        score_scale=None,  # rely on learned logit_scale for inference (more standard)
        hf_cache_dir=plip_cfg.get("hf_cache_dir", None),
    )

    # Data
    test_dir = _resolve_test_dir(cfg)
    data = cfg.get("data", {})
    wds = data.get("webdataset", {})
    pattern = str(wds.get("pattern", "shard-*.tar"))
    image_key = str(wds.get("image_key", "img.jpg;jpg;jpeg;png"))
    meta_key = str(wds.get("meta_key", "meta.json;json"))
    class_field = str(wds.get("class_field", "class_label"))
    bs = int(data.get("batch_size", 256))
    nw = int(data.get("num_workers", 8))
    max_patches = int(data.get("max_patches", 0))

    loader = make_wds_loader_with_keys(
        test_dir=str(test_dir),
        pattern=pattern,
        image_key=image_key,
        meta_key=meta_key,
        preprocess_fn=plip.preprocess,
        num_workers=nw,
        batch_size=bs,
    )

    keys: List[str] = []
    labels: List[str] = []
    chunks: List[np.ndarray] = []
    n_seen = 0
    for batch in loader:
        if batch is None:
            continue
        imgs, metas, bkeys = batch
        if imgs is None:
            continue
        # metas likely list[dict] when bs>1
        if isinstance(metas, dict):
            metas_list = [metas] * int(imgs.shape[0])
        else:
            metas_list = list(metas)

        img_feats = encode_images(plip, imgs)
        logits = score(plip, img_feats, tf.to(device=img_feats.device, dtype=img_feats.dtype))
        logits_np = logits.detach().cpu().float().numpy()

        # store
        for m, k in zip(metas_list, list(bkeys)):
            keys.append(str(k))
            labels.append(str(m.get(class_field, "UNKNOWN")))
        chunks.append(logits_np.astype(np.float32, copy=False))
        n_seen += int(logits_np.shape[0])
        if max_patches > 0 and n_seen >= max_patches:
            break

    scores = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, len(sel_idxs)), dtype=np.float32)
    keys_arr = np.asarray(keys, dtype=object)
    labels_arr = np.asarray(labels, dtype=object)

    # Save
    np.save(ARTIFACTS_DIR / "scores_fp32.npy", scores.astype(np.float32))
    np.save(ARTIFACTS_DIR / "keys.npy", keys_arr)
    np.save(ARTIFACTS_DIR / "labels.npy", labels_arr)
    atomic_write_json(ARTIFACTS_DIR / "selected_concepts.json", {"selected": sel_concepts})
    atomic_write_text(ARTIFACTS_DIR / "config_resolved.yaml", yaml.safe_dump(cfg, sort_keys=False))

    summary = {
        "n_samples": int(scores.shape[0]),
        "n_concepts_scored": int(scores.shape[1]),
        "use_shortlist_only": bool(use_shortlist_only),
        "test_dir": str(test_dir),
        "shortlist_yaml": str(shortlist_yaml),
    }
    atomic_write_json(ARTIFACTS_DIR / "summary.json", summary)

    # Plots (optional)
    out_cfg = cfg.get("output", {})
    if bool(out_cfg.get("plots", True)) and scores.shape[0] > 0:
        formats = tuple(out_cfg.get("formats", ["pdf", "png"]))
        dpi = int(out_cfg.get("plots_dpi", 300))
        concept_short = [c["concept_short_name"] for c in sel_concepts]
        _save_heatmap_mean_by_class(
            scores=scores,
            labels=labels_arr,
            concept_short=concept_short,
            out_base=PLOTS_DIR / "heatmap_mean_score_class_x_concept",
            formats=formats,
            dpi=dpi,
        )

    log.info("NO-ROI done (canonical): %s", ARTIFACTS_DIR.parent)
    log.info("  - scores: %s", ARTIFACTS_DIR / "scores_fp32.npy")
    log.info("  - keys  : %s", ARTIFACTS_DIR / "keys.npy")
    log.info("  - labels: %s", ARTIFACTS_DIR / "labels.npy")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
>>

__init__.py codice <<
# empty – marks "explainability" as a package
>>

paths.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


def get_repo_root() -> Path:
    """
    Resolve repository root.
    .../rcc-ssrl/src/explainability/paths.py -> parents[2] == .../rcc-ssrl
    """
    return Path(__file__).resolve().parents[2]


REPO_ROOT: Path = get_repo_root()
SRC_DIR: Path = REPO_ROOT / "src"
EXPLAINABILITY_DIR: Path = SRC_DIR / "explainability"

# Canonical config location
EXPLAINABILITY_CONFIGS_DIR: Path = EXPLAINABILITY_DIR / "configs"

# Canonical concept calibration metadata/config (inside repo)
CONCEPT_DIR: Path = EXPLAINABILITY_DIR / "concept"
CONCEPT_CALIBRATION_DIR: Path = CONCEPT_DIR / "calibration"
CONCEPT_CALIBRATION_METADATA_DIR: Path = CONCEPT_CALIBRATION_DIR / "metadata"
CALIBRATION_RESOLVED_CONFIG_YAML: Path = CONCEPT_CALIBRATION_METADATA_DIR / "config_resolved.yaml"

# ------------------------------------------------------------
# Canonical class naming (used for gating & reports)
# ------------------------------------------------------------

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
    # common variants (safe additions)
    "CCRCC": "ccRCC",
    "ccrcc": "ccRCC",
    "PRCC": "pRCC",
    "prcc": "pRCC",
}
DEFAULT_CLASSES = ["ccRCC", "pRCC", "CHROMO", "ONCO", "NOT_TUMOR"]

# Class naming utilities live here (single source of truth)
RCC_CLASS_ID_TO_NAME: Dict[int, str] = {
    0: "ccRCC",
    1: "pRCC",
    2: "chRCC",
    3: "oncocytoma",
    4: "other",
}

def rcc_class_name(class_id: int) -> str:
    return RCC_CLASS_ID_TO_NAME.get(int(class_id), f"class_{class_id}")


def _p(s: Union[str, Path]) -> Path:
    return Path(s).expanduser().resolve()


def _split_csv_env(v: Optional[str]) -> Optional[List[str]]:
    if not v:
        return None
    items = [x.strip() for x in v.split(",") if x.strip()]
    return items or None


def canonical_class_name(name: str) -> str:
    if name is None:
        return ""
    raw = str(name).strip()
    if not raw:
        return ""
    if raw in CLASS_ALIASES:
        return CLASS_ALIASES[raw]
    up = raw.upper()
    if up in CLASS_ALIASES:
        return CLASS_ALIASES[up]
    return raw


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _try_import_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except Exception:
        return None


def _default_project_root_from_file() -> Path:
    # paths.py is in: <repo>/src/explainability/paths.py => parents[2] == <repo>
    return get_repo_root()


# ------------------------------------------------------------
# Small structured containers for phase outputs
# ------------------------------------------------------------

@dataclass(frozen=True)
class PhaseOut:
    root: Path
    metadata: Path
    analysis: Path
    plots: Path
    report: Path

    def ensure(self) -> None:
        for d in (self.root, self.metadata, self.analysis, self.plots, self.report):
            _ensure_dir(d)


@dataclass(frozen=True)
class PaperBundle:
    root: Path
    metrics: Path
    plots: Path
    tables: Path
    manifest_json: Path

    def ensure(self) -> None:
        for d in (self.root, self.metrics, self.plots, self.tables):
            _ensure_dir(d)


# ------------------------------------------------------------
# Main "single source of truth" paths
# ------------------------------------------------------------

@dataclass(frozen=True)
class XAIPaths:
    # ---- repo roots ----
    project_root: Path
    explain_src_root: Path  # .../src/explainability

    # ---- central configs (YAML only) ----
    configs_dir: Path
    cfg_calibration_yaml: Path
    cfg_concept_no_roi_yaml: Path
    cfg_concept_roi_yaml: Path
    cfg_spatial_yaml: Path
    cfg_comparision_yaml: Path
    cfg_concepts_list_yaml: Path
    cfg_concepts_shortlist_yaml: Path  # produced/overwritten by calibration

    # ---- paper central bundle ----
    paper: PaperBundle

    # ---- artifact root (heavy outputs live here; env-overridable) ----
    xai_root: Path

    # ---- GLOBAL concept calibration (canonical, not per-model) ----
    concept_global_root: Path
    calib_out: PhaseOut
    calib_exemplars_dir: Path
    calib_onco_audit_dir: Path
    calib_shortlist_dir: Path
    calib_shortlist_json: Path
    calib_shortlist_yaml: Path
    calib_metrics_csv: Path
    calib_top_concepts_json: Path
    calib_validated_marker: Path
    calib_gaps_json: Path
    calib_gaps_md: Path

    # ---- GLOBAL caches (non model-dependent) ----
    concept_datasets_root: Path
    no_roi_patch_cache: Path
    roi_patch_cache: Path

    @staticmethod
    def from_env() -> "XAIPaths":
        # repo roots
        project_root = _p(os.environ.get("PROJECT_ROOT", str(_default_project_root_from_file())))
        explain_src_root = project_root / "src" / "explainability"

        # central configs
        configs_dir = explain_src_root / "configs"
        cfg_calibration_yaml = configs_dir / "calibration.yaml"
        cfg_comparision_yaml = configs_dir / "comparision.yaml"
        cfg_concept_no_roi_yaml = configs_dir / "concept_no_roi.yaml"
        cfg_concept_roi_yaml = configs_dir / "concept_roi.yaml"
        cfg_spatial_yaml = configs_dir / "config_xai.yaml"
        cfg_concepts_list_yaml = configs_dir / "concepts_list.yaml"
        cfg_concepts_shortlist_yaml = configs_dir / "concepts_shortlist.yaml"

        # paper bundle (centralized)
        paper_root = explain_src_root / "paper-productions"
        paper = PaperBundle(
            root=paper_root,
            metrics=paper_root / "metrics",
            plots=paper_root / "plots",
            tables=paper_root / "tables",
            manifest_json=paper_root / "manifest.json",
        )

        # heavy outputs root (outside repo by default)
        xai_root = _p(os.environ.get("XAI_ROOT", str(project_root / "outputs" / "xai")))

        # global concept root + calibration outputs
        concept_global_root = xai_root / "concept"
        calib_root = concept_global_root / "calibration"

        calib_out = PhaseOut(
            root=calib_root,
            metadata=calib_root / "metadata",
            analysis=calib_root / "analysis",
            plots=calib_root / "analysis" / "plots",
            report=calib_root / "analysis" / "report",
        )

        calib_exemplars_dir = calib_out.analysis / "exemplars"
        calib_onco_audit_dir = calib_out.analysis / "onco_audit"

        # shortlist: keep both JSON+YAML available (JSON is easiest to parse robustly)
        calib_shortlist_dir = configs_dir
        calib_shortlist_json = configs_dir / "concepts_shortlist.json"
        calib_shortlist_yaml = configs_dir / "concepts_shortlist.yaml"

        # canonical key files (paper-grade)
        calib_metrics_csv = calib_out.analysis / "metrics_per_class.csv"
        calib_top_concepts_json = calib_out.analysis / "top_concepts_by_class.json"
        calib_validated_marker = calib_out.analysis / "VALIDATED.ok"
        calib_gaps_json = calib_out.analysis / "shortlist_gaps.json"
        calib_gaps_md = calib_out.analysis / "shortlist_gaps.md"

        # global datasets cache (non model-dependent)
        concept_datasets_root = concept_global_root / "datasets"
        no_roi_patch_cache = concept_datasets_root / "no_roi_patches"
        roi_patch_cache = concept_datasets_root / "roi_patches"

        return XAIPaths(
            project_root=project_root,
            explain_src_root=explain_src_root,
            configs_dir=configs_dir,
            cfg_calibration_yaml=cfg_calibration_yaml,
            cfg_concept_no_roi_yaml=cfg_concept_no_roi_yaml,
            cfg_concept_roi_yaml=cfg_concept_roi_yaml,
            cfg_spatial_yaml=cfg_spatial_yaml,
            cfg_comparision_yaml=cfg_comparision_yaml,
            cfg_concepts_list_yaml=cfg_concepts_list_yaml,
            cfg_concepts_shortlist_yaml=cfg_concepts_shortlist_yaml,
            paper=paper,
            xai_root=xai_root,
            concept_global_root=concept_global_root,
            calib_out=calib_out,
            calib_exemplars_dir=calib_exemplars_dir,
            calib_onco_audit_dir=calib_onco_audit_dir,
            calib_shortlist_dir=calib_shortlist_dir,
            calib_shortlist_json=calib_shortlist_json,
            calib_shortlist_yaml=calib_shortlist_yaml,
            calib_metrics_csv=calib_metrics_csv,
            calib_top_concepts_json=calib_top_concepts_json,
            calib_validated_marker=calib_validated_marker,
            calib_gaps_json=calib_gaps_json,
            calib_gaps_md=calib_gaps_md,
            concept_datasets_root=concept_datasets_root,
            no_roi_patch_cache=no_roi_patch_cache,
            roi_patch_cache=roi_patch_cache,
        )

    # ------------------------------------------------------------
    # Ensure dirs (no side effects on import; call explicitly)
    # ------------------------------------------------------------

    def ensure_global_dirs(self) -> None:
        _ensure_dir(self.configs_dir)
        self.paper.ensure()
        self.calib_out.ensure()
        _ensure_dir(self.calib_exemplars_dir)
        _ensure_dir(self.calib_onco_audit_dir)
        _ensure_dir(self.calib_shortlist_dir)
        _ensure_dir(self.concept_datasets_root)
        _ensure_dir(self.no_roi_patch_cache)
        _ensure_dir(self.roi_patch_cache)

    # ------------------------------------------------------------
    # Model-dependent outputs: under the model run dir -> XAI/...
    # (you pass model_run_dir = training run folder / mlflow run folder)
    # ------------------------------------------------------------

    def model_xai_root(self, model_run_dir: Path) -> Path:
        return _p(model_run_dir) / "xai"

    def model_phase(self, model_run_dir: Path, phase: str) -> PhaseOut:
        """
        phase:
          - "concept_no_roi"
          - "concept_roi"
          - "spatial"
        """
        xai = self.model_xai_root(model_run_dir)
        if phase == "concept_no_roi":
            root = xai / "concept" / "no_roi"
        elif phase == "concept_roi":
            root = xai / "concept" / "roi"
        elif phase == "spatial":
            root = xai / "spatial"
        else:
            raise ValueError(f"Unknown phase: {phase}")

        return PhaseOut(
            root=root,
            metadata=root / "metadata",
            analysis=root / "analysis",
            plots=root / "analysis" / "plots",
            report=root / "analysis" / "report",
        )

    def ensure_model_dirs(self, model_run_dir: Path) -> None:
        for ph in ("concept_no_roi", "concept_roi", "spatial"):
            self.model_phase(model_run_dir, ph).ensure()

    # Convenience: commonly used output files per phase (model-dependent)
    def concept_no_roi_outputs(self, model_run_dir: Path) -> Dict[str, Path]:
        out = self.model_phase(model_run_dir, "concept_no_roi")
        return {
            "phase_root": out.root,
            "metadata_dir": out.metadata,
            "analysis_dir": out.analysis,
            "plots_dir": out.plots,
            "report_dir": out.report,
            "metrics_csv": out.analysis / "metrics_no_roi.csv",
            "top_concepts_json": out.analysis / "top_concepts_no_roi.json",
            "report_md": out.report / "report_no_roi.md",
            "config_resolved_yaml": out.metadata / "config_resolved.yaml",
        }

    def concept_roi_outputs(self, model_run_dir: Path) -> Dict[str, Path]:
        out = self.model_phase(model_run_dir, "concept_roi")
        return {
            "phase_root": out.root,
            "metadata_dir": out.metadata,
            "analysis_dir": out.analysis,
            "plots_dir": out.plots,
            "report_dir": out.report,
            "metrics_csv": out.analysis / "metrics_roi.csv",
            "top_concepts_json": out.analysis / "top_concepts_roi.json",
            "report_md": out.report / "report_roi.md",
            "config_resolved_yaml": out.metadata / "config_resolved.yaml",
        }

    def spatial_outputs(self, model_run_dir: Path) -> Dict[str, Path]:
        out = self.model_phase(model_run_dir, "spatial")
        return {
            "phase_root": out.root,
            "metadata_dir": out.metadata,
            "analysis_dir": out.analysis,
            "plots_dir": out.plots,
            "report_dir": out.report,
            "metrics_csv": out.analysis / "metrics_spatial.csv",
            "rollout_index_json": out.metadata / "rollout_index.json",
            "overlays_dir": out.analysis / "overlays",
            "report_md": out.report / "report_spatial.md",
            "config_resolved_yaml": out.metadata / "config_resolved.yaml",
        }

    # ------------------------------------------------------------
    # Expected classes (override via env if needed)
    # ------------------------------------------------------------

    def expected_classes(self) -> List[str]:
        env = _split_csv_env(os.environ.get("XAI_EXPECTED_CLASSES"))
        if env:
            return [canonical_class_name(x) for x in env]
        return list(DEFAULT_CLASSES)

    # ------------------------------------------------------------
    # Shortlist loading + gating
    # ------------------------------------------------------------

    def _load_shortlist_obj(self) -> Dict[str, Any]:
        """
        Loads shortlist from:
          1) calibration analysis shortlist JSON (preferred)
          2) calibration analysis shortlist YAML
          3) central configs concepts_shortlist.yaml (fallback)
        """
        if self.calib_shortlist_json.exists():
            return json.loads(self.calib_shortlist_json.read_text())

        yaml_mod = _try_import_yaml()
        if yaml_mod is not None:
            if self.calib_shortlist_yaml.exists():
                return yaml_mod.safe_load(self.calib_shortlist_yaml.read_text()) or {}
            if self.cfg_concepts_shortlist_yaml.exists():
                return yaml_mod.safe_load(self.cfg_concepts_shortlist_yaml.read_text()) or {}

        raise FileNotFoundError(
            "Missing shortlist artifacts. Expected one of:\n"
            f"- {self.calib_shortlist_json}\n"
            f"- {self.calib_shortlist_yaml}\n"
            f"- {self.cfg_concepts_shortlist_yaml} (fallback)\n"
            "Run calibration + build_shortlist first."
        )

    @staticmethod
    def _extract_by_class(shortlist_obj: Any) -> Dict[str, Any]:
        """
        Accepts multiple formats, returns mapping:
          { "ONCO": <concepts>, "CHROMO": <concepts>, ... }
        where <concepts> can be list/dict/etc.
        """
        if shortlist_obj is None:
            return {}

        if isinstance(shortlist_obj, dict):
            for k in ("by_class", "shortlist_by_class", "classes", "per_class"):
                if k in shortlist_obj and isinstance(shortlist_obj[k], dict):
                    return shortlist_obj[k]
            # Some pipelines store directly as {class: concepts}
            # Heuristic: if keys look like classes and values like list/dict, accept as-is
            return shortlist_obj

        if isinstance(shortlist_obj, list):
            # list of {class:..., concepts:[...]} or {class:..., items:{...}}
            out: Dict[str, Any] = {}
            for item in shortlist_obj:
                if not isinstance(item, dict):
                    continue
                c = item.get("class") or item.get("label") or item.get("name")
                if not c:
                    continue
                concepts = item.get("concepts") or item.get("items") or item.get("values") or []
                out[str(c)] = concepts
            return out

        return {}

    @staticmethod
    def _count_concepts(concepts: Any) -> int:
        if concepts is None:
            return 0
        if isinstance(concepts, list):
            return len(concepts)
        if isinstance(concepts, dict):
            return len(concepts.keys())
        return 0

    def assert_calibration_ready(
        self,
        expected_classes: Optional[List[str]] = None,
        min_concepts_per_class: int = 5,
        require_validated_marker: bool = True,
        write_gap_reports: bool = True,
    ) -> None:
        """
        Hard gate BEFORE concept no-ROI/ROI.
        Conditions:
          - metrics_per_class.csv exists (calibration produced analysis)
          - VALIDATED.ok exists (optional but strongly recommended)
          - shortlist covers all expected classes with >= min_concepts_per_class
        If fails, raises RuntimeError and writes shortlist_gaps.(json|md) in calibration analysis.
        """
        expected = [canonical_class_name(c) for c in (expected_classes or self.expected_classes())]

        if not self.calib_metrics_csv.exists():
            raise FileNotFoundError(f"Missing calibration metrics: {self.calib_metrics_csv}")

        if require_validated_marker and not self.calib_validated_marker.exists():
            raise RuntimeError(
                "Calibration NOT validated.\n"
                f"Missing marker: {self.calib_validated_marker}\n"
                "Expected outcome: STOP before no-ROI/ROI.\n"
                "Fix: run calibration_validation (it must write VALIDATED.ok) then rebuild shortlist."
            )

        sl_obj = self._load_shortlist_obj()
        by_class_raw = self._extract_by_class(sl_obj)

        # canonicalize keys
        by_class: Dict[str, Any] = {}
        for k, v in by_class_raw.items():
            ck = canonical_class_name(str(k))
            if not ck:
                continue
            by_class[ck] = v

        missing: List[str] = []
        too_small: List[Tuple[str, int]] = []

        for c in expected:
            concepts = by_class.get(c)
            n = self._count_concepts(concepts)
            if n <= 0:
                missing.append(c)
            elif n < int(min_concepts_per_class):
                too_small.append((c, n))

        if missing or too_small:
            gap_payload = {
                "expected_classes": expected,
                "min_concepts_per_class": int(min_concepts_per_class),
                "missing_classes": missing,
                "too_small": [{"class": c, "n": n} for (c, n) in too_small],
                "notes": [
                    "Pipeline must STOP before no-ROI/ROI when shortlist coverage is incomplete.",
                    "Fix options: relax shortlist thresholds; expand ontology/synonyms; fix NaN metrics; rerun calibration+validation+build_shortlist.",
                ],
            }

            if write_gap_reports:
                _ensure_dir(self.calib_out.analysis)
                self.calib_gaps_json.write_text(json.dumps(gap_payload, indent=2))

                lines = []
                lines.append("# Shortlist coverage FAILED\n")
                lines.append(f"- min_concepts_per_class: {min_concepts_per_class}")
                lines.append(f"- expected_classes: {expected}\n")
                if missing:
                    lines.append("## Missing classes")
                    for c in missing:
                        lines.append(f"- {c}")
                    lines.append("")
                if too_small:
                    lines.append("## Too small")
                    for c, n in too_small:
                        lines.append(f"- {c}: {n} (< {min_concepts_per_class})")
                    lines.append("")
                lines.append("## Expected outcome")
                lines.append("- STOP before running concept no-ROI/ROI.\n")
                lines.append("## Fix")
                lines.append("- Relax shortlist thresholds OR expand ontology/synonyms; then rerun calibration+validation+build_shortlist.\n")
                self.calib_gaps_md.write_text("\n".join(lines))

            msg = ["Shortlist coverage check FAILED:"]
            if missing:
                msg.append(f"- Missing concepts for classes: {missing}")
            for c, n in too_small:
                msg.append(f"- Class {c} has only {n} concepts (< {min_concepts_per_class})")
            msg.append("Expected outcome: stop BEFORE no-ROI/ROI.")
            msg.append(f"See: {self.calib_gaps_json} and {self.calib_gaps_md}")
            raise RuntimeError("\n".join(msg))

    # ------------------------------------------------------------
    # Optional helper: dump for debugging
    # ------------------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "explain_src_root": str(self.explain_src_root),
            "xai_root": str(self.xai_root),
            "configs_dir": str(self.configs_dir),
            "paper_root": str(self.paper.root),
            "calibration_root": str(self.calib_out.root),
        }


# ------------------------------------------------------------
# Global singleton (single source of truth)
# ------------------------------------------------------------

PATHS = XAIPaths.from_env()
CONFIG_DIR: Path = PATHS.configs_dir


@dataclass(frozen=True)
class CalibrationLayout:
    root: Path
    metadata_dir: Path
    analysis_dir: Path
    report_dir: Path
    configs_dir: Path
    shortlist_dir: Path
    shortlist_json: Path
    shortlist_yaml: Path


def _calibration_layout(paths: XAIPaths) -> CalibrationLayout:
    return CalibrationLayout(
        root=paths.calib_out.root,
        metadata_dir=paths.calib_out.metadata,
        analysis_dir=paths.calib_out.analysis,
        report_dir=paths.calib_out.report,
        configs_dir=paths.configs_dir,
        shortlist_dir=paths.calib_shortlist_dir,
        shortlist_json=paths.calib_shortlist_json,
        shortlist_yaml=paths.calib_shortlist_yaml,
    )


CALIBRATION_PATHS = _calibration_layout(PATHS)


def ensure_calibration_layout(layout: CalibrationLayout = CALIBRATION_PATHS) -> CalibrationLayout:
    layout_dirs = (
        layout.root,
        layout.metadata_dir,
        layout.analysis_dir,
        layout.report_dir,
        layout.configs_dir,
        layout.shortlist_dir,
    )
    for d in layout_dirs:
        _ensure_dir(d)
    return layout


@dataclass(frozen=True)
class NoROILayout:
    root: Path
    artifacts_dir: Path
    plots_dir: Path
    logs_dir: Path


def _no_roi_layout(paths: XAIPaths) -> NoROILayout:
    root = paths.concept_global_root / "no_roi"
    return NoROILayout(
        root=root,
        artifacts_dir=root / "artifacts",
        plots_dir=root / "plots",
        logs_dir=root / "logs",
    )


NO_ROI_PATHS = _no_roi_layout(PATHS)


def ensure_no_roi_layout(layout: NoROILayout = NO_ROI_PATHS) -> NoROILayout:
    for d in (layout.root, layout.artifacts_dir, layout.plots_dir, layout.logs_dir):
        _ensure_dir(d)
    return layout


@dataclass(frozen=True)
class RoiLayout:
    root: Path
    artifacts_dir: Path
    rois_dir: Path
    figures_dir: Path
    logs_dir: Path


def roi_layout(model_root: Path) -> RoiLayout:
    root = _p(model_root) / "xai" / "concept" / "roi"
    return RoiLayout(
        root=root,
        artifacts_dir=root / "artifacts",
        rois_dir=root / "rois",
        figures_dir=root / "figures",
        logs_dir=root / "logs",
    )


def ensure_roi_layout(model_root: Path) -> RoiLayout:
    layout = roi_layout(model_root)
    for d in (layout.root, layout.artifacts_dir, layout.rois_dir, layout.figures_dir, layout.logs_dir):
        _ensure_dir(d)
    return layout


@dataclass(frozen=True)
class ComparisonLayout:
    root: Path
    figures_dir: Path
    summary_csv: Path
    report_md: Path


def comparison_layout(model_id: str, paths: XAIPaths = PATHS) -> ComparisonLayout:
    root = paths.concept_global_root / "comparision" / str(model_id)
    tables_dir = root / "tables"
    return ComparisonLayout(
        root=root,
        figures_dir=root / "figures",
        summary_csv=tables_dir / "roi_vs_no_roi_summary.csv",
        report_md=root / "report.md",
    )


def ensure_comparison_layout(model_id: str, paths: XAIPaths = PATHS) -> ComparisonLayout:
    layout = comparison_layout(model_id, paths=paths)
    _ensure_dir(layout.figures_dir)
    _ensure_dir(layout.summary_csv.parent)
    _ensure_dir(layout.report_md.parent)
    return layout
>>

run_xai_pipeline.sh codice <<
#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# XAI pipeline (canonical, phased, deterministic)
#
# Requested phase logic (single source of truth):
#   Phase 1: Calibration on TRAIN+VAL  -> deep validation -> shortlist + report -> strict validation
#   Phase 2: Concept XAI on TEST (NO-ROI) (model-independent, computed once)
#   Phase 3: Spatial XAI on TEST (per model / ablation) (produces rollout + ROI masks)
#   Phase 4: ROI creation + Concept XAI on TEST (ROI) (model-dependent)
#   Phase 5: Compare NO-ROI vs ROI (per model)
#
# This script enforces input/output order and avoids hard-failing on missing
# optional config files by falling back to defaults/env when possible.
# ============================================================================

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] [INFO] $*"; }
err() { echo "[$(ts)] [ERROR] $*" >&2; }
die() { err "$*"; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

CFG_DIR="${PROJECT_ROOT}/src/explainability/configs"
OUT_ROOT="${PROJECT_ROOT}/outputs/xai"

# Canonical calibration layout (matches explainability.paths)
CAL_META="${OUT_ROOT}/concept/calibration/metadata"
CAL_ANAL="${OUT_ROOT}/concept/calibration/analysis"
CAL_REPORT="${CAL_ANAL}/report"
SHORTLIST_YAML="${CFG_DIR}/concepts_shortlist.yaml"

LOG_LEVEL="${LOG_LEVEL:-INFO}"

# ---- Config candidates (do NOT assume files exist) -------------------------
CAL_CFG="${CAL_CFG:-${CFG_DIR}/calibration.yaml}"

# NO-ROI config is OPTIONAL: the runner can operate from env/defaults.
NO_ROI_CFG_DEFAULT_A="${CFG_DIR}/concept_no_roi.yaml"          # old name (missing in your repo)
NO_ROI_CFG_DEFAULT_B="${CFG_DIR}/spatial-concept.yaml"         # existing (often contains shared WDS/plip fields)
NO_ROI_CFG_DEFAULT_C="${CFG_DIR}/config_concept_plip.yaml"     # existing (plip-centric)

SPATIAL_CFG="${SPATIAL_CFG:-${CFG_DIR}/spatial.yaml}"
ROI_CFG="${ROI_CFG:-${CFG_DIR}/spatial-concept.yaml}"

SHORTLIST_K_PRIMARY="${SHORTLIST_K_PRIMARY:-8}"
SHORTLIST_K_CONFOUNDS="${SHORTLIST_K_CONFOUNDS:-5}"
SHORTLIST_RANK_BY="${SHORTLIST_RANK_BY:-auc_ovr}"
SHORTLIST_MIN_AUC="${SHORTLIST_MIN_AUC:-0.60}"

STRICT_VALIDATE="${STRICT_VALIDATE:-1}"   # 1 => --strict, 0 => non-strict

PHASES_DEFAULT="calibration audit no_roi spatial roi comparision"
PHASES="${XAI_PHASES:-$PHASES_DEFAULT}"

resolve_first_existing() {
  # usage: resolve_first_existing <path1> <path2> ...
  for p in "$@"; do
    if [[ -n "${p}" && -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

require_file() {
  local p="$1"
  local label="$2"
  [[ -f "$p" ]] || die "Missing file (${label}): ${p}"
}

require_dir() {
  local p="$1"
  local label="$2"
  [[ -d "$p" ]] || die "Missing dir (${label}): ${p}"
}

run_py() {
  log "RUN: $*"
  "$@"
}

phase_banner() {
  local name="$1"
  local desc="$2"
  echo
  echo "============================================================"
  echo "[${name}] ${desc}"
  echo "============================================================"
}

log "Host: $(hostname -f 2>/dev/null || hostname)"
log "Project root: ${PROJECT_ROOT}"
log "Configs: ${CFG_DIR}"
log "Outputs: ${OUT_ROOT}"
log "Planned phases: ${PHASES}"

# Export common paths for python runners that support env overrides.
export CALIBRATION_METADATA_DIR="${CAL_META}"
export CALIBRATION_ANALYSIS_DIR="${CAL_ANAL}"
export CALIBRATION_REPORT_DIR="${CAL_REPORT}"
export CONCEPT_SHORTLIST_YAML="${SHORTLIST_YAML}"

# ----------------------------------------------------------------------------
# Phase 1 (Calibration + audit bundle): calibration -> deep validation -> shortlist -> validate
# ----------------------------------------------------------------------------
if [[ " ${PHASES} " == *" calibration "* || " ${PHASES} " == *" audit "* ]]; then
  phase_banner "PHASE 1" "Calibration + deep validation + shortlist/report + validation"

  require_file "${CAL_CFG}" "CAL_CFG"

  log "Inputs : CAL_CFG=${CAL_CFG}"
  log "Outputs: CAL_META=${CAL_META} ; CAL_ANAL=${CAL_ANAL}"

  run_py python -m explainability.concept.calibration.run_calibration \
    --config "${CAL_CFG}" --log-level "${LOG_LEVEL}"

  require_dir "${CAL_META}" "CAL_META"

  run_py python -m explainability.concept.calibration.calibration_validation \
    --metadata-dir "${CAL_META}" \
    --out-dir "${CAL_ANAL}" \
    --compute-auc \
    --require-onco \
    --log-level "${LOG_LEVEL}"

  require_file "${CAL_ANAL}/metrics_per_class.csv" "metrics_per_class.csv"

  run_py python -m explainability.concept.calibration.build_shortlist \
    --metrics-csv "${CAL_ANAL}/metrics_per_class.csv" \
    --out-dir "${CFG_DIR}" \
    --report-dir "${CAL_REPORT}" \
    --rank-by "${SHORTLIST_RANK_BY}" \
    --k-primary "${SHORTLIST_K_PRIMARY}" \
    --k-confounds "${SHORTLIST_K_CONFOUNDS}" \
    --min-auc "${SHORTLIST_MIN_AUC}" \
    --write-tex

  require_file "${SHORTLIST_YAML}" "SHORTLIST_YAML"

  if [[ "${STRICT_VALIDATE}" == "1" ]]; then
    run_py python -m explainability.concept.calibration.validate_artifacts --strict
  else
    run_py python -m explainability.concept.calibration.validate_artifacts
  fi
fi

# ----------------------------------------------------------------------------
# Phase 2 (Concept NO-ROI on TEST): model-independent, computed once
# ----------------------------------------------------------------------------
if [[ " ${PHASES} " == *" no_roi "* ]]; then
  phase_banner "PHASE 2" "Concept XAI on TEST (NO-ROI, model-independent)"

  # Config is OPTIONAL. If missing, run_no_roi will fall back to env/defaults.
  NO_ROI_CFG_RESOLVED="$(resolve_first_existing \
    "${NO_ROI_CFG:-}" \
    "${NO_ROI_CFG_DEFAULT_A}" \
    "${NO_ROI_CFG_DEFAULT_B}" \
    "${NO_ROI_CFG_DEFAULT_C}" \
    || true)"

  log "Inputs : calibration_metadata=${CAL_META} ; shortlist=${SHORTLIST_YAML} ; TEST via (config or WDS_TEST_DIR)"
  if [[ -n "${NO_ROI_CFG_RESOLVED}" ]]; then
    log "Config  : NO_ROI_CFG=${NO_ROI_CFG_RESOLVED}"
    run_py python -m explainability.concept.run_no_roi \
      --config "${NO_ROI_CFG_RESOLVED}" \
      --log-level "${LOG_LEVEL}"
  else
    log "Config  : NO_ROI_CFG=<none> (falling back to env/defaults; set WDS_TEST_DIR at minimum)"
    run_py python -m explainability.concept.run_no_roi \
      --log-level "${LOG_LEVEL}"
  fi
fi

# ----------------------------------------------------------------------------
# Phase 3 (Spatial on TEST): per model/ablation
# ----------------------------------------------------------------------------
if [[ " ${PHASES} " == *" spatial "* ]]; then
  phase_banner "PHASE 3" "Spatial XAI on TEST (per model/ablation)"

  # You MUST provide which model runs to process.
  # Keep it explicit to avoid silently running on the wrong checkpoint.
  #
  # Expected env:
  #   SPATIAL_RUNS="/abs/path/runA,/abs/path/runB,..."
  #
  # If your spatial runner instead reads everything from the YAML, set:
  #   SPATIAL_RUNS="__FROM_CONFIG__"
  #
  require_file "${SPATIAL_CFG}" "SPATIAL_CFG"

  if [[ -z "${SPATIAL_RUNS:-}" ]]; then
    log "SKIP spatial: SPATIAL_RUNS is empty. Set SPATIAL_RUNS to run dirs (comma-separated) or '__FROM_CONFIG__'."
  elif [[ "${SPATIAL_RUNS}" == "__FROM_CONFIG__" ]]; then
    run_py python -m explainability.spatial.xai_spatial \
      --config "${SPATIAL_CFG}" \
      --log-level "${LOG_LEVEL}"
  else
    IFS=',' read -r -a RUNS <<< "${SPATIAL_RUNS}"
    for r in "${RUNS[@]}"; do
      rr="$(echo "${r}" | xargs)"
      [[ -z "${rr}" ]] && continue
      # NOTE: we pass the run dir explicitly; if your spatial runner uses a different flag,
      # adjust here (this is intentionally the only place you should touch).
      run_py python -m explainability.spatial.xai_spatial \
        --config "${SPATIAL_CFG}" \
        --model-run "${rr}" \
        --log-level "${LOG_LEVEL}"
    done
  fi
fi

# ----------------------------------------------------------------------------
# Phase 4 (ROI + Concept ROI): depends on spatial outputs
# ----------------------------------------------------------------------------
if [[ " ${PHASES} " == *" roi "* ]]; then
  phase_banner "PHASE 4" "ROI creation + Concept XAI on TEST (ROI, model-dependent)"

  require_file "${ROI_CFG}" "ROI_CFG"
  run_py python -m explainability.concept.run_roi \
    --config "${ROI_CFG}" \
    --log-level "${LOG_LEVEL}"
fi

# ----------------------------------------------------------------------------
# Phase 5 (Comparison): compare no-roi vs roi
# ----------------------------------------------------------------------------
if [[ " ${PHASES} " == *" comparision "* ]]; then
  phase_banner "PHASE 5" "Comparison: NO-ROI vs ROI (per model)"

  # Keep backward-compat spelling: run_comparision.py + folder comparision/
  require_file "${ROI_CFG}" "ROI_CFG"
  run_py python -m explainability.concept.run_comparision \
    --config "${ROI_CFG}" \
    --log-level "${LOG_LEVEL}"
fi

log "Done."
>>

spatial/eval_utils.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Common utilities for explainability:
 - logging and seeding
 - basic image preprocessing
 - evaluation artifacts loading (predictions + logits)
 - selection of TP/FP/FN and low-confidence cases
 - WebDataset loader with keys (supports batch_size>1)
 - atomic writers (json/csv) for idempotent pipelines
"""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms, datasets

try:
    import webdataset as wds
    HAVE_WDS = True
except Exception:
    HAVE_WDS = False


# -------------------------------------------------------------------------
# Logging / reproducibility
# -------------------------------------------------------------------------
def setup_logger(name: str = "explainability") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------------
# Atomic writers (avoid partial files in HPC preemptions)
# -------------------------------------------------------------------------
def ensure_dir(p: str | Path) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding=encoding) as f:
        f.write(text)
        tmp = Path(f.name)
    tmp.replace(path)


def atomic_write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    atomic_write_text(Path(path), json.dumps(obj, indent=indent, ensure_ascii=False) + "\n")


def atomic_write_csv(path: str | Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), newline="") as f:
        tmp = Path(f.name)
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    tmp.replace(path)


# -------------------------------------------------------------------------
# Image transforms
# -------------------------------------------------------------------------
def build_preprocess(img_size: int, imagenet_norm: bool = True) -> transforms.Compose:
    ops: List[Any] = [
        transforms.Resize(
            img_size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ]
    if imagenet_norm:
        ops.append(
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            )
        )
    return transforms.Compose(ops)


def tensor_to_pil(t: torch.Tensor, imagenet_norm: bool = True) -> Image.Image:
    t = t.detach().cpu()
    if imagenet_norm:
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        t = t * std + mean
    t = t.clamp(0.0, 1.0)
    return transforms.ToPILImage()(t)


# -------------------------------------------------------------------------
# Eval artifacts (predictions.csv + logits_test.npy)
# -------------------------------------------------------------------------
def load_eval_artifacts(
    eval_dir: str | Path,
    pred_csv: str,
    logits_npy: str,
    logger: logging.Logger,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[List[str]],
    Optional[List[Dict[str, Any]]],
]:
    """Load y_true / y_pred / confidence / wds_key / full rows from eval output."""
    eval_dir = Path(eval_dir)
    y_true = y_pred = conf = None
    keys: Optional[List[str]] = None
    meta_rows: Optional[List[Dict[str, Any]]] = None

    pcsv = eval_dir / pred_csv
    if pcsv.exists():
        yt, yp, kk, rows = [], [], [], []
        with pcsv.open() as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            has_key = "wds_key" in fields
            for row in reader:
                t = row.get("y_true", "")
                yt.append(int(t) if str(t).strip() != "" else -1)
                yp.append(int(row["y_pred"]))
                kk.append(row["wds_key"] if has_key else None)
                rows.append(row)
        y_true = np.array(yt)
        y_pred = np.array(yp)
        keys = kk if any(k is not None for k in kk) else None
        meta_rows = rows
        logger.info(f"Loaded predictions.csv with {len(yp)} rows from {pcsv}")
    else:
        logger.warning(f"predictions.csv not found: {pcsv}")

    plog = eval_dir / logits_npy
    if plog.exists():
        logits = np.load(plog)
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = ex / ex.sum(axis=1, keepdims=True)
        conf = probs.max(axis=1)
        logger.info(f"Loaded logits from: {plog}")
    else:
        logger.warning(f"logits npy not found: {plog}")

    return y_true, y_pred, conf, keys, meta_rows


# -------------------------------------------------------------------------
# Selection logic (TP / FP / FN / low-confidence)
# -------------------------------------------------------------------------
def select_items(
    y_true: Optional[np.ndarray],
    y_pred: Optional[np.ndarray],
    conf: Optional[np.ndarray],
    keys: Optional[List[str]],
    n_classes: int,
    cfg_sel: Dict[str, Any],
    logger: logging.Logger,
):
    """
    Select indices to explain and track selection reasons.

    Returns
    -------
    targets : List[str] or List[int]
        Selected wds_keys (if keys is not None) or raw indices.
    reasons : Dict[str, List[str]] or Dict[int, List[str]]
        Map from wds_key (or index) -> list of selection reasons.
    """
    if y_pred is None:
        logger.warning("No predictions available; selection is empty.")
        return [], {}

    def pick(arr, k, by_conf=None, reverse=True):
        if len(arr) == 0 or k <= 0:
            return []
        idx = np.asarray(arr, dtype=int)
        if by_conf is not None:
            # safety: conf shape check
            if by_conf.shape[0] <= idx.max():
                logger.warning(
                    "Confidence array shorter than indices; ignoring confidence ordering."
                )
            else:
                order = np.argsort(by_conf[idx])
                if reverse:
                    order = order[::-1]
                idx = idx[order]
        return idx[:k].tolist()

    items: List[int] = []
    reason_by_idx: Dict[int, set[str]] = {}

    def add_reason(idx: int, reason: str):
        if idx not in reason_by_idx:
            reason_by_idx[idx] = set()
        reason_by_idx[idx].add(reason)

    # ------------------------------------------------------------------
    # Per-class TP / FP / FN
    # ------------------------------------------------------------------
    for c in range(n_classes):
        idx_c = np.where(y_true == c)[0] if y_true is not None else np.array([], dtype=int)

        # High-confidence TP
        if idx_c.size > 0:
            tpc = idx_c[y_pred[idx_c] == c]
        else:
            tpc = np.array([], dtype=int)

        chosen_tp = pick(
            tpc,
            cfg_sel["per_class"].get("topk_tp", 0),
            by_conf=conf,
            reverse=True,
        )
        items += chosen_tp
        for i in chosen_tp:
            add_reason(i, "tp_high_conf")

        # FN
        if idx_c.size > 0:
            fnc = idx_c[y_pred[idx_c] != c]
        else:
            fnc = np.array([], dtype=int)

        chosen_fn = pick(
            fnc,
            cfg_sel["per_class"].get("topk_fn", 0),
            by_conf=conf,
            reverse=False,  # lowest confidence among wrong
        )
        items += chosen_fn
        for i in chosen_fn:
            # reverse=False => low confidence among wrong predictions
            add_reason(i, "fn_low_conf")

        # FP
        idx_pred_c = np.where(y_pred == c)[0]
        if y_true is not None and idx_pred_c.size > 0:
            fpc = idx_pred_c[y_true[idx_pred_c] != c]
        else:
            fpc = idx_pred_c

        chosen_fp = pick(
            fpc,
            cfg_sel["per_class"].get("topk_fp", 0),
            by_conf=conf,
            reverse=True,
        )
        items += chosen_fp
        for i in chosen_fp:
            add_reason(i, "fp_high_conf")

    # ------------------------------------------------------------------
    # Globally low-confidence cases (optional)
    # ------------------------------------------------------------------
    if conf is not None and "global_low_conf" in cfg_sel:
        n_low = cfg_sel["global_low_conf"].get("topk", 0)
        if n_low > 0:
            order = np.argsort(conf)  # ascending → lowest confidence first
            chosen_low = order[:n_low].tolist()
            items += chosen_low
            for i in chosen_low:
                add_reason(i, "low_conf")

    # Dedup preserving order
    seen = set()
    unique_items: List[int] = []
    for i in items:
        if i not in seen:
            seen.add(i)
            unique_items.append(i)

    if keys is not None:
        targets = [keys[i] for i in unique_items]
        reasons = {
            keys[i]: sorted(list(reason_by_idx.get(i, [])))
            for i in unique_items
        }
    else:
        targets = unique_items
        reasons = {
            i: sorted(list(reason_by_idx.get(i, [])))
            for i in unique_items
        }

    logger.info(f"Selected {len(targets)} items for XAI.")
    return targets, reasons


# -------------------------------------------------------------------------
# WebDataset helper: filter by key set in one streaming pass
# -------------------------------------------------------------------------
def iter_wds_filtered_by_keys(
    loader,
    wanted: set[str],
    *,
    key_prefix_strip: Optional[str] = None,
):
    """
    Iterate a WDS loader and yield only samples whose key is in `wanted`.

    Notes
    -----
    - Works for batch_size==1 (single sample) and batch_size>1 (batched).
    - key_prefix_strip: if provided, strips e.g. "test::" from keys before matching.
    """
    if not wanted:
        return

    def _canon_key(k: Any) -> str:
        s = str(k)
        if key_prefix_strip and s.startswith(key_prefix_strip):
            s = s[len(key_prefix_strip):]
        return s

    for batch in loader:
        if batch is None:
            continue
        # batch_size==1: (img, meta, key)
        if isinstance(batch, (list, tuple)) and len(batch) == 3 and not torch.is_tensor(batch[2]):
            img, meta, key = batch
            kk = _canon_key(key)
            if kk in wanted:
                yield img, meta, kk
            continue

        # batch_size>1: (imgs[B,...], metas, keys)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            imgs, metas, keys = batch
            # keys might be list[str] or tuple[str]
            for i, k in enumerate(list(keys)):
                kk = _canon_key(k)
                if kk in wanted:
                    mi = metas[i] if isinstance(metas, (list, tuple)) else metas
                    yield imgs[i], mi, kk


# -------------------------------------------------------------------------
# Data loaders
# -------------------------------------------------------------------------
def make_wds_loader_with_keys(
    test_dir: str,
    pattern: str,
    image_key: str,
    meta_key: str,
    preprocess_fn,
    num_workers: int,
    batch_size: int = 1,
):
    """
    Create a WebDataset loader that yields:
      - batch_size==1: (image_tensor, meta, key)
      - batch_size>1 : (images_tensor[B,...], metas, keys)
    """
    if not HAVE_WDS:
        raise RuntimeError("webdataset not available; install it for explainability.")
    import glob

    shard_glob = str(Path(test_dir) / pattern)
    shards = sorted(glob.glob(shard_glob))
    if not shards:
        raise FileNotFoundError(f"No shards found: {shard_glob}")

    ds = (
        wds.WebDataset(
            shards,
            shardshuffle=False,
            handler=wds.warn_and_continue,
            empty_check=False,
        )
        .decode("pil")
        .to_tuple(image_key, meta_key, "__key__", handler=wds.warn_and_continue)
        .map_tuple(preprocess_fn, lambda x: x, lambda x: x)
    )

    def _collate_first(batch):
        if not batch:
            return None
        return batch[0]

    collate_fn = _collate_first if int(batch_size) == 1 else None
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(batch_size),
        num_workers=min(num_workers, len(shards)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    return loader


def make_imgfolder_loader(
    test_dir: str, preprocess_fn, batch_size: int, num_workers: int
):
    """Fallback loader for ImageFolder datasets (not WebDataset)."""
    ds = datasets.ImageFolder(test_dir, transform=preprocess_fn)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return ds, loader
>>

spatial/__init__.py codice <<
# empty on purpose – marks "common" as a package
>>

spatial/ssl_linear_loader.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to build an SSL classifier from:
 - ResNetBackbone or ViTBackbone weights saved inside SSL checkpoints
 - A linear head checkpoint saved by the linear probe trainer.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ---- Gestione dipendenza TIMM per ViT ----
try:
    import timm
    HAVE_TIMM = True
except ImportError:
    HAVE_TIMM = False

# ---- Backbone Definitions ----

class ResNetBackbone(nn.Module):
    def __init__(self, name: str="resnet50", pretrained: bool=False):
        super().__init__()
        from torchvision import models
        if "34" in name: m = models.resnet34(weights=None)
        else:            m = models.resnet50(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.out_dim = m.fc.in_features
        
    def _fwd(self, x): 
        x = self.stem(x); l1 = self.layer1(x); l2 = self.layer2(l1); l3 = self.layer3(l2); l4 = self.layer4(l3); return l4
    
    def forward_global(self, x): 
        feats = self._fwd(x)
        return torch.flatten(F.adaptive_avg_pool2d(feats, 1), 1)

class _VitBackbone(nn.Module):
    def __init__(self, name: str="vit_small_patch16_224"):
        super().__init__()
        if not HAVE_TIMM:
            raise RuntimeError("timm is required for ViT backbones. Install it with `pip install timm`.")
        # dynamic_img_size=True è fondamentale per XAI/Inference su size diverse
        self.model = timm.create_model(name, pretrained=False, num_classes=0, dynamic_img_size=True)
        self.out_dim = self.model.num_features

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)
        # Output timm standard: [B, T, C]. MoCo v3 usa il CLS token (indice 0)
        if feats.dim() == 3:
            return feats[:, 0] 
        # Fallback per architetture che fanno pool interno
        return torch.flatten(torch.nn.functional.adaptive_avg_pool2d(feats, 1), 1)

# ---- Loader Logic ----

_PREFIXES = ("stu.", "backbone_q.", "student.", "backbone.", "module.stu.", "module.backbone_q.", "base_encoder.")

def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

def _best_substate(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # pick the sub-dict with most keys among known prefixes
    cands = [(_strip_prefix(sd, p), p) for p in _PREFIXES]
    # Add the raw dict as a candidate (prefix "")
    cands.append((sd, ""))
    
    best_dict, best_prefix = max(cands, key=lambda x: len(x[0]))
    return best_dict

def _load_torch_state(path: str) -> Dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu") # Fallback older pytorch
        
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    return payload if isinstance(payload, dict) else {}

class SSLLinearClassifier(nn.Module):
    """
    Compose a Backbone (ResNet or ViT) with a linear head.
    """
    def __init__(self, backbone_name: str="resnet50", num_classes: int=5):
        super().__init__()
        if "vit" in backbone_name.lower():
            self.backbone = _VitBackbone(backbone_name)
        else:
            self.backbone = ResNetBackbone(backbone_name, pretrained=False)
            
        self.head = nn.Linear(self.backbone.out_dim, num_classes)

    def load_backbone_from_ssl(self, ssl_backbone_ckpt: str) -> Tuple[int, int]:
        sd = _load_torch_state(ssl_backbone_ckpt)
        sd = _best_substate(sd)
        
        # Gestione specifica per ViT timm vs MoCo naming
        # A volte MoCo salva come 'module.base_encoder.model.blocks...' vs 'blocks...'
        if isinstance(self.backbone, _VitBackbone):
            new_sd = {}
            for k, v in sd.items():
                # Rimuovi 'model.' se presente (comune in wrapper timm salvati male)
                if k.startswith("model."):
                    k = k[6:]
                new_sd[k] = v
            sd = new_sd

        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        return len(missing), len(unexpected)

    def load_head_from_probe(self, ssl_head_ckpt: str) -> Tuple[int, int]:
        hd = torch.load(ssl_head_ckpt, map_location="cpu")
        if isinstance(hd, dict) and "state_dict" in hd:
            hd = hd["state_dict"]
        missing, unexpected = self.head.load_state_dict(hd, strict=False)
        return len(missing), len(unexpected)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_global(x)
        return self.head(feats)
>>

spatial/xai_spatial.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial XAI on RCC test set (TP/FP/FN selection via predictions.csv).

Generates:
- GradCAM / IG / Occlusion (if enabled and dependencies are available)
- Attention Rollout for ViT (via monkey patching timm Attention blocks).

This script is config-driven and can be:
- run standalone: python xai_spatial.py --config CONFIG_PATH
- called programmatically from the orchestrator:
    from explainability.spatial.xai_generate import main as spatial_xai_main
    spatial_xai_main(["--config", str(config_path)])
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402

from explainability.common.eval_utils import (
    setup_logger,
    set_seed,
    build_preprocess,
    tensor_to_pil,
    load_eval_artifacts,
    select_items,
    make_wds_loader_with_keys,
    make_imgfolder_loader,
)
from explainability.common.ssl_linear_loader import SSLLinearClassifier

# Optional dependencies
try:
    import webdataset as wds  # noqa: F401

    HAVE_WDS = True
except Exception:
    HAVE_WDS = False

try:
    from torchcam.methods import GradCAM  # noqa: F401

    HAVE_TCAM = True
except Exception:
    HAVE_TCAM = False

try:
    from captum.attr import IntegratedGradients, Occlusion  # noqa: F401

    HAVE_CAPTUM = True
except Exception:
    HAVE_CAPTUM = False


# -------------------------------------------------------------------------
# Heatmap overlay utilities
# -------------------------------------------------------------------------
def overlay_heatmap(pil_img: Image.Image, heatmap, alpha: float = 0.5) -> Image.Image:
    """Overlay a normalized heatmap on top of a PIL image."""
    heatmap = np.array(heatmap)
    if heatmap.ndim > 2:
        heatmap = np.squeeze(heatmap)
    if heatmap.ndim == 3:
        if heatmap.shape[0] in (1, 3, 4):
            heatmap = heatmap.mean(axis=0)
        else:
            heatmap = heatmap.mean(axis=-1)

    if heatmap.ndim != 2:
        return pil_img.convert("RGBA")

    hmin, hmax = float(heatmap.min()), float(heatmap.max())
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)

    heat_rgba = Image.fromarray((cm.jet(heatmap) * 255).astype(np.uint8)).convert("RGBA")
    base = pil_img.convert("RGBA")
    if heat_rgba.size != base.size:
        heat_rgba = heat_rgba.resize(base.size, Image.BILINEAR)

    return Image.blend(base, heat_rgba, alpha=alpha)


def _parse_maybe_json_or_literal(s):
    import ast

    if isinstance(s, (bytes, bytearray)):
        s = s.decode("utf-8")
    if isinstance(s, str):
        s = s.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            return ast.literal_eval(s)
        except Exception:
            return s
    return s


# -------------------------------------------------------------------------
# ViT Attention Rollout (monkey patching)
# -------------------------------------------------------------------------
class ViTAttentionRollout:
    """
    Compute Attention Rollout for timm ViT models via Monkey Patching.

    It replaces the forward() of attention blocks to manually compute and
    capture the attention matrices, bypassing Flash Attention / SDPA paths
    that hide the weights.
    """

    def __init__(self, model, head_fusion: str = "mean", discard_ratio: float = 0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []
        self.original_forwards = {}  # type: ignore[var-annotated]

    def _manual_attention_forward(self, module, x, attn_mask=None, **kwargs):
        """
        Replacement forward method for timm Attention blocks.
        Replicates standard ViT attention logic but captures the weights.
        """
        B, N, C = x.shape

        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # [B, H, N, D]

        if hasattr(module, "q_norm") and module.q_norm is not None:
            q = module.q_norm(q)
        if hasattr(module, "k_norm") and module.k_norm is not None:
            k = module.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * module.scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)

        self.attentions.append(attn.detach().cpu())

        attn = module.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = module.proj(x)
        x = module.proj_drop(x)
        return x

    def _patch_model(self) -> None:
        self.attentions = []
        self.original_forwards = {}

        for _, m in self.model.named_modules():
            if hasattr(m, "qkv") and hasattr(m, "scale"):
                self.original_forwards[m] = m.forward

                def make_wrapper(mod):
                    def wrapped(*args, **kwargs):
                        if not args:
                            raise RuntimeError(
                                "Attention forward called without positional input tensor."
                            )
                        x = args[0]
                        return self._manual_attention_forward(mod, x, **kwargs)

                    return wrapped

                m.forward = make_wrapper(m)

    def _unpatch_model(self) -> None:
        for m, original in self.original_forwards.items():
            m.forward = original
        self.original_forwards = {}

    def __call__(self, input_tensor: torch.Tensor):
        self._patch_model()
        try:
            with torch.no_grad():
                _ = self.model(input_tensor)
        finally:
            self._unpatch_model()

        if not self.attentions:
            print("[DEBUG] Rollout Error: No attention blocks captured via patching.")
            return None

        all_attn = torch.stack(self.attentions).squeeze(1)  # [L, H, T, T]

        if self.head_fusion == "mean":
            all_attn = torch.mean(all_attn, dim=1)
        elif self.head_fusion == "max":
            all_attn = torch.max(all_attn, dim=1)[0]
        elif self.head_fusion == "min":
            all_attn = torch.min(all_attn, dim=1)[0]

        num_tokens = all_attn.shape[1]
        eye = torch.eye(num_tokens).to(all_attn.device)
        joint_attentions = eye

        for layer_attn in all_attn:
            if self.discard_ratio > 0:
                flat = layer_attn.view(num_tokens, -1)
                val, _ = torch.topk(
                    flat,
                    int(flat.shape[1] * (1 - self.discard_ratio)),
                    dim=1,
                )
                threshold = val[:, -1].unsqueeze(1)
                layer_attn = torch.where(
                    layer_attn >= threshold,
                    layer_attn,
                    torch.zeros_like(layer_attn),
                )

            layer_attn = layer_attn / (layer_attn.sum(dim=-1, keepdims=True) + 1e-9)
            aug_attn = 0.5 * layer_attn + 0.5 * eye
            joint_attentions = torch.matmul(aug_attn, joint_attentions)

        mask = joint_attentions[0, 1:]
        grid_size = int(np.sqrt(mask.shape[0]))
        if grid_size * grid_size != mask.shape[0]:
            if grid_size * grid_size == mask.shape[0] - 1:
                mask = mask[1:]
            else:
                return None

        mask = mask.reshape(grid_size, grid_size).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)
        return mask


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    import yaml

    logger = setup_logger("xai_spatial.sh")

    parser = argparse.ArgumentParser(description="Spatial XAI for SSL RCC model")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(open(args.config, "r"))

    set_seed(int(cfg["experiment"]["seed"]))

    device = torch.device(
        cfg.get("runtime", {}).get("device", "cuda")
        if torch.cuda.is_available()
        else "cpu"
    )

    run_id = cfg["experiment"].get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(cfg["experiment"]["outputs_root"]) / cfg["model"]["name"] / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"[Spatial XAI] Output dir: {out_root}")

    # Eval artifacts
    y_true, y_pred, conf, keys, meta_rows = load_eval_artifacts(
        cfg["evaluation_inputs"]["eval_run_dir"],
        cfg["evaluation_inputs"]["predictions_csv"],
        cfg["evaluation_inputs"]["logits_npy"],
        logger,
    )

    arch_hint = cfg["model"].get("arch_hint", "ssl_linear").lower()
    if arch_hint != "ssl_linear":
        logger.error("Only arch_hint=ssl_linear is supported in this script.")
        return

    class_order: List[str] = cfg["labels"]["class_order"]
    num_classes = len(class_order)

    # Model
    model = SSLLinearClassifier(
        backbone_name=cfg["model"].get("backbone_name", "resnet50"),
        num_classes=num_classes,
    )
    mb, ub = model.load_backbone_from_ssl(cfg["model"]["ssl_backbone_ckpt"])
    mh, uh = model.load_head_from_probe(cfg["model"]["ssl_head_ckpt"])
    logger.info(
        f"Loaded SSL backbone (missing={mb}, unexpected={ub}) "
        f"and linear head (missing={mh}, unexpected={uh})."
    )

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)

    img_size = int(cfg["data"]["img_size"])
    imagenet_norm = bool(cfg["data"].get("imagenet_norm", False))
    preprocess_fn = build_preprocess(img_size, imagenet_norm)

    if cfg["data"]["backend"].lower() == "webdataset":
        w = cfg["data"]["webdataset"]
        loader = make_wds_loader_with_keys(
            w["test_dir"],
            w["pattern"],
            w["image_key"],
            w["meta_key"],
            preprocess_fn,
            int(cfg["data"]["num_workers"]),
        )
    else:
        _, loader = make_imgfolder_loader(
            cfg["data"]["imagefolder"]["test_dir"],
            preprocess_fn,
            int(cfg["data"]["batch_size"]),
            int(cfg["data"]["num_workers"]),
        )

    n_classes = num_classes

    targets, sel_reasons = select_items(
        y_true,
        y_pred,
        conf,
        keys,
        n_classes,
        cfg["selection"],
        logger,
    )
    target_set = set(targets)
    logger.info(f"[Spatial XAI] Targets selected: {len(target_set)}")

    import time
    t_start = time.time()
    total_targets = len(target_set)
    if total_targets == 0:
        logger.warning("[Spatial XAI] No targets selected, exiting early.")
        return

    methods = set(cfg["xai"]["methods"])
    use_ig = HAVE_CAPTUM and ("ig" in methods)
    use_occ = HAVE_CAPTUM and ("occlusion" in methods)
    has_gradcam = HAVE_TCAM and (("gradcam" in methods) or ("gradcam++" in methods))
    use_rollout = "attn_rollout" in methods

    if use_rollout:
        logger.info("Attention Rollout ENABLED (monkey patching mode).")

    if use_ig:
        ig = IntegratedGradients(model)  # type: ignore[valid-type]
    if use_occ:
        occl = Occlusion(model)  # noqa: F841

    target_layer = None
    if has_gradcam:
        tname = cfg["xai"]["gradcam"]["target_layer"]
        try:
            modules_dict = dict(model.named_modules())
            if tname in modules_dict:
                target_layer = modules_dict[tname]
            else:
                curr = model
                for part in tname.split("."):
                    curr = getattr(curr, part)
                target_layer = curr
        except Exception:
            target_layer = None
        if target_layer is None:
            logger.warning(f"Target layer {tname} not found for GradCAM.")

    index_csv = open(out_root / "index.csv", "w", newline="")
    writer = csv.writer(index_csv)
    writer.writerow(
        [
            "global_idx",
            "wds_key",
            "true",
            "pred",
            "conf",
            "methods",
            "png_paths",
            "selection_reason",
        ]
    )

    produced = 0
    global_idx = 0

    row_by_key = {r["wds_key"]: r for r in (meta_rows or [])} if meta_rows else {}
    idx_by_key = {k: i for i, k in enumerate(keys)} if keys is not None else {}
    class_order = cfg["labels"]["class_order"]

    rollout_instance = None
    if use_rollout:
        try:
            rollout_instance = ViTAttentionRollout(
                model.backbone.model,
                discard_ratio=cfg["xai"]["attn_rollout"]["discard_ratio"],
            )
        except AttributeError as e:
            logger.error(f"Cannot initialize Rollout: model structure error. {e}")
            use_rollout = False

    # Conteggi per validazione/summary
    method_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    for batch in loader:
        if cfg["data"]["backend"].lower() == "webdataset":
            img_t, meta_any, key = batch
            if keys is not None and key not in target_set:
                continue

            meta = meta_any if isinstance(meta_any, dict) else {}
            if isinstance(meta_any, (str, bytes)):
                meta = _parse_maybe_json_or_literal(meta_any) or {}

            row = row_by_key.get(key, {})

            true_id = int(row.get("y_true", -1)) if row else -1
            true_txt = (
                class_order[true_id]
                if 0 <= true_id < n_classes
                else meta.get("class_label", "")
            )

            pred_id = int(row.get("y_pred", -1)) if row else -1
            pred_txt = (
                class_order[pred_id] if 0 <= pred_id < n_classes else str(pred_id)
            )

            idx_eval = idx_by_key.get(key, None)
            if conf is not None and idx_eval is not None:
                prob = float(conf[idx_eval])
            else:
                prob = float("nan")

            sel_reason_list = sel_reasons.get(key, []) if keys is not None else []
            sel_reason_str = "|".join(sel_reason_list) if sel_reason_list else ""
            for r in sel_reason_list:
                reason_counts[r] += 1
        else:
            img_t, lbl = batch
            key = None
            true_id = int(lbl.item())
            true_txt = class_order[true_id]
            pred_id = -1
            pred_txt = ""
            prob = float("nan")
            sel_reason_str = ""

        x = img_t.to(device).unsqueeze(0) if img_t.ndim == 3 else img_t.to(device)
        x.requires_grad_(True)

        logits = model(x)

        if pred_id >= 0:
            target_class = pred_id
        else:
            target_class = int(torch.argmax(logits, dim=1).item())

        out_dir = out_root / f"idx_{global_idx:07d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        pil_in = tensor_to_pil(x[0], imagenet_norm=imagenet_norm)
        orig_path = out_dir / "input.png"
        pil_in.save(orig_path)

        if sel_reason_str:
            (out_dir / "selection_reason.txt").write_text(sel_reason_str + "\n")

        png_paths, used = [], []

        if use_ig:
            try:
                attr = ig.attribute(
                    x,
                    target=target_class,
                    n_steps=cfg["xai"]["ig"]["steps"],
                )
                heat = (
                    attr.abs()
                    .mean(dim=1)
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )
                over = overlay_heatmap(pil_in, heat, alpha=0.5)
                path = out_dir / "ig.png"
                over.save(path)
                png_paths.append(str(path))
                used.append("ig")
            except Exception as e:
                logger.warning(f"IG failed: {e}")

        if has_gradcam and target_layer is not None:
            try:
                cam_method = GradCAM(model, target_layer=target_layer)
                with torch.enable_grad():
                    model.zero_grad()
                    sc = model(x)
                maps = cam_method(class_idx=target_class, scores=sc)
                raw_map = maps[0].detach().cpu()
                if raw_map.ndim == 1:
                    seq_len = raw_map.numel()
                    grid = int(np.sqrt(seq_len))
                    if grid * grid == seq_len:
                        raw_map = raw_map.reshape(grid, grid)
                    elif grid * grid == seq_len - 1:
                        raw_map = raw_map[1:].reshape(grid, grid)
                heat = raw_map.numpy()
                over = overlay_heatmap(pil_in, heat, alpha=0.5)
                path = out_dir / "gradcam.png"
                over.save(path)
                png_paths.append(str(path))
                used.append("gradcam")
                for h in cam_method.hook_handles:
                    h.remove()
            except Exception as e:
                logger.warning(f"GradCAM failed: {e}")

        if use_rollout and rollout_instance is not None:
            try:
                mask = rollout_instance(x)
                if mask is not None:
                    mask_np = np.array(mask.detach().cpu() if hasattr(mask, "detach") else mask)
                    over = overlay_heatmap(pil_in, mask_np, alpha=0.6)
                    path = out_dir / "attn_rollout.png"
                    over.save(path)
                    np.save(out_dir / "attn_rollout.npy", mask_np)
                    png_paths.append(str(path))
                    used.append("rollout")
            except Exception as e:
                logger.warning(f"Rollout failed: {e}")

        for m in used:
            method_counts[m] += 1

        writer.writerow(
            [
                global_idx,
                key,
                true_txt,
                pred_txt,
                prob,
                ";".join(used),
                ";".join(png_paths),
                sel_reason_str,
            ]
        )


        global_idx += 1
        produced += 1

        # progress + ETA
        if produced % 10 == 0 or produced == total_targets:
            elapsed = time.time() - t_start
            avg_per_item = elapsed / max(1, produced)
            remaining = total_targets - produced
            eta = remaining * avg_per_item
            logger.info(
                f"[PROGRESS] Spatial XAI: {produced}/{total_targets} "
                f"({100.0*produced/total_targets:.1f}%), "
                f"elapsed={elapsed/60:.1f} min, "
                f"avg={avg_per_item:.2f} s/item, "
                f"ETA~{eta/60:.1f} min"
            )

        if keys is not None and produced >= len(targets):
            break

    index_csv.close()
    total_elapsed = time.time() - t_start
    logger.info(
        f"[Spatial XAI] Done. Produced {produced} cases in {total_elapsed/60:.1f} min."
    )

    # ----------------- VALIDAZIONE / SEGNALAZIONI -----------------
    if produced == 0:
        logger.warning(
            "[Spatial XAI] No outputs produced (produced=0). "
            "Controlla selection config / eval artifacts."
        )
    else:
        if method_counts:
            logger.info("[Spatial XAI] Method usage:")
            for m, cnt in method_counts.items():
                logger.info(f"  - {m}: {cnt} patches")
        if reason_counts:
            logger.info("[Spatial XAI] Selection reasons distribution:")
            for r, cnt in reason_counts.items():
                logger.info(f"  - {r}: {cnt} patches")


if __name__ == "__main__":
    main()
>>

