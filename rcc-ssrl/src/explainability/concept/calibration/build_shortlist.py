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
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from explainability.utils.bootstrap import bootstrap_package

bootstrap_package(__file__, globals())

from . import ensure_layout, SHORTLIST_DIR, REPORT_DIR, ANALYSIS_DIR
from .utils import copy_plot_files, format_tex_table
from explainability.utils.class_utils import canon_class


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
    if primary.empty:
        report_lines.append("_None after filtering._")
    else:
        try:
            report_lines.append(primary[cols].to_markdown(index=False))
        except ImportError:
            report_lines.append(primary[cols].to_csv(index=False))
        report_lines.append("\n### Confounders (high rank but primary_class mismatch)\n")
    if confounds.empty:
        report_lines.append("_None after filtering._")
    else:
        try:
            report_lines.append(confounds[cols].to_markdown(index=False))
        except ImportError:
            report_lines.append(confounds[cols].to_csv(index=False))
        report_lines.append("")

        if args.write_tex:
            tex_cols = [c for c in ["concept_short_name", "concept_name", rank_col, "cohen_d", "auc_ovr"] if c in primary.columns]
            tex = format_tex_table(
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
    copy_plot_files(plots_dir, fig_dir)

    log.info("Wrote shortlist JSON: %s", out_json)
    log.info("Wrote shortlist YAML (OFFICIAL): %s", out_yaml)
    log.info("Wrote report.md: %s", out_md)
    log.info("Wrote flat CSV: %s", flat_csv)
    if args.write_tex:
        log.info("Wrote LaTeX tables under: %s", tex_dir_report)


if __name__ == "__main__":
    main()
