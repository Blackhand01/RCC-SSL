concept/calibration/build_shortlist.py codice <<
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
from ..utils.class_utils import canon_class


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
        report_lines.append(primary[cols].to_markdown(index=False) if not primary.empty else "_None after filtering._")
        report_lines.append("\n### Confounders (high rank but primary_class mismatch)\n")
        report_lines.append(confounds[cols].to_markdown(index=False) if not confounds.empty else "_None after filtering._")
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
>>

concept/calibration/calibration.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

"""
Unified CLI for PLIP concept calibration.

Subcommands:
  - calibrate     Run calibration on TRAIN/VAL and write canonical metadata.
  - deep-validate Run deep validation and produce analysis outputs.
  - check         Validate calibration/deep-validation/report/shortlist artifacts.
"""

import argparse
import csv
import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
import torch

from explainability.utils.bootstrap import bootstrap_package

bootstrap_package(__file__, globals())

from explainability.utils.class_utils import canon_class
from ..plip.ontology_io import load_ontology, concepts_to_prompt_lists, concepts_to_dicts
from ..plip.plip_model import load_plip, encode_text, encode_images, score
from ..plip.wds_loader import build_wds_loader
from . import ensure_layout, METADATA_DIR, ANALYSIS_DIR
from ...paths import resolve_config, CALIBRATION_CONFIG_YAML, OUTPUT_DIR
from .utils import (
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_CALIB_DIR,
    DEFAULT_REPORT_DIR,
    DEFAULT_SHORTLIST_YAML,
    as_float,
    augment_selection_with_primary_concepts,
    build_metrics_tables,
    build_selection_from_delta,
    build_selection_union,
    check_calibration,
    check_deep_validation,
    check_report,
    check_shortlist,
    check_spatial_concept_light_outputs,
    compute_auc_ap_for_selected,
    compute_fast_stats,
    ensure_dir,
    get_plt,
    guess_class_names,
    is_constant,
    load_concepts,
    normalize_labels,
    plot_bar,
    plot_barh,
    plot_heatmap,
    resolve_calibration_dir,
    write_exemplars,
)


# ---------------------------------------------------------------------
# Calibration (metadata)
# ---------------------------------------------------------------------


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


def run_calibration(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("plip_calibration")

    ensure_layout()
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
            if is_constant(s):
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


# ---------------------------------------------------------------------
# Deep validation (analysis)
# ---------------------------------------------------------------------


def run_deep_validation(args: argparse.Namespace) -> None:
    if args.quiet_tokenizers:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("plip_deep_validation")

    ensure_layout()

    # Resolve metadata dir (canonical by default)
    if args.metadata_dir is not None:
        cal = args.metadata_dir
    elif args.cal_run is not None:
        cal = resolve_calibration_dir(args.cal_run)
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

    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    concepts = load_concepts(concepts_path)
    scores = np.load(scores_path, mmap_mode="r")  # (N,C)
    labels_raw = np.load(labels_path, allow_pickle=True)
    keys = np.load(keys_path, allow_pickle=True) if keys_path.exists() else None

    class_names_guess = guess_class_names(cfg, labels_raw)
    labels, class_names = normalize_labels(labels_raw, class_names_guess)

    n = labels.shape[0]
    n_concepts = len(concepts)
    if scores.ndim != 2 or scores.shape[0] != n or scores.shape[1] != n_concepts:
        raise RuntimeError(f"Shape mismatch: scores {scores.shape} labels {labels.shape} concepts {n_concepts}")

    out_dir = args.out_dir if args.out_dir is not None else ANALYSIS_DIR
    ensure_dir(out_dir)
    plot_dir = out_dir / "plots"
    ensure_dir(plot_dir)

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
    selected = augment_selection_with_primary_concepts(concepts, class_names, selected)
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
        r["auc_ovr"] = (float("nan") if m is None else as_float(m.get("auc_ovr", float("nan"))))
        r["ap_ovr"] = (float("nan") if m is None else as_float(m.get("ap_ovr", float("nan"))))
        r["auc_valid"] = (0.0 if m is None else as_float(m.get("auc_valid", 1.0 if math.isfinite(r["auc_ovr"]) else 0.0), 0.0))
        r["ap_valid"] = (0.0 if m is None else as_float(m.get("ap_valid", 1.0 if math.isfinite(r["ap_ovr"]) else 0.0), 0.0))
        r["auc_ap_reason"] = (float("nan") if m is None else as_float(m.get("reason", float("nan"))))

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
        ensure_dir(onco_audit_dir)
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
    plot_heatmap(
        mats["means"],
        row_labels=ylabels,
        col_labels=xlabels,
        title="Mean score (class x concept)",
        out_base=plot_dir / "heatmap_mean_score",
    )
    plot_heatmap(
        mats["delta"],
        row_labels=ylabels,
        col_labels=xlabels,
        title="Delta mean vs rest (class x concept)",
        out_base=plot_dir / "heatmap_delta_mean",
    )
    plot_heatmap(
        mats["top1_freq"],
        row_labels=ylabels,
        col_labels=xlabels,
        title="Top-1 freq (class x concept)",
        out_base=plot_dir / "heatmap_top1_freq",
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap(
        mats["topk_freq"],
        row_labels=ylabels,
        col_labels=xlabels,
        title=f"Top-{args.topk} freq (class x concept)",
        out_base=plot_dir / f"heatmap_top{args.topk}_freq",
        vmin=0.0,
        vmax=1.0,
    )

    # Per-class bar plots (top delta and top auc if available)
    for k, cls in enumerate(class_names):
        order = np.argsort(mats["delta"][k])[::-1][:topn]
        vals = mats["delta"][k, order]
        labs = [concepts[j].short_name for j in order]
        plot_barh(
            vals,
            labs,
            plot_dir / f"bar_{cls}_top_delta",
            title=f"{cls}: top-{topn} concepts by delta(mean)",
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
                plot_barh(
                    auc_vals_np[ord2],
                    [auc_labs[i] for i in ord2],
                    plot_dir / f"bar_{cls}_top_auc",
                    title=f"{cls}: top-{min(topn, len(ord2))} concepts by AUC(OVR)",
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
    from sklearn.metrics import precision_recall_curve, roc_curve

    plt = get_plt()
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


# ---------------------------------------------------------------------
# Artifact checks
# ---------------------------------------------------------------------


def run_check(args: argparse.Namespace) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PLIP concept calibration pipeline (unified CLI).")
    sub = p.add_subparsers(dest="command", required=True)

    # calibrate
    p_cal = sub.add_parser(
        "calibrate",
        help="Run calibration on TRAIN/VAL and write canonical metadata artifacts.",
    )
    p_cal.add_argument(
        "--config",
        required=False,
        type=Path,
        default=CALIBRATION_CONFIG_YAML,
        help="YAML config path (default: central configs/calibration.yaml).",
    )
    p_cal.add_argument("--log-level", default="INFO", type=str, help="Logging level (DEBUG, INFO, WARNING).")
    p_cal.set_defaults(func=run_calibration)

    # deep-validate
    p_val = sub.add_parser(
        "deep-validate",
        help="Run deep validation and write analysis outputs (metrics/plots/exemplars).",
    )
    p_val.add_argument(
        "--cal-run",
        default=None,
        type=Path,
        help="(Backward compat) Calibration dir. Default: canonical calibration/metadata/.",
    )
    p_val.add_argument(
        "--metadata-dir",
        default=None,
        type=Path,
        help="Canonical metadata dir (default: output/calibration/metadata).",
    )
    p_val.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="Canonical analysis dir (default: output/calibration/analysis).",
    )
    p_val.add_argument("--topk", type=int, default=5, help="Top-k for topk_freq (default: 5)")
    p_val.add_argument("--chunk-size", type=int, default=16384, help="Chunk size for memmap-friendly passes")
    p_val.add_argument(
        "--compute-auc",
        dest="compute_auc",
        action="store_true",
        default=True,
        help="Compute AUC/AP (bounded) for selected concepts per class (default: ON).",
    )
    p_val.add_argument(
        "--no-compute-auc",
        dest="compute_auc",
        action="store_false",
        help="Disable AUC/AP computation (debug only).",
    )
    p_val.add_argument("--require-auc", action="store_true", help="Fail if AUC/AP are missing/invalid for most selected pairs.")
    p_val.add_argument(
        "--auc-topm-per-class",
        type=int,
        default=25,
        help="Compute AUC/AP for union of top-M per class (delta/cohen_d/top1_freq). (default: 25)",
    )
    p_val.add_argument("--plots-topn", type=int, default=12, help="Top-N concepts to plot per class (default: 12)")
    p_val.add_argument("--max-exemplars", type=int, default=40, help="Top exemplars (keys) per (class, concept) (default: 40)")
    p_val.add_argument("--quiet-tokenizers", action="store_true", help="Set TOKENIZERS_PARALLELISM=false inside the process")
    p_val.add_argument("--allow-nonfinite", action="store_true", help="Allow non-finite scores (they will be replaced with 0.0). Default: fail-fast.")
    p_val.add_argument("--require-onco", action="store_true", default=True, help="Fail if ONCO has 0 diagnostic concepts in top list (default: ON).")
    p_val.add_argument("--no-require-onco", dest="require_onco", action="store_false", help="Disable ONCO gating.")
    p_val.add_argument("--log-level", default="INFO", type=str, help="Logging level (DEBUG, INFO, WARNING).")
    p_val.set_defaults(func=run_deep_validation)

    # check
    p_chk = sub.add_parser(
        "check",
        help="Validate calibration/deep-validation/report/shortlist artifacts.",
    )
    p_chk.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    p_chk.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p_chk.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p_chk.add_argument("--shortlist-yaml", type=Path, default=DEFAULT_SHORTLIST_YAML)
    p_chk.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing optional dirs (plots/figures/paper_tables) and low per-class coverage.",
    )
    p_chk.add_argument(
        "--min-valid-per-class",
        type=int,
        default=10,
        help="Minimum rows per class with numeric auc_ovr+ap_ovr required (strict only).",
    )
    p_chk.add_argument(
        "--check-spatial-concept",
        action="store_true",
        help="Also validate light spatial/roi outputs and heavy indexed paths for the unified spatial+concept pipeline.",
    )
    p_chk.set_defaults(func=run_check)

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
>>

concept/calibration/__init__.py codice <<
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

concept/calibration/utils.py codice <<
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
>>

concept/__init__.py codice <<
"""
Canonical (fixed) filesystem layout for PLIP concept calibration + validation.

Paths (single source of truth, see explainability.paths):
  A) Calibration unified metadata (TRAIN+VAL scored separately, output merged):
     src/explainability/output/calibration/metadata/

  B) Deep validation (analysis of the calibration):
     src/explainability/output/calibration/analysis/

  C) Paper-ready report:
     src/explainability/output/calibration/analysis/report/

  D) Final shortlist for test (no-ROI and ROI):
     src/explainability/output/calibration/analysis/concepts_shortlist.yaml

Additional canonical pipelines:
  E) Concept XAI on TEST (NO-ROI, model-independent, computed once):
     src/explainability/output/no_roi/

  F) Concept XAI on TEST (ROI, model-dependent, uses spatial masks from a model):
     src/explainability/output/roi/<MODEL_ID>/

  G) Comparison ROI vs NO-ROI (paper-ready, per model):
     src/explainability/output/roi-no_roi-comparision/<MODEL_ID>/

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
import copy
import csv
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from explainability.utils.bootstrap import bootstrap_package

bootstrap_package(__file__, globals())

from ..utils.class_utils import load_shortlist_idx
from .plip.plip_model import load_plip, encode_images
from .plip.scoring import score
from .plip.wds_loader import build_wds_loader
from .calibration.utils import (
    Concept,
    as_float,
    augment_selection_with_primary_concepts,
    build_metrics_tables,
    build_selection_from_delta,
    build_selection_union,
    compute_auc_ap_for_selected,
    compute_fast_stats,
    ensure_dir,
    get_plt,
    guess_class_names,
    load_concepts,
    normalize_labels,
    plot_barh,
    plot_heatmap,
    write_exemplars,
)
from ..paths import CALIBRATION_PATHS, ensure_no_roi_layout, NO_ROI_CONFIG_YAML, resolve_config


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp", delete=False) as tf:
        tmp = Path(tf.name)
        tf.write(data)
        tf.flush()
        os.fsync(tf.fileno())
    tmp.replace(path)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    _atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")



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
    concepts: List[Concept],
    shortlist_yaml: Path,
    use_shortlist_only: bool,
    log: logging.Logger,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    concept_to_idx = {c.short_name: c.idx for c in concepts if c.short_name}

    if not use_shortlist_only:
        idxs = list(range(len(concepts)))
    else:
        if not shortlist_yaml.exists():
            raise FileNotFoundError(f"Shortlist YAML not found: {shortlist_yaml}")
        shortlist = load_shortlist_idx(shortlist_yaml, concept_to_idx, log=log)
        union: set[int] = set()
        for _cls, d in shortlist.items():
            union.update(d.get("primary", []))
            union.update(d.get("confounds", []))
        idxs = sorted(union)
        if not idxs:
            raise RuntimeError(f"Shortlist produced 0 indices (ontology mismatch?): {shortlist_yaml}")
        log.info("Selected concepts (union shortlist): %d / %d", len(idxs), len(concepts))

    sel: List[Dict[str, Any]] = []
    for local_idx, global_idx in enumerate(idxs):
        c = concepts[global_idx]
        sel.append(
            {
                "concept_idx": int(local_idx),
                "concept_idx_global": int(global_idx),
                "concept_id": c.id,
                "concept_short_name": c.short_name,
                "concept_name": c.name,
                "group": c.group,
                "primary_class": c.primary_class,
            }
        )
    return idxs, sel


def _build_selected_concepts(
    concepts: List[Concept],
    selected_global_idxs: Sequence[int],
) -> Tuple[List[Concept], List[int]]:
    selected: List[Concept] = []
    global_idx_by_local: List[int] = []
    for local_idx, global_idx in enumerate(selected_global_idxs):
        c = concepts[global_idx]
        selected.append(
            Concept(
                idx=local_idx,
                id=c.id,
                short_name=c.short_name,
                name=c.name,
                group=c.group,
                primary_class=c.primary_class,
                prompt=c.prompt,
            )
        )
        global_idx_by_local.append(int(global_idx))
    return selected, global_idx_by_local


def _load_selected_concepts(
    path: Path,
    concepts: List[Concept],
    log: logging.Logger,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"selected_concepts.json not found: {path}")
    obj = json.loads(path.read_text())
    selected = obj.get("selected", [])
    if not isinstance(selected, list) or not selected:
        raise RuntimeError(f"selected_concepts.json is empty/invalid: {path}")

    concept_to_idx = {c.short_name: c.idx for c in concepts if c.short_name}
    sel_idxs: List[int] = []
    sel_meta: List[Dict[str, Any]] = []
    for local_idx, entry in enumerate(selected):
        if not isinstance(entry, dict):
            continue
        global_idx = entry.get("concept_idx_global", None)
        if global_idx is None:
            global_idx = entry.get("concept_idx", None)
        if global_idx is None:
            sn = str(entry.get("concept_short_name") or "").strip()
            if sn and sn in concept_to_idx:
                global_idx = concept_to_idx[sn]
        if global_idx is None:
            raise RuntimeError(f"selected_concepts.json missing global idx for entry {local_idx}: {path}")
        global_idx = int(global_idx)
        if global_idx < 0 or global_idx >= len(concepts):
            raise RuntimeError(f"selected_concepts.json has out-of-range index {global_idx}: {path}")
        c = concepts[global_idx]
        sel_idxs.append(global_idx)
        sel_meta.append(
            {
                "concept_idx": int(local_idx),
                "concept_idx_global": int(global_idx),
                "concept_id": c.id,
                "concept_short_name": c.short_name,
                "concept_name": c.name,
                "group": c.group,
                "primary_class": c.primary_class,
            }
        )
    log.info("Loaded selected concepts from: %s", path)
    return sel_idxs, sel_meta


def _load_subset_keys(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Subset keys file not found: {path}")
    keys = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    return keys or None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Concept NO-ROI on TEST (model-independent, canonical output).")
    ap.add_argument(
        "--config",
        type=Path,
        default=NO_ROI_CONFIG_YAML,
        help=(
            "YAML config (default: central configs/no_roi.yaml). "
            "If missing/unreadable, the runner falls back to env/defaults."
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
    ap.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Stop after N patches (subset). Overrides data.max_patches.",
    )
    ap.add_argument(
        "--subset-prob",
        type=float,
        default=None,
        help="Randomly keep each patch with probability p in (0,1].",
    )
    ap.add_argument(
        "--subset-keys",
        type=Path,
        default=None,
        help="File with WebDataset keys (one per line) to restrict to a subset.",
    )
    ap.add_argument(
        "--subset-seed",
        type=int,
        default=0,
        help="Seed for subset sampling when --subset-prob is used.",
    )
    ap.add_argument(
        "--compute-auc",
        dest="compute_auc",
        action="store_true",
        default=None,
        help="Compute AUC/AP for selected concepts (requires scikit-learn).",
    )
    ap.add_argument(
        "--no-compute-auc",
        dest="compute_auc",
        action="store_false",
        help="Disable AUC/AP computation.",
    )
    ap.add_argument(
        "--no-metrics",
        dest="compute_metrics",
        action="store_false",
        default=True,
        help="Skip metrics/plots/exemplars (scores only).",
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


def _merge_missing(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """
    Recursively merge entries from src into dst without overwriting existing keys.
    Useful to apply profile defaults.
    """
    for k, v in src.items():
        if isinstance(v, dict):
            cur = dst.get(k)
            if isinstance(cur, dict):
                _merge_missing(cur, v)
            elif k not in dst:
                dst[k] = copy.deepcopy(v)
        elif k not in dst:
            dst[k] = v


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("concept_no_roi")

    layout = ensure_no_roi_layout()

    cfg: Dict[str, Any] = {}
    cfg_path = resolve_config(args.config) if args.config is not None else None
    if cfg_path is not None and cfg_path.exists():
        cfg = _safe_load_yaml(cfg_path, log=log)
    else:
        log.warning("NO-ROI config missing; continuing with env/defaults: %s", cfg_path)

    # If config defines profiles.no_roi, merge its defaults into the top-level cfg
    # without overwriting explicit root keys. This allows using shared configs
    # that bundle ROI + NO-ROI under profiles.
    profile_no_roi = cfg.get("profiles", {}).get("no_roi", {})
    if isinstance(profile_no_roi, dict):
        _merge_missing(cfg, profile_no_roi)

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

    loader = build_wds_loader(
        split_dir=test_dir,
        pattern=pattern,
        image_key=image_key,
        meta_key=meta_key,
        preprocess=plip.preprocess,
        batch_size=bs,
        num_workers=nw,
        return_raw=False,
    )

    keys: List[str] = []
    labels: List[str] = []
    chunks: List[np.ndarray] = []
    n_seen = 0
    for batch in loader:
        if batch is None:
            continue
        imgs, metas, bkeys, _raw = batch
        if imgs is None:
            continue
        metas_list = list(metas) if isinstance(metas, (list, tuple)) else [metas] * int(imgs.shape[0])

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

concept/run_no_roi_test.sbatch codice <<
#!/usr/bin/env bash
#SBATCH --job-name=no_roi_test
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/xai/no_roi.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/xai/no_roi.%j.err
#SBATCH -p gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --exclude=compute-5-14,compute-5-11,compute-3-12

set -euo pipefail

export PROJECT_ROOT="/home/mla_group_01/rcc-ssrl"
export VENV_DIR="/home/mla_group_01/rcc-ssrl/.venvs/xai"

mkdir -p "$PROJECT_ROOT/src/logs/xai" \
         "$PROJECT_ROOT/src/explainability/output/no_roi/logs"

# --- CRITICAL: do not leak ~/.local into Python ---
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Keep imports robust regardless of repo layout
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Hugging Face cache (use HF_HOME; TRANSFORMERS_CACHE is deprecated)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# Activate the *only* environment you should be using
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[FATAL] Venv not found: $VENV_DIR" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# Extra safety: ensure the venv python is actually used
export PYTHON_BIN="$VENV_DIR/bin/python"
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] VENV_DIR=$VENV_DIR"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -c 'import sys; print("[INFO] sys.executable=", sys.executable)'

# Delegate to the deterministic runner
bash "$PROJECT_ROOT/src/explainability/concept/run_no_roi_test.sh"
>>

concept/run_no_roi_test.sh codice <<
#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# Run PLIP Concept XAI on TEST WITHOUT ROI (NO-ROI)
# Canonical output under:
#   $PROJECT_ROOT/src/explainability/output/no_roi/
# ------------------------------------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-/home/mla_group_01/rcc-ssrl}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venvs/xai}"

CONFIG_NO_ROI="${CONFIG_NO_ROI:-$PROJECT_ROOT/src/explainability/configs/no_roi.yaml}"

# Canonical calibration outputs
CALIB_METADATA_DIR_DEFAULT="$PROJECT_ROOT/src/explainability/output/calibration/metadata"
SHORTLIST_YAML_DEFAULT="$PROJECT_ROOT/src/explainability/output/calibration/analysis/concepts_shortlist.yaml"

CALIBRATION_METADATA_DIR="${CALIBRATION_METADATA_DIR:-$CALIB_METADATA_DIR_DEFAULT}"
CONCEPT_SHORTLIST_YAML="${CONCEPT_SHORTLIST_YAML:-$SHORTLIST_YAML_DEFAULT}"

# Optional: override TEST dir via env (runner can also read it from YAML)
WDS_TEST_DIR="${WDS_TEST_DIR:-}"

# Subset knobs (optional). They map to CLI flags supported by run_no_roi.py.
MAX_PATCHES="${MAX_PATCHES:-}"
SUBSET_PROB="${SUBSET_PROB:-}"
SUBSET_KEYS="${SUBSET_KEYS:-}"
SUBSET_SEED="${SUBSET_SEED:-0}"

# Overwrite canonical no_roi artifacts (optional)
OVERWRITE="${OVERWRITE:-0}"

# --- CRITICAL: do not leak ~/.local into Python ---
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Make imports robust regardless of repo layout
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Hugging Face cache (prefer HF_HOME; avoids TRANSFORMERS_CACHE warning)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
# HF_HOME is the supported knob for cache root :contentReference[oaicite:2]{index=2}

# ------------------------------------------------------------------
# Hard preflight (fail fast)
# ------------------------------------------------------------------
if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "[FATAL] PROJECT_ROOT not found: $PROJECT_ROOT" >&2
  exit 2
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[FATAL] Venv not found: $VENV_DIR" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[FATAL] PYTHON_BIN not executable: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$PROJECT_ROOT/src/explainability/output/no_roi/logs"

# Check canonical inputs exist
if [[ ! -d "$CALIBRATION_METADATA_DIR" ]]; then
  echo "[FATAL] Calibration metadata dir not found: $CALIBRATION_METADATA_DIR" >&2
  exit 2
fi
for req in "concepts.json" "text_features.pt"; do
  if [[ ! -f "$CALIBRATION_METADATA_DIR/$req" ]]; then
    echo "[FATAL] Missing calibration artifact: $CALIBRATION_METADATA_DIR/$req" >&2
    exit 2
  fi
done
if [[ ! -f "$CONCEPT_SHORTLIST_YAML" ]]; then
  echo "[FATAL] Shortlist YAML not found: $CONCEPT_SHORTLIST_YAML" >&2
  exit 2
fi
if [[ -n "$WDS_TEST_DIR" ]] && [[ ! -d "$WDS_TEST_DIR" ]]; then
  echo "[FATAL] WDS_TEST_DIR does not exist: $WDS_TEST_DIR" >&2
  exit 2
fi

# ------------------------------------------------------------------
# Binary-compat sanity checks (this is your actual failure)
# ------------------------------------------------------------------
"$PYTHON_BIN" - <<'PY'
import sys
print("[INFO] sys.executable =", sys.executable)
try:
    import numpy as np
    print("[INFO] numpy =", np.__version__)
    major = int(np.__version__.split(".", 1)[0])
    if major >= 2:
        raise SystemExit(
            "[FATAL] Detected NumPy >= 2 in the active env. "
            "Your stack (torch/sklearn wheels) is very likely built against NumPy 1.x.\n"
            "Fix inside the venv:\n"
            "  python -m pip install -U --force-reinstall 'numpy<2'\n"
            "  python -m pip install -U --force-reinstall scikit-learn\n"
        )
except Exception as e:
    raise SystemExit(f"[FATAL] NumPy import failed: {e}")

# sklearn import is where your traceback dies
try:
    import sklearn
    print("[INFO] sklearn =", sklearn.__version__)
except Exception as e:
    raise SystemExit(
        "[FATAL] scikit-learn import failed (binary mismatch vs NumPy).\n"
        "Fix inside the venv:\n"
        "  python -m pip install -U --force-reinstall 'numpy<2'\n"
        "  python -m pip install -U --force-reinstall scikit-learn\n"
        f"Original error: {e}"
    )

# torch sanity
try:
    import torch
    print("[INFO] torch =", torch.__version__)
    print("[INFO] cuda available =", torch.cuda.is_available())
except Exception as e:
    raise SystemExit(f"[FATAL] torch import failed: {e}")
PY

# ------------------------------------------------------------------
# Build CLI args
# ------------------------------------------------------------------
args=()
args+=( "--config" "$CONFIG_NO_ROI" )
args+=( "--calibration-metadata-dir" "$CALIBRATION_METADATA_DIR" )
args+=( "--shortlist-yaml" "$CONCEPT_SHORTLIST_YAML" )

if [[ -n "$WDS_TEST_DIR" ]]; then
  args+=( "--test-dir" "$WDS_TEST_DIR" )
fi
if [[ -n "$MAX_PATCHES" ]]; then
  args+=( "--max-patches" "$MAX_PATCHES" )
fi
if [[ -n "$SUBSET_PROB" ]]; then
  args+=( "--subset-prob" "$SUBSET_PROB" "--subset-seed" "$SUBSET_SEED" )
fi
if [[ -n "$SUBSET_KEYS" ]]; then
  args+=( "--subset-keys" "$SUBSET_KEYS" )
fi
if [[ "$OVERWRITE" == "1" ]]; then
  args+=( "--overwrite" )
fi

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
echo "[INFO] CONFIG_NO_ROI=$CONFIG_NO_ROI"
echo "[INFO] CALIBRATION_METADATA_DIR=$CALIBRATION_METADATA_DIR"
echo "[INFO] CONCEPT_SHORTLIST_YAML=$CONCEPT_SHORTLIST_YAML"
if [[ -n "$WDS_TEST_DIR" ]]; then echo "[INFO] WDS_TEST_DIR=$WDS_TEST_DIR"; fi
if [[ -n "$MAX_PATCHES" ]]; then echo "[INFO] MAX_PATCHES=$MAX_PATCHES"; fi
if [[ -n "$SUBSET_PROB" ]]; then echo "[INFO] SUBSET_PROB=$SUBSET_PROB (seed=$SUBSET_SEED)"; fi
if [[ -n "$SUBSET_KEYS" ]]; then echo "[INFO] SUBSET_KEYS=$SUBSET_KEYS"; fi
echo "[INFO] OVERWRITE=$OVERWRITE"

cd "$PROJECT_ROOT"

# ------------------------------------------------------------------
# Execute robustly (try both common module layouts; fallback to file)
# ------------------------------------------------------------------
if "$PYTHON_BIN" -c "import importlib; importlib.import_module('src.explainability.concept.run_no_roi')" >/dev/null 2>&1; then
  exec "$PYTHON_BIN" -u -m src.explainability.concept.run_no_roi "${args[@]}"
elif "$PYTHON_BIN" -c "import importlib; importlib.import_module('explainability.concept.run_no_roi')" >/dev/null 2>&1; then
  exec "$PYTHON_BIN" -u -m explainability.concept.run_no_roi "${args[@]}"
else
  exec "$PYTHON_BIN" -u "$PROJECT_ROOT/src/explainability/concept/run_no_roi.py" "${args[@]}"
fi
>>

configs/calibration.yaml codice <<
# Canonical calibration config (TRAIN + optional VAL kept separate).
# Dynamic inputs:
#   - export WDS_CALIB_TRAIN_DIR (required)
#   - export WDS_CALIB_VAL_DIR   (optional, but recommended)

experiment:
  name: "plip_calibration"
  seed: 1337
  # The runner ENFORCES canonical output via explainability.paths (metadata/).
  # Keep this field for backward compatibility with older configs, but it is ignored.
  outputs_root: null
  run_id: null
  use_runs: false

data:
  backend: "webdataset"
  num_workers: 8
  batch_size: 256
  max_patches: 0
  webdataset:
    train_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/train
    val_dir:   /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/val
    pattern: "shard-*.tar"
    image_key: "img.jpg;jpg;jpeg;png"
    meta_key: "meta.json;json"
    class_field: "class_label"

concepts:
  ontology_yaml: /home/mla_group_01/rcc-ssrl/src/explainability/configs/concepts_list.yaml
  score_scale: 100.0

plip:
  model_id: "vinid/plip"
  model_local_dir: null
  device: "cuda"
  precision: "fp16"
  hf_cache_dir: null

output:
  save_arrays: true
  save_topk_jsonl: false
  plots: true
>>

configs/concepts_list.yaml codice <<
version: 2
name: "rcc_histology_plip_v2"

global_instructions: >-
  Text prompts are short English descriptions of high-power H&E renal tumor
  patches. Each concept should be visually recognizable on a 224x224 tile.
  Prompts are designed for PLIP (CLIP-like). This file keeps the original
  32 concepts unchanged and extends them with extra concepts and prompt
  variants as multiple prompts per concept (no duplicate/alias concepts).

concepts:
  - id: 1
    name: "Diffuse clear cytoplasm"
    short_name: "ccrcc_clear_diffuse"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompts:
      - >-
        A high power H&E image of clear cell renal cell carcinoma with tumor cells
        showing diffusely clear, optically empty cytoplasm and small round nuclei.
      - >-
        High power H&E of clear cell RCC: sheets of tumor cells with optically
        empty, lipid-like clear cytoplasm and delicate cell membranes.

  - id: 2
    name: "Clear cells with delicate capillaries"
    short_name: "ccrcc_clear_capillaries"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompts:
      - >-
        A renal tumor H&E patch of clear cell renal cell carcinoma with nests of
        clear cells separated by a fine, delicate chicken-wire capillary network.
      - >-
        H&E renal tumor patch: clear cell RCC nests traversed by a fine
        branching capillary network, like chicken-wire vasculature.

  - id: 3
    name: "Alveolar or nested architecture"
    short_name: "ccrcc_alveolar_nested"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompts:
      - >-
        An H&E image of clear cell renal cell carcinoma showing discrete
        alveolar or nested groups of tumor cells surrounded by thin capillaries
        and small clear spaces between nests.
      - >-
        H&E renal tumor: clear cell RCC with rounded nests separated by
        capillary-rich septa, giving an alveolar nesting pattern.

  - id: 4
    name: "Mixed clear and eosinophilic cytoplasm"
    short_name: "ccrcc_mixed_cytoplasm"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompts:
      - >-
        A renal cell carcinoma H&E patch where most tumor cells have clear
        cytoplasm but a substantial subset shows more eosinophilic, granular
        cytoplasm within the same nests.

  - id: 5
    name: "Solid clear cell sheets"
    short_name: "ccrcc_solid_clear"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompts:
      - >-
        A high power H&E image of clear cell renal cell carcinoma with broad
        solid sheets of clear tumor cells and only sparse or inconspicuous
        capillaries.

  - id: 6
    name: "Microcystic clear cell pattern"
    short_name: "ccrcc_microcystic"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompts:
      - >-
        A renal tumor H&E patch showing clear cell renal cell carcinoma with
        numerous small microcystic or vacuolated spaces between clear tumor
        cells.

  - id: 7
    name: "Papillary fronds with fibrovascular cores"
    short_name: "prcc_papillary_fronds"
    group: "pRCC"
    primary_class: "pRCC"
    prompts:
      - >-
        A high power H&E image of papillary renal cell carcinoma with multiple
        papillary fronds, each containing a fibrovascular core lined by tumor
        epithelium.
      - >-
        H&E of papillary RCC showing true papillae with fibrovascular cores,
        lined by neoplastic epithelium in multiple finger-like projections.

  - id: 8
    name: "Type 1 papillary cytology"
    short_name: "prcc_type1"
    group: "pRCC"
    primary_class: "pRCC"
    prompts:
      - >-
        An H&E patch of papillary renal cell carcinoma type 1 with slender
        papillae lined by small cuboidal cells, scant pale cytoplasm and
        low-grade, basophilic nuclei.
      - >-
        H&E papillary RCC type 1: low-grade nuclei, small cuboidal cells, pale
        cytoplasm, papillae that look slender and delicate.

  - id: 9
    name: "Type 2 papillary cytology"
    short_name: "prcc_type2"
    group: "pRCC"
    primary_class: "pRCC"
    prompts:
      - >-
        An H&E patch of papillary renal cell carcinoma type 2 with papillae
        lined by larger eosinophilic tumor cells, higher nuclear grade and more
        prominent nucleoli.
      - >-
        H&E papillary RCC type 2: larger eosinophilic cells, higher nuclear
        grade, prominent nucleoli along papillary surfaces.

  - id: 10
    name: "Foamy macrophages in papillary cores"
    short_name: "prcc_foamy_macrophages"
    group: "pRCC"
    primary_class: "pRCC"
    prompts:
      - >-
        A papillary renal cell carcinoma H&E image where numerous foamy
        macrophages with vacuolated cytoplasm aggregate within papillary
        fibrovascular cores or luminal spaces.

  - id: 11
    name: "Psammoma bodies"
    short_name: "prcc_psammoma_bodies"
    group: "pRCC"
    primary_class: "pRCC"
    prompts:
      - >-
        A renal tumor H&E patch showing papillary structures with multiple small,
        round, concentrically laminated psammoma bodies scattered in papillae
        or stroma.

  - id: 12
    name: "Tubulopapillary architecture"
    short_name: "prcc_tubulopapillary"
    group: "pRCC"
    primary_class: "pRCC"
    prompts:
      - >-
        An H&E image of papillary renal cell carcinoma with complex short
        papillae and fused tubulopapillary structures rather than long, delicate
        papillary fronds.

  - id: 13
    name: "Hobnail nuclei along papillary lumens"
    short_name: "prcc_hobnail_nuclei"
    group: "pRCC"
    primary_class: "pRCC"
    prompts:
      - >-
        A papillary renal cell carcinoma H&E patch where tumor nuclei protrude
        into the papillary lumen, creating a hobnail or pseudostratified
        appearance along the luminal surface.

  - id: 14
    name: "Plant-cell borders"
    short_name: "chrcc_plant_cell_borders"
    group: "chRCC"
    primary_class: "CHROMO"
    prompts:
      - >-
        A chromophobe renal cell carcinoma H&E image showing polygonal tumor
        cells arranged in a mosaic with thick, sharply outlined plant-cell
        borders between adjacent cells.
      - >-
        H&E chromophobe RCC: mosaic of polygonal cells with thick, sharply
        outlined cell borders creating a plant-cell appearance.

  - id: 15
    name: "Perinuclear halos"
    short_name: "chrcc_perinuclear_halos"
    group: "chRCC"
    primary_class: "CHROMO"
    prompts:
      - >-
        A chromophobe renal cell carcinoma H&E patch with many tumor cells
        showing a clear perinuclear halo and more eosinophilic cytoplasm toward
        the cell periphery.
      - >-
        H&E chromophobe RCC: polygonal eosinophilic cells with conspicuous
        perinuclear clearing (halos) and crisp cytoplasmic borders.

  - id: 16
    name: "Pale reticulated cytoplasm"
    short_name: "chrcc_pale_reticulated"
    group: "chRCC"
    primary_class: "CHROMO"
    prompts:
      - >-
        An H&E image of chromophobe renal cell carcinoma where tumor cells have
        pale, finely reticulated eosinophilic cytoplasm, sometimes with subtle
        vesicular or cobweb-like texture.

  - id: 17
    name: "Raisinoid wrinkled nuclei"
    short_name: "chrcc_raisinoid_nuclei"
    group: "chRCC"
    primary_class: "CHROMO"
    prompts:
      - >-
        A chromophobe renal cell carcinoma H&E patch dominated by tumor cells
        with irregular, wrinkled raisinoid nuclei and relatively dense
        chromatin.

  - id: 18
    name: "Oncocytic chromophobe variant"
    short_name: "chrcc_oncocytic_variant"
    group: "chRCC"
    primary_class: "CHROMO"
    prompts:
      - >-
        An H&E image of chromophobe renal cell carcinoma with more dense
        eosinophilic cytoplasm but still showing perinuclear halos or plant-cell
        borders in many cells.

  - id: 19
    name: "Compact oncocytic cytoplasm"
    short_name: "onco_oncocytic_cytoplasm"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        A renal oncocytoma H&E patch with sheets of tumor cells having abundant,
        dense, finely granular, intensely eosinophilic oncocytic cytoplasm and
        relatively uniform round nuclei.
      - >-
        H&E oncocytoma: uniform tumor cells with dense granular eosinophilic
        cytoplasm (oncocytic), bland round nuclei, minimal pleomorphism.

  - id: 20
    name: "Archipelagenous oncocytoma islands"
    short_name: "onco_archipelagenous"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        An H&E image of renal oncocytoma showing round or oval islands of
        oncocytic tumor cells separated by pale, loose or myxoid stroma giving
        an archipelagenous pattern.
      - >-
        H&E renal oncocytoma: round islands of oncocytic cells separated by pale
        edematous or myxoid stroma, an archipelago pattern.

  - id: 21
    name: "Central fibrous scar"
    short_name: "onco_central_scar"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        A renal oncocytoma H&E patch containing dense central fibrous tissue with
        entrapped thick-walled vessels and surrounding nodules of oncocytic
        tumor cells.

  - id: 22
    name: "Thick-walled vessels in oncocytoma"
    short_name: "onco_thick_vessels"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        An H&E image of renal oncocytoma highlighting thick-walled hyalinized
        vessels running between nests of oncocytic tumor cells in loose
        stroma.

  - id: 23
    name: "Low-grade nuclei (ISUP 1–2)"
    short_name: "grade_low_isup12"
    group: "Grading"
    primary_class: null
    prompts:
      - >-
        A high power H&E patch of renal cell carcinoma with mostly small,
        uniform nuclei and inconspicuous or small nucleoli, consistent with low
        ISUP grade 1 or 2.

  - id: 24
    name: "High-grade nuclei (ISUP 3–4)"
    short_name: "grade_high_isup34"
    group: "Grading"
    primary_class: null
    prompts:
      - >-
        A renal cell carcinoma H&E patch with enlarged pleomorphic nuclei and
        prominent nucleoli easily visible at low power, compatible with high
        ISUP grade 3 or 4.

  - id: 25
    name: "Coagulative tumor necrosis"
    short_name: "necrosis_coagulative"
    group: "Necrosis"
    primary_class: null
    prompts:
      - >-
        An H&E image of renal cell carcinoma showing a sharp transition from
        viable tumor to areas of coagulative necrosis with eosinophilic ghost
        cells and granular debris.

  - id: 26
    name: "Sarcomatoid spindle-cell areas"
    short_name: "dediff_sarcomatoid"
    group: "Dedifferentiation"
    primary_class: null
    prompts:
      - >-
        A renal cell carcinoma H&E patch containing malignant spindle-cell
        fascicles with high nuclear atypia and loss of epithelial architecture,
        representing sarcomatoid transformation.

  - id: 27
    name: "Rhabdoid tumor cells"
    short_name: "dediff_rhabdoid"
    group: "Dedifferentiation"
    primary_class: null
    prompts:
      - >-
        An H&E image of renal cell carcinoma with scattered large rhabdoid cells
        having eccentric nuclei, prominent nucleoli and dense eosinophilic
        cytoplasmic inclusions.

  - id: 28
    name: "Tumor-associated hemorrhage"
    short_name: "other_hemorrhage"
    group: "Other"
    primary_class: null
    prompts:
      - >-
        A renal tumor H&E patch showing extensive fresh or organizing hemorrhage
        within and around nests of carcinoma cells.

  - id: 29
    name: "Lymphocytic inflammatory infiltrate"
    short_name: "other_lymphocytes"
    group: "Other"
    primary_class: null
    prompts:
      - >-
        An H&E image of renal cell carcinoma with a dense lymphocytic infiltrate
        in the tumor stroma or at the interface between tumor and surrounding
        kidney.

  - id: 30
    name: "Thick fibrous septa"
    short_name: "other_fibrous_septa"
    group: "Other"
    primary_class: null
    prompts:
      - >-
        A renal tumor H&E patch where nests or sheets of carcinoma cells are
        separated by broad, collagen-rich fibrous septa.

  - id: 31
    name: "Cystic or tubular spaces with eosinophilic fluid"
    short_name: "other_cystic_eosinophilic"
    group: "Other"
    primary_class: null
    prompts:
      - >-
        An H&E image of renal cell carcinoma containing cystic or tubular
        spaces lined by tumor cells and filled with eosinophilic or proteinaceous
        luminal material.

  - id: 32
    name: "Peritumoral normal kidney tissue"
    short_name: "other_peritumoral_kidney"
    group: "Other"
    primary_class: null
    prompts:
      - >-
        An H&E patch showing the interface between renal cell carcinoma and
        adjacent non-neoplastic kidney parenchyma with preserved tubules and
        glomeruli.

  # -------------------------
  # NEW PRIMARY DIAGNOSTIC CONCEPTS (to improve subtype separation)
  # -------------------------
  - id: 42
    name: "Tubular or acinar clear cell pattern"
    short_name: "ccrcc_tubular_acinar"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompts:
      - >-
        H&E clear cell RCC showing tubular or acinar structures lined by clear
        cells, with small lumina and delicate intervening capillaries.

  - id: 43
    name: "Macrocystic change in clear cell RCC"
    short_name: "ccrcc_macrocystic_change"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompts:
      - >-
        H&E clear cell RCC with large cystic spaces or pseudocysts, lined by
        clear tumor cells and thin septa with small vessels.

  - id: 45
    name: "Hemosiderin-laden macrophages in papillary RCC"
    short_name: "prcc_hemosiderin_macrophages"
    group: "pRCC"
    primary_class: "pRCC"
    prompts:
      - >-
        H&E papillary RCC with coarse golden-brown hemosiderin pigment in
        macrophages within papillary cores or adjacent hemorrhagic stroma.

  - id: 46
    name: "Hyalinized or sclerotic papillary cores"
    short_name: "prcc_hyalinized_cores"
    group: "pRCC"
    primary_class: "pRCC"
    prompts:
      - >-
        H&E papillary RCC showing papillae with sclerotic or hyalinized
        fibrovascular cores, sometimes thick and glassy beneath epithelium.

  - id: 47
    name: "Papillary RCC with crowded pseudostratified lining"
    short_name: "prcc_crowded_pseudostratified"
    group: "pRCC"
    primary_class: "pRCC"
    prompts:
      - >-
        H&E papillary RCC with crowded epithelial lining and nuclear
        pseudostratification along papillary surfaces, with true fibrovascular cores.

  - id: 48
    name: "Chromophobe RCC solid sheets with pale eosinophilic cells"
    short_name: "chrcc_solid_sheets_pale"
    group: "chRCC"
    primary_class: "CHROMO"
    prompts:
      - >-
        H&E chromophobe RCC showing broad solid sheets of pale eosinophilic
        polygonal cells with distinct cell borders and low mitotic activity.

  - id: 49
    name: "Chromophobe RCC binucleated tumor cells"
    short_name: "chrcc_binucleation"
    group: "chRCC"
    primary_class: "CHROMO"
    prompts:
      - >-
        H&E chromophobe RCC with frequent binucleated cells, raisinoid nuclear
        irregularity, and sharply outlined cell membranes.

  - id: 50
    name: "Chromophobe RCC cytoplasmic microvesicles / reticular clearing"
    short_name: "chrcc_microvesicular_cytoplasm"
    group: "chRCC"
    primary_class: "CHROMO"
    prompts:
      - >-
        H&E chromophobe RCC: tumor cells with pale, microvesicular or reticulated
        cytoplasm and perinuclear clearing, giving a finely vacuolated look.

  - id: 51
    name: "Oncocytoma nested / trabecular architecture"
    short_name: "onco_nested_trabecular"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        H&E renal oncocytoma showing nests and trabeculae of oncocytic cells
        separated by delicate stroma, without prominent perinuclear halos.

  - id: 52
    name: "Oncocytoma tubulocystic pattern"
    short_name: "onco_tubulocystic"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        H&E oncocytoma with small tubules or cystic spaces lined by oncocytic
        cells, with granular eosinophilic cytoplasm and bland nuclei.
      - >-
        H&E oncocytoma showing a tubulocystic pattern with variably sized cysts
        and small tubules lined by oncocytic cells.

  - id: 53
    name: "Oncocytoma edematous / myxoid stroma"
    short_name: "onco_edematous_stroma"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        H&E renal oncocytoma with pale edematous or myxoid stroma separating
        nests of oncocytic cells, producing a loose stromal background.

  - id: 54
    name: "Oncocytoma bland round nuclei"
    short_name: "onco_bland_round_nuclei"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        H&E oncocytoma: uniform round nuclei with minimal pleomorphism in
        oncocytic cells, lacking raisinoid wrinkling or strong perinuclear halos.

  # -------------------------
  # NEW NOT_TUMOR PRIMARY CONCEPTS (drive NOT_TUMOR separation)
  # -------------------------
  - id: 55
    name: "Normal renal glomerulus and Bowman space"
    short_name: "notumor_glomerulus_bowman"
    group: "NOT_TUMOR"
    primary_class: "NOT_TUMOR"
    prompts:
      - >-
        H&E normal kidney cortex showing a glomerular tuft with Bowman space and
        surrounding tubules, no atypical tumor nests.

  - id: 56
    name: "Proximal tubules with brush border"
    short_name: "notumor_proximal_tubules"
    group: "NOT_TUMOR"
    primary_class: "NOT_TUMOR"
    prompts:
      - >-
        H&E normal kidney with proximal convoluted tubules: eosinophilic
        cytoplasm and fuzzy luminal brush borders, orderly tubular profiles.

  - id: 57
    name: "Distal tubules / collecting ducts"
    short_name: "notumor_distal_collecting"
    group: "NOT_TUMOR"
    primary_class: "NOT_TUMOR"
    prompts:
      - >-
        H&E normal kidney with distal tubules or collecting ducts: paler cuboidal
        cells, cleaner lumina, and regular tubular arrangement.

  - id: 58
    name: "Renal medulla with collecting ducts and vasa recta"
    short_name: "notumor_medulla_collecting_vasa_recta"
    group: "NOT_TUMOR"
    primary_class: "NOT_TUMOR"
    prompts:
      - >-
        H&E renal medulla showing parallel collecting ducts and thin vascular
        channels (vasa recta), lacking tumor cell nests or papillary fronds.

  - id: 59
    name: "Arteriole / small artery in kidney parenchyma"
    short_name: "notumor_arteriole"
    group: "NOT_TUMOR"
    primary_class: "NOT_TUMOR"
    prompts:
      - >-
        H&E kidney parenchyma containing a small artery or arteriole with a
        muscular wall and endothelial-lined lumen, adjacent normal tubules.

  - id: 60
    name: "Chronic interstitial fibrosis with tubular atrophy"
    short_name: "notumor_interstitial_fibrosis_atrophy"
    group: "NOT_TUMOR"
    primary_class: "NOT_TUMOR"
    prompts:
      - >-
        H&E non-neoplastic kidney showing interstitial fibrosis and atrophic
        tubules, with preserved overall architecture and no malignant nests.

  - id: 61
    name: "Perirenal adipose tissue"
    short_name: "notumor_adipose"
    group: "NOT_TUMOR"
    primary_class: "NOT_TUMOR"
    prompts:
      - >-
        H&E showing mature adipose tissue with large clear fat vacuoles and thin
        septa, without clusters of atypical carcinoma cells.

  - id: 62
    name: "Urothelium / renal pelvis-type lining (if present)"
    short_name: "notumor_urothelium"
    group: "NOT_TUMOR"
    primary_class: "NOT_TUMOR"
    prompts:
      - >-
        H&E showing urothelial-type lining epithelium with multiple cell layers
        and a smooth surface, without papillary RCC tumor fronds.

  # -------------------------
  # OPTIONAL EXTRA ORTHOGONAL ATTRIBUTES (primary_class = null)
  # -------------------------
  - id: 63
    name: "Neutrophil-rich acute inflammation"
    short_name: "other_neutrophils_acute"
    group: "Other"
    primary_class: null
    prompts:
      - >-
        H&E patch with abundant neutrophils and acute inflammatory exudate in
        stroma or lumina, not a specific RCC subtype feature.

  - id: 64
    name: "Fibrovascular scar-like stroma (non-specific)"
    short_name: "other_scar_like_stroma"
    group: "Other"
    primary_class: null
    prompts:
      - >-
        H&E image with dense collagenized scar-like stroma and few cells,
        separating adjacent tissue elements; non-specific across diagnoses.

  - id: 65
    name: "Hyalinized thick basement membranes (non-specific)"
    short_name: "other_hyaline_basement_membranes"
    group: "Other"
    primary_class: null
    prompts:
      - >-
        H&E patch showing thick, hyalinized basement membrane-like material
        around tubules or small structures, not defining a subtype by itself.

  # -------------------------
  # NEW ONCO PRIMARY CONCEPTS (discriminative vs all other classes)
  # -------------------------
  - id: 66
    name: "Organoid packed nests (oncocytoma)"
    short_name: "onco_organoid_packed_nests"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        H&E renal oncocytoma with an organoid growth pattern of tightly packed
        nests of oncocytic cells and thin intervening stroma.

  - id: 67
    name: "Packed nests with small capillaries (oncocytoma)"
    short_name: "onco_nests_small_capillaries"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        H&E oncocytoma showing tight nests of oncocytic cells with many small
        capillaries in the stroma between nests.

  - id: 68
    name: "Less dense nests in edematous stroma (oncocytoma)"
    short_name: "onco_sparse_nests_edematous_stroma"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        H&E renal oncocytoma with oncocytic nests that are less densely arranged
        within abundant pale edematous or myxoid stroma.

  - id: 69
    name: "Peripheral rim around edematous stroma (oncocytoma)"
    short_name: "onco_peripheral_rim_edematous"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        H&E oncocytoma with a peripheral rim of oncocytic tumor cells surrounding
        abundant pale edematous stromal tissue.

  - id: 70
    name: "Polygonal oncocytic cells with prominent nucleoli"
    short_name: "onco_polygonal_cells_prominent_nucleoli"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        H&E oncocytoma with large polygonal cells, dense granular eosinophilic
        cytoplasm, round nuclei, and conspicuous nucleoli.

  - id: 71
    name: "Cystic oncocytoma pattern"
    short_name: "onco_cystic_pattern"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        H&E renal oncocytoma with cystic spaces and oncocytic tumor cells forming
        cystic structures separated by thin septa.

  - id: 73
    name: "Oncocytoma nests with round nuclei and no halos"
    short_name: "onco_round_nuclei_no_halos"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        H&E renal oncocytoma with nests of oncocytic cells, predominantly round
        nuclei, and no obvious perinuclear halos.

  - id: 74
    name: "Oncocytoma with uncommon binucleation and no zonal necrosis"
    short_name: "onco_uncommon_binucleation_no_necrosis"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        H&E oncocytoma with nests of oncocytic cells, binucleation uncommon, and
        no zonal coagulative necrosis.

  - id: 75
    name: "Moderately circumscribed oncocytoma nests"
    short_name: "onco_moderately_circumscribed_nests"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompts:
      - >-
        H&E renal oncocytoma showing a moderately circumscribed tumor composed of
        oncocytic nests in a clean stromal background.
>>

configs/concepts_shortlist.yaml codice <<
version: 1
rank_by: auc_ovr
k_primary: 8
k_confounds: 5
classes:
  CHROMO:
    primary:
    - chrcc_pale_reticulated
    - chrcc_oncocytic_variant
    - chrcc_perinuclear_halos
    - chrcc_microvesicular_cytoplasm
    - chrcc_binucleation
    confounds:
    - notumor_glomerulus_bowman
    - prcc_psammoma_bodies
    - ccrcc_mixed_cytoplasm
    - grade_high_isup34
    - onco_round_nuclei_no_halos
  NOT_TUMOR:
    primary:
    - notumor_interstitial_fibrosis_atrophy
    - other_lymphocytes
    - other_scar_like_stroma
    - notumor_distal_collecting
    - other_hyaline_basement_membranes
    - other_peritumoral_kidney
    - other_fibrous_septa
    - other_hemorrhage
    confounds:
    - prcc_hyalinized_cores
    - necrosis_coagulative
    - onco_peripheral_rim_edematous
    - dediff_sarcomatoid
  ONCO:
    primary:
    - onco_round_nuclei_no_halos
    - onco_bland_round_nuclei
    - onco_nested_trabecular
    - onco_uncommon_binucleation_no_necrosis
    - onco_oncocytic_cytoplasm
    - onco_polygonal_cells_prominent_nucleoli
    - onco_tubulocystic
    - onco_nests_small_capillaries
    confounds:
    - prcc_hobnail_nuclei
    - ccrcc_mixed_cytoplasm
    - prcc_psammoma_bodies
    - prcc_type2
    - grade_high_isup34
  ccRCC:
    primary:
    - ccrcc_solid_clear
    - ccrcc_macrocystic_change
    - ccrcc_clear_diffuse
    - ccrcc_clear_capillaries
    - ccrcc_tubular_acinar
    - ccrcc_alveolar_nested
    - ccrcc_microcystic
    confounds:
    - chrcc_solid_sheets_pale
    - dediff_rhabdoid
    - other_neutrophils_acute
    - grade_low_isup12
    - onco_archipelagenous
  pRCC:
    primary:
    - prcc_papillary_fronds
    - prcc_type1
    - prcc_crowded_pseudostratified
    - prcc_hemosiderin_macrophages
    - prcc_hyalinized_cores
    - prcc_tubulopapillary
    - prcc_foamy_macrophages
    - prcc_hobnail_nuclei
    confounds:
    - other_peritumoral_kidney
    - notumor_medulla_collecting_vasa_recta
    - other_hemorrhage
    - other_lymphocytes
    - onco_thick_vessels
relaxed_filters:
  ONCO:
  - primary_empty_after_filter_backoff_to_raw_rank
notes:
- 'primary concepts: primary_class matches class (NOT_TUMOR: primary_class==NOT_TUMOR
  OR legacy primary_class null AND group==Other)'
- 'confounds: high-ranked concepts where primary_class mismatches the class (possible
  leakage / dataset bias / prompt ambiguity)'
>>

configs/config_concept_plip.yaml codice <<
concepts:
  ontology_yaml: /home/mla_group_01/rcc-ssrl/src/explainability/configs/concepts_list.yaml
  score_scale: 100.0

plip:
  model_id: vinid/plip
  device: cuda
  precision: fp16
  model_local_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/hf_models/plip_vinid
  hf_cache_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/hf_cache

data:
  batch_size: 256
  num_workers: 8
  max_patches: 0
  webdataset:
    pattern: "shard-*.tar"
    image_key: "img.jpg;jpg;jpeg;png"
    meta_key: "meta.json;json"
    class_field: "class_label"

output:
  save_arrays: true
  plots: true
  formats: ["pdf", "png"]
  plots_dpi: 300
>>

configs/no_roi.yaml codice <<
# Canonical NO-ROI config (model-independent concept scoring on TEST).
#
# Minimal required runtime input:
#   - export WDS_TEST_DIR (unless you set data.webdataset.test_dir below)
#
# Optional overrides:
#   - export CALIBRATION_METADATA_DIR
#   - export CONCEPT_SHORTLIST_YAML

profiles:
  no_roi:
    inputs:
      # If null, runner uses (in priority): CLI override -> env -> canonical explainability.paths
      calibration_metadata_dir: null
      shortlist_yaml: null
      use_shortlist_only: true

    data:
      backend: "webdataset"
      num_workers: 8
      batch_size: 256
      max_patches: 0
      webdataset:
        test_dir: null
        pattern: "shard-*.tar"
        image_key: "img.jpg;jpg;jpeg;png"
        meta_key: "meta.json;json"
        class_field: "class_label"

    plip:
      model_id: "vinid/plip"
      model_local_dir: null
      device: "cuda"
      precision: "fp16"
      hf_cache_dir: null

    output:
      plots: true
      formats: ["pdf", "png"]
      plots_dpi: 300
>>

configs/roi.yaml codice <<
# Concept ROI scoring — per model, uses spatial outputs (attention rollout masks).
# Unified defaults: NO-ROI profile is included under profiles.no_roi.

inputs:
  calibration_metadata_dir: /home/mla_group_01/rcc-ssrl/src/explainability/output/calibration/metadata
  shortlist_yaml: /home/mla_group_01/rcc-ssrl/src/explainability/output/calibration/analysis/concepts_shortlist.yaml

plip:
  model_id: vinid/plip
  device: cuda
  precision: fp16
  model_local_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/hf_models/plip_vinid
  hf_cache_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/hf_cache

roi:
  mode: crop
  quantile: 0.90
  min_area_frac: 0.01
  pad_frac: 0.05
  save_overlay: true
  save_roi_crops: true

selection:
  per_class:
    topk_tp: 2
    topk_fp: 2
    topk_fn: 2
  global_low_conf:
    topk: 0

xai:
  attn_rollout:
    discard_ratio: 0.90

data:
  backend: "webdataset"
  img_size: 224
  imagenet_norm: true
  num_workers: 8
  webdataset:
    test_dir: ""
    pattern: "shard-*.tar"
    image_key: "img.jpg;jpg;jpeg;png"
    meta_key: "meta.json;json"
  imagefolder:
    test_dir: ""
    batch_size: 64

profiles:
  no_roi:
    inputs:
      calibration_metadata_dir: /home/mla_group_01/rcc-ssrl/src/explainability/output/calibration/metadata
      shortlist_yaml: /home/mla_group_01/rcc-ssrl/src/explainability/output/calibration/analysis/concepts_shortlist.yaml
      use_shortlist_only: true
    plip:
      model_id: vinid/plip
      device: cuda
      precision: fp16
      model_local_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/hf_models/plip_vinid
      hf_cache_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/hf_cache
    data:
      batch_size: 256
      num_workers: 8
      max_patches: 0
      webdataset:
        test_dir: /beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test
        pattern: "shard-*.tar"
        image_key: "img.jpg;jpg;jpeg;png"
        meta_key: "meta.json;json"
        class_field: "class_label"
    output:
      save_arrays: true
      plots: true
      formats: ["pdf", "png"]
      plots_dpi: 300
>>

configs/spatial.yaml codice <<
# Mirror of config_xai.yaml (unified spatial config)

experiment:
  name: "xai_ssl_rcc_vit_spatial"
  seed: 1337
  outputs_root: "/home/mla_group_01/rcc-ssrl/src/explainability/output/spatial"
  run_id: "default"

evaluation_inputs:
  eval_run_dir: null
  predictions_csv: "predictions.csv"
  logits_npy: "logits_test.npy"

data:
  backend: "webdataset"
  img_size: 224
  imagenet_norm: true
  num_workers: 8
  batch_size: 1
  webdataset:
    test_dir: null          # overridden
    pattern: "shard-*.tar"
    image_key: "img.jpg;jpg;jpeg;png"
    meta_key: "meta.json;json"
  imagefolder:
    test_dir: null          # optional

labels:
  class_order: ["ccRCC", "pRCC", "CHROMO", "ONCO", "NOT_TUMOR"]

model:
  name: null               # overridden per ablation
  arch_hint: "ssl_linear"
  backbone_name: "vit_small_patch16_224"
  ssl_backbone_ckpt: null  # overridden per ablation
  ssl_head_ckpt: null      # overridden per ablation

selection:
  per_class:
    topk_tp: 5
    topk_fp: 3
    topk_fn: 3
  global_low_conf:
    topk: 3
  min_per_class: 10

xai:
  methods: ["attn_rollout", "gradcam", "ig"]

  gradcam:
    target_layer: "backbone.model.norm"

  ig:
    steps: 25
    baseline: "black"

  occlusion:
    window: 32
    stride: 16

  attn_rollout:
    head_fusion: "mean"
    discard_ratio: 0.9

runtime:
  device: "cuda"
  precision: "fp32"
>>

__init__.py codice <<
# empty – marks "explainability" as a package
>>

paths.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single source of truth for explainability filesystem layout + central configs.

Goals:
  - One canonical layout (no timestamped runs) with optional env override.
  - Centralised config directory: src/explainability/configs/
  - Outputs stay under src/explainability/output/ by default.
  - Avoid hard-coded absolute paths inside code; compute relative to repo.

Back-compat:
  - Default artifact root is src/explainability/output unless XAI_ROOT is set.

New (unified pipeline):
  - Light outputs (stats-only) remain under src/explainability/output/...
  - Heavy per-patch artifacts (input/rollout/ROI/overlays) live under each model root on scratch:
      <MODEL_ROOT>/attention_rollout_concept/run_<RUN_ID>/
  - Experiment discovery helpers for scratch model runs live here as the single source of truth.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# Repo / package roots
# ---------------------------------------------------------------------

EXPLAINABILITY_DIR = Path(__file__).resolve().parent  # .../src/explainability
SRC_DIR = EXPLAINABILITY_DIR.parent                   # .../src
REPO_ROOT = SRC_DIR.parent                            # .../ (repo root)

# Centralised configs directory
CONFIG_DIR = EXPLAINABILITY_DIR / "configs"
# Canonical output root (default)
OUTPUT_DIR = EXPLAINABILITY_DIR / "output"

# Scratch model root (defaults for the RCC cluster)
MODELS_ROOT_DEFAULT = Path(
    "/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/models"
)


def _env_path(key: str) -> Optional[Path]:
    v = os.getenv(key, "").strip()
    if not v:
        return None
    return Path(v)


# Canonical artifacts root.
# Default keeps outputs under src/explainability/output unless XAI_ROOT is set.
XAI_ROOT = _env_path("XAI_ROOT") or OUTPUT_DIR


def resolve_config(path_or_name: Union[str, Path]) -> Path:
    """
    Resolve a config file path.
      - If an existing path is provided -> return it.
      - Else interpret it as a filename under CONFIG_DIR.
    """
    p = Path(path_or_name)
    if p.exists():
        return p
    return CONFIG_DIR / str(path_or_name)


def resolve_models_root(models_root: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve the scratch models root:
      - explicit models_root arg wins
      - else env MODELS_ROOT
      - else MODELS_ROOT_DEFAULT
    """
    if models_root is not None:
        return Path(models_root).expanduser()
    return (_env_path("MODELS_ROOT") or MODELS_ROOT_DEFAULT).expanduser()


# ---------------------------------------------------------------------
# Layout dataclasses
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationLayout:
    root_dir: Path
    metadata_dir: Path
    analysis_dir: Path
    report_dir: Path
    shortlist_dir: Path
    shortlist_yaml: Path
    shortlist_json: Path

    @property
    def configs_dir(self) -> Path:
        # Legacy alias (shortlist artifacts no longer live under configs/).
        return self.shortlist_dir


@dataclass(frozen=True)
class NoRoiLayout:
    root_dir: Path
    artifacts_dir: Path
    plots_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class SpatialLayout:
    """
    Model-dependent spatial XAI layout.
    Stored under <XAI_ROOT>/spatial/<MODEL_ID>/...
    """
    root_dir: Path
    artifacts_dir: Path
    plots_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class RoiConceptLayout:
    """
    Model-dependent concept XAI with ROI masks (depends on spatial outputs).
    Light artifacts (arrays/JSON) are stored under <XAI_ROOT>/roi/<MODEL_ID>/...
    Heavy ROI crops/overlays are stored under <MODEL_ROOT>/xai/roi/...
    """
    root_dir: Path
    artifacts_dir: Path
    rois_dir: Path
    figures_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class ComparisonLayout:
    root_dir: Path
    figures_dir: Path
    summary_csv: Path
    report_md: Path


@dataclass(frozen=True)
class SpatialConceptHeavyLayout:
    """
    Heavy, per-patch artifacts for unified spatial+concept XAI.
    Stored under: <MODEL_ROOT>/attention_rollout_concept/run_<RUN_ID>/
    """
    root_dir: Path
    selection_dir: Path
    items_dir: Path
    selection_json: Path
    summary_csv: Path
    summary_json: Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Canonical layout builders
# ---------------------------------------------------------------------

def calibration_layout(root: Optional[Path] = None) -> CalibrationLayout:
    """
    Canonical calibration + deep validation layout.
    shortlist_dir is under the analysis output (no artifacts in configs/).
    """
    base = (Path(root) if root is not None else XAI_ROOT) / "calibration"
    meta = base / "metadata"
    analysis = base / "analysis"
    report = analysis / "report"
    shortlist_dir = analysis
    return CalibrationLayout(
        root_dir=base,
        metadata_dir=meta,
        analysis_dir=analysis,
        report_dir=report,
        shortlist_dir=shortlist_dir,
        shortlist_yaml=shortlist_dir / "concepts_shortlist.yaml",
        shortlist_json=shortlist_dir / "concepts_shortlist.json",
    )


def no_roi_layout(root: Optional[Path] = None) -> NoRoiLayout:
    """
    Canonical NO-ROI concept scoring on TEST (model-independent).
    """
    base = (Path(root) if root is not None else XAI_ROOT) / "no_roi"
    return NoRoiLayout(
        root_dir=base,
        artifacts_dir=base / "artifacts",
        plots_dir=base / "plots",
        logs_dir=base / "logs",
    )

def _model_id(model_root: Union[str, Path]) -> str:
    return Path(model_root).name


def model_xai_root(model_root: Union[str, Path]) -> Path:
    """
    Legacy helper retained for compatibility with older code.
    Prefer spatial_layout/roi_concept_layout for canonical outputs under XAI_ROOT.
    """
    return Path(model_root) / "xai"


def spatial_layout(model_root: Union[str, Path]) -> SpatialLayout:
    base = XAI_ROOT / "spatial" / _model_id(model_root)
    return SpatialLayout(
        root_dir=base,
        artifacts_dir=base / "artifacts",
        plots_dir=base / "plots",
        logs_dir=base / "logs",
    )


def roi_concept_layout(model_root: Union[str, Path]) -> RoiConceptLayout:
    model_root_p = Path(model_root)
    base = XAI_ROOT / "roi" / _model_id(model_root_p)
    # Heavy outputs (per-sample crops/overlays) must not live under the repo output.
    heavy = model_root_p / "xai" / "roi"
    return RoiConceptLayout(
        root_dir=base,
        artifacts_dir=base / "artifacts",
        rois_dir=heavy / "rois",
        figures_dir=heavy / "figures",
        logs_dir=base / "logs",
    )


def comparison_layout(model_id: str) -> ComparisonLayout:
    base = XAI_ROOT / "roi-no_roi-comparision" / str(model_id)
    tables_dir = base / "tables"
    return ComparisonLayout(
        root_dir=base,
        figures_dir=base / "figures",
        summary_csv=tables_dir / "roi_vs_no_roi_summary.csv",
        report_md=base / "report.md",
    )


def spatial_concept_heavy_layout(model_root: Union[str, Path], run_id: str) -> SpatialConceptHeavyLayout:
    """
    Heavy artifacts layout for unified spatial+concept XAI (per model root).
    """
    mr = Path(model_root)
    root = mr / "attention_rollout_concept" / f"run_{str(run_id)}"
    selection_dir = root / "selection"
    items_dir = root / "items"
    return SpatialConceptHeavyLayout(
        root_dir=root,
        selection_dir=selection_dir,
        items_dir=items_dir,
        selection_json=selection_dir / "xai_selection.json",
        summary_csv=root / "xai_summary.csv",
        summary_json=root / "xai_summary.json",
    )


def ensure_spatial_concept_heavy_layout(layout: SpatialConceptHeavyLayout) -> SpatialConceptHeavyLayout:
    _ensure_dir(layout.root_dir)
    _ensure_dir(layout.selection_dir)
    _ensure_dir(layout.items_dir)
    return layout


# ---------------------------------------------------------------------
# Ensure helpers (used by runners)
# ---------------------------------------------------------------------

def ensure_calibration_layout(layout: Optional[CalibrationLayout] = None) -> CalibrationLayout:
    l = layout or calibration_layout()
    _ensure_dir(l.root_dir)
    _ensure_dir(l.metadata_dir)
    _ensure_dir(l.analysis_dir)
    _ensure_dir(l.report_dir)
    _ensure_dir(l.shortlist_dir)
    return l


def ensure_no_roi_layout(layout: Optional[NoRoiLayout] = None) -> NoRoiLayout:
    l = layout or no_roi_layout()
    _ensure_dir(l.root_dir)
    _ensure_dir(l.artifacts_dir)
    _ensure_dir(l.plots_dir)
    _ensure_dir(l.logs_dir)
    return l


def ensure_spatial_layout(layout: SpatialLayout) -> SpatialLayout:
    _ensure_dir(layout.artifacts_dir)
    _ensure_dir(layout.plots_dir)
    _ensure_dir(layout.logs_dir)
    return layout


def ensure_roi_concept_layout(model_root: Union[str, Path, RoiConceptLayout]) -> RoiConceptLayout:
    if isinstance(model_root, RoiConceptLayout):
        layout = model_root
    else:
        layout = roi_concept_layout(model_root)
    _ensure_dir(layout.artifacts_dir)
    # Heavy dirs live under model_root (scratch) - still ensure them.
    _ensure_dir(layout.rois_dir)
    _ensure_dir(layout.figures_dir)
    _ensure_dir(layout.logs_dir)
    return layout


def ensure_roi_layout(model_root: Union[str, Path]) -> RoiConceptLayout:
    """
    Backward-compatible alias used by run_spatial-concept.py.
    """
    return ensure_roi_concept_layout(roi_concept_layout(model_root))


def ensure_comparison_layout(model_id: str) -> ComparisonLayout:
    l = comparison_layout(model_id)
    _ensure_dir(l.figures_dir)
    _ensure_dir(l.summary_csv.parent)
    _ensure_dir(l.report_md.parent)
    return l


def get_heavy_xai_dir(model_root: Union[str, Path], run_id: str, *, kind: str = "spatial_concept") -> Path:
    """
    Resolve heavy XAI directory under a model root.
    kind:
      - spatial_concept -> <MODEL_ROOT>/attention_rollout_concept/run_<RUN_ID>/
    """
    kind = str(kind).strip().lower()
    mr = Path(model_root)
    if kind in ("spatial_concept", "attention_rollout_concept", "roi"):
        return mr / "attention_rollout_concept" / f"run_{str(run_id)}"
    # Default fallback: keep heavy XAI under model_root/xai/<kind>/run_<id>
    return mr / "xai" / kind / f"run_{str(run_id)}"


def get_item_out_dir(model_root: Union[str, Path], run_id: str, idx: int, *, kind: str = "spatial_concept") -> Path:
    """
    Item output dir for a single selected sample under the heavy layout.
    Canonical name uses 8 digits: idx_00001234
    """
    base = get_heavy_xai_dir(model_root, run_id, kind=kind)
    return Path(base) / "items" / f"idx_{int(idx):08d}"


def get_light_stats_dir(kind: str, model_id: str) -> Path:
    """
    Resolve the canonical light (stats-only) output directory inside the repo.
    """
    kind_norm = str(kind).strip().lower()
    mid = str(model_id)
    if kind_norm in ("spatial", "spatial_stats", "stats_spatial"):
        return OUTPUT_DIR / "spatial" / mid
    if kind_norm in ("roi", "roi_stats", "stats_roi"):
        return OUTPUT_DIR / "roi" / mid
    if kind_norm in ("roi-no_roi-comparision", "comparision", "comparison"):
        return OUTPUT_DIR / "roi-no_roi-comparision" / mid
    if kind_norm in ("no_roi", "no-roi"):
        return OUTPUT_DIR / "no_roi"
    if kind_norm in ("calibration",):
        return OUTPUT_DIR / "calibration"
    return OUTPUT_DIR / kind_norm / mid


# ---------------------------------------------------------------------
# Experiment discovery + resolvers (scratch models)
# ---------------------------------------------------------------------

def iter_exp_roots(models_root: Union[str, Path], exp_prefix: str) -> Iterator[Path]:
    """
    Iterate experiment roots under models_root matching exp_prefix (sorted by name).
    """
    root = Path(models_root)
    if not root.exists() or not root.is_dir():
        return iter(())
    exps = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(str(exp_prefix))]
    exps = sorted(exps, key=lambda p: p.name)
    return iter(exps)


def iter_ablation_dirs(exp_root: Union[str, Path]) -> Iterator[Path]:
    """
    Iterate ablation dirs under an exp root (sorted by name).
    Expected pattern: exp_*_ablXX
    """
    er = Path(exp_root)
    if not er.exists() or not er.is_dir():
        return iter(())
    abls = [p for p in er.iterdir() if p.is_dir() and ("_abl" in p.name)]
    abls = sorted(abls, key=lambda p: p.name)
    return iter(abls)


def resolve_checkpoints(ablation_dir: Union[str, Path]) -> Optional[Dict[str, Path]]:
    """
    Resolve required checkpoints under an ablation dir.
    Returns dict with keys:
      - ssl_backbone_ckpt
      - ssl_head_ckpt
    """
    ad = Path(ablation_dir)
    ckpt_dir = ad / "checkpoints"
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        return None

    # Backbone: *_ssl_best.pt but NOT *_ssl_linear_best.pt
    backbone = sorted(
        [p for p in ckpt_dir.glob("*_ssl_best.pt") if "linear" not in p.name.lower()],
        key=lambda p: p.name,
    )
    head = sorted(list(ckpt_dir.glob("*_ssl_linear_best.pt")), key=lambda p: p.name)
    if not backbone or not head:
        return None

    return {
        "ssl_backbone_ckpt": backbone[-1],
        "ssl_head_ckpt": head[-1],
    }


def resolve_latest_eval_dir(ablation_dir: Union[str, Path], pattern: str = "*_ssl_linear_best*") -> Optional[Path]:
    """
    Resolve latest eval dir for an ablation:
      <ablation_dir>/eval/<something matching pattern>/<TIMESTAMP>/
    Chooses latest TIMESTAMP (lexicographic), and if multiple parents match, chooses
    the latest (parent, timestamp) lexicographically.
    """
    ad = Path(ablation_dir)
    eval_root = ad / "eval"
    if not eval_root.exists() or not eval_root.is_dir():
        return None

    parents = sorted(
        [p for p in eval_root.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, pattern)],
        key=lambda p: p.name,
    )
    candidates: List[Tuple[str, str, Path]] = []
    for par in parents:
        ts_dirs = sorted([d for d in par.iterdir() if d.is_dir()], key=lambda p: p.name)
        if not ts_dirs:
            continue
        ts = ts_dirs[-1]
        candidates.append((par.name, ts.name, ts))
    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda t: (t[0], t[1]))
    return candidates[-1][2]


# ---------------------------------------------------------------------
# Canonical exported constants
# ---------------------------------------------------------------------

CALIBRATION_PATHS = calibration_layout()
NO_ROI_PATHS = no_roi_layout()

# Central config file defaults (optional convenience)
CALIBRATION_CONFIG_YAML = CONFIG_DIR / "calibration.yaml"
NO_ROI_CONFIG_YAML = CONFIG_DIR / "no_roi.yaml"
SPATIAL_CONFIG_YAML = CONFIG_DIR / "spatial.yaml"
SPATIAL_CONCEPT_CONFIG_YAML = CONFIG_DIR / "roi.yaml"
CONCEPT_PLIP_CONFIG_YAML = CONFIG_DIR / "config_concept_plip.yaml"
CONCEPTS_LIST_YAML = CONFIG_DIR / "concepts_list.yaml"
CONCEPT_SHORTLIST_YAML_CFG = CONFIG_DIR / "concepts_shortlist.yaml"
CONCEPT_SHORTLIST_JSON_CFG = CONFIG_DIR / "concepts_shortlist.json"
CONCEPT_SHORTLIST_FLAT_CSV_CFG = CONFIG_DIR / "concepts_shortlist_flat.csv"


__all__ = [
    "EXPLAINABILITY_DIR",
    "SRC_DIR",
    "REPO_ROOT",
    "MODELS_ROOT_DEFAULT",
    "resolve_models_root",
    "XAI_ROOT",
    "CONFIG_DIR",
    "OUTPUT_DIR",
    "resolve_config",
    "CalibrationLayout",
    "NoRoiLayout",
    "SpatialLayout",
    "RoiConceptLayout",
    "ComparisonLayout",
    "SpatialConceptHeavyLayout",
    "CALIBRATION_PATHS",
    "NO_ROI_PATHS",
    "CALIBRATION_CONFIG_YAML",
    "NO_ROI_CONFIG_YAML",
    "SPATIAL_CONFIG_YAML",
    "SPATIAL_CONCEPT_CONFIG_YAML",
    "CONCEPT_PLIP_CONFIG_YAML",
    "CONCEPTS_LIST_YAML",
    "CONCEPT_SHORTLIST_YAML_CFG",
    "CONCEPT_SHORTLIST_JSON_CFG",
    "CONCEPT_SHORTLIST_FLAT_CSV_CFG",
    "ensure_calibration_layout",
    "ensure_no_roi_layout",
    "model_xai_root",
    "spatial_layout",
    "roi_concept_layout",
    "ensure_spatial_layout",
    "ensure_roi_concept_layout",
    "ensure_roi_layout",
    "comparison_layout",
    "ensure_comparison_layout",
    "spatial_concept_heavy_layout",
    "ensure_spatial_concept_heavy_layout",
    "get_heavy_xai_dir",
    "get_item_out_dir",
    "get_light_stats_dir",
    "iter_exp_roots",
    "iter_ablation_dirs",
    "resolve_checkpoints",
    "resolve_latest_eval_dir",
]
>>

run_comparision.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

"""
Compare ROI vs NO-ROI concept scores.

Inputs:
  - NO-ROI canonical:
      src/explainability/output/no_roi/artifacts/{scores_fp32.npy, keys.npy}
  - ROI per model:
      src/explainability/output/roi/<MODEL_ID>/artifacts/{scores_fp32.npy, keys.npy, selected_concepts.json}

Outputs (requested canonical path, typo kept):
  - src/explainability/output/roi-no_roi-comparision/<MODEL_ID>/
      tables/roi_vs_no_roi_summary.csv
      figures/top_abs_delta.png/pdf
      report.md
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

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

from ..spatial.eval_utils import atomic_write_text
from ..paths import (
    CONFIG_DIR,
    NO_ROI_PATHS,
    ensure_comparison_layout,
    roi_concept_layout,
)
import yaml


def _load_selected_concepts(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text())
    sel = obj.get("selected", [])
    if not isinstance(sel, list) or not sel:
        raise RuntimeError(f"Invalid selected_concepts.json: {path}")
    return sel


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ROI vs NO-ROI comparison (paper-ready).")
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument(
        "--no-roi-root",
        type=Path,
        default=NO_ROI_PATHS.root_dir,
        help="NO-ROI canonical root.",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "comparision.yaml",
        help="Config YAML (optional, default: explainability/configs/comparision.yaml)",
    )
    ap.add_argument("--model-id", type=str, default=None, help="Override model id (folder name).")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--log-level", type=str, default="INFO")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("comparision_roi_no_roi")

    topk_cfg = None
    if args.config and args.config.exists():
        try:
            cfg = yaml.safe_load(args.config.read_text())
            topk_cfg = int(cfg.get("topk", args.topk))
        except Exception as e:
            log.warning("Failed to read comparison config %s: %s", args.config, e)

    model_root = args.model_root.resolve()
    if not model_root.exists():
        raise FileNotFoundError(f"model_root not found: {model_root}")

    model_id = args.model_id or model_root.name
    lay = ensure_comparison_layout(model_id=model_id)
    roi_root = roi_concept_layout(Path(model_id)).root_dir

    # NO-ROI
    no_roi_art = args.no_roi_root / "artifacts"
    no_scores_p = no_roi_art / "scores_fp32.npy"
    no_keys_p = no_roi_art / "keys.npy"
    if not no_scores_p.exists() or not no_keys_p.exists():
        raise FileNotFoundError(
            f"NO-ROI artifacts missing under {no_roi_art} (run no_roi first)."
        )
    no_scores = np.load(no_scores_p)
    no_keys = np.load(no_keys_p, allow_pickle=True).astype(object).tolist()
    no_map = {str(k): i for i, k in enumerate(no_keys)}

    # ROI
    roi_art = roi_root / "artifacts"
    roi_scores_p = roi_art / "scores_fp32.npy"
    roi_keys_p = roi_art / "keys.npy"
    sel_json_p = roi_art / "selected_concepts.json"
    if not roi_scores_p.exists() or not roi_keys_p.exists() or not sel_json_p.exists():
        raise FileNotFoundError(
            f"ROI artifacts missing under {roi_art} (run ROI stage first)."
        )
    roi_scores = np.load(roi_scores_p)
    roi_keys = np.load(roi_keys_p, allow_pickle=True).astype(object).tolist()
    sel = _load_selected_concepts(sel_json_p)
    concept_names = [str(x.get('concept_short_name')) for x in sel]

    if roi_scores.ndim != 2 or no_scores.ndim != 2:
        raise RuntimeError("scores arrays must be 2D")
    if roi_scores.shape[1] != len(concept_names):
        raise RuntimeError(
            f"ROI scores dim mismatch: scores.shape[1]={roi_scores.shape[1]} vs selected={len(concept_names)}"
        )
    if no_scores.shape[1] != roi_scores.shape[1]:
        # This should not happen if both use shortlist union; fail loudly.
        raise RuntimeError(
            f"NO-ROI concept dim != ROI concept dim: no={no_scores.shape[1]} roi={roi_scores.shape[1]}"
        )

    # Align keys: compare only intersection (ROI keys are subset)
    aligned_roi = []
    aligned_no = []
    aligned_keys = []
    missing = 0
    for i, k in enumerate(roi_keys):
        kk = str(k)
        j = no_map.get(kk, None)
        if j is None:
            missing += 1
            continue
        aligned_roi.append(roi_scores[i])
        aligned_no.append(no_scores[j])
        aligned_keys.append(kk)

    if not aligned_keys:
        raise RuntimeError(
            f"No overlapping keys between ROI and NO-ROI. missing_roi_in_no_roi={missing}"
        )

    A = np.asarray(aligned_roi, dtype=np.float32)
    B = np.asarray(aligned_no, dtype=np.float32)
    D = A - B

    mean_delta = D.mean(axis=0)
    mean_abs = np.abs(D).mean(axis=0)

    df = pd.DataFrame(
        {
            "concept_short_name": concept_names,
            "mean_delta_roi_minus_no_roi": mean_delta.astype(np.float64),
            "mean_abs_delta": mean_abs.astype(np.float64),
        }
    ).sort_values("mean_abs_delta", ascending=False)

    lay.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(lay.summary_csv, index=False)

    # Plot top-K by abs delta
    topk = max(5, int(topk_cfg if topk_cfg is not None else args.topk))
    df_top = df.head(topk)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, max(4, 0.35 * len(df_top))))
        ax = fig.add_subplot(111)
        ax.barh(np.arange(len(df_top)), df_top["mean_abs_delta"].values)
        ax.set_yticks(np.arange(len(df_top)))
        ax.set_yticklabels(df_top["concept_short_name"].tolist(), fontsize=9)
        ax.set_xlabel("mean |ROI - NO-ROI|")
        ax.set_title(f"{model_id}: ROI vs NO-ROI — top-{topk} concepts by mean absolute delta")
        fig.tight_layout()
        fig.savefig(lay.figures_dir / "top_abs_delta.png", dpi=300, bbox_inches="tight")
        fig.savefig(lay.figures_dir / "top_abs_delta.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning("Plot failed: %s", str(e))

    report = []
    report.append("# ROI vs NO-ROI comparison\n")
    report.append(f"- model_id: `{model_id}`")
    report.append(f"- model_root: `{model_root}`")
    report.append(f"- no_roi_root: `{args.no_roi_root}`")
    report.append(f"- n_overlap: **{len(aligned_keys)}** (missing={missing})\n")
    report.append("## Summary\n")
    report.append(f"- table: `{lay.summary_csv}`")
    report.append(f"- figure: `{lay.figures_dir / 'top_abs_delta.png'}`\n")
    report.append("## Top concepts by mean_abs_delta\n")
    report.append(df_top.to_markdown(index=False))
    report.append("")

    atomic_write_text(lay.report_md, "\n".join(report) + "\n")

    log.info("Comparision done: %s", lay.root)
    log.info("  - %s", lay.summary_csv)
    log.info("  - %s", lay.report_md)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
>>

run_spatial-concept.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

"""
Unified spatial + concept XAI over all SSL model ablations.

Heavy artifacts (per-patch images, rollout, ROI, per-item concept scores) are written under each model root:
  <MODEL_ROOT>/attention_rollout_concept/run_<RUN_ID>/

Light summaries (CSV/JSON + pointers to heavy artifacts) are written under the repo:
  src/explainability/output/spatial/<MODEL_ID>/spatial_concept/
  src/explainability/output/roi/<MODEL_ID>/spatial_concept/

This runner:
  - scans scratch models root for experiment runs (exp_<DATE>_...) and their ablations
  - resolves checkpoints + latest eval dir
  - selects a small subset of test samples (TP/FP/FN/low-conf) using existing selection utils
  - computes ViT attention rollout heatmaps
  - derives deterministic ROI from rollout
  - scores PLIP concepts on the ROI using the precomputed calibration text features + shortlist
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

from explainability.utils.bootstrap import bootstrap_package

bootstrap_package(__file__, globals())

from explainability.paths import (  # noqa: E402
    CALIBRATION_PATHS,
    CONCEPT_PLIP_CONFIG_YAML,
    CONCEPT_SHORTLIST_YAML_CFG,
    MODELS_ROOT_DEFAULT,
    SPATIAL_CONCEPT_CONFIG_YAML,
    ensure_spatial_concept_heavy_layout,
    get_item_out_dir,
    get_light_stats_dir,
    iter_ablation_dirs,
    iter_exp_roots,
    resolve_checkpoints,
    resolve_latest_eval_dir,
    spatial_concept_heavy_layout,
)
from explainability.spatial.eval_utils import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    build_preprocess,
    ensure_dir,
    iter_wds_filtered_by_keys,
    load_eval_artifacts,
    make_imgfolder_loader,
    make_wds_loader_with_keys,
    select_items,
    tensor_to_pil,
)
from explainability.spatial.ssl_linear_loader import SSLLinearClassifier  # noqa: E402
from explainability.spatial.attention_rollout import ViTAttentionRollout, overlay_heatmap  # noqa: E402

from explainability.concept.utils.class_utils  import (  # noqa: E402
    canon_class,
    concept_indices_for_patch,
    idx_to_class,
    load_class_names,
    load_shortlist_idx,
)
from explainability.utils.roi_utils import extract_bbox_from_mask  # noqa: E402
from explainability.concept.plip.plip_model import encode_images, load_plip  # noqa: E402
from explainability.concept.plip.scoring import score  # noqa: E402


def _safe_load_yaml(path: Optional[Path], log: logging.Logger) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        if not path.exists():
            return {}
        obj = yaml.safe_load(path.read_text())
        return obj if isinstance(obj, dict) else {}
    except Exception as e:
        log.warning("Failed to read YAML: %s (%s)", path, e)
        return {}


def _deep_get(cfg: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _parse_quantile(thr: float) -> float:
    """
    Accept either quantile in [0,1] or percentile in (1,100].
    """
    t = float(thr)
    if t > 1.0:
        t = t / 100.0
    return _clamp01(t)


def _min_area_frac(min_area: float, img_w: int, img_h: int) -> float:
    """
    Accept either:
      - fraction in (0,1]
      - absolute pixels (>1) converted to fraction.
    """
    v = float(min_area)
    if v <= 0.0:
        return 0.0
    if v <= 1.0:
        return float(v)
    denom = float(max(1, int(img_w) * int(img_h)))
    return float(v) / denom


def _load_text_features(text_features_pt: Path) -> Optional[torch.Tensor]:
    try:
        obj = torch.load(text_features_pt, map_location="cpu")
    except Exception:
        return None
    if torch.is_tensor(obj):
        tf = obj
    elif isinstance(obj, dict):
        tf = obj.get("text_features", None) or obj.get("features", None)
    else:
        tf = None
    if tf is None or (not torch.is_tensor(tf)) or tf.ndim != 2:
        return None
    return tf


def _load_concepts(concepts_json: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        raw = json.loads(concepts_json.read_text())
    except Exception:
        return None
    if isinstance(raw, dict) and "concepts" in raw:
        concepts = raw["concepts"]
    else:
        concepts = raw
    if not isinstance(concepts, list) or not concepts:
        return None
    out: List[Dict[str, Any]] = []
    for c in concepts:
        if isinstance(c, dict):
            out.append(c)
    return out if out else None


def _concept_to_idx(concepts: List[Dict[str, Any]]) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for i, c in enumerate(concepts):
        sn = str(c.get("short_name") or c.get("concept_short_name") or "").strip()
        if sn:
            m[sn] = int(i)
    return m


def _plip_crop_size(plip_obj: Any) -> Tuple[int, int]:
    """
    Best-effort read of PLIP/CLIP crop size for producing roi.png aligned with PLIP input.
    """
    ip = getattr(getattr(plip_obj, "processor", None), "image_processor", None)
    if ip is None:
        return 224, 224
    cs = getattr(ip, "crop_size", None)
    if isinstance(cs, dict):
        h = cs.get("height") or cs.get("shortest_edge") or cs.get("size") or 224
        w = cs.get("width") or cs.get("shortest_edge") or cs.get("size") or 224
        return int(h), int(w)
    if isinstance(cs, (list, tuple)) and len(cs) == 2:
        return int(cs[0]), int(cs[1])
    if isinstance(cs, (int, float)):
        v = int(cs)
        return v, v
    return 224, 224


def _load_eval_cfg(eval_dir: Path, log: logging.Logger) -> Dict[str, Any]:
    for name in ("config_resolved.yaml", "config_eval.yaml", "config.yaml"):
        p = eval_dir / name
        if not p.exists():
            continue
        try:
            obj = yaml.safe_load(p.read_text())
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            log.warning("Failed to parse %s (%s)", p, e)
            continue
    return {}


def _resolve_dataset_spec(
    eval_cfg: Dict[str, Any], cfg_fallback: Dict[str, Any], log: logging.Logger
) -> Optional[Dict[str, Any]]:
    """
    Resolve a minimal dataset spec for loading test images.
    Priority: eval_cfg -> fallback cfg -> env.
    """
    cfg: Dict[str, Any] = {}
    cfg.update(cfg_fallback or {})
    for k, v in (eval_cfg or {}).items():
        cfg[k] = v

    backend = str(_deep_get(cfg, ("data", "backend"), "")).strip().lower()
    if not backend:
        if isinstance(_deep_get(cfg, ("data", "webdataset"), None), dict):
            backend = "webdataset"
        elif isinstance(_deep_get(cfg, ("data", "imagefolder"), None), dict):
            backend = "imagefolder"

    img_size = _as_int(_deep_get(cfg, ("data", "img_size"), 224), 224)
    imagenet_norm = bool(_deep_get(cfg, ("data", "imagenet_norm"), False))
    num_workers = _as_int(_deep_get(cfg, ("data", "num_workers"), 8), 8)

    if backend == "webdataset":
        wds_cfg = _deep_get(cfg, ("data", "webdataset"), {}) if isinstance(_deep_get(cfg, ("data", "webdataset"), None), dict) else {}
        test_dir = str(wds_cfg.get("test_dir") or "").strip()
        if not test_dir:
            test_dir = os.getenv("WDS_TEST_DIR", "").strip()
        if not test_dir:
            log.warning("Missing test_dir for WebDataset (config.data.webdataset.test_dir or env WDS_TEST_DIR).")
            return None
        return {
            "backend": "webdataset",
            "test_dir": test_dir,
            "pattern": str(wds_cfg.get("pattern", "shard-*.tar")),
            "image_key": str(wds_cfg.get("image_key", "img.jpg;jpg;jpeg;png")),
            "meta_key": str(wds_cfg.get("meta_key", "meta.json;json")),
            "img_size": int(img_size),
            "imagenet_norm": bool(imagenet_norm),
            "num_workers": int(num_workers),
        }

    if backend == "imagefolder":
        test_dir = ""
        if isinstance(_deep_get(cfg, ("data", "imagefolder"), None), dict):
            ifd = _deep_get(cfg, ("data", "imagefolder"), {})
            test_dir = str(ifd.get("test_dir") or "").strip()
        if not test_dir:
            test_dir = str(_deep_get(cfg, ("dataset", "test_dir"), "") or "").strip()
        if not test_dir:
            log.warning("Missing test_dir for ImageFolder backend.")
            return None
        return {
            "backend": "imagefolder",
            "test_dir": test_dir,
            "img_size": int(img_size),
            "imagenet_norm": bool(imagenet_norm),
            "num_workers": int(num_workers),
            "batch_size": _as_int(_deep_get(cfg, ("data", "batch_size"), 64), 64),
        }

    if backend:
        log.warning("Unsupported backend='%s' (expected webdataset|imagefolder).", backend)
    else:
        log.warning("Unable to infer dataset backend from eval config.")
    return None


def _resolve_backbone_name(eval_cfg: Dict[str, Any]) -> Optional[str]:
    for kp in [
        ("model", "backbone_name"),
        ("model", "backbone"),
        ("model", "arch"),
        ("model", "name"),
        ("backbone_name",),
    ]:
        v = _deep_get(eval_cfg, kp, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _load_state_dict_for_probe(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    return payload if isinstance(payload, dict) else {}


def _guess_vit_backbone_name_from_ckpt(ssl_backbone_ckpt: Path) -> Optional[str]:
    """
    Best-effort guess for common timm ViT names from checkpoint shapes.
    Falls back to None if it cannot detect ViT-like keys.
    """
    sd = _load_state_dict_for_probe(ssl_backbone_ckpt)
    if not sd:
        return None

    embed_dim = None
    patch = None
    for k, v in sd.items():
        if not hasattr(v, "shape"):
            continue
        if str(k).endswith("patch_embed.proj.weight") and len(v.shape) == 4:
            embed_dim = int(v.shape[0])
            patch = int(v.shape[-1])
            break
    if embed_dim is None or patch is None:
        any_blocks = any(".blocks." in str(k) or str(k).startswith("blocks.") for k in sd.keys())
        return "vit_base_patch16_224" if any_blocks else None

    depth = 0
    for k in sd.keys():
        parts = str(k).split(".")
        for i, p in enumerate(parts):
            if p == "blocks" and i + 1 < len(parts):
                try:
                    bi = int(parts[i + 1])
                    depth = max(depth, bi + 1)
                except Exception:
                    continue

    if patch == 16:
        if embed_dim <= 400:
            return "vit_small_patch16_224"
        if embed_dim <= 800:
            return "vit_base_patch16_224"
        if embed_dim <= 1100:
            return "vit_large_patch16_224"
        return "vit_huge_patch16_224" if depth >= 24 else "vit_large_patch16_224"

    if patch == 14:
        if embed_dim <= 800:
            return "vit_base_patch14_224"
        if embed_dim <= 1100:
            return "vit_large_patch14_224"
        return "vit_huge_patch14_224"

    return "vit_base_patch16_224"


def _canon_key_strip_prefix(k: str) -> str:
    """
    Canonicalize a key possibly prefixed with split provenance, e.g. 'test::abcd' -> 'abcd'.
    """
    s = str(k)
    if "::" in s:
        pref = s.split("::", 1)[0].strip().lower()
        if pref in ("train", "val", "test"):
            return s.split("::", 1)[1]
    return s


def _build_rollout_mask_binary(mask_2d: np.ndarray, q: float) -> np.ndarray:
    m = np.asarray(mask_2d, dtype=np.float32)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.min(m)) if m.size else 0.0
    mx = float(np.max(m)) if m.size else 0.0
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) <= 1e-12:
        m = np.zeros_like(m, dtype=np.float32)
    else:
        m = (m - mn) / (mx - mn)
    thr = float(np.quantile(m, q)) if m.size else 1.0
    bw = (m >= thr)
    if not bw.any() and m.size:
        bw = (m == float(np.max(m)))
    return bw.astype(bool)


def _apply_mask_to_image(img: Image.Image, mask_bool: np.ndarray) -> Image.Image:
    w, h = img.size
    m_img = Image.fromarray(mask_bool.astype(np.uint8) * 255).resize((w, h), resample=Image.NEAREST)
    m = (np.asarray(m_img) > 0)
    arr = np.asarray(img.convert("RGB"))
    out = arr.copy()
    out[~m] = 0
    return Image.fromarray(out, mode="RGB")


def _write_light_outputs(
    *,
    model_id: str,
    heavy_run_dir: Path,
    summary_rows: List[Dict[str, Any]],
    summary_payload: Dict[str, Any],
    log: logging.Logger,
) -> None:
    for kind in ("spatial", "roi"):
        base = get_light_stats_dir(kind, model_id)
        out_dir = base / "spatial_concept"
        ensure_dir(out_dir)

        latest = {
            "version": 1,
            "kind": "spatial_concept",
            "model_id": str(model_id),
            "heavy_run_dir": str(heavy_run_dir),
            "updated_at": datetime.now().isoformat(),
            "summary_csv": str(out_dir / "xai_summary.csv"),
            "summary_json": str(out_dir / "xai_summary.json"),
        }
        atomic_write_json(out_dir / "latest_run.json", latest)

        fieldnames = list(summary_rows[0].keys()) if summary_rows else [
            "idx",
            "key",
            "true_class",
            "pred_class",
            "conf",
            "reasons",
            "input_png",
            "attn_rollout_npy",
            "attn_rollout_png",
            "roi_png",
            "roi_bbox_json",
            "concept_scores_json",
            "top_concepts",
        ]
        atomic_write_csv(out_dir / "xai_summary.csv", summary_rows, fieldnames=fieldnames)
        atomic_write_json(out_dir / "xai_summary.json", summary_payload)
        log.info("Wrote light %s stats: %s", kind, out_dir)


def _process_one_ablation(
    ablation_dir: Path,
    *,
    run_id: str,
    cfg_sel: Dict[str, Any],
    roi_mode: str,
    roi_quantile: float,
    roi_min_area: float,
    roi_pad_frac: float,
    rollout_discard_ratio: float,
    plip_obj: Optional[Any],
    plip_tf_all: Optional[torch.Tensor],
    plip_concepts: Optional[List[Dict[str, Any]]],
    plip_shortlist: Optional[Dict[str, Dict[str, List[int]]]],
    dry_run: bool,
    log: logging.Logger,
    cfg_fallback: Dict[str, Any],
) -> None:
    model_id = ablation_dir.name
    ckpts = resolve_checkpoints(ablation_dir)
    if not ckpts:
        log.warning("Skip %s: missing checkpoints under %s/checkpoints", model_id, ablation_dir)
        return

    eval_dir = resolve_latest_eval_dir(ablation_dir)
    if eval_dir is None:
        log.warning("Skip %s: missing eval dir under %s/eval", model_id, ablation_dir)
        return

    heavy = ensure_spatial_concept_heavy_layout(spatial_concept_heavy_layout(ablation_dir, run_id))
    heavy_run_dir = heavy.root_dir

    y_true, y_pred, conf, keys, _meta_rows = load_eval_artifacts(
        eval_dir,
        pred_csv="predictions.csv",
        logits_npy="logits_test.npy",
        logger=log,
    )
    if y_pred is None:
        log.warning("Skip %s: y_pred not available (missing/invalid predictions.csv)", model_id)
        return

    eval_cfg = _load_eval_cfg(eval_dir, log=log)
    class_names = load_class_names(eval_dir)
    if class_names is None:
        v = _deep_get(eval_cfg, ("labels", "class_order"), None)
        if isinstance(v, list) and all(isinstance(x, str) for x in v) and len(v) >= 2:
            class_names = list(v)

    n_classes = len(class_names) if class_names else 0
    if n_classes <= 0:
        logits_path = eval_dir / "logits_test.npy"
        if logits_path.exists():
            try:
                logits = np.load(logits_path, mmap_mode="r")
                if logits.ndim == 2:
                    n_classes = int(logits.shape[1])
            except Exception:
                pass
    if n_classes <= 0:
        try:
            n_classes = int(np.max(np.asarray(y_pred, dtype=np.int64))) + 1
        except Exception:
            n_classes = 0
    if n_classes <= 0:
        log.warning("Skip %s: could not infer n_classes", model_id)
        return

    targets, reasons = select_items(
        y_true=y_true,
        y_pred=y_pred,
        conf=conf,
        keys=keys,
        n_classes=n_classes,
        cfg_sel=cfg_sel,
        logger=log,
    )

    selected_indices: List[int] = []
    selected_keys: Optional[List[str]] = None
    idx_by_key: Dict[str, int] = {}
    if keys is not None:
        for i, k in enumerate(keys):
            if k is None:
                continue
            kk = _canon_key_strip_prefix(str(k))
            if kk not in idx_by_key:
                idx_by_key[kk] = int(i)
        selected_keys = [_canon_key_strip_prefix(str(k)) for k in targets]
        for k in selected_keys:
            ii = idx_by_key.get(k, None)
            if ii is not None:
                selected_indices.append(int(ii))
    else:
        selected_indices = [int(i) for i in targets]

    selection_payload: Dict[str, Any] = {
        "version": 1,
        "run_id": str(run_id),
        "model_id": str(model_id),
        "model_root": str(ablation_dir),
        "eval_dir": str(eval_dir),
        "checkpoints": {
            "ssl_backbone_ckpt": str(ckpts["ssl_backbone_ckpt"]),
            "ssl_head_ckpt": str(ckpts["ssl_head_ckpt"]),
        },
        "selection_cfg": cfg_sel,
        "n_classes": int(n_classes),
        "selected_indices": selected_indices,
        "selected_keys": selected_keys,
        "reasons": reasons,
        "created_at": datetime.now().isoformat(),
    }
    atomic_write_json(heavy.selection_json, selection_payload)

    if dry_run:
        log.info("[DRY-RUN] %s -> heavy: %s", model_id, heavy_run_dir)
        return

    ds_spec = _resolve_dataset_spec(eval_cfg, cfg_fallback, log=log)
    if ds_spec is None:
        log.warning("Skip %s: could not resolve test dataset spec from eval config/env.", model_id)
        return

    img_size = int(ds_spec.get("img_size", 224))
    imagenet_norm = bool(ds_spec.get("imagenet_norm", False))
    preprocess_fn = build_preprocess(img_size, imagenet_norm)

    backbone_name = _resolve_backbone_name(eval_cfg)
    if not backbone_name:
        backbone_name = _guess_vit_backbone_name_from_ckpt(ckpts["ssl_backbone_ckpt"]) or "vit_base_patch16_224"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = SSLLinearClassifier(backbone_name=backbone_name, num_classes=int(n_classes))
        mb, ub = model.load_backbone_from_ssl(str(ckpts["ssl_backbone_ckpt"]))
        mh, uh = model.load_head_from_probe(str(ckpts["ssl_head_ckpt"]))
        log.info(
            "[%s] Loaded model backbone=%s (missing=%d unexpected=%d) head(missing=%d unexpected=%d)",
            model_id,
            backbone_name,
            mb,
            ub,
            mh,
            uh,
        )
    except Exception as e:
        log.warning("Skip %s: failed to build/load SSLLinearClassifier (%s)", model_id, e)
        return

    model = model.to(device).eval()

    rollout = None
    if hasattr(getattr(model, "backbone", None), "model"):
        try:
            rollout = ViTAttentionRollout(getattr(model.backbone, "model"), discard_ratio=float(rollout_discard_ratio))
        except Exception as e:
            log.warning("[%s] Attention rollout init failed (%s). Will fall back to full-image ROI.", model_id, e)
            rollout = None
    else:
        log.warning("[%s] Backbone is not ViT-like (no .model). Will fall back to full-image ROI.", model_id)

    plip_h, plip_w = (224, 224)
    if plip_obj is not None:
        try:
            plip_h, plip_w = _plip_crop_size(plip_obj)
        except Exception:
            plip_h, plip_w = (224, 224)

    wanted_indices_set = set(int(i) for i in selected_indices)
    wanted_keys_set = set(selected_keys) if selected_keys else set()

    rows: List[Dict[str, Any]] = []

    def _emit_stub_concept_scores(out_path: Path, *, err: str, meta: Dict[str, Any]) -> None:
        payload = {"error": err, "meta": meta, "scores": {}, "topk": []}
        atomic_write_json(out_path, payload)

    def _score_concepts_for_roi(
        roi_img: Image.Image,
        *,
        idx_eval: int,
        key: Optional[str],
        true_cls: Optional[str],
        pred_cls: Optional[str],
        conf_val: Optional[float],
        out_json: Path,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        meta = {
            "idx": int(idx_eval),
            "key": (None if key is None else str(key)),
            "true_class": true_cls,
            "pred_class": pred_cls,
            "conf": (None if conf_val is None else float(conf_val)),
        }
        if plip_obj is None or plip_tf_all is None or plip_concepts is None or plip_shortlist is None:
            _emit_stub_concept_scores(out_json, err="plip_or_calibration_missing", meta=meta)
            return [], {}

        true_c = canon_class(true_cls) if true_cls else None
        pred_c = canon_class(pred_cls) if pred_cls else None
        idxs = concept_indices_for_patch(plip_shortlist, true_c, pred_c)
        if not idxs:
            _emit_stub_concept_scores(out_json, err="no_concepts_selected_for_patch", meta=meta)
            return [], {}

        tf_sub = plip_tf_all[torch.tensor(idxs, dtype=torch.long)]

        try:
            x = plip_obj.preprocess(roi_img).unsqueeze(0)
            img_feats = encode_images(plip_obj, x)
            logits = score(plip_obj, img_feats, tf_sub.to(device=img_feats.device, dtype=img_feats.dtype))
            scores_np = logits.detach().cpu().float().numpy().reshape(-1)
        except Exception as e:
            _emit_stub_concept_scores(out_json, err=f"plip_scoring_failed: {e}", meta=meta)
            return [], {}

        selected: List[Dict[str, Any]] = []
        mapping: Dict[str, float] = {}
        pred_set_primary = set(plip_shortlist.get(pred_c, {}).get("primary", [])) if pred_c else set()
        pred_set_conf = set(plip_shortlist.get(pred_c, {}).get("confounds", [])) if pred_c else set()
        true_set_primary = set(plip_shortlist.get(true_c, {}).get("primary", [])) if true_c else set()

        for j, sc in zip(idxs, scores_np.tolist()):
            cc = plip_concepts[j]
            sn = str(cc.get("short_name") or cc.get("concept_short_name") or f"c{j}")
            kind = "other"
            if j in pred_set_primary or j in true_set_primary:
                kind = "primary"
            elif j in pred_set_conf:
                kind = "confound"
            entry = {
                "concept_idx": int(j),
                "concept_short_name": sn,
                "concept_name": str(cc.get("name") or cc.get("concept_name") or sn),
                "group": cc.get("group", None),
                "primary_class": cc.get("primary_class", None),
                "kind": kind,
                "score": float(sc),
            }
            selected.append(entry)
            mapping[sn] = float(sc)

        topn = 10
        selected_sorted = sorted(selected, key=lambda d: float(d.get("score", 0.0)), reverse=True)
        topk = selected_sorted[: min(topn, len(selected_sorted))]

        payload = {
            "version": 1,
            "meta": meta,
            "n_concepts_scored": int(len(selected)),
            "scores": mapping,
            "selected": selected,
            "topk": topk,
        }
        atomic_write_json(out_json, payload)
        return topk, mapping

    def _process_sample(
        img_t: torch.Tensor,
        *,
        idx_eval: int,
        key: Optional[str],
    ) -> None:
        item_dir = get_item_out_dir(ablation_dir, run_id, int(idx_eval), kind="spatial_concept")
        item_dir.mkdir(parents=True, exist_ok=True)

        true_idx = None if y_true is None else int(y_true[idx_eval])
        pred_idx = int(y_pred[idx_eval])
        conf_val = None if conf is None else float(conf[idx_eval])

        true_cls = idx_to_class(true_idx, class_names) if class_names else (None if true_idx is None else str(true_idx))
        pred_cls = idx_to_class(pred_idx, class_names) if class_names else str(pred_idx)

        rr: List[str] = []
        if selected_keys is not None and key is not None:
            rr = list(reasons.get(key, [])) if isinstance(reasons, dict) else []
        else:
            rr = list(reasons.get(int(idx_eval), [])) if isinstance(reasons, dict) else []

        pil_in = tensor_to_pil(img_t, imagenet_norm=imagenet_norm)
        (item_dir / "input.png").write_bytes(b"")  # pre-touch for NFS quirks
        pil_in.save(item_dir / "input.png")

        try:
            if rollout is not None:
                x = img_t.unsqueeze(0).to(device)
                m = rollout(x)
                mask_np = np.asarray(m, dtype=np.float32) if m is not None else np.zeros((1, 1), dtype=np.float32)
            else:
                mask_np = np.zeros((1, 1), dtype=np.float32)
        except Exception as e:
            log.warning("[%s] rollout failed idx=%d key=%s (%s) -> fallback full ROI", model_id, idx_eval, key, e)
            mask_np = np.zeros((1, 1), dtype=np.float32)

        np.save(item_dir / "attn_rollout.npy", mask_np.astype(np.float32))
        try:
            over = overlay_heatmap(pil_in, mask_np, alpha=0.6)
            over.save(item_dir / "attn_rollout.png")
        except Exception as e:
            log.warning("[%s] overlay heatmap failed idx=%d (%s)", model_id, idx_eval, e)

        q = float(roi_quantile)
        min_area_frac = _min_area_frac(float(roi_min_area), pil_in.size[0], pil_in.size[1])
        try:
            rb = extract_bbox_from_mask(
                mask_np,
                img_w=pil_in.size[0],
                img_h=pil_in.size[1],
                quantile=q,
                min_area_frac=float(min_area_frac),
                pad_frac=float(roi_pad_frac),
            )
            x0, y0, x1, y1 = rb.as_xyxy()
            rb_meta = {"method": rb.method, "threshold": float(rb.threshold)}
        except Exception as e:
            log.warning("[%s] ROI bbox extraction failed idx=%d (%s) -> full image", model_id, idx_eval, e)
            x0, y0, x1, y1 = 0, 0, pil_in.size[0] - 1, pil_in.size[1] - 1
            rb_meta = {"method": "fallback_full_exception", "threshold": float("nan")}

        roi_bbox = {
            "x0": int(x0),
            "y0": int(y0),
            "x1": int(x1),
            "y1": int(y1),
            "roi_mode": str(roi_mode),
            "quantile": float(q),
            "min_area_frac": float(min_area_frac),
            "pad_frac": float(roi_pad_frac),
            "plip_crop_h": int(plip_h),
            "plip_crop_w": int(plip_w),
            "mask_shape": list(mask_np.shape),
            "img_w": int(pil_in.size[0]),
            "img_h": int(pil_in.size[1]),
            **rb_meta,
        }
        atomic_write_json(item_dir / "roi_bbox.json", roi_bbox)

        roi_crop = pil_in.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))
        roi_img: Image.Image
        roi_mask_img: Optional[Image.Image] = None
        if roi_mode in ("mask", "both"):
            bw = _build_rollout_mask_binary(mask_np, q)
            roi_mask_img = _apply_mask_to_image(pil_in, bw)
        if roi_mode == "mask":
            roi_img = roi_mask_img if roi_mask_img is not None else pil_in
        else:
            roi_img = roi_crop

        try:
            roi_img_resized = roi_img.resize((int(plip_w), int(plip_h)), resample=Image.BICUBIC)
        except Exception:
            roi_img_resized = roi_img

        roi_img_resized.save(item_dir / "roi.png")
        if roi_mode == "both" and roi_mask_img is not None:
            try:
                roi_mask_img.resize((int(plip_w), int(plip_h)), resample=Image.BICUBIC).save(item_dir / "roi_mask.png")
            except Exception:
                pass

        concept_json = item_dir / "concept_scores.json"
        topk, _mapping = _score_concepts_for_roi(
            roi_img_resized.convert("RGB"),
            idx_eval=int(idx_eval),
            key=key,
            true_cls=true_cls,
            pred_cls=pred_cls,
            conf_val=conf_val,
            out_json=concept_json,
        )

        rel = Path("items") / f"idx_{int(idx_eval):08d}"
        top_str = "|".join([f"{d.get('concept_short_name')}:{float(d.get('score', 0.0)):.4f}" for d in topk]) if topk else ""
        row = {
            "idx": int(idx_eval),
            "key": ("" if key is None else str(key)),
            "true_class": ("" if true_cls is None else str(true_cls)),
            "pred_class": ("" if pred_cls is None else str(pred_cls)),
            "conf": ("" if conf_val is None else f"{float(conf_val):.6f}"),
            "reasons": "|".join([str(x) for x in rr]) if rr else "",
            "input_png": str(rel / "input.png"),
            "attn_rollout_npy": str(rel / "attn_rollout.npy"),
            "attn_rollout_png": str(rel / "attn_rollout.png"),
            "roi_png": str(rel / "roi.png"),
            "roi_bbox_json": str(rel / "roi_bbox.json"),
            "concept_scores_json": str(rel / "concept_scores.json"),
            "top_concepts": top_str,
        }
        rows.append(row)

    if ds_spec["backend"] == "webdataset":
        test_dir = Path(str(ds_spec["test_dir"]))
        if not test_dir.exists():
            log.warning("Skip %s: WebDataset test_dir not found: %s", model_id, test_dir)
            return
        loader = make_wds_loader_with_keys(
            test_dir=str(test_dir),
            pattern=str(ds_spec.get("pattern", "shard-*.tar")),
            image_key=str(ds_spec.get("image_key", "img.jpg;jpg;jpeg;png")),
            meta_key=str(ds_spec.get("meta_key", "meta.json;json")),
            preprocess_fn=preprocess_fn,
            num_workers=int(ds_spec.get("num_workers", 8)),
            batch_size=1,
        )

        if selected_keys is not None and wanted_keys_set:
            wanted = set(_canon_key_strip_prefix(k) for k in wanted_keys_set)
            found = set()
            for img_t, _meta, kk in iter_wds_filtered_by_keys(loader, wanted):
                kkc = _canon_key_strip_prefix(str(kk))
                idx_eval = idx_by_key.get(kkc, None)
                if idx_eval is None:
                    continue
                found.add(kkc)
                _process_sample(img_t, idx_eval=int(idx_eval), key=kkc)
            missing = sorted(list(wanted - found))
            if missing:
                log.warning("[%s] %d selected keys not found in test loader (showing up to 8): %s", model_id, len(missing), missing[:8])
        else:
            seen = 0
            for batch in loader:
                if batch is None:
                    continue
                img_t, _meta, kk = batch
                if int(seen) in wanted_indices_set:
                    _process_sample(img_t, idx_eval=int(seen), key=_canon_key_strip_prefix(str(kk)))
                seen += 1

    elif ds_spec["backend"] == "imagefolder":
        test_dir = Path(str(ds_spec["test_dir"]))
        if not test_dir.exists():
            log.warning("Skip %s: ImageFolder test_dir not found: %s", model_id, test_dir)
            return
        ds, _loader = make_imgfolder_loader(
            test_dir=str(test_dir),
            preprocess_fn=preprocess_fn,
            batch_size=int(ds_spec.get("batch_size", 64)),
            num_workers=int(ds_spec.get("num_workers", 8)),
        )
        for idx_eval in sorted(wanted_indices_set):
            try:
                img_t, _lbl = ds[int(idx_eval)]
            except Exception as e:
                log.warning("[%s] Failed to load imagefolder idx=%d (%s)", model_id, idx_eval, e)
                continue
            _process_sample(img_t, idx_eval=int(idx_eval), key=None)

    rows = sorted(rows, key=lambda r: int(r.get("idx", 0)))

    fieldnames = list(rows[0].keys()) if rows else [
        "idx",
        "key",
        "true_class",
        "pred_class",
        "conf",
        "reasons",
        "input_png",
        "attn_rollout_npy",
        "attn_rollout_png",
        "roi_png",
        "roi_bbox_json",
        "concept_scores_json",
        "top_concepts",
    ]
    atomic_write_csv(heavy.summary_csv, rows, fieldnames=fieldnames)

    summary_payload = {
        "version": 1,
        "run_id": str(run_id),
        "model_id": str(model_id),
        "model_root": str(ablation_dir),
        "eval_dir": str(eval_dir),
        "n_items": int(len(rows)),
        "selection_json": str(Path("selection") / "xai_selection.json"),
        "items": rows,
        "created_at": datetime.now().isoformat(),
    }
    atomic_write_json(heavy.summary_json, summary_payload)

    log.info("[%s] Wrote heavy run: %s (n_items=%d)", model_id, heavy_run_dir, len(rows))

    try:
        _write_light_outputs(
            model_id=model_id,
            heavy_run_dir=heavy_run_dir,
            summary_rows=rows,
            summary_payload=summary_payload,
            log=log,
        )
    except Exception as e:
        log.warning("[%s] Failed to write light outputs (%s)", model_id, e)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Unified spatial+concept XAI over all SSL ablations.")
    ap.add_argument("--models-root", type=Path, default=MODELS_ROOT_DEFAULT)
    ap.add_argument("--exp-prefix", type=str, default="exp_20251109_")
    ap.add_argument(
        "--config",
        type=Path,
        default=SPATIAL_CONCEPT_CONFIG_YAML,
        help="Unified pipeline config (default: explainability/configs/roi.yaml).",
    )
    ap.add_argument("--run-id", type=str, default=None, help="Deterministic run id (default: timestamp).")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--max-per-class-tp", type=int, default=None)
    ap.add_argument("--max-per-class-fp", type=int, default=None)
    ap.add_argument("--max-per-class-fn", type=int, default=None)
    ap.add_argument("--global-low-conf", type=int, default=None)

    ap.add_argument("--roi-mode", type=str, default=None, choices=["crop", "mask", "both"])
    ap.add_argument(
        "--roi-threshold",
        type=float,
        default=None,
        help="Quantile in [0,1] (e.g. 0.9) or percentile in (1,100] (e.g. 90).",
    )
    ap.add_argument(
        "--roi-min-area",
        type=float,
        default=None,
        help="Min area as fraction (<=1) or pixels (>1).",
    )
    ap.add_argument("--log-level", type=str, default="INFO")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("attention_rollout_concept_all")

    models_root = args.models_root.expanduser().resolve()
    exp_prefix = str(args.exp_prefix)
    run_id = args.run_id or _now_run_id()

    cfg = _safe_load_yaml(args.config, log=log)

    sel_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    per = sel_cfg.get("per_class", {}) if isinstance(sel_cfg.get("per_class", {}), dict) else {}
    gl = sel_cfg.get("global_low_conf", {}) if isinstance(sel_cfg.get("global_low_conf", {}), dict) else {}

    topk_tp = args.max_per_class_tp if args.max_per_class_tp is not None else _as_int(per.get("topk_tp", 2), 2)
    topk_fp = args.max_per_class_fp if args.max_per_class_fp is not None else _as_int(per.get("topk_fp", 2), 2)
    topk_fn = args.max_per_class_fn if args.max_per_class_fn is not None else _as_int(per.get("topk_fn", 2), 2)
    topk_low = args.global_low_conf if args.global_low_conf is not None else _as_int(gl.get("topk", 0), 0)

    cfg_sel: Dict[str, Any] = {
        "per_class": {
            "topk_tp": int(max(0, topk_tp)),
            "topk_fp": int(max(0, topk_fp)),
            "topk_fn": int(max(0, topk_fn)),
        }
    }
    if int(topk_low) > 0:
        cfg_sel["global_low_conf"] = {"topk": int(topk_low)}

    roi_cfg = cfg.get("roi", {}) if isinstance(cfg.get("roi", {}), dict) else {}
    roi_mode = str(args.roi_mode or roi_cfg.get("mode", "crop")).strip().lower()
    if roi_mode not in ("crop", "mask", "both"):
        roi_mode = "crop"
    thr = args.roi_threshold if args.roi_threshold is not None else _as_float(roi_cfg.get("threshold", roi_cfg.get("quantile", 0.90)), 0.90)
    roi_quantile = _parse_quantile(float(thr))
    roi_min_area = float(args.roi_min_area if args.roi_min_area is not None else _as_float(roi_cfg.get("min_area", roi_cfg.get("min_area_frac", 0.01)), 0.01))
    roi_pad_frac = float(_as_float(roi_cfg.get("pad_frac", 0.05), 0.05))

    rollout_discard_ratio = float(
        _as_float(
            _deep_get(cfg, ("xai", "attn_rollout", "discard_ratio"), _deep_get(cfg, ("spatial", "attn_rollout", "discard_ratio"), 0.90)),
            0.90,
        )
    )

    plip_obj = None
    plip_tf_all = None
    plip_concepts = None
    plip_shortlist = None

    if args.dry_run:
        log.info("[DRY-RUN] Skipping PLIP/calibration loading.")
    else:
        cal_dir = CALIBRATION_PATHS.metadata_dir
        tf_path = cal_dir / "text_features.pt"
        concepts_json = cal_dir / "concepts.json"
        if tf_path.exists() and concepts_json.exists():
            plip_tf_all = _load_text_features(tf_path)
            plip_concepts = _load_concepts(concepts_json)
            if plip_tf_all is None or plip_concepts is None:
                log.warning("Calibration artifacts unreadable (tf=%s concepts=%s). Concept stage will emit stubs.", tf_path, concepts_json)
        else:
            log.warning("Calibration artifacts missing under %s. Concept stage will emit stubs.", cal_dir)

        shortlist_yaml = CALIBRATION_PATHS.shortlist_yaml
        if not shortlist_yaml.exists():
            shortlist_yaml = CONCEPT_SHORTLIST_YAML_CFG
        if plip_concepts is not None and shortlist_yaml.exists():
            try:
                plip_shortlist = load_shortlist_idx(shortlist_yaml, _concept_to_idx(plip_concepts), log=log)
            except Exception as e:
                log.warning("Failed to load shortlist (%s): %s. Concept stage will emit stubs.", shortlist_yaml, e)
                plip_shortlist = None
        else:
            if not shortlist_yaml.exists():
                log.warning("Shortlist YAML not found (%s). Concept stage will emit stubs.", shortlist_yaml)

        cfg_plip = _safe_load_yaml(CONCEPT_PLIP_CONFIG_YAML, log=log)
        cfg_plip2 = cfg.get("plip", {}) if isinstance(cfg.get("plip", {}), dict) else {}
        plip_cfg: Dict[str, Any] = {}
        if isinstance(cfg_plip.get("plip", None), dict):
            plip_cfg.update(cfg_plip.get("plip", {}))
        else:
            plip_cfg.update(cfg_plip)
        plip_cfg.update(cfg_plip2)

        if plip_tf_all is not None and plip_shortlist is not None and plip_concepts is not None:
            try:
                plip_obj = load_plip(
                    model_id=str(plip_cfg.get("model_id", "vinid/plip")),
                    model_local_dir=plip_cfg.get("model_local_dir", None),
                    device=str(plip_cfg.get("device", "cuda")),
                    precision=str(plip_cfg.get("precision", "fp16")),
                    score_scale=None,
                    hf_cache_dir=plip_cfg.get("hf_cache_dir", None),
                )
                log.info("Loaded PLIP model_id=%s device=%s", plip_obj.model_id, plip_obj.device)
            except Exception as e:
                log.warning("Failed to load PLIP (%s). Concept stage will emit stubs.", e)
                plip_obj = None

    if not models_root.exists() or not models_root.is_dir():
        log.warning("models_root not found: %s", models_root)
        return

    exp_roots = list(iter_exp_roots(models_root, exp_prefix))
    if not exp_roots:
        log.warning("No experiment roots found under %s with prefix '%s'.", models_root, exp_prefix)
        return

    log.info("Scanning models_root=%s exp_prefix=%s run_id=%s", models_root, exp_prefix, run_id)
    log.info("Selection cfg: %s", cfg_sel)
    log.info("ROI cfg: mode=%s quantile=%.3f min_area=%s pad=%.3f", roi_mode, roi_quantile, roi_min_area, roi_pad_frac)

    cfg_fallback = cfg if isinstance(cfg, dict) else {}

    n_done = 0
    n_skip = 0
    for er in exp_roots:
        abls = list(iter_ablation_dirs(er))
        if not abls:
            continue
        for ad in abls:
            try:
                _process_one_ablation(
                    ad,
                    run_id=run_id,
                    cfg_sel=cfg_sel,
                    roi_mode=roi_mode,
                    roi_quantile=float(roi_quantile),
                    roi_min_area=float(roi_min_area),
                    roi_pad_frac=float(roi_pad_frac),
                    rollout_discard_ratio=float(rollout_discard_ratio),
                    plip_obj=plip_obj,
                    plip_tf_all=plip_tf_all,
                    plip_concepts=plip_concepts,
                    plip_shortlist=plip_shortlist,
                    dry_run=bool(args.dry_run),
                    log=log,
                    cfg_fallback=cfg_fallback,
                )
                n_done += 1
            except Exception as e:
                n_skip += 1
                log.warning("Ablation failed (skipping) dir=%s (%s)", ad, e)
                continue

    log.info("Done. processed=%d skipped=%d run_id=%s", n_done, n_skip, run_id)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
>>

run_xai_pipeline.sbatch codice <<
#!/usr/bin/env bash
#SBATCH -J xai_all
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/xai/run_all.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/xai/run_all.%j.err
#SBATCH -p gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00

# Escludi i nodi noti problematici (aggiunto compute-4-15)
#SBATCH --exclude=compute-5-14,compute-5-11,compute-3-12,compute-4-13,compute-4-15

set -euo pipefail

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VENV_PATH="${VENV_PATH:-/home/mla_group_01/rcc-ssrl/.venvs/xai}"
export MODELS_ROOT="${MODELS_ROOT:-/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/models}"
export EXP_PREFIX="${EXP_PREFIX:-exp_20251109_}"
export XAI_CONFIG="${XAI_CONFIG:-/home/mla_group_01/rcc-ssrl/src/explainability/configs/roi.yaml}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "[INFO] Host: $(hostname)"
echo "[INFO] xai_all wrapper: delegating configuration to run_xai_pipeline.sh"
echo "[INFO] MODELS_ROOT=${MODELS_ROOT}"
echo "[INFO] EXP_PREFIX=${EXP_PREFIX}"
echo "[INFO] XAI_CONFIG=${XAI_CONFIG}"
echo "[INFO] LOG_LEVEL=${LOG_LEVEL}"

SCRIPT="/home/mla_group_01/rcc-ssrl/src/explainability/run_xai_pipeline.sh"

if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERROR] Orchestrator script not found: $SCRIPT" >&2
  exit 1
fi

chmod +x "$SCRIPT"

SRUN_ARGS=( srun --ntasks=1 )

echo "[INFO] Command: ${SRUN_ARGS[*]} $SCRIPT"
"${SRUN_ARGS[@]}" "$SCRIPT"

echo "[OK] xai_all job completed (run_xai_pipeline.sh exited with code $?)"
>>

run_xai_pipeline.sh codice <<
>>

spatial/attention_rollout_job.sbatch codice <<
#!/usr/bin/env bash
#SBATCH --job-name=attn_rollout
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/xai/attn_rollout.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/xai/attn_rollout.%j.err
#SBATCH -p gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --exclude=compute-5-14,compute-5-11,compute-3-12

set -euo pipefail

PROJECT_ROOT="/home/mla_group_01/rcc-ssrl"
mkdir -p "${PROJECT_ROOT}/src/logs/xai" || true

module purge || true

# -----------------------------
# REQUIRED: eval + ckpts (i-JEPA abl01)
# -----------------------------
ABL_ROOT="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/models/exp_20251109_181534_i_jepa/exp_i_jepa_abl01"

EVAL_RUN_DIR="${EVAL_RUN_DIR:-${ABL_ROOT}/eval/i_jepa_ssl_linear_best/20251113_120919}"
SSL_BACKBONE_CKPT="${SSL_BACKBONE_CKPT:-${ABL_ROOT}/checkpoints/i_jepa__ssl_best.pt}"
SSL_HEAD_CKPT="${SSL_HEAD_CKPT:-${ABL_ROOT}/checkpoints/i_jepa__ssl_linear_best.pt}"

# -----------------------------
# DATA (WebDataset test)
# -----------------------------
DATA_BACKEND="${DATA_BACKEND:-webdataset}"

TEST_WDS_DIR="${TEST_WDS_DIR:-/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test}"
WDS_PATTERN="${WDS_PATTERN:-*.tar}"
# FIX: le chiavi reali nel tar sono "img.jpg" e "meta.json" (fallback compat: "jpg"/"json")
WDS_IMAGE_KEY="${WDS_IMAGE_KEY:-img.jpg;jpg}"
WDS_META_KEY="${WDS_META_KEY:-meta.json;json}"

# -----------------------------
# MODEL
# -----------------------------
MODEL_NAME="${MODEL_NAME:-i_jepa_ssl_vit}"
BACKBONE_NAME="${BACKBONE_NAME:-vit_small_patch16_224}"

# -----------------------------
# OUTPUTS
# -----------------------------
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs/xai_spatial}"
RUN_ID="${RUN_ID:-i_jepa_abl01_$(date +%Y%m%d_%H%M%S)}"

# -----------------------------
# XAI controls
# -----------------------------
XAI_METHODS="${XAI_METHODS:-attn_rollout}"
ATNN_DISCARD_RATIO="${ATNN_DISCARD_RATIO:-0.9}"

# subset vs full-test
FULL_TEST="${FULL_TEST:-0}"
TOPK_TP="${TOPK_TP:-6}"
TOPK_FN="${TOPK_FN:-6}"
TOPK_FP="${TOPK_FP:-6}"
TOPK_LOWCONF="${TOPK_LOWCONF:-20}"

# perf/runtime
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMG_SIZE="${IMG_SIZE:-224}"
IMAGENET_NORM="${IMAGENET_NORM:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-1337}"

# venv (opzionale)
VENV_PATH="${VENV_PATH:-/home/mla_group_01/rcc-ssrl/.venvs/xai}"
if [[ -n "${VENV_PATH}" && -d "${VENV_PATH}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

# -----------------------------
# Sanity checks (fail fast)
# -----------------------------
if [[ ! -d "${EVAL_RUN_DIR}" ]]; then
  echo "[FATAL] EVAL_RUN_DIR non esiste: ${EVAL_RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_RUN_DIR}/predictions.csv" ]] || [[ ! -f "${EVAL_RUN_DIR}/logits_test.npy" ]]; then
  echo "[FATAL] EVAL_RUN_DIR deve contenere predictions.csv e logits_test.npy: ${EVAL_RUN_DIR}" >&2
  ls -la "${EVAL_RUN_DIR}" || true
  exit 3
fi
if [[ ! -f "${SSL_BACKBONE_CKPT}" ]]; then
  echo "[FATAL] SSL_BACKBONE_CKPT non trovato: ${SSL_BACKBONE_CKPT}" >&2
  exit 4
fi
if [[ ! -f "${SSL_HEAD_CKPT}" ]]; then
  echo "[FATAL] SSL_HEAD_CKPT non trovato: ${SSL_HEAD_CKPT}" >&2
  exit 5
fi
if [[ "${DATA_BACKEND}" != "webdataset" ]]; then
  echo "[FATAL] Questo job e' configurato per webdataset. DATA_BACKEND=${DATA_BACKEND}" >&2
  exit 6
fi
if [[ ! -d "${TEST_WDS_DIR}" ]]; then
  echo "[FATAL] TEST_WDS_DIR non esiste: ${TEST_WDS_DIR}" >&2
  exit 7
fi

# -----------------------------
# Run
# -----------------------------
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

export PROJECT_ROOT VENV_PATH
export EVAL_RUN_DIR SSL_BACKBONE_CKPT SSL_HEAD_CKPT
export DATA_BACKEND TEST_WDS_DIR WDS_PATTERN WDS_IMAGE_KEY WDS_META_KEY
export MODEL_NAME BACKBONE_NAME
export OUTPUTS_ROOT RUN_ID
export XAI_METHODS ATNN_DISCARD_RATIO
export FULL_TEST TOPK_TP TOPK_FN TOPK_FP TOPK_LOWCONF
export DEVICE NUM_WORKERS IMG_SIZE IMAGENET_NORM BATCH_SIZE SEED

exec "${PROJECT_ROOT}/src/explainability/spatial/run_attention_rollout.sh"
>>

spatial/attention_rollout.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial XAI on RCC test set (TP/FP/FN selection via predictions.csv).

Generates:
- GradCAM / IG / Occlusion (if enabled and dependencies are available)
- Attention Rollout for ViT (via monkey patching timm Attention blocks).

This script is config-driven and can be:
- run standalone: python attention_rollout.py --config CONFIG_PATH
- called programmatically from the orchestrator:
    from explainability.spatial.xai_generate import main as spatial_xai_main
    spatial_xai_main(["--config", str(config_path)])
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402

from explainability.spatial.eval_utils import (
    setup_logger,
    set_seed,
    build_preprocess,
    tensor_to_pil,
    load_eval_artifacts,
    select_items,
    make_wds_loader_with_keys,
    make_imgfolder_loader,
    canonicalize_key,
)
from explainability.spatial.ssl_linear_loader import SSLLinearClassifier

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
            logging.getLogger(__name__).warning(
                "Rollout Error: no attention blocks captured via patching."
            )
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

    logger = setup_logger("attention_rollout.sh")

    parser = argparse.ArgumentParser(description="Spatial XAI for SSL RCC model")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

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
            batch_size=int(cfg["data"].get("batch_size", 1)),
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

    index_csv = (out_root / "index.csv").open("w", newline="", encoding="utf-8")
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
    global_idx = 0  # output index (0..produced-1)
    dataset_idx = 0  # only used for imagefolder (selection by numeric index)

    row_by_key: Dict[str, Dict[str, Any]] = {}
    if meta_rows:
        for r in meta_rows:
            k = canonicalize_key(r.get("wds_key", ""))
            if k:
                row_by_key[k] = r
    idx_by_key = {canonicalize_key(k): i for i, k in enumerate(keys)} if keys is not None else {}
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

    def _iter_samples_from_batch(batch_any):
        """
        Normalize DataLoader batch to an iterator of (img_t, meta_any, key_str_or_none).
        Supports webdataset batch_size==1 and batch_size>1.
        """
        if cfg["data"]["backend"].lower() != "webdataset":
            img_t, lbl = batch_any
            yield img_t, {"class_id": int(lbl)}, None
            return

        imgs, metas, keys_any = batch_any
        if torch.is_tensor(imgs) and imgs.ndim == 3:
            yield imgs, metas, canonicalize_key(keys_any)
            return

        if torch.is_tensor(imgs) and imgs.ndim == 4:
            bsz = int(imgs.shape[0])
            keys_list = list(keys_any) if isinstance(keys_any, (list, tuple)) else [keys_any] * bsz
            metas_list = list(metas) if isinstance(metas, (list, tuple)) else [metas] * bsz
            for i in range(bsz):
                yield imgs[i], metas_list[i], canonicalize_key(keys_list[i])
            return

        return

    for batch in loader:
        for img_t, meta_any, key in _iter_samples_from_batch(batch):
            if cfg["data"]["backend"].lower() == "webdataset":
                if key is None or key == "":
                    continue
                if keys is not None and key not in target_set:
                    continue
            else:
                if targets and dataset_idx not in target_set:
                    dataset_idx += 1
                    continue
                dataset_idx += 1

            meta = meta_any if isinstance(meta_any, dict) else {}
            if isinstance(meta_any, (str, bytes)):
                meta = _parse_maybe_json_or_literal(meta_any) or {}

            row = row_by_key.get(key or "", {}) if key else {}

            true_id = int(row.get("y_true", -1)) if row else -1
            true_txt = (
                class_order[true_id]
                if 0 <= true_id < n_classes
                else str(meta.get("class_label", ""))
            )

            pred_id = int(row.get("y_pred", -1)) if row else -1
            pred_txt = (
                class_order[pred_id] if 0 <= pred_id < n_classes else str(pred_id)
            )

            idx_eval = idx_by_key.get(key or "", None) if key else None
            if conf is not None and idx_eval is not None:
                prob = float(conf[idx_eval])
            else:
                prob = float("nan")

            sel_reason_list = sel_reasons.get(key, []) if (keys is not None and key) else []
            sel_reason_str = "|".join(sel_reason_list) if sel_reason_list else ""
            for r in sel_reason_list:
                reason_counts[r] += 1

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
                (out_dir / "selection_reason.txt").write_text(sel_reason_str + "\n", encoding="utf-8")

            png_paths: List[str] = []
            used: List[str] = []

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
                    for h in getattr(cam_method, "hook_handles", []):
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
                        used.append("attn_rollout")
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

            if produced >= total_targets:
                break
        if produced >= total_targets:
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
from typing import Any, Dict, List, Optional, Tuple, Union

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


def canonicalize_key(k: Any) -> str:
    """
    Canonicalize keys coming from predictions.csv and WebDataset __key__.
    - decode bytes
    - strip common dataset prefixes like "test::", "val::", "train::"
    """
    if k is None:
        return ""
    if isinstance(k, (bytes, bytearray)):
        try:
            k = k.decode("utf-8")
        except Exception:
            k = str(k)
    s = str(k).strip()
    for pfx in ("test::", "val::", "train::"):
        if s.startswith(pfx):
            s = s[len(pfx):]
            break
    return s


# -------------------------------------------------------------------------
# Atomic writers (avoid partial files in HPC preemptions)
# -------------------------------------------------------------------------
def ensure_dir(p: Union[str, Path]) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def atomic_write_text(path: Union[str, Path], text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding=encoding) as f:
        f.write(text)
        tmp = Path(f.name)
    tmp.replace(path)


def atomic_write_json(path: Union[str, Path], obj: Any, *, indent: int = 2) -> None:
    atomic_write_text(Path(path), json.dumps(obj, indent=indent, ensure_ascii=False) + "\n")


def atomic_write_csv(path: Union[str, Path], rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
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
    eval_dir: Union[str, Path],
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
        with pcsv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            has_key = "wds_key" in fields
            for row in reader:
                t = row.get("y_true", "")
                yt.append(int(t) if str(t).strip() != "" else -1)
                yp.append(int(row["y_pred"]))
                if has_key:
                    row["wds_key"] = canonicalize_key(row.get("wds_key", ""))
                    kk.append(row["wds_key"])
                else:
                    kk.append(None)
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
def _upgrade_wds_field_key(k: str, kind: str) -> str:
    """
    Backward-compatible key upgrade:
    - legacy configs often use "jpg"/"json"
    - dataset uses multi-extension fields "img.jpg" and "meta.json"
    WebDataset supports alternative extensions via ';' (e.g., "img.jpg;jpg").
    """
    kk = str(k).strip()
    if kind == "image" and kk == "jpg":
        return "img.jpg;jpg"
    if kind == "meta" and kk == "json":
        return "meta.json;json"
    return kk


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

    image_key = _upgrade_wds_field_key(image_key, "image")
    meta_key = _upgrade_wds_field_key(meta_key, "meta")

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

spatial/run_attention_rollout.sh codice <<
#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# User-tunable (override via env)
# -----------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/home/mla_group_01/rcc-ssrl}"
VENV_PATH="${VENV_PATH:-/home/mla_group_01/rcc-ssrl/.venvs/xai}"
PYTHON_BIN="${PYTHON_BIN:-python}"

ATTN_SCRIPT="${ATTN_SCRIPT:-$PROJECT_ROOT/src/explainability/spatial/attention_rollout.py}"

# Eval artifacts
EVAL_RUN_DIR="${EVAL_RUN_DIR:-}"                         # REQUIRED
PREDICTIONS_CSV="${PREDICTIONS_CSV:-predictions.csv}"
LOGITS_NPY="${LOGITS_NPY:-logits_test.npy}"

# Checkpoints
MODEL_NAME="${MODEL_NAME:-ssl_vit}"
BACKBONE_NAME="${BACKBONE_NAME:-vit_small_patch16_224}"
SSL_BACKBONE_CKPT="${SSL_BACKBONE_CKPT:-}"               # REQUIRED
SSL_HEAD_CKPT="${SSL_HEAD_CKPT:-}"                       # REQUIRED

# Data
DATA_BACKEND="${DATA_BACKEND:-webdataset}"               # webdataset | imagefolder
TEST_WDS_DIR="${TEST_WDS_DIR:-}"                         # REQUIRED if webdataset
WDS_PATTERN="${WDS_PATTERN:-*.tar}"
# FIX: dataset usa chiavi multi-estensione (es. "img.jpg", "meta.json")
WDS_IMAGE_KEY="${WDS_IMAGE_KEY:-img.jpg;jpg}"
WDS_META_KEY="${WDS_META_KEY:-meta.json;json}"
TEST_IMAGEFOLDER_DIR="${TEST_IMAGEFOLDER_DIR:-}"         # REQUIRED if imagefolder

IMG_SIZE="${IMG_SIZE:-224}"
IMAGENET_NORM="${IMAGENET_NORM:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Runtime
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1337}"

# Outputs
OUTPUTS_ROOT="${OUTPUTS_ROOT:-$PROJECT_ROOT/outputs/xai_spatial}"
RUN_ID="${RUN_ID:-}"

# XAI
XAI_METHODS="${XAI_METHODS:-attn_rollout}"
ATNN_DISCARD_RATIO="${ATNN_DISCARD_RATIO:-0.9}"
GRADCAM_TARGET_LAYER="${GRADCAM_TARGET_LAYER:-backbone.model.blocks.11}"
IG_STEPS="${IG_STEPS:-32}"

# Selection
FULL_TEST="${FULL_TEST:-0}"
TOPK_TP="${TOPK_TP:-6}"
TOPK_FN="${TOPK_FN:-6}"
TOPK_FP="${TOPK_FP:-6}"
TOPK_LOWCONF="${TOPK_LOWCONF:-20}"

# Labels
CLASS_ORDER_JSON="${CLASS_ORDER_JSON:-[\"ccRCC\",\"pRCC\",\"chRCC\",\"oncocytoma\",\"unclassified\"]}"

# -----------------------------
# Basic checks
# -----------------------------
if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "[FATAL] PROJECT_ROOT not found: $PROJECT_ROOT" >&2
  exit 2
fi
if [[ ! -f "$ATTN_SCRIPT" ]]; then
  echo "[FATAL] attention_rollout.py not found: $ATTN_SCRIPT" >&2
  exit 2
fi
if [[ -z "${EVAL_RUN_DIR}" || ! -d "${EVAL_RUN_DIR}" ]]; then
  echo "[FATAL] Set EVAL_RUN_DIR to a valid eval directory containing predictions/logits." >&2
  echo "        Current EVAL_RUN_DIR: ${EVAL_RUN_DIR:-<empty>}" >&2
  echo "        Expected files: ${PREDICTIONS_CSV} and ${LOGITS_NPY}" >&2
  exit 2
fi
if [[ -z "${SSL_BACKBONE_CKPT}" || ! -f "${SSL_BACKBONE_CKPT}" ]]; then
  echo "[FATAL] Set SSL_BACKBONE_CKPT to an existing file." >&2
  exit 2
fi
if [[ -z "${SSL_HEAD_CKPT}" || ! -f "${SSL_HEAD_CKPT}" ]]; then
  echo "[FATAL] Set SSL_HEAD_CKPT to an existing file." >&2
  exit 2
fi

if [[ "${DATA_BACKEND}" == "webdataset" ]]; then
  if [[ -z "${TEST_WDS_DIR}" || ! -d "${TEST_WDS_DIR}" ]]; then
    echo "[FATAL] DATA_BACKEND=webdataset requires TEST_WDS_DIR (directory with .tar shards)." >&2
    exit 2
  fi
elif [[ "${DATA_BACKEND}" == "imagefolder" ]]; then
  if [[ -z "${TEST_IMAGEFOLDER_DIR}" || ! -d "${TEST_IMAGEFOLDER_DIR}" ]]; then
    echo "[FATAL] DATA_BACKEND=imagefolder requires TEST_IMAGEFOLDER_DIR." >&2
    exit 2
  fi
else
  echo "[FATAL] Unknown DATA_BACKEND=${DATA_BACKEND} (use webdataset|imagefolder)." >&2
  exit 2
fi

# -----------------------------
# Optional tee logs
# -----------------------------
if [[ -n "${LOG_DIR:-}" ]]; then
  mkdir -p "${LOG_DIR}"
  exec > >(tee -a "${LOG_DIR}/run_attention_rollout.${RUN_ID:-local}.$(date +%Y%m%d_%H%M%S).out") 2>&1
fi

# -----------------------------
# Env / venv
# -----------------------------
cd "$PROJECT_ROOT"
if [[ -n "${VENV_PATH}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# -----------------------------
# Build config YAML (generated)
# -----------------------------
mkdir -p "${OUTPUTS_ROOT}"
CFG_DIR="${OUTPUTS_ROOT}/_configs"
mkdir -p "${CFG_DIR}"

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

CFG_PATH="${CFG_DIR}/attn_rollout_${MODEL_NAME}_${RUN_ID}.yaml"

# EXPORT: tutto ciò che il blocco Python legge da os.environ
export PROJECT_ROOT DATA_BACKEND XAI_METHODS FULL_TEST TOPK_TP TOPK_FN TOPK_FP TOPK_LOWCONF
export CLASS_ORDER_JSON SEED RUN_ID OUTPUTS_ROOT DEVICE MODEL_NAME BACKBONE_NAME
export SSL_BACKBONE_CKPT SSL_HEAD_CKPT EVAL_RUN_DIR PREDICTIONS_CSV LOGITS_NPY
export IMG_SIZE IMAGENET_NORM BATCH_SIZE NUM_WORKERS
export TEST_WDS_DIR WDS_PATTERN WDS_IMAGE_KEY WDS_META_KEY TEST_IMAGEFOLDER_DIR
export ATNN_DISCARD_RATIO GRADCAM_TARGET_LAYER IG_STEPS
export CFG_PATH

"${PYTHON_BIN}" - <<'PY'
import json, os
from pathlib import Path
import yaml

def env(name, default=None):
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default

def env_int(name, default):
    return int(env(name, str(default)))

def env_float(name, default):
    return float(env(name, str(default)))

def env_bool01(name, default):
    v = str(env(name, str(default))).strip().lower()
    return v in ("1","true","yes","y","on")

data_backend = str(env("DATA_BACKEND", "webdataset")).lower()
xai_methods = [m.strip() for m in str(env("XAI_METHODS", "attn_rollout")).split(",") if m.strip()]

full_test = env_bool01("FULL_TEST", 0)
if full_test:
    topk_tp = 10**9
    topk_fn = 10**9
    topk_fp = 10**9
    topk_low = 0
else:
    topk_tp = env_int("TOPK_TP", 6)
    topk_fn = env_int("TOPK_FN", 6)
    topk_fp = env_int("TOPK_FP", 6)
    topk_low = env_int("TOPK_LOWCONF", 20)

class_order = json.loads(str(env("CLASS_ORDER_JSON", "[]")))
if not class_order:
    raise SystemExit("CLASS_ORDER_JSON is empty; set it to the correct class order.")

cfg = {
  "experiment": {
    "seed": env_int("SEED", 1337),
    "run_id": str(env("RUN_ID")),
    "outputs_root": str(env("OUTPUTS_ROOT")),
  },
  "runtime": {"device": str(env("DEVICE", "cuda"))},
  "model": {
    "name": str(env("MODEL_NAME", "ssl_vit")),
    "arch_hint": "ssl_linear",
    "backbone_name": str(env("BACKBONE_NAME", "vit_small_patch16_224")),
    "ssl_backbone_ckpt": str(env("SSL_BACKBONE_CKPT")),
    "ssl_head_ckpt": str(env("SSL_HEAD_CKPT")),
  },
  "evaluation_inputs": {
    "eval_run_dir": str(env("EVAL_RUN_DIR")),
    "predictions_csv": str(env("PREDICTIONS_CSV", "predictions.csv")),
    "logits_npy": str(env("LOGITS_NPY", "logits_test.npy")),
  },
  "labels": {"class_order": class_order},
  "data": {
    "backend": data_backend,
    "img_size": env_int("IMG_SIZE", 224),
    "imagenet_norm": env_bool01("IMAGENET_NORM", 1),
    "batch_size": env_int("BATCH_SIZE", 1),
    "num_workers": env_int("NUM_WORKERS", 8),
  },
  "selection": {
    "per_class": {"topk_tp": topk_tp, "topk_fn": topk_fn, "topk_fp": topk_fp},
    "global_low_conf": {"topk": topk_low},
  },
  "xai": {
    "methods": xai_methods,
    "attn_rollout": {"discard_ratio": env_float("ATNN_DISCARD_RATIO", 0.9)},
    "gradcam": {"target_layer": str(env("GRADCAM_TARGET_LAYER", "backbone.model.blocks.11"))},
    "ig": {"steps": env_int("IG_STEPS", 32)},
  },
}

if data_backend == "webdataset":
    cfg["data"]["webdataset"] = {
      "test_dir": str(env("TEST_WDS_DIR")),
      "pattern": str(env("WDS_PATTERN", "*.tar")),
      "image_key": str(env("WDS_IMAGE_KEY", "img.jpg;jpg")),
      "meta_key": str(env("WDS_META_KEY", "meta.json;json")),
    }
else:
    cfg["data"]["imagefolder"] = {"test_dir": str(env("TEST_IMAGEFOLDER_DIR"))}

out = Path(str(env("CFG_PATH")))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(str(out))
PY

echo "[INFO] Config written: ${CFG_PATH}"
echo "[INFO] Running: ${PYTHON_BIN} ${ATTN_SCRIPT} --config ${CFG_PATH}"
exec "${PYTHON_BIN}" "${ATTN_SCRIPT}" --config "${CFG_PATH}"
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

utils/bootstrap.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

"""
Helper to run package modules as scripts without breaking relative imports.

Usage (top of a file):
  from explainability.utils.bootstrap import bootstrap_package
  bootstrap_package(__file__, globals())
"""

from pathlib import Path
import sys
from typing import Dict


def bootstrap_package(file: str, g: Dict) -> None:
    """
    If executed as a script (no __package__), add .../src to sys.path and set __package__
    so relative imports work.
    """
    if g.get("__package__"):
        return
    this = Path(file).resolve()

    src_dir = this
    while src_dir.name != "src" and src_dir.parent != src_dir:
        src_dir = src_dir.parent
    if src_dir.name != "src":
        return

    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    rel = this.relative_to(src_dir).with_suffix("")   # explainability/...
    pkg = ".".join(rel.parts[:-1])                    # explainability.<...>
    g["__package__"] = pkg
>>

utils/class_utils.py codice <<
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
>>

utils/__init__.py codice <<
"""
Small utilities used across the explainability pipeline.
"""
>>

utils/roi_utils.py codice <<
#!/usr/bin/env python3
from __future__ import annotations

"""
ROI utilities:
  - robust bbox extraction from 2D rollout masks (supports low-res 14x14 etc)
  - safe scaling to image pixel coordinates
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class RoiBox:
    x0: int
    y0: int
    x1: int
    y1: int
    method: str
    threshold: float

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return int(self.x0), int(self.y0), int(self.x1), int(self.y1)


def _normalize_mask(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape={m.shape}")
    mn = float(np.nanmin(m))
    mx = float(np.nanmax(m))
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) <= 1e-12:
        return np.zeros_like(m, dtype=np.float32)
    out = (m - mn) / (mx - mn)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def extract_bbox_from_mask(
    mask_2d: np.ndarray,
    *,
    img_w: int,
    img_h: int,
    quantile: float = 0.90,
    min_area_frac: float = 0.01,
    pad_frac: float = 0.05,
) -> RoiBox:
    """
    Extract a single bbox around the most salient region.

    - quantile: threshold at q-quantile of normalized mask.
    - min_area_frac: if bbox is too small, fall back to full image.
    - pad_frac: expand bbox by this fraction of its size (clamped).
    """
    m = _normalize_mask(mask_2d)
    thr = float(np.quantile(m, quantile)) if m.size > 0 else 1.0
    bw = (m >= thr)
    ys, xs = np.where(bw)

    # fallback: no pixels above threshold
    if xs.size == 0 or ys.size == 0:
        return RoiBox(0, 0, img_w - 1, img_h - 1, method="fallback_full", threshold=thr)

    # bbox in mask coords
    x0m, x1m = int(xs.min()), int(xs.max())
    y0m, y1m = int(ys.min()), int(ys.max())

    mh, mw = m.shape[0], m.shape[1]
    sx = float(img_w) / float(max(1, mw))
    sy = float(img_h) / float(max(1, mh))

    x0 = int(np.floor(x0m * sx))
    x1 = int(np.ceil((x1m + 1) * sx) - 1)
    y0 = int(np.floor(y0m * sy))
    y1 = int(np.ceil((y1m + 1) * sy) - 1)

    # clamp
    x0 = max(0, min(img_w - 1, x0))
    x1 = max(0, min(img_w - 1, x1))
    y0 = max(0, min(img_h - 1, y0))
    y1 = max(0, min(img_h - 1, y1))

    # padding
    bwx = max(1, x1 - x0 + 1)
    bwy = max(1, y1 - y0 + 1)
    px = int(round(pad_frac * bwx))
    py = int(round(pad_frac * bwy))
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - py)
    x1 = min(img_w - 1, x1 + px)
    y1 = min(img_h - 1, y1 + py)

    area = float((x1 - x0 + 1) * (y1 - y0 + 1))
    full = float(img_w * img_h)
    if full > 0 and (area / full) < float(min_area_frac):
        return RoiBox(0, 0, img_w - 1, img_h - 1, method="fallback_small_bbox", threshold=thr)

    return RoiBox(x0, y0, x1, y1, method="quantile_bbox", threshold=thr)
>>

