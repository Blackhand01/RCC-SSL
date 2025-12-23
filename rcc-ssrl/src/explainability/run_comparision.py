#!/usr/bin/env python3
from __future__ import annotations

"""
Compare ROI vs NO-ROI concept scores.

Inputs:

  NO-ROI (model-independent, canonico):
    - NO_ROI_PATHS.root_dir / artifacts/
        scores_fp32.npy
        keys.npy
        selected_concepts.json

  ROI (per modello, prodotti da run_spatial-concept.py):
    - src/explainability/output/roi/<MODEL_ID>/spatial_concept/latest_run.json
      {
        "heavy_run_dir": ".../attention_rollout_concept/run_<RUN_ID>/",
        "summary_json": ".../src/explainability/output/roi/<MODEL_ID>/spatial_concept/xai_summary.json",
        ...
      }
    - summary_json contiene "items", ognuno con:
        key
        concept_scores_json (path relativo a heavy_run_dir)

Outputs canonici:
  - XAI_ROOT/roi-no_roi-comparision/<MODEL_ID>/
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

from ..spatial.eval_utils import atomic_write_text  # noqa: E402
from ..paths import (  # noqa: E402
    CONFIG_DIR,
    NO_ROI_PATHS,
    ensure_comparison_layout,
    get_light_stats_dir,
)
import yaml  # noqa: E402


def _load_selected_concepts(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text())
    sel = obj.get("selected", [])
    if not isinstance(sel, list) or not sel:
        raise RuntimeError(f"Invalid selected_concepts.json: {path}")
    return sel


def _canon_key_strip_prefix(k: str) -> str:
    """
    Canonicalizza chiavi tipo 'test::abcd' -> 'abcd' (come in run_spatial-concept).
    """
    s = str(k)
    if "::" in s:
        pref = s.split("::", 1)[0].strip().lower()
        if pref in ("train", "val", "test"):
            return s.split("::", 1)[1]
    return s


def _load_latest_roi_summary(model_id: str, log: logging.Logger) -> Tuple[Path, List[Dict[str, Any]]]:
    """
    Trova heavy_run_dir + lista items ROI a partire da output/roi/<MODEL_ID>/spatial_concept/latest_run.json
    """
    base = get_light_stats_dir("roi", model_id)
    out_dir = base / "spatial_concept"
    latest_path = out_dir / "latest_run.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"latest_run.json non trovato per ROI: {latest_path} "
            "(esegui prima run_spatial-concept.py)"
        )

    latest = json.loads(latest_path.read_text())
    heavy_run_dir = Path(latest.get("heavy_run_dir", "")).expanduser()
    if not heavy_run_dir.exists():
        raise FileNotFoundError(f"heavy_run_dir non esistente: {heavy_run_dir}")

    summary_json_path = Path(latest.get("summary_json", ""))
    if not summary_json_path.is_absolute():
        summary_json_path = out_dir / summary_json_path
    if not summary_json_path.exists():
        raise FileNotFoundError(f"summary_json non trovato: {summary_json_path}")

    summary = json.loads(summary_json_path.read_text())
    items = summary.get("items", [])
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"xai_summary.json vuoto o invalido: {summary_json_path}")
    log.info("Trovati %d item ROI (summary=%s)", len(items), summary_json_path)
    return heavy_run_dir, items


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ROI vs NO-ROI comparison (paper-ready).")
    ap.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="Root di un'ablation (es: .../exp_*/exp_*_abl01)",
    )
    ap.add_argument(
        "--no-roi-root",
        type=Path,
        default=NO_ROI_PATHS.root_dir,
        help="Root NO-ROI canonico (default: NO_ROI_PATHS.root_dir).",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DIR / "comparision.yaml",
        help="Config YAML (opzionale, default: explainability/configs/comparision.yaml)",
    )
    ap.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override model id (default: nome della cartella model-root).",
    )
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

    # Config opzionale per topk
    topk_cfg = None
    if args.config and args.config.exists():
        try:
            cfg = yaml.safe_load(args.config.read_text())
            if isinstance(cfg, dict) and "topk" in cfg:
                topk_cfg = int(cfg["topk"])
        except Exception as e:
            log.warning("Failed to read comparison config %s: %s", args.config, e)

    model_root = args.model_root.resolve()
    if not model_root.exists():
        raise FileNotFoundError(f"model_root not found: {model_root}")
    model_id = args.model_id or model_root.name

    lay = ensure_comparison_layout(model_id=model_id)

    # ---------------- NO-ROI canonical ----------------
    no_roi_root = args.no_roi_root or NO_ROI_PATHS.root_dir
    no_roi_art = no_roi_root / "artifacts"
    no_scores_p = no_roi_art / "scores_fp32.npy"
    no_keys_p = no_roi_art / "keys.npy"
    sel_json_p = no_roi_art / "selected_concepts.json"

    for p in (no_scores_p, no_keys_p, sel_json_p):
        if not p.exists():
            raise FileNotFoundError(f"NO-ROI artifact mancante: {p}")

    no_scores = np.load(no_scores_p)
    no_keys = np.load(no_keys_p, allow_pickle=True).astype(object).tolist()
    sel = _load_selected_concepts(sel_json_p)
    concept_names = [str(x.get("concept_short_name")) for x in sel]
    if no_scores.ndim != 2:
        raise RuntimeError(f"NO-ROI scores_fp32 deve essere 2D, shape={no_scores.shape}")

    n_concepts = len(concept_names)
    if no_scores.shape[1] != n_concepts:
        raise RuntimeError(
            f"NO-ROI scores dim mismatch: scores.shape[1]={no_scores.shape[1]} vs selected={n_concepts}"
        )

    idx_by_name = {name: i for i, name in enumerate(concept_names)}
    no_map = {_canon_key_strip_prefix(str(k)): i for i, k in enumerate(no_keys)}

    # ---------------- ROI (run_spatial-concept) ----------------
    heavy_run_dir, items = _load_latest_roi_summary(model_id, log)

    roi_vecs: List[np.ndarray] = []
    roi_keys: List[str] = []
    missing_scores = 0

    for row in items:
        key = str(row.get("key", "")).strip()
        if not key:
            continue
        rel_cs = row.get("concept_scores_json", "")
        if not rel_cs:
            missing_scores += 1
            continue
        cs_path = heavy_run_dir / rel_cs
        if not cs_path.exists():
            log.warning("concept_scores.json mancante per key=%s: %s", key, cs_path)
            missing_scores += 1
            continue

        try:
            obj = json.loads(cs_path.read_text())
        except Exception as e:
            log.warning("concept_scores.json illeggibile (%s): %s", cs_path, e)
            missing_scores += 1
            continue

        scores_map = obj.get("scores", {})
        if not isinstance(scores_map, dict) or not scores_map:
            missing_scores += 1
            continue

        meta = obj.get("meta", {}) or {}
        key_meta = str(meta.get("key") or key).strip()
        kcanon = _canon_key_strip_prefix(key_meta)

        v = np.full((n_concepts,), np.nan, dtype=np.float32)
        for sname, sval in scores_map.items():
            j = idx_by_name.get(str(sname))
            if j is None:
                continue
            try:
                v[j] = float(sval)
            except Exception:
                continue

        roi_vecs.append(v)
        roi_keys.append(kcanon)

    if not roi_vecs:
        raise RuntimeError(
            f"Nessun vettore ROI valido trovato (items={len(items)}, missing_scores={missing_scores})."
        )

    # Allineamento chiavi
    aligned_roi: List[np.ndarray] = []
    aligned_no: List[np.ndarray] = []
    aligned_keys: List[str] = []
    no_missing = 0

    for k, v in zip(roi_keys, roi_vecs):
        j = no_map.get(k, None)
        if j is None:
            no_missing += 1
            continue
        aligned_roi.append(v)
        aligned_no.append(no_scores[j])
        aligned_keys.append(k)

    if not aligned_keys:
        raise RuntimeError(
            f"Nessuna chiave in comune tra ROI e NO-ROI. "
            f"ROI_valid={len(roi_vecs)} missing_in_no_roi={no_missing}"
        )

    A = np.asarray(aligned_roi, dtype=np.float32)
    B = np.asarray(aligned_no, dtype=np.float32)
    if A.shape != B.shape:
        raise RuntimeError(f"Shape mismatch ROI vs NO-ROI: {A.shape} vs {B.shape}")

    D = A - B
    mean_delta = np.nanmean(D, axis=0)
    mean_abs = np.nanmean(np.abs(D), axis=0)

    # ---------------- Output tabella + figura ----------------
    df = pd.DataFrame(
        {
            "concept_short_name": concept_names,
            "mean_delta_roi_minus_no_roi": mean_delta.astype(np.float64),
            "mean_abs_delta": mean_abs.astype(np.float64),
        }
    ).sort_values("mean_abs_delta", ascending=False)

    lay.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(lay.summary_csv, index=False)

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
        ax.set_title(
            f"{model_id}: ROI vs NO-ROI - top-{topk} concepts by mean absolute delta"
        )
        fig.tight_layout()
        lay.figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(lay.figures_dir / "top_abs_delta.png", dpi=300, bbox_inches="tight")
        fig.savefig(lay.figures_dir / "top_abs_delta.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.warning("Plot failed: %s", str(e))

    # ---------------- Report markdown ----------------
    report = []
    report.append("# ROI vs NO-ROI comparison\n")
    report.append(f"- model_id: `{model_id}`")
    report.append(f"- model_root: `{model_root}`")
    report.append(f"- no_roi_root: `{no_roi_root}`")
    report.append(
        f"- n_overlap: **{len(aligned_keys)}** "
        f"(roi_valid={len(roi_vecs)}, missing_in_no_roi={no_missing})\n"
    )
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
