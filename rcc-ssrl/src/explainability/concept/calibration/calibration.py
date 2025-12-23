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
