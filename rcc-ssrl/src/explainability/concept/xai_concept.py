#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concept-based XAI on RCC test set.

- Reuses SSLLinearClassifier (ResNet/ViT + linear head) from spatial XAI.
- Loads predictions.csv + logits_test.npy from evaluation run.
- Selects a subset of test patches (TP/FP/FN + low-confidence TP).
- Uses a concept bank defined as: concept_name -> list of WebDataset keys.
- Builds a feature centroid per concept and computes similarity between
  each selected patch feature and all concept centroids.

Outputs:
- Per-patch directory: input.png + concept_scores.json
- Global CSV index: index.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import yaml

try:
    import webdataset as wds  # noqa: F401
    HAVE_WDS = True
except Exception:
    HAVE_WDS = False

from explainability.common.eval_utils import (
    setup_logger,
    set_seed,
    build_preprocess,
    tensor_to_pil,
    load_eval_artifacts,
    select_items,
    make_wds_loader_with_keys,
)
from explainability.common.ssl_linear_loader import SSLLinearClassifier


# -------------------------------------------------------------------------
# Concept bank loading
# -------------------------------------------------------------------------
def load_concept_bank(cfg_concepts: Dict[str, Any], log: logging.Logger):
    meta_csv = cfg_concepts["meta_csv"]
    name_col = cfg_concepts["concept_name_col"]
    key_col = cfg_concepts["key_col"]
    group_col = cfg_concepts.get("group_col")
    class_col = cfg_concepts.get("class_col")

    concept_to_keys: Dict[str, List[str]] = {}
    concept_meta: Dict[str, Dict[str, Any]] = {}
    concept_keys: Set[str] = set()

    with open(meta_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cname = row[name_col]
            key = row[key_col]
            if not cname or not key:
                continue
            concept_to_keys.setdefault(cname, []).append(key)
            concept_keys.add(key)
            if cname not in concept_meta:
                concept_meta[cname] = {
                    "group": row.get(group_col) if group_col in (reader.fieldnames or []) else None,
                    "class_label": row.get(class_col) if class_col in (reader.fieldnames or []) else None,
                }

    log.info(
        f"Loaded concept bank: {len(concept_to_keys)} concepts, {len(concept_keys)} unique keys."
    )
    return concept_to_keys, concept_meta, concept_keys


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    log = setup_logger()

    parser = argparse.ArgumentParser(description="Concept-based XAI for SSL RCC model")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(open(args.config, "r"))

    # Seed / device
    set_seed(int(cfg["experiment"]["seed"]))
    device = torch.device(
        cfg.get("runtime", {}).get("device", "cuda")
        if torch.cuda.is_available()
        else "cpu"
    )

    # Output root
    out_root = (
        Path(cfg["experiment"]["outputs_root"])
        / cfg["model"]["name"]
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_root.mkdir(parents=True, exist_ok=True)
    log.info(f"[Concept XAI] Output dir: {out_root}")

    # Eval artifacts
    y_true, y_pred, conf, keys, meta_rows = load_eval_artifacts(
        cfg["evaluation_inputs"]["eval_run_dir"],
        cfg["evaluation_inputs"]["predictions_csv"],
        cfg["evaluation_inputs"]["logits_npy"],
        log,
    )
    if y_pred is None or keys is None:
        raise RuntimeError("Concept XAI requires predictions with wds_key column.")

    class_order: List[str] = cfg["labels"]["class_order"]
    num_classes = len(class_order)

    # Model
    model = SSLLinearClassifier(
        backbone_name=cfg["model"].get("backbone_name", "resnet50"),
        num_classes=num_classes,
    )
    mb, ub = model.load_backbone_from_ssl(cfg["model"]["ssl_backbone_ckpt"])
    mh, uh = model.load_head_from_probe(cfg["model"]["ssl_head_ckpt"])
    log.info(
        f"Loaded backbone from SSL (missing={mb}, unexpected={ub}); "
        f"head from probe (missing={mh}, unexpected={uh})."
    )

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Selection (same as spatial)

    targets, sel_reasons = select_items(
        y_true,
        y_pred,
        conf,
        keys,
        num_classes,
        cfg["selection"],
        log,
    )
    target_set: Set[str] = set(targets)
    log.info(f"[Concept XAI] Selected {len(target_set)} target patches.")

    import time
    t_start = time.time()
    total_targets = len(target_set)
    if total_targets == 0:
        log.warning("[Concept XAI] No targets selected, exiting early.")
        return

    # Concept bank
    concept_to_keys, concept_meta, concept_keys = load_concept_bank(
        cfg["concepts"], log
    )

    # Build set of all keys for which we need features
    all_needed_keys: Set[str] = set(target_set) | set(concept_keys)
    log.info(
        f"Total keys that require features: {len(all_needed_keys)} "
        f"(targets={len(target_set)}, concepts={len(concept_keys)})"
    )

    # Data loader (WebDataset only)
    if cfg["data"]["backend"].lower() != "webdataset":
        raise RuntimeError("Concept XAI currently supports only webdataset backend.")
    if not HAVE_WDS:
        raise RuntimeError("webdataset is not installed; required for concept XAI.")

    img_size = int(cfg["data"]["img_size"])
    imagenet_norm = bool(cfg["data"].get("imagenet_norm", False))
    preprocess_fn = build_preprocess(img_size, imagenet_norm)

    w = cfg["data"]["webdataset"]
    num_workers = int(cfg["data"]["num_workers"])

    train_dir = w.get("train_dir")
    test_dir = w.get("test_dir")

    dirs = []
    if train_dir:
        dirs.append(train_dir)
    if test_dir:
        dirs.append(test_dir)

    if not dirs:
        raise RuntimeError("Concept XAI: no webdataset.train_dir or test_dir specified in config.")

    # Map keys to eval indices (per conf) come prima
    idx_by_key = {k: i for i, k in enumerate(keys)}

    feat_by_key: Dict[str, np.ndarray] = {}
    input_tensor_by_key: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for d in dirs:
            try:
                loader = make_wds_loader_with_keys(
                    d,
                    w["pattern"],
                    w["image_key"],
                    w["meta_key"],
                    preprocess_fn,
                    num_workers,
                )
            except FileNotFoundError as e:
                log.warning(f"[Concept XAI] No shards found in {d}: {e}")
                continue

            log.info(f"[Concept XAI] Scanning WebDataset shards in {d}")
            for batch in loader:
                if batch is None:
                    continue
                img_t, meta_any, key = batch

                # Se il key non serve né per concetti né per target, skip
                if key not in all_needed_keys:
                    continue

                x = img_t.to(device)
                if x.ndim == 3:
                    x = x.unsqueeze(0)

                feats = model.backbone.forward_global(x)  # [1, D]
                feat_by_key[key] = feats.squeeze(0).cpu().numpy()

                # Solo per target, salviamo anche il tensor per salvare input.png
                if key in target_set and key not in input_tensor_by_key:
                    input_tensor_by_key[key] = img_t  # tensor già preprocessato

    missing = all_needed_keys - set(feat_by_key.keys())
    if missing:
        log.warning(f"Missing features for {len(missing)} keys (will be ignored).")

    # Build concept centroids
    min_patches_per_concept = int(cfg["concepts"]["min_patches_per_concept"])
    centroids: Dict[str, np.ndarray] = {}
    dims = None

    for cname, ckeys in concept_to_keys.items():
        feats = [
            feat_by_key[k]
            for k in ckeys
            if k in feat_by_key
        ]
        if len(feats) < min_patches_per_concept:
            log.warning(
                f"Concept '{cname}' has only {len(feats)} usable patches "
                f"(min required={min_patches_per_concept}); skipping."
            )
            continue
        mat = np.stack(feats, axis=0)
        centroid = mat.mean(axis=0)
        centroids[cname] = centroid
        dims = centroid.shape[0]

    if not centroids:
        raise RuntimeError("No valid concept centroids could be built.")

    concept_names = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in concept_names], axis=0)  # [C, D]

    # Normalize for cosine similarity if requested
    sim_type = cfg["concepts"].get("similarity", "cosine").lower()
    if sim_type == "cosine":
        centroid_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-9
        centroid_matrix = centroid_matrix / centroid_norms

    # Global index CSV
    index_csv = open(out_root / "index.csv", "w", newline="")
    writer = csv.writer(index_csv)
    writer.writerow(
        [
            "global_idx",
            "wds_key",
            "true_label",
            "pred_label",
            "conf",
            "selection_reason",
            "top_concepts",  # pipe-separated: name:score
        ]
    )

    # Quick lookup for meta_rows by key
    row_by_key = {r["wds_key"]: r for r in (meta_rows or [])} if meta_rows else {}

    topk_per_patch = int(cfg["concepts"]["topk_per_patch"])

    # per summary finale
    reason_counts: Counter[str] = Counter()

    # Iterate over targets in a deterministic order
    produced = 0
    for global_idx, key in enumerate(sorted(target_set)):
        if key not in feat_by_key:
            log.warning(f"Skipping target {key}: no feature available.")
            continue

        feat = feat_by_key[key]  # [D]
        if sim_type == "cosine":
            fnorm = np.linalg.norm(feat) + 1e-9
            feat_vec = feat / fnorm
        else:
            feat_vec = feat

        scores = centroid_matrix @ feat_vec  # [C]
        # Top-k concepts
        order = np.argsort(scores)[::-1]
        k_top = order[:topk_per_patch]

        # Labels / conf from eval
        row = row_by_key.get(key, {})
        true_id = int(row.get("y_true", -1)) if row else -1
        pred_id = int(row.get("y_pred", -1)) if row else -1

        true_label = (
            class_order[true_id] if 0 <= true_id < num_classes else row.get("true_label", "")
        )
        pred_label = (
            class_order[pred_id] if 0 <= pred_id < num_classes else row.get("pred_label", "")
        )

        idx_eval = idx_by_key.get(key, None)
        if conf is not None and idx_eval is not None:
            prob = float(conf[idx_eval])
        else:
            prob = float("nan")

        sel_reason_list = sel_reasons.get(key, [])
        sel_reason_str = "|".join(sel_reason_list) if sel_reason_list else ""
        for r in sel_reason_list:
            reason_counts[r] += 1

        # Output dir
        out_dir = out_root / f"idx_{global_idx:07d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save input.png (if tensor available)
        if key in input_tensor_by_key:
            pil_in = tensor_to_pil(
                input_tensor_by_key[key],
                imagenet_norm=imagenet_norm,
            )
            pil_in.save(out_dir / "input.png")

        # Save concept_scores.json
        concept_scores = []
        for i in k_top:
            name = concept_names[i]
            sc = float(scores[i])
            meta = concept_meta.get(name, {})
            concept_scores.append(
                {
                    "name": name,
                    "score": sc,
                    "group": meta.get("group"),
                    "class_label": meta.get("class_label"),
                }
            )

        json_payload = {
            "wds_key": key,
            "true_label": true_label,
            "pred_label": pred_label,
            "conf": prob,
            "selection_reason": sel_reason_list,
            "similarity": sim_type,
            "concept_scores": concept_scores,
        }
        with open(out_dir / "concept_scores.json", "w") as jf:
            json.dump(json_payload, jf, indent=2)

        # Write index.csv row
        top_concepts_str = "|".join(
            f"{concept_names[i]}:{scores[i]:.5f}" for i in k_top
        )
        writer.writerow(
            [
                global_idx,
                key,
                true_label,
                pred_label,
                prob,
                sel_reason_str,
                top_concepts_str,
            ]
        )


        produced += 1

        if produced % 10 == 0 or produced == total_targets:
            elapsed = time.time() - t_start
            avg_per_item = elapsed / max(1, produced)
            remaining = total_targets - produced
            eta = remaining * avg_per_item
            log.info(
                f"[PROGRESS] Concept XAI: {produced}/{total_targets} "
                f"({100.0*produced/total_targets:.1f}%), "
                f"elapsed={elapsed/60:.1f} min, "
                f"avg={avg_per_item:.2f} s/item, "
                f"ETA~{eta/60:.1f} min"
            )

    index_csv.close()
    total_elapsed = time.time() - t_start
    log.info(
        f"[Concept XAI] Done. Produced {produced} patches with concept scores "
        f"in {total_elapsed/60:.1f} min."
    )

    # ----------------- VALIDAZIONE / SEGNALAZIONI -----------------
    if produced == 0:
        log.warning(
            "[Concept XAI] No concept scores produced (produced=0). "
            "Controlla concept bank / selection."
        )
    else:
        log.info(f"[Concept XAI] Concepts with centroids: {len(centroids)}")
        if reason_counts:
            log.info("[Concept XAI] Selection reasons distribution:")
            for r, cnt in reason_counts.items():
                log.info(f"  - {r}: {cnt} patches")


if __name__ == "__main__":
    main()
