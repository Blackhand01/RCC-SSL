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


def _save_heatmap_mean_by_class(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    concept_short: Sequence[str],
    out_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if scores.size == 0:
        return

    labels_arr = np.asarray(labels, dtype=object)
    # Preserve first-seen order of labels
    classes = [str(c) for c in dict.fromkeys(labels_arr.tolist())]
    if not classes:
        return

    n_classes = len(classes)
    n_concepts = scores.shape[1]
    mat = np.zeros((n_classes, n_concepts), dtype=np.float32)
    for i, cls in enumerate(classes):
        mask = labels_arr == cls
        if not np.any(mask):
            continue
        mat[i] = scores[mask].mean(axis=0)

    out_base.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(
            max(8.0, 0.3 * n_concepts),
            max(4.0, 0.4 * n_classes),
        )
    )
    im = ax.imshow(mat, aspect="auto")
    ax.set_yticks(np.arange(n_classes))
    ax.set_yticklabels(classes)
    ax.set_xticks(np.arange(len(concept_short)))
    ax.set_xticklabels(concept_short, rotation=90, fontsize=6)
    ax.set_xlabel("Concept")
    ax.set_ylabel("Class label")
    ax.set_title("NO-ROI: mean concept score per class")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    for fmt in formats:
        ext = str(fmt).lstrip(".")
        out_path = out_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)


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

    concepts = load_concepts(concepts_json)
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
    max_patches = (
        int(args.max_patches)
        if getattr(args, "max_patches", None) is not None
        else int(data.get("max_patches", 0))
    )

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
