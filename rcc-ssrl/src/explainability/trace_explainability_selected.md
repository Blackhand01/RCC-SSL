explainability/run_spatial-concept.py codice <<
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
import csv
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
    canonicalize_key,
)
from explainability.spatial.ssl_linear_loader import SSLLinearClassifier  # noqa: E402
from explainability.spatial.attention_rollout import ViTAttentionRollout, overlay_heatmap  # noqa: E402

from explainability.utils.class_utils import (  # noqa: E402
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

def _ensure_2d_mask(mask_any: Any) -> np.ndarray:
    """
    Coerce common rollout shapes to a 2D float32 array (H,W).
    Accepts (H,W), (1,H,W), (H,W,1), (B,H,W), etc.
    """
    m = np.asarray(mask_any, dtype=np.float32)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    if m.ndim == 2:
        return m
    m = np.squeeze(m)
    if m.ndim == 2:
        return m
    if m.ndim == 3:
        # channel-first or channel-last -> average channels; otherwise average first dim
        if m.shape[0] in (1, 3, 4) and m.shape[1] != m.shape[0]:
            return m.mean(axis=0).astype(np.float32)
        if m.shape[-1] in (1, 3, 4) and m.shape[-2] != m.shape[-1]:
            return m.mean(axis=-1).astype(np.float32)
        return m.mean(axis=0).astype(np.float32)
    while m.ndim > 2:
        m = m.mean(axis=0)
    return np.asarray(m, dtype=np.float32)


def _apply_mask_to_image(img: Image.Image, mask_bool: np.ndarray) -> Image.Image:
    w, h = img.size
    m_img = Image.fromarray(mask_bool.astype(np.uint8) * 255).resize((w, h), resample=Image.NEAREST)
    m = (np.asarray(m_img) > 0)
    arr = np.asarray(img.convert("RGB"))
    out = arr.copy()
    out[~m] = 0
    return Image.fromarray(out, mode="RGB")


def _load_precomputed_attn_rollout_index(
    *,
    spatial_run_dir: Path,
    log: logging.Logger,
) -> Dict[str, Path]:
    """
    Build key -> attn_rollout.npy mapping from attention_rollout.py outputs:
      <spatial_run_dir>/index.csv
      <spatial_run_dir>/idx_0000000/attn_rollout.npy
    """
    index_csv = spatial_run_dir / "index.csv"
    if not index_csv.exists():
        return {}

    m: Dict[str, Path] = {}
    try:
        with index_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = canonicalize_key(str(row.get("wds_key", "") or "").strip())
                if not k:
                    continue
                try:
                    gi = int(str(row.get("global_idx", "")).strip())
                except Exception:
                    continue
                npy = spatial_run_dir / f"idx_{gi:07d}" / "attn_rollout.npy"
                if npy.exists():
                    m[k] = npy
    except Exception as e:
        log.warning("Failed to read precomputed attn_rollout index %s (%s)", index_csv, e)
        return {}

    return m


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
    reuse_attn_rollout: bool,
    attn_rollout_outputs_root: Optional[Path],
    attn_rollout_run_id: Optional[str],
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

    # Optional reuse of precomputed attention rollout masks from spatial/attention_rollout.py
    precomputed_rollout_by_key: Dict[str, Path] = {}
    if reuse_attn_rollout:
        if attn_rollout_outputs_root is None:
            log.warning("[%s] reuse_attn_rollout enabled but attn_rollout_outputs_root is None -> fallback to on-the-fly rollout.", model_id)
        else:
            rid = str(attn_rollout_run_id or run_id)
            spatial_run_dir = Path(attn_rollout_outputs_root) / model_id / rid
            if spatial_run_dir.exists():
                precomputed_rollout_by_key = _load_precomputed_attn_rollout_index(spatial_run_dir=spatial_run_dir, log=log)
                if precomputed_rollout_by_key:
                    log.info(
                        "[%s] Reusing precomputed attn_rollout masks: %d (spatial_run_dir=%s)",
                        model_id,
                        len(precomputed_rollout_by_key),
                        spatial_run_dir,
                    )
                else:
                    log.warning("[%s] reuse_attn_rollout enabled but no masks indexed (index.csv missing/empty) in %s", model_id, spatial_run_dir)
            else:
                log.warning("[%s] reuse_attn_rollout enabled but spatial_run_dir not found: %s", model_id, spatial_run_dir)

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
            kk = canonicalize_key(str(k))
            if kk not in idx_by_key:
                idx_by_key[kk] = int(i)
        selected_keys = [canonicalize_key(str(k)) for k in targets]
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
            # Prefer precomputed rollout (3/5) keyed by WebDataset key
            if key is not None and precomputed_rollout_by_key:
                kcanon = canonicalize_key(str(key))
                p = precomputed_rollout_by_key.get(kcanon, None)
                if p is not None and p.exists():
                    try:
                        mask_np = _ensure_2d_mask(np.load(p))
                    except Exception as e:
                        log.warning("[%s] Failed to load precomputed attn_rollout.npy (%s): %s", model_id, p, e)
                        mask_np = None  # fall back below
                else:
                    mask_np = None
            else:
                mask_np = None

            if mask_np is not None:
                mask_np = _ensure_2d_mask(mask_np)
            else:
                if rollout is not None:
                    x = img_t.unsqueeze(0).to(device)
                    m = rollout(x)
                    mask_np = _ensure_2d_mask(m) if m is not None else np.zeros((1, 1), dtype=np.float32)
                else:
                    mask_np = np.zeros((1, 1), dtype=np.float32)
        except Exception as e:
            log.warning("[%s] rollout failed idx=%d key=%s (%s) -> fallback full ROI", model_id, idx_eval, key, e)
            mask_np = np.zeros((1, 1), dtype=np.float32)

        mask_np = _ensure_2d_mask(mask_np)
        np.save(item_dir / "attn_rollout.npy", mask_np.astype(np.float32, copy=False))
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
            wanted = set(canonicalize_key(k) for k in wanted_keys_set)
            found = set()
            for img_t, _meta, kk in iter_wds_filtered_by_keys(loader, wanted):
                kkc = canonicalize_key(str(kk))
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
                    _process_sample(img_t, idx_eval=int(seen), key=canonicalize_key(str(kk)))
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

    ap.add_argument(
        "--reuse-attn-rollout",
        action="store_true",
        help=(
            "Reuse precomputed attention rollout masks produced by "
            "src/explainability/spatial/attention_rollout.py. "
            "Expects: <attn-rollout-outputs-root>/<MODEL_ID>/<attn-rollout-run-id>/index.csv "
            "and idx_*/attn_rollout.npy."
        ),
    )
    ap.add_argument(
        "--attn-rollout-outputs-root",
        type=Path,
        default=None,
        help="Root directory containing attention_rollout.py outputs (e.g. /home/mla_group_01/rcc-ssrl/src/explainability/output/spatial).",
    )
    ap.add_argument(
        "--attn-rollout-run-id",
        type=str,
        default=None,
        help="Run id folder used by attention_rollout.py (if omitted, falls back to --run-id).",
    )
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
                    reuse_attn_rollout=bool(args.reuse_attn_rollout),
                    attn_rollout_outputs_root=(args.attn_rollout_outputs_root.expanduser().resolve() if args.attn_rollout_outputs_root is not None else None),
                    attn_rollout_run_id=(str(args.attn_rollout_run_id) if args.attn_rollout_run_id is not None else None),
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

explainability/concept/run_no_roi.py codice <<
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
>>

