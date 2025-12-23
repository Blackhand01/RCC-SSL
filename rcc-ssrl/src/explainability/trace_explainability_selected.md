class_utils.py codice <<
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

roi_utils.py codice <<
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
                k = _canon_key_strip_prefix(str(row.get("wds_key", "") or "").strip())
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
            # Prefer precomputed rollout (3/5) keyed by WebDataset key
            if key is not None and precomputed_rollout_by_key:
                kcanon = _canon_key_strip_prefix(str(key))
                p = precomputed_rollout_by_key.get(kcanon, None)
                if p is not None and p.exists():
                    try:
                        mask_np = np.load(p)
                    except Exception as e:
                        log.warning("[%s] Failed to load precomputed attn_rollout.npy (%s): %s", model_id, p, e)
                        mask_np = None  # fall back below
                else:
                    mask_np = None
            else:
                mask_np = None

            if mask_np is not None:
                mask_np = np.asarray(mask_np, dtype=np.float32)
            else:
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

run_comparision.py codice <<
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

