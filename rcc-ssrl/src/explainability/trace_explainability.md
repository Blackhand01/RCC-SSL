common/eval_utils.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities for explainability:
- logging and seeding
- basic image preprocessing
- evaluation artifacts loading (predictions + logits)
- selection of TP/FP/FN and low-confidence cases
- WebDataset loader with keys
"""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
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
            add_reason(i, "fn_high_conf")

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
# Data loaders
# -------------------------------------------------------------------------
def make_wds_loader_with_keys(
    test_dir: str,
    pattern: str,
    image_key: str,
    meta_key: str,
    preprocess_fn,
    num_workers: int,
):
    """Create a WebDataset loader that yields (image_tensor, meta, key)."""
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

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=min(num_workers, len(shards)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_first,
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

common/__init__.py codice <<
# empty on purpose – marks "common" as a package
>>

common/ssl_linear_loader.py codice <<
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

concept/config_concept.yaml codice <<
experiment:
  name: "xai_ssl_rcc_vit_concept"
  seed: 1337
  # sovrascritto dall'orchestratore per ogni ablation
  outputs_root: null

evaluation_inputs:
  # sovrascritto dall'orchestratore per ogni ablation
  eval_run_dir: null
  predictions_csv: "predictions.csv"
  logits_npy: "logits_test.npy"

data:
  backend: "webdataset"
  img_size: 224
  imagenet_norm: false
  num_workers: 4
  webdataset:
    train_dir: "/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/train"
    test_dir: "/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test"
    pattern: "shard-*.tar"
    image_key: "img.jpg;jpg;jpeg;png"
    meta_key: "meta.json;json"

labels:
  class_order: ["ccRCC", "pRCC", "CHROMO", "ONCO", "NOT_TUMOR"]

model:
  name: null                 # sovrascritto per ablation
  backbone_name: "vit_small_patch16_224"
  ssl_backbone_ckpt: null    # sovrascritto per ablation
  ssl_head_ckpt: null        # sovrascritto per ablation

selection:
  per_class:
    topk_tp: 5
    topk_fp: 3
    topk_fn: 3
  global_low_conf:
    topk: 3
  min_per_class: 10

concepts:
  # default, sovrascrivibile via env CONCEPT_BANK_CSV
  meta_csv: "/home/mla_group_01/rcc-ssrl/src/explainability/concept/ontology/concepts_rcc_debug.csv"

  # mappatura colonne concept bank
  concept_name_col: "concept_name"
  key_col: "wds_key"
  group_col: "group"
  class_col: "class_label"
  present_col: "present"
  confidence_col: "confidence"
  rationale_col: "rationale"

  similarity: "cosine"
  topk_per_patch: 5
  min_patches_per_concept: 5

runtime:
  device: "cuda"
>>

concept/__init__.py codice <<
# empty – marks "concept" as a package
>>

concept/ontology/build_concept_bank.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump grezzo delle risposte VLM per tutte le coppie (patch, concept).

Input:
- Ontology YAML con i concetti (name, group, primary_class, prompt)
- CSV di candidate patches: image_path, wds_key, class_label
- Modello HF locale (LLaVA-Med) via VLMClientHF

Output:
- concepts_rcc_*.csv con colonne:
    concept_name, wds_key, group, class_label, user_question, assistant_answer
  (tutte le risposte del VLM, senza filtri/soglie, con prompt ripulito)
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _normalize_whitespace(s: str) -> str:
    """Collassa spazi / newline multipli in un'unica riga pulita."""
    return " ".join(str(s).split())


# ----------------------------------------------------------------------
# Ontology loading + validation
# ----------------------------------------------------------------------
def load_ontology(path: str | Path) -> List[Dict[str, Any]]:
    """Load ontology YAML e valida i campi minimi per ciascun concept."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ontology YAML not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "concepts" not in data:
        raise ValueError(f"Ontology YAML must contain a 'concepts' list: {path}")

    concepts_raw = data["concepts"]
    if not isinstance(concepts_raw, list) or not concepts_raw:
        raise ValueError(f"'concepts' must be a non-empty list in {path}")

    concepts: List[Dict[str, Any]] = []
    for idx, c in enumerate(concepts_raw):
        if not isinstance(c, dict):
            raise ValueError(f"Concept #{idx} is not a dict in {path}")

        name = str(c.get("name", "")).strip()
        prompt = str(c.get("prompt", "")).strip()

        if not name:
            raise ValueError(f"Concept #{idx} in {path} has empty 'name'")
        if not prompt:
            raise ValueError(f"Concept '{name}' in {path} has empty 'prompt'")

        concepts.append(
            {
                "name": name,
                "prompt": prompt,
                "group": c.get("group"),
                "primary_class": c.get("primary_class"),
            }
        )

    return concepts


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Dump raw VLM outputs for RCC concept bank."
    )
    parser.add_argument("--ontology", required=True, help="Ontology YAML path")
    parser.add_argument(
        "--images-csv",
        required=True,
        help="CSV with columns: image_path,wds_key,class_label",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Eren-Senoglu/llava-med-v1.5-mistral-7b-hf",
        help=(
            "HF model id or local path for the VLM "
            "(e.g. 'Eren-Senoglu/llava-med-v1.5-mistral-7b-hf' or a local directory). "
            "If you launch via run_full_xai.sh, this is overridden by VLM_MODEL_PATH."
        ),
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path for concept bank (concepts_rcc_*.csv)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If > 0, limit the number of candidate patches processed (debug).",
    )
    args = parser.parse_args(argv)

    # Ontology
    concepts = load_ontology(args.ontology)

    # VLM client
    try:
        from explainability.concept.ontology.vlm_client_hf import VLMClientHF
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "HF backend requested but transformers/torch dependencies are missing. "
            "Install them in the same venv used for explainability."
        ) from exc

    vlm = VLMClientHF(args.model_name)

    # ------------------------------------------------------------------
    # Read candidate patches + validation
    # ------------------------------------------------------------------
    images_csv_path = Path(args.images_csv)
    if not images_csv_path.is_file():
        raise FileNotFoundError(f"Images CSV not found: {images_csv_path}")

    rows: List[Dict[str, str]] = []
    with images_csv_path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required_cols = {"image_path", "wds_key", "class_label"}
        missing = required_cols - set(fieldnames)
        if missing:
            raise ValueError(
                f"Images CSV {images_csv_path} is missing required columns: {sorted(missing)}"
            )

        for r in reader:
            image_path = (r.get("image_path") or "").strip()
            wds_key = (r.get("wds_key") or "").strip()
            class_label = (r.get("class_label") or "").strip()

            if not image_path or not wds_key or not class_label:
                continue

            rows.append(
                {
                    "image_path": image_path,
                    "wds_key": wds_key,
                    "class_label": class_label,
                }
            )

    if not rows:
        raise RuntimeError(
            f"Concept bank: no valid candidate patches in {images_csv_path}. "
            "Stage 0a (build_concept_candidates) probably failed or produced an empty CSV."
        )

    # Debug mode: limit number of patches (e.g. 100) to reduce queries
    if args.max_images and args.max_images > 0:
        rng = random.Random(1337)
        rng.shuffle(rows)
        rows = rows[: args.max_images]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    total_rows = len(rows)
    total_concepts = len(concepts)
    total_planned_queries = total_rows * total_concepts
    print(
        f"[INFO] Concept bank RAW: candidates={total_rows}, concepts={total_concepts}, "
        f"max_queries={total_planned_queries}, model_name={args.model_name}"
    )

    total_queries = 0
    written_rows = 0
    skipped_empty_answer = 0
    log_every = 200  # stampa ogni N query

    # ------------------------------------------------------------------
    # Write output CSV
    # ------------------------------------------------------------------
    with out_path.open("w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(
            [
                "concept_name",
                "wds_key",
                "group",
                "class_label",
                "user_question",
                "assistant_answer",
            ]
        )

        for r_idx, r in enumerate(rows):
            img_path = r["image_path"]
            key = r["wds_key"]
            patch_class = r.get("class_label", "")

            for c_idx, c in enumerate(concepts):
                concept_name = c["name"]
                concept_group = c.get("group")
                concept_prompt = c["prompt"]

                # user_question costruita in modo deterministico (non prendiamo il prompt echiato dal modello)
                user_question = _normalize_whitespace(
                    f"For this RCC patch, is the concept '{concept_name}' present? "
                    f"Definition: {concept_prompt}"
                )

                t0 = time.time()
                try:
                    ans = vlm.ask_concept(img_path, concept_name, concept_prompt)
                except RuntimeError as e:
                    total_queries += 1
                    dt = time.time() - t0
                    if getattr(vlm, "debug", False):
                        print(
                            f"[BANK DEBUG] RuntimeError for key={key}, concept={concept_name}, "
                            f"class={patch_class}, dt={dt:.2f}s\n{e}\n{'-'*80}"
                        )
                    if total_queries <= 10:
                        print(
                            f"[WARN] VLM error for key={key}, concept={concept_name}: {e}"
                        )
                    continue

                dt = time.time() - t0
                total_queries += 1

                if total_queries % log_every == 0:
                    elapsed = time.time() - t_start
                    avg_per_query = elapsed / max(1, total_queries)
                    remaining = total_planned_queries - total_queries
                    est_remain = remaining * avg_per_query
                    print(
                        f"[PROGRESS] queries={total_queries}/{total_planned_queries} "
                        f"({100.0*total_queries/total_planned_queries:.1f}%), "
                        f"elapsed={elapsed/60:.1f} min, "
                        f"avg_per_query={avg_per_query:.2f} s, "
                        f"ETA~{est_remain/60:.1f} min"
                    )

                # Risposta vuota o non nel formato atteso
                if not ans or not isinstance(ans, dict):
                    if total_queries <= 10:
                        print(
                            f"[DEBUG] Empty/invalid VLM answer (non-dict) for key={key}, "
                            f"concept={concept_name}"
                        )
                    continue

                # VLMClientHF ora restituisce: {"concept", "user_question", "assistant_answer"}
                assistant_answer = _normalize_whitespace(
                    ans.get("assistant_answer", "") or ""
                )
                user_question_full = _normalize_whitespace(
                    ans.get("user_question", user_question) or ""
                )

                if not assistant_answer:
                    skipped_empty_answer += 1
                    if skipped_empty_answer <= 10:
                        print(
                            f"[DEBUG] Skipping empty assistant_answer for key={key}, "
                            f"concept={concept_name}"
                        )
                    continue

                writer.writerow(
                    [
                        concept_name,
                        key,
                        concept_group,
                        patch_class,
                        user_question_full,
                        assistant_answer,
                    ]
                )
                written_rows += 1

    total_elapsed = time.time() - t_start
    print(
        f"[SUMMARY] Concept bank RAW dump: "
        f"candidates={total_rows}, concepts={total_concepts}, "
        f"queries={total_queries}, written_rows={written_rows}, "
        f"skipped_empty_answer={skipped_empty_answer}, "
        f"elapsed={total_elapsed/60:.1f} min"
    )

    if written_rows == 0:
        raise RuntimeError(
            f"No valid rows written to concept bank CSV {out_path}. "
            "Check VLM responses and ontology/prompts."
        )

    print(f"[OK] Concept bank written to {out_path}")


if __name__ == "__main__":
    main()
>>

concept/ontology/build_concept_candidates.py codice <<
#!/usr/bin/env python3
"""
Build RCC concept candidate CSV from a WebDataset split.

Given a train WebDataset (shard-*.tar) with entries like:
  <wds_key>.img.jpg
  <wds_key>.meta.json   (must contain class_label)

This script:
  - extracts up to N patches per class into PNGs under --images-root/<class_label>/
  - writes a CSV with columns: image_path, wds_key, class_label

No VLM calls happen here: this is Stage 0a (dataset-level candidates). Stage 0b
will consume the CSV to query the VLM and build the concept bank.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build RCC concept candidate CSV from TRAIN WebDataset."
    )
    p.add_argument(
        "--train-dir",
        required=True,
        type=Path,
        help="Root of train WebDataset split (folder with shard-*.tar).",
    )
    p.add_argument(
        "--pattern",
        default="shard-*.tar",
        type=str,
        help="Glob pattern for shards (default: shard-*.tar).",
    )
    p.add_argument(
        "--image-key",
        default="img.jpg;jpg;jpeg;png",
        type=str,
        help="WebDataset image key(s) (default: img.jpg;jpg;jpeg;png).",
    )
    p.add_argument(
        "--meta-key",
        default="meta.json;json",
        type=str,
        help="WebDataset metadata key(s) (default: meta.json;json).",
    )
    p.add_argument(
        "--out-csv",
        required=True,
        type=Path,
        help="Output CSV path (concept_candidates_rcc.csv).",
    )
    p.add_argument(
        "--images-root",
        required=True,
        type=Path,
        help="Root directory where PNG crops for VLM will be stored.",
    )
    p.add_argument(
        "--max-patches-per-class",
        type=int,
        default=2000,
        help="Maximum number of candidate patches per class_label (default: 2000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed (currently unused, kept for compatibility).",
    )
    return p.parse_args()


def _suffixes(raw: str) -> List[str]:
    """Return list of suffixes with leading dot, splitting on ';'."""
    out: List[str] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        out.append(part if part.startswith(".") else f".{part}")
    return out


def _find_meta_member(tf: tarfile.TarFile, base: str, meta_suffixes: Sequence[str]) -> tarfile.TarInfo | None:
    for suf in meta_suffixes:
        candidate = f"{base}{suf}"
        try:
            return tf.getmember(candidate)
        except KeyError:
            continue
    return None


def _iter_image_members(tf: tarfile.TarFile, image_suffixes: Sequence[str]) -> Iterable[tarfile.TarInfo]:
    for member in tf.getmembers():
        for suf in image_suffixes:
            if member.name.endswith(suf):
                yield member
                break


# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    shards = sorted(args.train_dir.glob(args.pattern))
    if not shards:
        raise FileNotFoundError(f"No shards found under {args.train_dir} matching {args.pattern}")

    image_suffixes = _suffixes(args.image_key)
    meta_suffixes = _suffixes(args.meta_key)

    args.images_root.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    per_class_counts: Dict[str, int] = {}
    written_rows = 0

    with args.out_csv.open("w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["image_path", "wds_key", "class_label"])

        for shard in shards:
            print(f"[INFO] Processing shard {shard.name}")
            with tarfile.open(shard) as tf:
                for img_member in _iter_image_members(tf, image_suffixes):
                    # strip the image suffix to get base key
                    for suf in image_suffixes:
                        if img_member.name.endswith(suf):
                            base = img_member.name[: -len(suf)]
                            break
                    else:
                        continue

                    meta_member = _find_meta_member(tf, base, meta_suffixes)
                    if meta_member is None:
                        continue

                    with tf.extractfile(meta_member) as mf:
                        if mf is None:
                            continue
                        try:
                            meta = json.load(mf)
                        except Exception:
                            continue

                    class_label = str(meta.get("class_label", "")).strip()
                    if not class_label:
                        continue

                    # enforce per-class cap
                    current = per_class_counts.get(class_label, 0)
                    if args.max_patches_per_class > 0 and current >= args.max_patches_per_class:
                        continue

                    # derive wds_key and output path
                    wds_key = base
                    out_dir = args.images_root / class_label
                    out_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"{wds_key.replace('/', '_')}.png"
                    out_img_path = out_dir / filename

                    with tf.extractfile(img_member) as imf:
                        if imf is None:
                            continue
                        try:
                            img = Image.open(io.BytesIO(imf.read())).convert("RGB")
                        except Exception:
                            continue

                    img.save(out_img_path, format="PNG")

                    writer.writerow([out_img_path.as_posix(), wds_key, class_label])
                    per_class_counts[class_label] = current + 1
                    written_rows += 1

    if written_rows == 0:
        raise RuntimeError(
            f"No candidate patches were written to {args.out_csv}. "
            "Check that shards contain the expected keys."
        )

    print(
        f"[OK] concept_candidates CSV written to {args.out_csv} "
        f"({written_rows} rows, classes={len(per_class_counts)})"
    )


if __name__ == "__main__":
    main()
>>

concept/ontology/__init__.py codice <<
# empty – marks "ontology" as a package
>>

concept/ontology/ontology_rcc_debug.yaml codice <<
version: 3
name: "rcc_histology_4_concepts_core"

concepts:
  - id: 1
    name: "Clear cytoplasm (ccRCC)"
    short_name: "clear_cytoplasm"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompt: >
      Viable renal tumour cells with abundant optically clear or glassy cytoplasm and sharp cell borders,
      usually forming nests or alveoli typical of clear cell renal cell carcinoma.
      Exclude adipocytes and stromal fat, artefactual perinuclear clearing, and sheets of foamy macrophages.
>>

concept/ontology/ontology_rcc_v1.yaml codice <<
version: 1
name: "rcc_histology_10_core_concepts"

concepts:
  - id: 1
    name: "Clear cytoplasm (ccRCC)"
    short_name: "clear_cytoplasm"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompt: >
      Identify viable renal tumour cells with abundant optically clear or glassy cytoplasm and sharp cell borders,
      usually forming nests or alveoli. Exclude adipocytes and stromal fat, benign tubules, foamy macrophages and
      artefactual clearing. If these specific tumour cells are not clearly present, treat the concept as absent.

  - id: 2
    name: "Delicate branching capillary network (ccRCC)"
    short_name: "delicate_capillary_network"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompt: >
      Identify a delicate, thin-walled, branching capillary network investing nests or alveoli of tumour cells,
      creating a fine chicken-wire vascular pattern typical of clear cell RCC. Exclude thick fibrous septa and large
      muscular vessels. If this fine capillary meshwork is not obvious, treat the concept as absent.

  - id: 3
    name: "Papillary fronds with fibrovascular cores (pRCC)"
    short_name: "papillary_fronds"
    group: "pRCC"
    primary_class: "pRCC"
    prompt: >
      Identify true papillary fronds: finger-like projections with central fibrovascular cores containing loose stroma
      and vessels, lined by one or more layers of tumour cells. Exclude folded flat epithelium, simple tubules and
      solid nests. If you only see flat or tubular epithelium without clear fibrovascular cores, treat the concept as absent.

  - id: 4
    name: "Foamy macrophages in papillary cores (pRCC)"
    short_name: "foamy_macrophages"
    group: "pRCC"
    primary_class: "pRCC"
    prompt: >
      Identify clusters or sheets of foamy macrophages with finely vacuolated cytoplasm and small dense nuclei within
      papillary fibrovascular cores or lumina. Do not call clear tumour cells, necrotic debris or artefactual vacuoles
      foamy macrophages. If macrophages are sparse or ambiguous, err on absent/equivocal.

  - id: 5
    name: "Perinuclear halos (chRCC)"
    short_name: "perinuclear_halos"
    group: "chRCC"
    primary_class: "CHROMO"
    prompt: >
      Identify solid sheets or nests of tumour cells with distinct perinuclear clearing or halos surrounding the nucleus
      within pale to eosinophilic cytoplasm, typical of chromophobe RCC. Exclude random vacuoles, degenerative change
      and mucin. If halos are not a consistent, repeated pattern, treat the concept as absent.

  - id: 6
    name: "Plant-cell-like borders (chRCC)"
    short_name: "plant_cell_borders"
    group: "chRCC"
    primary_class: "CHROMO"
    prompt: >
      Identify tumour cells with thick, sharply delineated polygonal cell borders creating a plant-cell or mosaic
      appearance in sheets of chromophobe RCC. Exclude indistinct thin borders of clear cell RCC or ordinary tubular
      epithelium. If borders are not clearly thick and polygonal, treat the concept as absent.

  - id: 7
    name: "Oncocytic cytoplasm (oncocytoma)"
    short_name: "oncocytic_cytoplasm"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompt: >
      Identify tumour cells with abundant, dense, finely granular, deeply eosinophilic cytoplasm (oncocytes) and round,
      centrally placed nuclei with smooth membranes, typical of renal oncocytoma. Exclude chromophobe-like cells with
      obvious perinuclear halos. If the cytoplasm is not densely granular and eosinophilic, treat the concept as absent.

  - id: 8
    name: "Archipelagenous architecture in oedematous stroma (oncocytoma)"
    short_name: "archipelagenous_architecture"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompt: >
      Identify round or oval nests and islands of oncocytic tumour cells scattered in loose, oedematous or myxoid stroma,
      producing an archipelagenous or archipelago-like pattern. Exclude true papillary projections and densely packed solid
      sheets without intervening stroma. If the islands in loose stroma are not clear, treat the concept as absent.

  - id: 9
    name: "Coagulative tumour necrosis"
    short_name: "coagulative_necrosis"
    group: "Necrosis"
    primary_class: null
    prompt: >
      Identify areas of coagulative tumour necrosis characterised by ghost outlines of tumour cells with loss of nuclei,
      increased eosinophilia and granular debris, usually sharply demarcated from viable tumour. Exclude simple haemorrhage,
      cyst contents, autolysis and cautery artefact. If the changes are subtle or purely haemorrhagic, treat the concept as absent.

  - id: 10
    name: "Sarcomatoid / rhabdoid dedifferentiation"
    short_name: "sarcomatoid_rhabdoid_dedifferentiation"
    group: "Dedifferentiation"
    primary_class: null
    prompt: >
      Identify high-grade dedifferentiation within renal cell carcinoma composed of malignant spindle cells in fascicles
      (sarcomatoid) and/or rhabdoid cells with eccentric nuclei, prominent nucleoli and dense eosinophilic cytoplasmic
      inclusions. Exclude benign stromal spindle cells and inflammatory infiltrates. If definite sarcomatoid or rhabdoid
      morphology is not obvious, treat the concept as absent.
>>

concept/ontology/ontology_rcc_v2.yaml codice <<
version: 2
name: "rcc_histology_18_concepts_v2"

concepts:
  - id: 1
    name: "Clear cytoplasm (ccRCC)"
    short_name: "clear_cytoplasm"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompt: "Identify viable renal tumour cells with abundant optically clear or glassy cytoplasm and sharp cell borders, usually forming nests or alveoli. Exclude adipocytes and stromal fat, artefactual perinuclear clearing, and foamy macrophages."

  - id: 2
    name: "Delicate branching capillary network (ccRCC)"
    short_name: "delicate_capillary_network"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompt: "Identify a delicate, thin-walled branching capillary network intimately investing nests or alveoli of tumour cells, creating a fine chicken-wire vascular pattern typical of clear cell RCC. Exclude thick fibrous septa, large muscular vessels, and non-tumour parenchymal vessels."

  - id: 3
    name: "Alveolar/nested architecture (ccRCC)"
    short_name: "alveolar_nested_architecture"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompt: "Identify alveolar, acinar, or nested arrangements of tumour cells separated by delicate vasculature or thin fibrous septa, with open lumina or sinusoid-like spaces between nests. Exclude papillary fronds with fibrovascular cores and flat tubules."

  - id: 4
    name: "Cytoplasmic vacuolization (ccRCC)"
    short_name: "cytoplasmic_vacuolization"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompt: "Identify viable tumour cells with multiple, sharply defined intracytoplasmic vacuoles within clear or eosinophilic cytoplasm, giving a bubbly appearance in the setting of clear cell RCC. Exclude foamy macrophages, mucin pools, and obvious fixation artefacts."

  - id: 5
    name: "Papillary fronds with fibrovascular cores (pRCC)"
    short_name: "papillary_fronds"
    group: "pRCC"
    primary_class: "pRCC"
    prompt: "Identify true papillary fronds: finger-like projections into spaces with central fibrovascular cores containing loose stroma and vessels, lined by one or more layers of tumour cells. Exclude folded flat epithelium, simple tubules, and solid nests."

  - id: 6
    name: "Foamy macrophages in papillary cores (pRCC)"
    short_name: "foamy_macrophages"
    group: "pRCC"
    primary_class: "pRCC"
    prompt: "Identify clusters or sheets of foamy macrophages with finely vacuolated cytoplasm and small dense nuclei located within papillary fibrovascular cores or lumina. Do not misinterpret clear tumour cells, necrotic debris, or artefactual vacuolation as foamy macrophages."

  - id: 7
    name: "Psammoma bodies (pRCC)"
    short_name: "psammoma_bodies"
    group: "pRCC"
    primary_class: "pRCC"
    prompt: "Identify round, concentrically laminated basophilic calcifications (psammoma bodies) within papillary cores or tumour stroma. Exclude coarse dystrophic calcification, bone formation, and foreign material."

  - id: 8
    name: "Hobnail nuclei / pseudostratification (pRCC)"
    short_name: "hobnail_nuclei"
    group: "pRCC"
    primary_class: "pRCC"
    prompt: "Identify hobnail tumour cells with apically protruding, hyperchromatic nuclei bulging into lumina, or crowded pseudostratified nuclei along papillary surfaces, often with nuclear atypia. Exclude evenly spaced single-layer cuboidal epithelium and benign reactive changes."

  - id: 9
    name: "Perinuclear halos (chRCC)"
    short_name: "perinuclear_halos"
    group: "chRCC"
    primary_class: "CHROMO"
    prompt: "Identify solid sheets or nests of tumour cells showing distinct perinuclear clearing or halos surrounding the nucleus within pale to eosinophilic cytoplasm, in keeping with chromophobe RCC. Exclude artefactual vacuoles, degenerative ballooning, and mucin."

  - id: 10
    name: "Raisinoid nuclei (chRCC)"
    short_name: "raisinoid_nuclei"
    group: "chRCC"
    primary_class: "CHROMO"
    prompt: "Identify tumour cells with irregular, wrinkled (raisinoid) nuclear membranes, hyperchromasia, and irregular nuclear contours typical of chromophobe RCC. Exclude smooth, round nuclei of oncocytoma and low-grade clear cell RCC."

  - id: 11
    name: "Plant-cell-like borders (chRCC)"
    short_name: "plant_cell_borders"
    group: "chRCC"
    primary_class: "CHROMO"
    prompt: "Identify tumour cells with thick, sharply delineated polygonal cell borders that create a plant-cell or mosaic appearance in sheets of chromophobe RCC. Exclude indistinct or very thin borders of clear cell RCC or ordinary tubular epithelium."

  - id: 12
    name: "Pale-to-eosinophilic granular cytoplasm (chRCC)"
    short_name: "pale_eosinophilic_cytoplasm"
    group: "chRCC"
    primary_class: "CHROMO"
    prompt: "Identify cytoplasm that is pale to lightly eosinophilic, finely granular or reticulated in chromophobe RCC cells, often combined with perinuclear halos and plant-cell-like borders. Exclude the uniformly dense, deeply eosinophilic granular cytoplasm of oncocytic tumours."

  - id: 13
    name: "Oncocytic cytoplasm (oncocytoma)"
    short_name: "oncocytic_cytoplasm"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompt: "Identify tumour cells with abundant, dense, finely granular, deeply eosinophilic cytoplasm (oncocytes) and round, centrally placed nuclei with smooth membranes, typically in renal oncocytoma. Exclude chromophobe-like cells with prominent perinuclear halos or reticulated cytoplasm."

  - id: 14
    name: "Archipelagenous architecture in oedematous stroma (oncocytoma)"
    short_name: "archipelagenous_architecture"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompt: "Identify round or oval nests and islands of oncocytic tumour cells scattered in loose, oedematous or myxoid stroma, producing an archipelagenous or archipelago-like pattern. Exclude true papillary projections and densely packed solid sheets without intervening stroma."

  - id: 15
    name: "Central fibrous scar / stellate fibrosis (oncocytoma)"
    short_name: "central_fibrous_scar"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompt: "Identify dense, hyalinised, stellate fibrous tissue within or near the centre of an oncocytic tumour, often with radiating fibrous bands and thick-walled vessels. Exclude peripheral fibrous capsule, nonspecific peritumoural fibrosis, and scar outside the tumour."

  - id: 16
    name: "ISUP nucleolar grade"
    short_name: "isup_nucleolar_grade"
    group: "Grading"
    primary_class: null
    prompt: "Assess nucleolar prominence in viable tumour cells according to ISUP criteria: grade 1 with inconspicuous or small nucleoli at high power, grade 2 with clearly visible nucleoli at 400x but not at 100x, grade 3 with prominent nucleoli visible at 100x, and grade 4 in the presence of extreme nuclear pleomorphism, tumour giant cells, sarcomatoid or rhabdoid morphology. Ignore crushed areas, necrosis, and non-tumour tissue."

  - id: 17
    name: "Coagulative tumour necrosis"
    short_name: "coagulative_necrosis"
    group: "Necrosis"
    primary_class: null
    prompt: "Identify areas of coagulative tumour necrosis characterised by ghost outlines of tumour cells with loss of nuclei, increased eosinophilia, and granular debris, usually sharply demarcated from viable tumour. Exclude simple haemorrhage, cyst contents, autolysis, and cautery artefact."

  - id: 18
    name: "Sarcomatoid / rhabdoid dedifferentiation"
    short_name: "sarcomatoid_rhabdoid_dedifferentiation"
    group: "Dedifferentiation"
    primary_class: null
    prompt: "Identify foci of high-grade dedifferentiation within renal cell carcinoma composed of malignant spindle cells in fascicles (sarcomatoid change) and/or rhabdoid cells with eccentric nuclei, prominent nucleoli, and dense eosinophilic cytoplasmic inclusions. Any definite sarcomatoid or rhabdoid component should be marked, and should not be confused with benign stromal spindle cells or inflammatory infiltrates."
>>

concept/ontology/vlm_client_hf.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client locale (no HTTP) per LLaVA-Med via Hugging Face.

Contratto:
- dato (image_path, concept_name, base_prompt) costruisce un prompt completo
  per il VLM (user_question) e ottiene una risposta testuale (assistant_answer).
- Ritorna un dict con:
    - concept
    - user_question  (prompt completo, lato "utente" - esattamente cio che va al modello)
    - assistant_answer (testo generato dal modello)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class VLMClientHF:
    """LLaVA-Med locale via Hugging Face (no controller/worker HTTP)."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: str = "float16",
        debug: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if dtype == "bfloat16":
            model_dtype = torch.bfloat16
        else:
            model_dtype = torch.float16

        # debug esplicito > env > default False
        if debug is None:
            self.debug = os.getenv("VLM_DEBUG", "0") == "1"
        else:
            self.debug = bool(debug)

        if self.debug:
            print(
                f"[VLM-HF DEBUG] Loading model '{self.model_name}' "
                f"on {self.device} dtype={model_dtype}"
            )

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                use_fast=True,
                trust_remote_code=True,
            )
        except ValueError as exc:
            raise RuntimeError(
                "AutoProcessor failed to load the HF checkpoint. "
                "If the model type is not recognized (e.g., llava_mistral), "
                "upgrade transformers or install with trust_remote_code support."
            ) from exc

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                dtype=model_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
        except ValueError as exc:
            raise RuntimeError(
                "AutoModelForImageTextToText failed to load the HF checkpoint. "
                "If you see 'model type llava_mistral not recognized', upgrade transformers "
                "or install from source, and ensure trust_remote_code is allowed."
            ) from exc
        self.model.eval()

        # Alcuni checkpoint HF (es. llava-med-v1.5-mistral-7b-hf) non popolano patch_size nel processor:
        # serve per espandere il token <image> in processing_llava.py. Recuperalo dalla vision_config se manca.
        if getattr(self.processor, "patch_size", None) is None:
            vision_cfg = getattr(self.model.config, "vision_config", None)
            patch_size = getattr(vision_cfg, "patch_size", None) if vision_cfg else None
            patch_size = patch_size or getattr(
                self.processor.image_processor, "patch_size", None
            )
            if patch_size is None:
                patch_size = 14  # fallback ragionevole per ViT-L/14
                if self.debug:
                    print(
                        "[VLM-HF DEBUG] patch_size non trovato nel processor; uso fallback 14"
                    )
            self.processor.patch_size = patch_size

    # ------------------------------------------------------------------
    # Prompt helper
    # ------------------------------------------------------------------
    def _build_prompt(self, concept_name: str, base_prompt: str) -> str:
        """
        Costruisce la user_question per classificare presenza/assenza del concetto.

        Se il processor supporta una chat_template, la usiamo; altrimenti
        cadiamo sul classico pattern <image> + USER/ASSISTANT.
        """
        system = (
            "You are a board-certified renal pathologist, acting as a classifier "
            "for histologic concepts in renal tumor histology images. "
            "Your task is to decide whether a specific histologic concept is present "
            "or absent in a single image patch. "
            "Respond with only one of the following options: 'Present', 'Absent', or 'Uncertain'."
        )

        user_instruction = (
            f"{system}\n\n"
            "Analyse the attached histology patch.\n"
            f"Target concept: {concept_name}\n"
            f"Definition of the concept:\n{base_prompt}\n\n"
            "Your task is to decide if this specific histologic concept is present "
            "or absent in this image patch derived from renal tumor tissue.\n"
            "Respond with only one of the following options: 'Present', 'Absent', or 'Uncertain'."
        )

        # Se esiste una chat template, usiamola (nuove versioni HF + Eren-Senoglu lo espongono)
        apply_chat_template = getattr(self.processor, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": user_instruction},
                        ],
                    }
                ]
                prompt = apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                if self.debug:
                    print(
                        "[VLM-HF DEBUG] apply_chat_template failed, falling back to "
                        f"manual prompt. Error: {exc}"
                    )
                prompt = f"<image>\nUSER: {user_instruction}\nASSISTANT:"
        else:
            # Fallback in stile LLaVA classico
            prompt = f"<image>\nUSER: {user_instruction}\nASSISTANT:"

        return prompt

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------
    def ask_concept(
        self,
        image_path: str | Path,
        concept_name: str,
        base_prompt: str,
        temperature: float = 0.2,
        max_new_tokens: int = 128,
    ) -> Optional[Dict[str, Any]]:
        """
        Esegue una singola query e ritorna:
        - user_question: prompt completo passato al modello
        - assistant_answer: testo grezzo generato dal modello
        """

        image_path = Path(image_path)
        if not image_path.is_file():
            if self.debug:
                print(f"[VLM-HF DEBUG] Image not found: {image_path}")
            return None

        image = Image.open(image_path).convert("RGB")
        user_question = self._build_prompt(concept_name, base_prompt)

        if self.debug:
            print(
                f"[VLM-HF DEBUG] >>> REQUEST\n"
                f"image={image_path}\n"
                f"concept={concept_name}\n"
                f"prompt_preview={user_question[:200]}...\n"
                f"{'-'*60}"
            )

        inputs = self.processor(
            text=[user_question],
            images=[image],
            return_tensors="pt",
        )

        if self.debug:
            pix = inputs.get("pixel_values", None)
            ids = inputs.get("input_ids", None)
            print(
                "[VLM-HF DEBUG] processor outputs: "
                f"pixel_values_shape={None if pix is None else tuple(pix.shape)}, "
                f"input_ids_shape={None if ids is None else tuple(ids.shape)}"
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generate_kwargs: Dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
            }
            if temperature and temperature > 0:
                generate_kwargs.update(
                    dict(
                        do_sample=True,
                        temperature=float(temperature),
                        top_p=0.9,
                    )
                )

            output_ids = self.model.generate(**inputs, **generate_kwargs)

        decoded = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )
        assistant_answer = decoded[0].strip() if decoded else "Error: no answer from VLM received"

        if self.debug:
            print(
                f"[VLM-HF DEBUG] <<< RAW COMPLETION for concept={concept_name}, "
                f"image={image_path}\n"
                f"{assistant_answer}\n{'='*80}"
            )

        return {
            "concept": concept_name,
            "user_question": user_question,
            "assistant_answer": assistant_answer,
        }


if __name__ == "__main__":
    # debug rapido
    print(json.dumps({"status": "ok"}))
>>

concept/xai_concept.py codice <<
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
>>

concept/xai_concept.sbatch codice <<
#!/usr/bin/env bash
#SBATCH -J xai_concept_vit
#SBATCH -o ${LOG_DIR}/xai_concept.%j.out
#SBATCH -e ${LOG_DIR}/xai_concept.%j.err
#SBATCH -p gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-/home/mla_group_01/rcc-ssrl/src/explainability/concept/config_concept.yaml}"
WORKDIR="/home/mla_group_01/rcc-ssrl/src/explainability/concept"

echo "[INFO] Host: $(hostname)"
echo "[INFO] CONFIG_PATH=${CONFIG_PATH}"
echo "[INFO] WORKDIR=${WORKDIR}"

module purge || true

if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

cd "$WORKDIR"
srun python3 xai_concept.py --config "$CONFIG_PATH"
>>

__init__.py codice <<
# empty – marks "explainability" as a package
>>

run_all_explainability.sbatch codice <<
#!/usr/bin/env bash
#SBATCH -J xai_all
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/xai/run_all.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/xai/run_all.%j.err
#SBATCH -p gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=compute-5-14,compute-5-11,compute-3-12

# This job wraps the run_full_xai.sh orchestrator and runs it inside an allocation.

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] xai_all wrapper: delegating configuration to run_full_xai.sh"

# Orchestrator reale (esiste già)
SCRIPT="/home/mla_group_01/rcc-ssrl/src/explainability/run_full_xai.sh"

if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERROR] Orchestrator script not found: $SCRIPT" >&2
  exit 1
fi

chmod +x "$SCRIPT"

# srun runs the job step inside the allocation provided by sbatch.
SRUN_ARGS=( srun --ntasks=1 )

echo "[INFO] Command: ${SRUN_ARGS[*]} $SCRIPT"
"${SRUN_ARGS[@]}" "$SCRIPT"

echo "[OK] xai_all job completed (run_full_xai.sh exited with code $?)"
>>

run_explainability.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main orchestrator to run spatial and concept explainability.

- EXP_ROOT: path a una cartella esperimento tipo:
    .../outputs/mlruns/exp_20251109_181551_ibot-v1

- All'interno, ablation folder tipo:
    exp_ibot_abl01, exp_ibot_abl02, ...

- MODEL_NAME (argomento --model-name):
    base SSL backbone name, es. "moco_v3", "dino_v3", "ibot", "i_jepa"

- Da MODEL_NAME deriviamo:
    HEAD_MODEL_NAME = MODEL_NAME + "_ssl_linear_best"

  che viene usato per:
    - cartella eval:
        {ablation}/eval/{HEAD_MODEL_NAME}/{timestamp}/
    - nome del modello nei config XAI:
        cfg["model"]["name"] = HEAD_MODEL_NAME
    - cartella XAI:
        {ablation}/xai/{HEAD_MODEL_NAME}/{timestamp}/
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

log = logging.getLogger("run_explainability")


def run_sbatch(batch_file: Path, config_path: Path, log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    os.environ["CONFIG_PATH"] = str(config_path)
    os.environ["LOG_DIR"] = log_dir
    log.info(
        f"[SBATCH] submitting {batch_file} "
        f"with CONFIG_PATH={config_path}, LOG_DIR={log_dir}"
    )
    res = subprocess.run(
        ["sbatch", str(batch_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    job_line = (res.stdout or "").strip()
    if job_line:
        log.info(f"[SBATCH] submission output: {job_line}")


def _find_latest_eval_run(ablation_dir: Path, eval_model_name: str) -> Optional[Path]:
    """
    Cerca l'ultimo run di eval sotto:
      {ablation}/eval/{eval_model_name}/{timestamp}/
    es: eval/ibot_ssl_linear_best/20251113_113536
    """
    base = ablation_dir / "eval" / eval_model_name
    if not base.is_dir():
        log.warning(f"[SKIP] Eval directory not found for {ablation_dir.name}: {base}")
        return None

    run_dirs = [d for d in base.iterdir() if d.is_dir()]
    if not run_dirs:
        log.warning(f"[SKIP] No eval runs found under {base}")
        return None

    latest = sorted(run_dirs)[-1]
    log.info(f"[EVAL] Using eval run {latest} for ablation {ablation_dir.name}")
    return latest


def _check_checkpoint(path: Path, label: str, ablation_name: str) -> bool:
    if path.is_file():
        log.info(f"[CKPT] {label} FOUND for {ablation_name}: {path}")
        return True
    log.warning(f"[CKPT] {label} MISSING for {ablation_name}: {path}")
    return False


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run explainability pipeline.")
    parser.add_argument(
        "--experiment-root",
        required=True,
        type=Path,
        help="Path to root dir with ablations folders (exp_*).",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        type=str,
        help="Base SSL backbone name (es. moco_v3, dino_v3, ibot, i_jepa).",
    )
    parser.add_argument(
        "--spatial-config-template",
        required=True,
        type=Path,
        help="Path to spatial/config_xai.yaml template",
    )
    parser.add_argument(
        "--concept-config-template",
        required=True,
        type=Path,
        help="Path to concept/config_concept.yaml template",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    if not args.experiment_root.is_dir():
        raise FileNotFoundError(f"Experiment root not found: {args.experiment_root}")

    ablation_folders = sorted(
        [
            p
            for p in args.experiment_root.iterdir()
            if p.is_dir() and p.name.startswith("exp_")
        ]
    )

    if not ablation_folders:
        log.error(f"No ablation folders found under {args.experiment_root}")
        return

    base_model_name = args.model_name
    head_model_name = f"{base_model_name}_ssl_linear_best"

    log.info(
        f"[INFO] BASE_MODEL_NAME={base_model_name}  HEAD_MODEL_NAME={head_model_name}"
    )
    log.info(f"[INFO] Found {len(ablation_folders)} ablations.")

    # Root log dir: <repo>/src/logs/xai/{MODEL_NAME}/...
    log_root = Path(__file__).resolve().parents[1] / "logs" / "xai"

    for ablation in ablation_folders:
        log.info("=" * 80)
        log.info(f"[ABLATION] Processing: {ablation}")

        eval_run = _find_latest_eval_run(ablation, head_model_name)
        if eval_run is None:
            log.warning(f"[ABLATION] Skipping {ablation.name}: no eval run available.")
            continue

        # checkpoint names: {MODEL_NAME}__ssl_best.pt, {MODEL_NAME}__ssl_linear_best.pt
        backbone_ckpt = ablation / "checkpoints" / f"{base_model_name}__ssl_best.pt"
        head_ckpt = ablation / "checkpoints" / f"{base_model_name}__ssl_linear_best.pt"

        has_backbone = _check_checkpoint(
            backbone_ckpt, "ssl_backbone_ckpt", ablation.name
        )
        has_head = _check_checkpoint(head_ckpt, "ssl_head_ckpt", ablation.name)

        if not (has_backbone and has_head):
            log.warning(
                f"[ABLATION] Skipping {ablation.name}: missing required checkpoints."
            )
            continue

        # ---------------------- SPATIAL CONFIG ----------------------
        spatial_config = yaml.safe_load(open(args.spatial_config_template))
        # outputs_root: {ablation}/xai  (NON più 06_xai/)
        spatial_config["experiment"]["outputs_root"] = str(ablation / "xai")
        spatial_config["model"]["name"] = head_model_name
        spatial_config["model"]["ssl_backbone_ckpt"] = str(backbone_ckpt)
        spatial_config["model"]["ssl_head_ckpt"] = str(head_ckpt)
        spatial_config["evaluation_inputs"]["eval_run_dir"] = str(eval_run)

        spatial_config_path = ablation / "xai" / "config_xai.yaml"
        spatial_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(spatial_config_path, "w") as f:
            yaml.dump(spatial_config, f)
        log.info(f"[CFG] Spatial config written to {spatial_config_path}")

        # ---------------------- CONCEPT CONFIG ----------------------
        concept_config = yaml.safe_load(open(args.concept_config_template))
        concept_config["experiment"]["outputs_root"] = str(ablation / "xai")
        concept_config["model"]["name"] = head_model_name
        concept_config["model"]["ssl_backbone_ckpt"] = str(backbone_ckpt)
        concept_config["model"]["ssl_head_ckpt"] = str(head_ckpt)
        concept_config["evaluation_inputs"]["eval_run_dir"] = str(eval_run)

        # Override del concept bank via env CONCEPT_BANK_CSV se presente
        concept_bank_csv_env = os.environ.get("CONCEPT_BANK_CSV")
        if concept_bank_csv_env:
            concept_config.setdefault("concepts", {})
            concept_config["concepts"]["meta_csv"] = concept_bank_csv_env
            log.info(
                f"[CFG] Overriding concepts.meta_csv with CONCEPT_BANK_CSV={concept_bank_csv_env}"
            )

        concept_config_path = ablation / "xai" / "config_concept.yaml"
        with open(concept_config_path, "w") as f:
            yaml.dump(concept_config, f)
        log.info(f"[CFG] Concept config written to {concept_config_path}")

        # Log dir per questa ablation:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = str(log_root / base_model_name / ablation.name / datetime_str)

        spatial_sbatch = (
            Path(__file__).parent / "spatial" / "xai_generate.sbatch"
        )
        concept_sbatch = Path(__file__).parent / "concept" / "xai_concept.sbatch"

        log.info(f"[LAUNCH] Spatial XAI sbatch for {ablation.name}")
        run_sbatch(spatial_sbatch, spatial_config_path, log_dir)

        log.info(f"[LAUNCH] Concept XAI sbatch for {ablation.name}")
        run_sbatch(concept_sbatch, concept_config_path, log_dir)

    log.info("[DONE] run_explainability completed.")
    log.info("       Check Slurm logs under logs/xai/{MODEL_NAME}/...")


if __name__ == "__main__":
    main()
>>

run_full_xai.sh codice <<
#!/usr/bin/env bash
# Orchestrate full explainability pipeline:
# - Stage 0: global concept bank (dataset-level, only if missing)
# - Stage 1/2: spatial + concept XAI for all ablations in an experiment
#
# VLM: ONLY local HF model via VLMClientHF (no HTTP server).

set -euo pipefail

# -----------------------------------------------------------------------------
# GLOBAL CONFIG (single source of truth for explainability)
# -----------------------------------------------------------------------------
REPO_ROOT="/home/mla_group_01/rcc-ssrl"
SRC_DIR="${REPO_ROOT}/src"
# --- Experiment-level config (EDIT HERE) ---
EXP_ROOT="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3"
MODEL_NAME="moco_v3"                    # base SSL backbone name (non *_ssl_linear_best)
BACKBONE_NAME="vit_small_patch16_224"

# Python environment (usato per Stage 0 + Stage 1/2)
VENV_PATH="/home/mla_group_01/rcc-ssrl/.venvs/xai"

# Dataset-level config (concept bank)
TRAIN_WDS_DIR="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/train"
TRAIN_WDS_PATTERN="shard-*.tar"
CANDIDATES_CSV="${SRC_DIR}/explainability/concept/ontology/concept_candidates_rcc.csv"
CANDIDATES_IMG_ROOT="${SRC_DIR}/explainability/concept/ontology/concept_candidates_images"

# Ontologia + concept bank usati per Stage 0 e Stage 2 
VERS="debug"  # ontology + concept bank version (EDIT HERE)
ONTOLOGY_YAML="${SRC_DIR}/explainability/concept/ontology/ontology_rcc_${VERS}.yaml"
CONCEPT_BANK_CSV="${SRC_DIR}/explainability/concept/ontology/concepts_rcc_${VERS}.csv"

# VLM (LLaVA-Med) – HF ONLY
VLM_MODEL_PATH="Eren-Senoglu/llava-med-v1.5-mistral-7b-hf"

# Flags opzionali (per futura estensione; al momento solo log)
ONLY_SPATIAL="${ONLY_SPATIAL:-0}"
ONLY_CONCEPT="${ONLY_CONCEPT:-0}"

# Export env necessari downstream (sbatch XAI + run_explainability.py)
export VENV_PATH
export CONCEPT_BANK_CSV

# -----------------------------------------------------------------------------
# LOGGING: tutti i log per modello vanno sotto logs/xai/${MODEL_NAME}
# -----------------------------------------------------------------------------
LOG_ROOT="${SRC_DIR}/logs/xai"
LOG_DIR="${LOG_ROOT}/${MODEL_NAME}"
mkdir -p "${LOG_DIR}"

LOG_SUFFIX="${SLURM_JOB_ID:-local_$$}"

echo "[INFO] run_full_xai.sh starting"
echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] EXP_ROOT=${EXP_ROOT}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] BACKBONE_NAME=${BACKBONE_NAME}"
echo "[INFO] VLM_MODEL_PATH=${VLM_MODEL_PATH}"
echo "[INFO] VENV_PATH=${VENV_PATH}"
echo "[INFO] ONTOLOGY_YAML=${ONTOLOGY_YAML}"
echo "[INFO] CONCEPT_BANK_CSV=${CONCEPT_BANK_CSV}"
echo "[INFO] TRAIN_WDS_DIR=${TRAIN_WDS_DIR}"
echo "[INFO] TRAIN_WDS_PATTERN=${TRAIN_WDS_PATTERN}"

# ------------------- env -------------------

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

if [[ -n "${VENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

# ------------------- STAGE 0: concept bank (solo se NON esiste o è vuota) -------------------

# Conta le righe se il file esiste
if [[ -f "${CONCEPT_BANK_CSV}" ]]; then
  num_lines=$(wc -l < "${CONCEPT_BANK_CSV}")
else
  num_lines=0
fi

if [[ "${num_lines}" -le 1 ]]; then
  echo "[WARN] Concept bank missing or empty (lines=${num_lines}); rebuilding Stage 0."

  # prima di lanciare Python, controlla che esistano i tar attesi
  if ! compgen -G "${TRAIN_WDS_DIR}/${TRAIN_WDS_PATTERN}" > /dev/null; then
    echo "[ERROR] No shards found under ${TRAIN_WDS_DIR}/${TRAIN_WDS_PATTERN}" >&2
    echo "[ERROR] - Controlla che TRAIN_WDS_DIR punti alla cartella corretta" >&2
    echo "[ERROR] - Controlla che il pattern TRAIN_WDS_PATTERN (default shard-*.tar) sia corretto" >&2
    exit 1
  fi

  # 0a) concept_candidates_rcc.csv (train WDS -> PNG + CSV)
  echo "[INFO] Stage 0a: building concept_candidates_rcc.csv"
  python3 -m explainability.concept.ontology.build_concept_candidates \
    --train-dir "${TRAIN_WDS_DIR}" \
    --pattern "${TRAIN_WDS_PATTERN}" \
    --image-key "img.jpg;jpg;jpeg;png" \
    --meta-key "meta.json;json" \
    --out-csv "${CANDIDATES_CSV}" \
    --images-root "${CANDIDATES_IMG_ROOT}"

  # 0b) concepts_rcc_*.csv (VLM HF su candidates)
  echo "[INFO] Stage 0b: building concept bank via local HF VLM"
  export VLM_DEBUG="${VLM_DEBUG:-1}"

  python3 -m explainability.concept.ontology.build_concept_bank \
    --ontology "${ONTOLOGY_YAML}" \
    --images-csv "${CANDIDATES_CSV}" \
    --model-name "${VLM_MODEL_PATH}" \
    --out-csv "${CONCEPT_BANK_CSV}" \
    --max-images 0

  # hard check: concept bank deve avere almeno header + 1 riga
  lines_after=$(wc -l < "${CONCEPT_BANK_CSV}")
  if [[ "${lines_after}" -le 1 ]]; then
    echo "[ERROR] Concept bank ${CONCEPT_BANK_CSV} still empty after Stage 0 (lines=${lines_after}). Aborting." >&2
    exit 1
  fi
else
  echo "[INFO] Concept bank found at ${CONCEPT_BANK_CSV} with ${num_lines} lines – skipping Stage 0."
fi


# ------------------- STAGE 1/2: spatial + concept XAI per esperimento -------------------

ORCH_CMD=( python3 -m explainability.run_explainability
  --experiment-root "${EXP_ROOT}"
  --model-name "${MODEL_NAME}"
  --spatial-config-template "${SRC_DIR}/explainability/spatial/config_xai.yaml"
  --concept-config-template "${SRC_DIR}/explainability/concept/config_concept.yaml"
)

echo "[INFO] Stage 1/2: running experiment-level explainability:"
echo "[INFO]   ${ORCH_CMD[*]}"
"${ORCH_CMD[@]}"

echo "[OK] run_full_xai.sh completed."
>>

spatial/config_xai.yaml codice <<
experiment:
  name: "xai_ssl_rcc_vit_spatial"
  seed: 1337
  # Overridden dall’orchestratore per ablation
  outputs_root: null

evaluation_inputs:
  # Overridden dall’orchestratore per ablation
  eval_run_dir: null
  predictions_csv: "predictions.csv"
  logits_npy: "logits_test.npy"

data:
  backend: "webdataset"
  img_size: 224
  imagenet_norm: false
  num_workers: 4
  batch_size: 1
  webdataset:
    test_dir: "/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test"
    pattern: "shard-*.tar"
    image_key: "img.jpg;jpg;jpeg;png"
    meta_key: "meta.json;json"

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

spatial/__init__.py codice <<
# empty – marks "spatial" as a package
>>

spatial/xai_generate.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial XAI on RCC test set (TP/FP/FN selection via predictions.csv).

Generates:
- GradCAM / IG / Occlusion (if enabled and dependencies are available)
- Attention Rollout for ViT (via monkey patching timm Attention blocks).

This script is config-driven and can be:
- run standalone: python xai_generate.py --config CONFIG_PATH
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

    logger = setup_logger("xai_spatial")

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

    out_root = (
        Path(cfg["experiment"]["outputs_root"])
        / cfg["model"]["name"]
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
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
                    over = overlay_heatmap(pil_in, mask, alpha=0.6)
                    path = out_dir / "attn_rollout.png"
                    over.save(path)
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

spatial/xai_generate.sbatch codice <<
#!/usr/bin/env bash
#SBATCH -J xai_spatial_vit
#SBATCH -o ${LOG_DIR}/xai_spatial.%j.out
#SBATCH -e ${LOG_DIR}/xai_spatial.%j.err
#SBATCH -p gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-/home/mla_group_01/rcc-ssrl/src/explainability/spatial/config_xai.yaml}"
WORKDIR="/home/mla_group_01/rcc-ssrl/src/explainability/spatial"

echo "[INFO] Host: $(hostname)"
echo "[INFO] CONFIG_PATH=${CONFIG_PATH}"
echo "[INFO] WORKDIR=${WORKDIR}"

module purge || true

# Attiva eventuale venv (passata via env VENV_PATH, propagata da run_full_xai)
if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

cd "$WORKDIR"
srun python3 xai_generate.py --config "$CONFIG_PATH"
>>

