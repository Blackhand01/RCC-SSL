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
  concept_name_col: "concept_name"
  key_col: "wds_key"
  group_col: "group"
  class_col: "class_label"

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
Build concept bank for RCC histology using a VLM.

Input:
- Ontology YAML with RCC concepts.
- CSV of candidate patches with columns:
    image_path, wds_key, class_label
  (produced automatically by build_concept_candidates.py)

- VLM server (e.g. LLaVA-Med) answering concept-level questions in JSON.

Output:
- concepts_rcc_debug.csv with columns:
    concept_name, wds_key, group, class_label

This file is pointed to by concepts.meta_csv in config_concept.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from explainability.concept.ontology.vlm_client import VLMClient  # backend HTTP esistente
from explainability.concept.ontology.vlm_client_hf import VLMClientHF  # nuovo backend HF locale


def load_ontology(path: str | Path) -> List[Dict[str, Any]]:
    data = yaml.safe_load(open(path, "r"))
    return data["concepts"]


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build RCC concept bank via VLM.")
    parser.add_argument("--ontology", required=True, help="Ontology YAML path")
    parser.add_argument(
        "--images-csv",
        required=True,
        help="CSV with columns: image_path,wds_key,class_label",
    )
    parser.add_argument(
        "--controller",
        required=False,
        help="VLM controller URL (e.g. http://localhost:10000) – usato SOLO se backend=http",
    )
    parser.add_argument(
        "--model-name",
        default="Eren-Senoglu/llava-med-v1.5-mistral-7b-hf",
        help="Nome del modello VLM. "
             "Se backend=hf: id Hugging Face (es. Eren-Senoglu/llava-med-v1.5-mistral-7b-hf); "
             "se backend=http: nome registrato sul server (es. llava-med-v1.5-mistral-7b).",
    )
    parser.add_argument(
        "--backend",
        choices=["http", "hf"],
        default="hf",
        help="Tipo di backend VLM: 'hf' = modello locale via Hugging Face (no HTTP), "
             "'http' = controller/worker HTTP (pipeline vecchia).",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path for concept bank (concepts_rcc_debug.csv)",
    )
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=0.6,
        help="Minimal confidence to accept concept as present",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If > 0, limit the number of candidate patches processed (debug).",
    )
    args = parser.parse_args(argv)

    concepts = load_ontology(args.ontology)

    # Scegli il backend in base al flag --backend
    if args.backend == "hf":
        # modello locale via HuggingFace, NESSUN controller HTTP
        vlm = VLMClientHF(
            model_name=args.model_name,
            device=None,        # auto: cuda se disponibile, altrimenti cpu
            dtype="float16",    # va bene con A40
            debug=False,        # o True se vuoi log verbosi
        )
    else:
        # backend http vecchio: richiede --controller
        if not args.controller:
            raise RuntimeError(
                "HTTP backend selected but --controller is None. "
                "Pass --controller http://host:port oppure usa --backend hf."
            )
        vlm = VLMClient(args.controller, args.model_name)

    # Read candidate patches
    rows: List[Dict[str, str]] = []
    with open(args.images_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("image_path") or not r.get("wds_key"):
                continue
            rows.append(r)

    if not rows:
        raise RuntimeError(
            f"Concept bank: no candidate patches in {args.images_csv}. "
            "Stage 0a (build_concept_candidates) probably failed or produced an empty CSV."
        )

    # Debug mode: limit number of patches (e.g. 100) to reduce queries
    if args.max_images and args.max_images > 0:
        rng = random.Random(1337)
        rng.shuffle(rows)
        rows = rows[: args.max_images]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_f = open(out_path, "w", newline="")
    writer = csv.writer(out_f)
    writer.writerow(["concept_name", "wds_key", "group", "class_label"])

    t_start = time.time()
    total_rows = len(rows)
    total_concepts = len(concepts)
    total_planned_queries = total_rows * total_concepts
    print(
        f"[INFO] Concept bank: candidates={total_rows}, concepts={total_concepts}, "
        f"max_queries={total_planned_queries}, presence_threshold={args.presence_threshold}"
    )

    accepted = 0
    total_queries = 0
    log_every = 200  # stampa ogni N query

    for r_idx, r in enumerate(rows):
        img = r["image_path"]
        key = r["wds_key"]
        cls = r.get("class_label", "")

        for c_idx, c in enumerate(concepts):
            cname = c["name"]
            group = c.get("group")
            base_prompt = c["prompt"]

            t0 = time.time()
            try:
                ans = vlm.ask_concept(img, cname, base_prompt)
            except RuntimeError as e:
                # Tipicamente: error_code != 0 dal worker (es. problemi interni llava).
                total_queries += 1
                dt = time.time() - t0

                if vlm.debug:
                    print(
                        f"[BANK DEBUG] RuntimeError for key={key}, concept={cname}, "
                        f"class={cls}, dt={dt:.2f}s\n{e}\n{'-'*80}"
                    )

                # Log minimale anche fuori da debug per i primi casi
                if total_queries <= 10:
                    print(
                        f"[WARN] VLM error for key={key}, concept={cname}: {e}"
                    )
                continue

            dt = time.time() - t0
            total_queries += 1

            # LOG DI DEBUG: prime N risposte parse-ate, anche se poi vengono scartate
            if vlm.debug and ans is not None and total_queries <= 20:
                print(
                    f"[BANK DEBUG] key={key}, concept={cname}, "
                    f"class={cls}, dt={dt:.2f}s\n"
                    f"{json.dumps(ans, indent=2)}\n"
                    f"{'-'*80}"
                )

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

            if not ans:
                # debug minimale
                if total_queries <= 10:
                    print(
                        f"[DEBUG] Empty/invalid VLM answer for key={key}, "
                        f"concept={cname}"
                    )
                continue

            present = bool(ans.get("present", False))
            try:
                confidence = float(ans.get("confidence", 0.0))
            except Exception:
                confidence = 0.0

            if present and confidence >= args.presence_threshold:
                writer.writerow([cname, key, group, cls])
                accepted += 1

    out_f.close()
    total_elapsed = time.time() - t_start
    print(
        f"[SUMMARY] Concept bank: accepted={accepted}, "
        f"queries={total_queries}, "
        f"elapsed={total_elapsed/60:.1f} min"
    )

    if accepted == 0:
        raise RuntimeError(
            "Concept bank is empty (no accepted concept/key pairs). "
            "Cause probabili: modello VLM non raggiungibile / non caricato, "
            "risposte non parse-abili come JSON, oppure tutte le decisioni present=False "
            f"(backend={args.backend}, model_name={args.model_name})."
        )


if __name__ == "__main__":
    main()
>>

concept/ontology/build_concept_candidates.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build concept candidate CSV for RCC concepts directly from TRAIN WebDataset.

Dataset-level (project-level), NOT experiment-level.

Input:
- train_dir: root of WebDataset train split
- pattern: glob for shards (e.g. 'shard-*.tar')
- image_key: e.g. 'img.jpg;jpg;jpeg;png'
- meta_key: e.g. 'meta.json;json'

Output:
- concept_candidates_rcc.csv with columns:
    image_path, wds_key, class_label

Additionally exports PNG crops to an images_root directory so the VLM
can read them from filesystem.

NOTE:
- This is STAGE 0, dataset-level, independent from a specific experiment.
- run_full_xai.sh will call this with default paths for the RCC project.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import webdataset as wds
from PIL import Image

from explainability.common.eval_utils import setup_logger, set_seed


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
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
        default=20, # 2000
        help="Maximum number of candidate patches per class_label.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed (used mostly for shuffle).",
    )
    return p.parse_args(argv)


def _parse_meta(meta_raw) -> Dict:
    if isinstance(meta_raw, dict):
        return meta_raw
    if isinstance(meta_raw, (bytes, bytearray)):
        try:
            return json.loads(meta_raw.decode("utf-8"))
        except Exception:
            return {}
    if isinstance(meta_raw, str):
        try:
            return json.loads(meta_raw)
        except Exception:
            return {}
    return {}


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    log = setup_logger("build_concept_candidates_train")
    set_seed(args.seed)

    shard_glob = str(args.train_dir / args.pattern)
    log.info(f"Reading train shards from: {shard_glob}")

    shards = list(Path(args.train_dir).glob(args.pattern))
    if not shards:
        raise FileNotFoundError(f"No shards found matching {shard_glob}")

    log.info(f"Found {len(shards)} shards.")

    ds = (
        wds.WebDataset(
            [str(s) for s in shards],
            shardshuffle=True,
            handler=wds.warn_and_continue,
        )
        .shuffle(10000)
        .decode("pil")
        .to_tuple(args.image_key, args.meta_key, "__key__", handler=wds.warn_and_continue)
    )

    args.images_root.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    import time
    t_start = time.time()
    log_every = 200  # stampa log ogni N esempi

    total_seen = 0
    rows: List[Dict[str, str]] = []
    class_counts: Dict[str, int] = {}

    for img, meta_raw, key in ds:
        total_seen += 1
        if total_seen % log_every == 0:
            elapsed = time.time() - t_start
            eps = elapsed / max(1, total_seen)
            msg = (
                f"[PROGRESS] seen={total_seen} examples, "
                f"class_counts={class_counts}, "
                f"elapsed={elapsed/60:.1f} min, "
                f"~{eps:.3f} s/example"
            )
            log.info(msg)

        meta = _parse_meta(meta_raw)
        cls = str(meta.get("class_label", "")).strip()
        if not cls:
            continue

        cnt = class_counts.get(cls, 0)
        if cnt >= args.max_patches_per_class:
            continue  # already enough for this class

        safe_key = key.replace("/", "_")
        class_dir = args.images_root / cls
        class_dir.mkdir(parents=True, exist_ok=True)

        out_img_path = class_dir / f"{safe_key}.png"

        if isinstance(img, Image.Image):
            pil_img = img.convert("RGB")
        else:
            pil_img = Image.fromarray(img)
        pil_img.save(out_img_path)

        rows.append(
            {
                "image_path": str(out_img_path),
                "wds_key": key,
                "class_label": cls,
            }
        )
        class_counts[cls] = cnt + 1

    if not rows:
        log.warning(
            "No candidate patches collected; nothing to write "
            f"(total_seen={total_seen}, class_counts={class_counts})."
        )
        return

    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "wds_key", "class_label"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    log.info(
        f"Wrote {len(rows)} candidate rows to {args.out_csv} "
        f"(per-class counts: {class_counts})"
    )

    total_elapsed = time.time() - t_start
    log.info(
        f"[SUMMARY] concept_candidates_rcc: rows={len(rows)}, "
        f"classes={list(class_counts.keys())}, "
        f"total_elapsed={total_elapsed/60:.1f} min"
    )


if __name__ == "__main__":
    main()
>>

concept/ontology/debug_llava.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to test VLM client HTTP calls directly.
Run this locally with one image to verify the worker is responding correctly.
"""

import base64
import json
import os
from pathlib import Path

import requests

CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://localhost:10000")
MODEL_NAME = "llava-med-v1.5-mistral-7b"

def to_b64(image_path: Path) -> str:
    """Convert image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def test_llava_call(image_path: Path, concept_name: str, base_prompt: str):
    """Test a single VLM call and print raw response."""
    image_b64 = to_b64(image_path)

    prompt = f"""<image>
You are a board-certified renal pathologist.

Analyse the attached histology patch.

Concept: {concept_name}
Question: {base_prompt}

Respond ONLY with a single line containing a valid JSON object with the following keys:
- "concept": string
- "present": boolean (true or false)
- "confidence": float between 0 and 1
- "rationale": string (max 20 words)

Example:
{{"concept": "Clear cytoplasm (ccRCC)", "present": true, "confidence": 0.73, "rationale": "Concise reason"}}

Now return ONLY the JSON object, with no additional text before or after."""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_b64],
        "temperature": 0.2,
        "max_new_tokens": 128,
        # mai None: evita crash del worker su KeywordsStoppingCriteria
        "stop": "###",
    }

    print(f"Testing with image: {image_path}")
    print(f"Concept: {concept_name}")
    print(f"Sending request to: {CONTROLLER_URL}/worker_generate_stream")

    try:
        r = requests.post(f"{CONTROLLER_URL}/worker_generate_stream", json=payload, stream=True, timeout=120)
        print(f"STATUS: {r.status_code}")

        if r.status_code != 200:
            print(f"ERROR RESPONSE: {r.text}")
            return

        text = ""
        for chunk in r.iter_lines(delimiter=b"\0"):
            if chunk:
                try:
                    data = json.loads(chunk.decode())
                    text = data.get("text", "")
                except json.JSONDecodeError:
                    continue

        print(f"TEXT FIELD (first 400 chars): {repr(text[:400])}")

        # Try to parse as JSON
        try:
            parsed = json.loads(text.strip())
            print("PARSED JSON:", json.dumps(parsed, indent=2))
        except json.JSONDecodeError as e:
            print(f"JSON PARSE ERROR: {e}")

    except Exception as e:
        print(f"HTTP ERROR: {e}")

if __name__ == "__main__":
    # Use first available image from concept candidates
    images_root = Path("concept_candidates_images")
    if not images_root.exists():
        print(f"Images directory {images_root} not found. Run build_concept_candidates.py first.")
        exit(1)

    # Find first image
    image_files = list(images_root.rglob("*.png"))
    if not image_files:
        print("No PNG images found in concept_candidates_images/")
        exit(1)

    test_image = image_files[0]
    concept_name = "Clear cytoplasm (ccRCC)"
    base_prompt = "Identify viable renal tumour cells with abundant optically clear or glassy cytoplasm and sharp cell borders, in keeping with clear cell renal cell carcinoma. Exclude adipocytes, stromal fat, artefactual perinuclear clearing, and foamy macrophages."

    test_llava_call(test_image, concept_name, base_prompt)
>>

concept/ontology/__init__.py codice <<
# empty – marks "ontology" as a package
>>

concept/ontology/ontology_rcc_debug.yaml codice <<
version: 1
name: "rcc_histology_debug_4_concepts"

concepts:
  - id: 1
    name: "Clear cytoplasm (ccRCC)"
    short_name: "clear_cytoplasm_ccrcc"
    group: "ccRCC"
    primary_class: "ccRCC"
    prompt: >
      Identify viable renal tumour cells with abundant optically clear or glassy
      cytoplasm and sharp cell borders, in keeping with clear cell renal cell carcinoma.
      Exclude adipocytes, stromal fat, artefactual perinuclear clearing, and foamy
      macrophages.

  - id: 2
    name: "Papillary fronds with fibrovascular cores (pRCC)"
    short_name: "papillary_fronds_prcc"
    group: "pRCC"
    primary_class: "pRCC"
    prompt: >
      Identify true papillary fronds: finger-like projections with central fibrovascular
      cores containing loose stroma and vessels, lined by tumour cells, in keeping with
      papillary renal cell carcinoma. Exclude folded flat epithelium and simple tubules.

  - id: 3
    name: "Perinuclear halos (chRCC)"
    short_name: "perinuclear_halos_chrcc"
    group: "chRCC"
    primary_class: "CHROMO"
    prompt: >
      Identify tumour cells arranged in sheets or nests showing distinct perinuclear
      clearing or halos within pale to eosinophilic cytoplasm, typical of chromophobe
      renal cell carcinoma. Exclude artefactual vacuoles and mucin.

  - id: 4
    name: "Oncocytic cytoplasm (oncocytoma)"
    short_name: "oncocytic_cytoplasm_onco"
    group: "Oncocytoma"
    primary_class: "ONCO"
    prompt: >
      Identify tumour cells with abundant, dense, finely granular, deeply eosinophilic
      cytoplasm (oncocytes) and round centrally placed nuclei with smooth membranes,
      typical of renal oncocytoma. Exclude chromophobe-like cells with obvious
      perinuclear halos.
>>

concept/ontology/vlm_client_hf.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client locale (no HTTP) per LLaVA-Med via Hugging Face.

- Carica il modello da Hugging Face (es. Eren-Senoglu/llava-med-v1.5-mistral-7b-hf).
- Usa AutoProcessor per gestire testo + immagine.
- Espone ask_concept(image_path, concept_name, base_prompt) con stesso contratto di VLMClient HTTP:
  ritorna dict {"concept", "present", "confidence", "rationale"} oppure None.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


class VLMClientHF:
    """
    LLaVA-Med locale via Hugging Face (no controller/worker HTTP).
    """

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
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

        # debug esplicito > env > default False
        if debug is None:
            self.debug = os.getenv("VLM_DEBUG", "0") == "1"
        else:
            self.debug = bool(debug)

        if self.debug:
            print(f"[VLM-HF DEBUG] Loading model '{self.model_name}' on {self.device} dtype={torch_dtype}")

        # AutoProcessor/AutoModel per LLaVA-HF (llava-med-v1.5-mistral-7b-hf)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Estrai un oggetto JSON da una stringa arbitraria (stessa logica del client HTTP).
        """
        text = text.strip()

        # Tentativo diretto
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Cerca blocco {...} più esterno
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Regex fallback
        import re

        json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}'
        matches = re.findall(json_pattern, text)
        for match in reversed(matches):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    def _build_prompt(self, concept_name: str, base_prompt: str) -> str:
        """
        Prompt per LLaVA-HF:
        - include un token <image> per la singola immagine
        - stile USER/ASSISTANT
        - risposta SOLO JSON.
        """
        system = (
            "You are a board-certified renal pathologist. "
            "Answer succinctly and return ONLY a JSON with fields: "
            "concept, present (true/false), confidence (0..1), rationale (<=20 words)."
        )

        user = (
            f"{system}\n\n"
            "Analyse the attached histology patch.\n"
            f"Concept: {concept_name}\n"
            f"Question: {base_prompt}\n"
            'Respond ONLY with a single JSON object with keys: '
            '"concept" (string), "present" (true/false), '
            '"confidence" (0..1 float), "rationale" (<=20 words). '
            'Example: {"concept": "Clear cytoplasm (ccRCC)", "present": true, '
            '"confidence": 0.73, "rationale": "Concise reason"}'
        )

        # 1 immagine -> 1 token <image>
        prompt = f"<image>\nUSER: {user}\nASSISTANT:"
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
        Esegue una singola query (immagine + concept + domanda) al modello HF locale.

        Ritorna:
            dict normalizzato {"concept", "present", "confidence", "rationale"}
            oppure None se il testo non è parse-abile come JSON.
        """
        image_path = Path(image_path)
        if not image_path.is_file():
            if self.debug:
                print(f"[VLM-HF DEBUG] Image not found: {image_path}")
            return None

        image = Image.open(image_path).convert("RGB")
        prompt_str = self._build_prompt(concept_name, base_prompt)

        if self.debug:
            print(
                f"[VLM-HF DEBUG] >>> REQUEST\n"
                f"image={image_path}\n"
                f"concept={concept_name}\n"
                f"prompt_preview={prompt_str[:200]}...\n"
                f"{'-'*60}"
            )

        inputs = self.processor(
            text=[prompt_str],
            images=[image],
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generate_kwargs: Dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
            }
            # Se temperature > 0, abilita sampling
            if temperature and temperature > 0:
                generate_kwargs.update(
                    dict(
                        do_sample=True,
                        temperature=float(temperature),
                        top_p=0.9,
                    )
                )

            output_ids = self.model.generate(**inputs, **generate_kwargs)

        # Decodifica output (completo; l'estrattore JSON filtrerà eventuale rumore)
        text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if self.debug:
            print(
                f"[VLM-HF DEBUG] <<< RAW COMPLETION for concept={concept_name}, image={image_path}\n"
                f"{text}\n{'='*80}"
            )

        raw = self._extract_json(text)
        if raw is None or not isinstance(raw, dict):
            if self.debug:
                print("[VLM-HF DEBUG] No valid JSON object found in completion.")
            return None

        # Normalizza
        present_val = raw.get("present", False)
        if isinstance(present_val, str):
            present = present_val.strip().lower() in (
                "true",
                "yes",
                "y",
                "present",
                "1",
            )
        else:
            present = bool(present_val)

        try:
            confidence = float(raw.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        result: Dict[str, Any] = {
            "concept": raw.get("concept", concept_name),
            "present": present,
            "confidence": confidence,
            "rationale": raw.get("rationale", ""),
        }

        if self.debug:
            result["raw_text"] = text
            result["raw_prompt"] = prompt_str
            print(f"[VLM-HF DEBUG] >>> PARSED\n{json.dumps(result, indent=2)}\n{'#'*80}")

        return result
>>

concept/ontology/vlm_client.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple HTTP client for a vision-language model (e.g. LLaVA-Med).

It expects the model to return a JSON string with:
{"concept": "<name>", "present": true/false, "confidence": 0..1, "rationale": "<<=20 words>"}
"""

from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from PIL import Image


class VLMClient:
    """Minimal client for a controller+worker style VLM server."""

    def __init__(
        self,
        controller_url: str,
        model_name: str,
        timeout: int = 120,
        debug: Optional[bool] = None,
        stop_string: Optional[str] = None,
    ) -> None:
        self.controller_url = controller_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        # llava.serve.model_worker expects a non-None `stop` string; if None is
        # passed, KeywordsStoppingCriteria tokenization raises and the worker
        # returns error_code=1 (server_error_msg). Use a safe default separator.
        self.stop_string = stop_string or "###"

        # debug esplicito > env > default False
        if debug is None:
            self.debug = os.getenv("VLM_DEBUG", "0") == "1"
        else:
            self.debug = bool(debug)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_image(pil_img: Image.Image) -> str:
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _build_prompt(self, concept_name: str, base_prompt: str) -> str:
        """
        Costruisce il prompt testuale per LLaVA-Med.

        PUNTO CHIAVE: inseriamo esplicitamente un token <image> in testa,
        così len(images) == prompt.count("<image>") e il worker non alza
        più `ValueError: Number of images does not match number of <image> tokens`.
        """
        system = (
            "You are a board-certified renal pathologist. "
            "Answer succinctly and return a JSON with fields: "
            "concept, present (true/false), confidence (0..1), rationale (<=20 words)."
        )

        user = (
            f"{system}\n\n"
            "Analyse the attached histology patch.\n"
            f"Concept: {concept_name}\n"
            f"Question: {base_prompt}\n"
            'Return ONLY a JSON object with keys: concept (string), present (true/false), '
            'confidence (0..1 float), rationale (<=20 words). '
            'Example: {"concept": "Clear cytoplasm (ccRCC)", "present": true, "confidence": 0.73, '
            '"rationale": "Concise reason"}'
        )

        # Formato compatibile con llava.serve.model_worker:
        # una immagine -> un solo token <image> nel prompt.
        return f"<image>\n{user}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from text, trying multiple strategies.
        Returns the parsed dict or None if no valid JSON found.
        """
        text = text.strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find outermost {...}
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Use regex to find potential JSON objects
        import re
        json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}'
        matches = re.findall(json_pattern, text)
        for match in reversed(matches):  # Try longest first
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    # ------------------------------------------------------------------
    # Main API
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
        Invia (image, concept, question) al VLM e ritorna un dict Python
        già normalizzato, oppure None in caso di problemi "soft".

        Se il worker risponde con error_code != 0, viene alzato RuntimeError
        (gestito dal chiamante in build_concept_bank).
        """
        image_path = str(image_path)
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            img_b64 = self._encode_image(im)

        prompt_str = self._build_prompt(concept_name, base_prompt)

        payload = {
            "model": self.model_name,
            "prompt": prompt_str,
            "images": [img_b64],
            "temperature": float(temperature),
            "max_new_tokens": int(max_new_tokens),
            # llava.serve.model_worker vuole una stringa non-None
            "stop": self.stop_string,
        }

        if self.debug:
            print(
                f"[VLM DEBUG] >>> REQUEST\n"
                f"concept={concept_name}\n"
                f"image={image_path}\n"
                f"payload_keys={list(payload.keys())}\n"
                f"prompt_preview={prompt_str[:200]}...\n"
                f"{'-'*60}"
            )

        # Usa l'endpoint streaming, che ritorna chunk JSON con "text"
        url = f"{self.controller_url}/worker_generate_stream"
        try:
            resp = requests.post(url, json=payload, stream=True, timeout=self.timeout)
            resp.raise_for_status()

            text = ""
            for chunk in resp.iter_lines(delimiter=b"\0"):
                if chunk:
                    try:
                        data = json.loads(chunk.decode())
                        text = data.get("text", "")
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            if self.debug:
                print(
                    f"[VLM DEBUG] HTTP error for concept={concept_name}, "
                    f"image={image_path}: {e}"
                )
            # Errore "hard" di rete -> nessuna risposta utile
            return None

        if self.debug:
            print(
                f"[VLM DEBUG] <<< RAW RESPONSE\n"
                f"status={resp.status_code}\n"
                f"response_type={type(data)}\n"
                f"response_keys={list(data.keys()) if isinstance(data, dict) else 'N/A'}\n"
                f"{'-'*60}"
            )

        # Gestione esplicita di error_code dal worker/controller
        if isinstance(data, dict) and data.get("error_code", 0) != 0:
            msg = data.get("text", "") or data.get("message", "")
            if self.debug:
                print(
                    f"[VLM DEBUG] Worker returned error_code={data.get('error_code')}: {msg}"
                )
            raise RuntimeError(
                f"VLM worker error (code={data.get('error_code')}): {msg}"
            )

        # Extract text from response
        if not isinstance(data, dict) or "text" not in data:
            if self.debug:
                print(
                    f"[VLM DEBUG] Invalid response format for concept={concept_name}, "
                    f"image={image_path}: {data}"
                )
            return None

        full = data["text"].strip()

        if self.debug:
            print(
                f"[VLM DEBUG] <<< RAW COMPLETION for concept={concept_name}, "
                f"image={image_path}\n{full}\n{'='*80}"
            )

        # Alcuni modelli possono aggiungere ```json ... ```: ripulisci
        for token in ("```json", "```JSON", "```"):
            if token in full:
                full = full.replace(token, "")
        full = full.strip()

        # Usa il parser robusto invece di duplicare la logica
        raw = self._extract_json(full)
        if raw is None:
            if self.debug:
                print("[VLM DEBUG] No valid JSON object found in completion.")
            return None

        if not isinstance(raw, dict):
            if self.debug:
                print(f"[VLM DEBUG] Parsed JSON is not a dict: {raw}")
            return None

        # Normalizza tipi e range così build_concept_bank può fidarsi
        present_val = raw.get("present", False)
        if isinstance(present_val, str):
            present = present_val.strip().lower() in (
                "true",
                "yes",
                "y",
                "present",
                "1",
            )
        else:
            present = bool(present_val)

        try:
            confidence = float(raw.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        result: Dict[str, Any] = {
            "concept": raw.get("concept", concept_name),
            "present": present,
            "confidence": confidence,
            "rationale": raw.get("rationale", ""),
        }

        # In debug aggiungi anche il testo grezzo e il prompt originali
        if self.debug:
            result["raw_text"] = full
            result["raw_prompt"] = prompt_str

            print(
                f"[VLM DEBUG] >>> PARSED\n"
                f"{json.dumps(result, indent=2)}\n"
                f"{'#'*80}"
            )

        return result
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

# This job wraps the *existing* run_full_xai.sh orchestrator and runs it inside an allocation.

set -euo pipefail

# --------- user-configurable defaults (can be overridden via sbatch --export=ALL,VAR=...) ----------
# N.B.: questi valori sono solo fallback; i reali possono essere esportati prima di chiamare sbatch.

EXP_ROOT="${EXP_ROOT:-/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3}"
MODEL_NAME="${MODEL_NAME:-moco_v3_ssl_linear_best}"
BACKBONE_NAME="${BACKBONE_NAME:-vit_small_patch16_224}"

ONLY_SPATIAL="${ONLY_SPATIAL:-0}"      # 1 to enable (non ancora usato in run_full_xai.sh)
ONLY_CONCEPT="${ONLY_CONCEPT:-0}"      # 1 to enable (non ancora usato in run_full_xai.sh)

WITH_VLM_AGG="${WITH_VLM_AGG:-0}"      # legacy; non usato dall’orchestratore attuale
MIN_VLM_CONF="${MIN_VLM_CONF:-0.7}"

# venv per rcc-ssrl (train + explainability)
VENV_PATH="${VENV_PATH:-}"

echo "[INFO] Host: $(hostname)"
echo "[INFO] EXP_ROOT=${EXP_ROOT}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] BACKBONE_NAME=${BACKBONE_NAME}"
echo "[INFO] ONLY_SPATIAL=${ONLY_SPATIAL} ONLY_CONCEPT=${ONLY_CONCEPT}"
echo "[INFO] WITH_VLM_AGG=${WITH_VLM_AGG} MIN_VLM_CONF=${MIN_VLM_CONF}"
echo "[INFO] VENV_PATH=${VENV_PATH}"

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
>>

run_explainability.py codice <<
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main orchestrator to run spatial and concept explainability.

Itera sulle ablation di un esperimento e per ognuna:
- costruisce config_xai.yaml (spatial) e config_concept.yaml (concept)
- lancia gli sbatch relativi.

Comportamento:
- Se per una certa ablation NON esiste la cartella di eval per il modello
  (eval/<model_name>/...), quella ablation viene SKIPPATA con un warning.
- Se esiste ma non contiene nessun run (sottocartella timestamp), viene skippata.
- Se mancano i checkpoint SSL backbone/head, l'ablation viene skippata.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import os

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
    subprocess.run(["sbatch", str(batch_file)], check=True)


def _derive_backbone_basename(model_name: str) -> str:
    suffix = "_ssl_linear_best"
    if model_name.endswith(suffix):
        return model_name[: -len(suffix)]
    return model_name


def _find_latest_eval_run(ablation_dir: Path, model_name: str) -> Optional[Path]:
    base = ablation_dir / "eval" / model_name
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
        help="Path to root dir with ablations folders",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        type=str,
        help="Folder/model name for XAI outputs (es. moco_v3_ssl_linear_best)",
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

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

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

    backbone_base = _derive_backbone_basename(args.model_name)
    log.info(f"[INFO] MODEL_NAME={args.model_name}  BACKBONE_BASE={backbone_base}")
    log.info(f"[INFO] Found {len(ablation_folders)} ablations.")

    # i template vengono letti ogni volta; per ora va bene così

    for ablation in ablation_folders:
        log.info("=" * 80)
        log.info(f"[ABLATION] Processing: {ablation}")

        eval_run = _find_latest_eval_run(ablation, args.model_name)
        if eval_run is None:
            log.warning(f"[ABLATION] Skipping {ablation.name}: no eval run available.")
            continue

        backbone_ckpt = ablation / "checkpoints" / f"{backbone_base}__ssl_best.pt"
        head_ckpt = ablation / "checkpoints" / f"{backbone_base}__ssl_linear_best.pt"

        has_backbone = _check_checkpoint(backbone_ckpt, "ssl_backbone_ckpt", ablation.name)
        has_head = _check_checkpoint(head_ckpt, "ssl_head_ckpt", ablation.name)

        if not (has_backbone and has_head):
            log.warning(
                f"[ABLATION] Skipping {ablation.name}: missing required checkpoints."
            )
            continue

        # ---------------------- SPATIAL CONFIG ----------------------
        spatial_config = yaml.safe_load(open(args.spatial_config_template))
        spatial_config["experiment"]["outputs_root"] = str(ablation / "06_xai")
        spatial_config["model"]["name"] = args.model_name
        spatial_config["model"]["ssl_backbone_ckpt"] = str(backbone_ckpt)
        spatial_config["model"]["ssl_head_ckpt"] = str(head_ckpt)
        spatial_config["evaluation_inputs"]["eval_run_dir"] = str(eval_run)

        spatial_config_path = ablation / "06_xai" / "config_xai.yaml"
        spatial_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(spatial_config_path, "w") as f:
            yaml.dump(spatial_config, f)
        log.info(f"[CFG] Spatial config written to {spatial_config_path}")

        # ---------------------- CONCEPT CONFIG ----------------------
        concept_config = yaml.safe_load(open(args.concept_config_template))
        concept_config["experiment"]["outputs_root"] = str(ablation / "06_xai")
        concept_config["model"]["name"] = args.model_name
        concept_config["model"]["ssl_backbone_ckpt"] = str(backbone_ckpt)
        concept_config["model"]["ssl_head_ckpt"] = str(head_ckpt)
        concept_config["evaluation_inputs"]["eval_run_dir"] = str(eval_run)

        # se CONCEPT_BANK_CSV è definita nell'env (es. da run_full_xai.sh),
        # forza l'uso di quel path per il concept bank
        concept_bank_csv_env = os.environ.get("CONCEPT_BANK_CSV")
        if concept_bank_csv_env:
            concept_config.setdefault("concepts", {})
            concept_config["concepts"]["meta_csv"] = concept_bank_csv_env
            log.info(
                f"[CFG] Overriding concepts.meta_csv with CONCEPT_BANK_CSV={concept_bank_csv_env}"
            )

        concept_config_path = ablation / "06_xai" / "config_concept.yaml"
        with open(concept_config_path, "w") as f:
            yaml.dump(concept_config, f)
        log.info(f"[CFG] Concept config written to {concept_config_path}")

        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"/home/mla_group_01/rcc-ssrl/src/logs/xai/{backbone_base}/{ablation.name}/{datetime_str}"
        spatial_sbatch = Path(__file__).parent / "spatial" / "xai_generate.sbatch"
        concept_sbatch = Path(__file__).parent / "concept" / "xai_concept.sbatch"

        log.info(f"[LAUNCH] Spatial XAI sbatch for {ablation.name}")
        run_sbatch(spatial_sbatch, spatial_config_path, log_dir)

        log.info(f"[LAUNCH] Concept XAI sbatch for {ablation.name}")
        run_sbatch(concept_sbatch, concept_config_path, log_dir)

    log.info("[DONE] run_explainability completed.")
    log.info("       Check Slurm logs under /home/mla_group_01/rcc-ssrl/src/logs/xai/...")


if __name__ == "__main__":
    main()
>>

run_full_xai.sh codice <<
#!/usr/bin/env bash
# Orchestrate full explainability pipeline:
# - Stage 0: global concept bank (dataset-level, only if missing)
# - Stage 1/2: spatial + concept XAI for all ablations in an experiment

# NOTE:
#  - opzionalmente può lanciare un server LLaVA-Med locale (controller + worker)
#    per la Stage 0b (build_concept_bank).
#  - abilita questo comportamento esportando:
#        START_LOCAL_VLM=1
#        VLM_MODEL_PATH=/path/or/hf/id/of/microsoft/llava-med-v1.5-mistral-7b  # opzionale
#  - va eseguito su un nodo con GPU (srun/sbatch), NON sul login node.

set -euo pipefail

# ------------------- defaults (override via env or args if vuoi, ma qui li teniamo fissi) -------------------

REPO_ROOT="/home/mla_group_01/rcc-ssrl"
SRC_DIR="${REPO_ROOT}/src"
LOG_DIR="${SRC_DIR}/logs/xai"
mkdir -p "${LOG_DIR}"
LOG_SUFFIX="${SLURM_JOB_ID:-local_$$}"
LLAVA_CTRL_LOG="${LLAVA_CTRL_LOG:-${LOG_DIR}/llava_controller.${LOG_SUFFIX}.log}"
LLAVA_WORKER_LOG="${LLAVA_WORKER_LOG:-${LOG_DIR}/llava_worker.${LOG_SUFFIX}.log}"
# Worker port overrideable to avoid collisions
VLM_WORKER_PORT="${VLM_WORKER_PORT:-40000}"

# Dataset-level (Stage 0)
TRAIN_WDS_DIR="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/train"
CANDIDATES_CSV="${SRC_DIR}/explainability/concept/ontology/concept_candidates_rcc.csv"
CANDIDATES_IMG_ROOT="${SRC_DIR}/explainability/concept/ontology/concept_candidates_images"

# default: file di debug a 4 concetti
ONTOLOGY_YAML_DEFAULT="${SRC_DIR}/explainability/concept/ontology/ontology_rcc_debug.yaml"
ONTOLOGY_YAML="${ONTOLOGY_YAML:-$ONTOLOGY_YAML_DEFAULT}"

CONCEPT_BANK_CSV_DEFAULT="${SRC_DIR}/explainability/concept/ontology/concepts_rcc_debug.csv"
CONCEPT_BANK_CSV="${CONCEPT_BANK_CSV:-$CONCEPT_BANK_CSV_DEFAULT}"

# Experiment-level (Stage 1/2)
EXP_ROOT_DEFAULT="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3"
MODEL_NAME_DEFAULT="moco_v3_ssl_linear_best"
BACKBONE_NAME_DEFAULT="vit_small_patch16_224"

# VLM config for concept bank
VLM_CONTROLLER_DEFAULT="http://localhost:10000"
VLM_MODEL_DEFAULT="Eren-Senoglu/llava-med-v1.5-mistral-7b-hf"  # HF-converted weights for local HF backend
PRESENCE_THRESHOLD_DEFAULT="0.3"

# Se START_LOCAL_VLM=1, run_full_xai lancerà un server LLaVA-Med locale
# (controller + model_worker) prima di build_concept_bank e lo killerà alla fine.
START_LOCAL_VLM="${START_LOCAL_VLM:-0}"

# Path o HF id del modello LLaVA-Med
VLM_MODEL_PATH_DEFAULT="microsoft/llava-med-v1.5-mistral-7b"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-$VLM_MODEL_PATH_DEFAULT}"
VLM_WARMUP_SECONDS="${VLM_WARMUP_SECONDS:-120}"

# Path al repo e al python di LLaVA-Med (override via env se necessario)
LLAVA_REPO_ROOT_DEFAULT="/home/mla_group_01/LLaVA-Med"
LLAVA_REPO_ROOT="${LLAVA_REPO_ROOT:-$LLAVA_REPO_ROOT_DEFAULT}"

LLAVA_PYTHON_BIN_DEFAULT="/home/mla_group_01/llava-med-venv/bin/python"
LLAVA_PYTHON_BIN="${LLAVA_PYTHON_BIN:-$LLAVA_PYTHON_BIN_DEFAULT}"

# ------------------- helper: LLaVA-Med server locale -------------------
start_local_vlm() {
  if [[ "${START_LOCAL_VLM}" != "1" ]]; then
    return 0
  fi

  echo "[INFO] Starting local LLaVA-Med controller on ${VLM_CONTROLLER}"
  echo "[INFO]   LLAVA_REPO_ROOT=${LLAVA_REPO_ROOT}"
  echo "[INFO]   LLAVA_PYTHON_BIN=${LLAVA_PYTHON_BIN}"

  # Parse controller host/port from VLM_CONTROLLER (e.g., http://localhost:11000)
  CTRL_HOST=$(echo "${VLM_CONTROLLER}" | sed -E 's#https?://([^:/]+).*#\1#')
  CTRL_PORT=$(echo "${VLM_CONTROLLER}" | awk -F: '{print $NF}')
  # Fallback if parsing fails
  if [[ -z "${CTRL_HOST}" || "${CTRL_HOST}" == "${VLM_CONTROLLER}" ]]; then
    CTRL_HOST="0.0.0.0"
  fi
  if ! [[ "${CTRL_PORT}" =~ ^[0-9]+$ ]]; then
    CTRL_PORT="10000"
  fi

  if [[ ! -x "${LLAVA_PYTHON_BIN}" ]]; then
    echo "[ERROR] LLAVA_PYTHON_BIN='${LLAVA_PYTHON_BIN}' non eseguibile; controlla il venv LLaVA-Med." >&2
    return 1
  fi

  if [[ ! -d "${LLAVA_REPO_ROOT}" ]]; then
    echo "[ERROR] LLAVA_REPO_ROOT='${LLAVA_REPO_ROOT}' non esiste; clona il repo LLaVA-Med lì o override via env." >&2
    return 1
  fi

  pushd "${LLAVA_REPO_ROOT}" >/dev/null

  echo "[INFO] Controller log: ${LLAVA_CTRL_LOG}"
  echo "[INFO] Worker log: ${LLAVA_WORKER_LOG}"

  VLM_DEBUG="${VLM_DEBUG:-1}" \
  "${LLAVA_PYTHON_BIN}" -m llava.serve.controller \
    --host "0.0.0.0" \
    --port "${CTRL_PORT}" \
    > "${LLAVA_CTRL_LOG}" 2>&1 &
  VLM_CTRL_PID=$!
  sleep 5

  VLM_DEBUG="${VLM_DEBUG:-1}" \
  "${LLAVA_PYTHON_BIN}" -m llava.serve.model_worker \
    --host "0.0.0.0" \
    --controller "${VLM_CONTROLLER}" \
    --port "${VLM_WORKER_PORT}" \
    --worker "http://127.0.0.1:${VLM_WORKER_PORT}" \
    --model-path "${VLM_MODEL_PATH}" \
    --multi-modal \
    > "${LLAVA_WORKER_LOG}" 2>&1 &
  VLM_WORKER_PID=$!

  popd >/dev/null

  echo "[INFO] Waiting ${VLM_WARMUP_SECONDS}s for VLM to load weights..."
  sleep "${VLM_WARMUP_SECONDS}"

  # health-check controller -> /list_models (max 5 tentativi)
  local hc_ok=0
  for i in {1..5}; do
    if curl -s -X POST "${VLM_CONTROLLER}/list_models" >/dev/null 2>&1; then
      hc_ok=1
      break
    fi
    sleep 2
  done
  if [[ "${hc_ok}" != "1" ]]; then
    echo "[ERROR] VLM controller health-check failed on ${VLM_CONTROLLER}. See logs: ${LLAVA_CTRL_LOG} ${LLAVA_WORKER_LOG}" >&2
    return 1
  fi
}

stop_local_vlm() {
  if [[ "${START_LOCAL_VLM}" != "1" ]]; then
    return 0
  fi
  echo "[INFO] Stopping local LLaVA-Med server"
  if [[ -n "${VLM_WORKER_PID:-}" ]]; then
    kill "${VLM_WORKER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${VLM_CTRL_PID:-}" ]]; then
    kill "${VLM_CTRL_PID}" 2>/dev/null || true
  fi
}

VENV_PATH="${VENV_PATH:-}"   # opzionale: export VENV_PATH=/path/to/venv
EXP_ROOT="${EXP_ROOT:-$EXP_ROOT_DEFAULT}"
MODEL_NAME="${MODEL_NAME:-$MODEL_NAME_DEFAULT}"
BACKBONE_NAME="${BACKBONE_NAME:-$BACKBONE_NAME_DEFAULT}"
VLM_CONTROLLER="${VLM_CONTROLLER:-$VLM_CONTROLLER_DEFAULT}"
VLM_MODEL="${VLM_MODEL:-$VLM_MODEL_DEFAULT}"
PRESENCE_THRESHOLD="${PRESENCE_THRESHOLD:-$PRESENCE_THRESHOLD_DEFAULT}"

# Flags esperimento (override con export ONLY_SPATIAL=1 etc se ti serve)
ONLY_SPATIAL="${ONLY_SPATIAL:-0}"
ONLY_CONCEPT="${ONLY_CONCEPT:-0}"

# ------------------- logging -------------------

echo "[INFO] run_full_xai.sh starting"
echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] EXP_ROOT=${EXP_ROOT}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] BACKBONE_NAME=${BACKBONE_NAME}"
echo "[INFO] VLM_CONTROLLER=${VLM_CONTROLLER}"
echo "[INFO] VLM_MODEL=${VLM_MODEL}"
echo "[INFO] PRESENCE_THRESHOLD=${PRESENCE_THRESHOLD}"

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

  # 0a) concept_candidates_rcc.csv (train WDS -> PNG + CSV)
  echo "[INFO] Stage 0a: building concept_candidates_rcc.csv"
  python3 -m explainability.concept.ontology.build_concept_candidates \
    --train-dir "${TRAIN_WDS_DIR}" \
    --pattern "shard-*.tar" \
    --image-key "img.jpg;jpg;jpeg;png" \
    --meta-key "meta.json;json" \
    --out-csv "${CANDIDATES_CSV}" \
    --images-root "${CANDIDATES_IMG_ROOT}"

  # 0b) concepts_rcc_debug.csv (VLM su candidates) – HF locale, nessun HTTP
  echo "[INFO] Stage 0b: building concepts_rcc_debug.csv via local HF LLaVA-Med (no HTTP)"
  export VLM_DEBUG="${VLM_DEBUG:-0}"  # metti 1 se vuoi log dettagliati dal client HF

  python3 -m explainability.concept.ontology.build_concept_bank \
    --ontology "${ONTOLOGY_YAML}" \
    --images-csv "${CANDIDATES_CSV}" \
    --model-name "${VLM_MODEL}" \
    --out-csv "${CONCEPT_BANK_CSV}" \
    --presence-threshold "${PRESENCE_THRESHOLD}" \
    --max-images 100 \
    --backend "hf"

  # hard check: concept bank deve avere almeno header + 1 riga
  lines_after=$(wc -l < "${CONCEPT_BANK_CSV}")
  if [[ "${lines_after}" -le 1 ]]; then
    echo "[ERROR] Concept bank ${CONCEPT_BANK_CSV} still empty after Stage 0 (lines=${lines_after}). Aborting."
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

# flags optionali: se vuoi supportare solo spatial/solo concept, estendi run_explainability.py di conseguenza.
# Al momento il tuo run_explainability.py non ha --only-spatial / --only-concept, quindi li ignoriamo.
if [[ "${ONLY_SPATIAL}" == "1" ]]; then
  echo "[WARN] ONLY_SPATIAL=1 set, but run_explainability.py non supporta ancora il flag; eseguo comunque full pipeline."
fi
if [[ "${ONLY_CONCEPT}" == "1" ]]; then
  echo "[WARN] ONLY_CONCEPT=1 set, ma run_explainability.py non supporta ancora il flag; eseguo comunque full pipeline."
fi

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

