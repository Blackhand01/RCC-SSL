#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-only evaluation per modelli già addestrati (encoder + classifier).
Patch: salva predictions.csv ARRICCHITO con wds_key + metadati per XAI alignment.
"""
import os, sys, json, argparse, logging, random, csv
from pathlib import Path
from datetime import datetime
import numpy as np
from ssl_linear_loader import SSLLinearClassifier  # local import

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms, datasets
import torchvision

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score,
    classification_report, average_precision_score
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# opzionali
try:
    import webdataset as wds
    HAVE_WDS = True
except Exception:
    HAVE_WDS = False

try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False


# ------------------------ utils base ------------------------
def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True

def setup_logger():
    log = logging.getLogger("eval")
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout); h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    if not log.handlers:
        log.addHandler(h)
    return log

def default_preprocess(img_size, imagenet_norm=False):
    from torchvision import transforms
    tfm = [
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),  # [0,1]
    ]
    if imagenet_norm:
        tfm.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    return transforms.Compose(tfm)


# ------------------------ plotting ------------------------
def plot_confmat(cm, class_names, out_png):
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def plot_umap_logits(logits, labels, class_names, out_png, umap_cfg):
    reducer = umap.UMAP(
        n_neighbors=umap_cfg["n_neighbors"],
        min_dist=umap_cfg["min_dist"],
        random_state=umap_cfg["random_state"]
    )
    X2 = reducer.fit_transform(logits)
    plt.figure(figsize=(7,6))
    palette = plt.cm.tab10.colors
    for cid, cname in enumerate(class_names):
        idx = labels == cid
        plt.scatter(X2[idx,0], X2[idx,1], s=3, alpha=0.7, label=cname, color=palette[cid % len(palette)])
    plt.legend(markerscale=4, fontsize=8, bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()


# ------------------------ model loaders ------------------------
def load_model_from_repo(repo_root, module_name, class_name, checkpoint, strict, log):
    sys.path.insert(0, os.path.join(repo_root, "src"))
    mod = __import__(module_name, fromlist=[class_name])
    ModelClass = getattr(mod, class_name)
    model = ModelClass()
    if checkpoint and os.path.isfile(checkpoint):
        sd = torch.load(checkpoint, map_location="cpu")
        for k in ["state_dict", "model", "module", "net"]:
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                sd = sd[k]; break
        sd = {k.replace("module.", ""): v for k, v in sd.items()} if isinstance(sd, dict) else sd
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        log.info(f"Checkpoint caricato. Missing:{len(missing)} Unexpected:{len(unexpected)}")
    else:
        log.warning("Checkpoint mancante o non leggibile.")
    return model

def load_fallback_resnet50(checkpoint, strict, num_classes=5, log=None, dropout_p=0.2):
    model = torchvision.models.resnet50(weights=None)
    if dropout_p > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(model.fc.in_features, num_classes)
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    if checkpoint and os.path.isfile(checkpoint):
        sd = torch.load(checkpoint, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        if log: 
            log.info(f"Checkpoint caricato correttamente. Missing:{len(missing)} Unexpected:{len(unexpected)}")
    return model


# ------------------------ helpers ------------------------
def softmax_logits(x):
    x = x - x.max(dim=1, keepdim=True).values
    return torch.exp(x) / torch.exp(x).sum(dim=1, keepdim=True)

def _parse_meta(meta_any):
    if isinstance(meta_any, (bytes, bytearray)):
        return json.loads(meta_any.decode("utf-8"))
    if isinstance(meta_any, str):
        return json.loads(meta_any)
    if isinstance(meta_any, dict):
        return meta_any
    if isinstance(meta_any, (list, tuple)) and len(meta_any) == 1:
        return _parse_meta(meta_any[0])
    return {}

# ------------------------ WebDataset loader (grezzo, con __key__) ------------------------
def make_wds_loader(test_dir, pattern, image_key, meta_key, class_order, preprocess, batch_size, num_workers):
    import os, glob
    if not HAVE_WDS:
        raise RuntimeError("webdataset non disponibile")

    shard_glob = os.path.join(test_dir, pattern)
    shards = sorted(glob.glob(shard_glob))
    if len(shards) == 0:
        raise FileNotFoundError(f"Nessun shard trovato con pattern: {shard_glob}")

    seen_keys = set()
    def _is_new(sample):
        # sample è un dict in questa fase
        k = sample.get("__key__")
        if k in seen_keys:
            return False
        seen_keys.add(k)
        return True

    def _is_valid_tuple(t):
        # dopo to_tuple: vogliamo (img, meta, key) tutti non-null
        return (
            isinstance(t, (tuple, list)) and len(t) >= 3
            and (t[0] is not None) and (t[1] is not None) and (t[2] is not None)
        )

    ds = (
        wds.WebDataset(
            shards,
            shardshuffle=False,
            handler=wds.warn_and_continue,
            empty_check=False
        )
        .decode("pil")
        .select(_is_new)
        .to_tuple(image_key, meta_key, "__key__", handler=wds.warn_and_continue)  # -> (img, meta, key)
        .select(_is_valid_tuple)                                                 # filtra qualsiasi anomalia
        .map_tuple(preprocess, lambda m: m, lambda k: k)                          # (img_t, meta_raw, key)
        .shuffle(0)
        .repeat(0)
    )

    eff_workers = min(num_workers, max(1, len(shards)))

    # batch_size=1 + collate_fn robusto: ritorna direttamente la tupla o None
    def _collate_first(batch):
        if not batch:
            return None
        item = batch[0]  # (img, meta, key)
        if not (isinstance(item, (tuple, list)) and len(item) >= 3):
            return None
        img, meta, key = item
        # Se l'immagine è 3D, aggiungi la batch dim
        if isinstance(img, torch.Tensor) and img.ndim == 3:
            img = img.unsqueeze(0)  # [1,C,H,W]
        return (img, meta, key)


    return DataLoader(
        ds,
        batch_size=1,  # <<< importante per avere una tupla (img, meta, key)
        num_workers=eff_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_first,
    )





# ------------------------ main ------------------------
def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    log = setup_logger()
    set_seed(cfg["experiment"]["seed"])

    device = torch.device(cfg["runtime"].get("device","cuda") if torch.cuda.is_available() else "cpu")
    img_size = int(cfg["data"]["img_size"])
    imagenet_norm = bool(cfg["data"].get("imagenet_norm", False))
    preprocess = default_preprocess(img_size, imagenet_norm)

    # output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = cfg["model"]["name"]
    out_root = Path(cfg["experiment"]["outputs_root"]) / model_tag / ts
    out_root.mkdir(parents=True, exist_ok=True)
    log.info(f"Output dir: {str(out_root)}")

    # classi
    class_names = cfg.get("labels", {}).get("class_order", ["ccRCC","pRCC","CHROMO","ONCO","NOT_TUMOR"])
    class_to_id = {c:i for i,c in enumerate(class_names)}

    # loader TEST
    backend = cfg["data"]["backend"].lower()
    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])

    if backend == "webdataset":
        w = cfg["data"]["webdataset"]
        test_loader = make_wds_loader(
        test_dir=w["test_dir"],
        pattern=w["pattern"],
        image_key=w["image_key"],
        meta_key=w["meta_key"],
        class_order=class_names,
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers
    )

    else:
        ds = datasets.ImageFolder(cfg["data"]["imagefolder"]["test_dir"], transform=preprocess)
        test_loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # modello
    model_cfg = cfg["model"]
    model = None
    arch_hint = model_cfg.get("arch_hint", "cnn").lower()
    if arch_hint == "ssl_linear":
        num_classes = len(class_names)
        model = SSLLinearClassifier(backbone_name=model_cfg.get("backbone_name", "resnet50"),
                                    num_classes=num_classes)
        allow_swap = bool(model_cfg.get("allow_arch_autoswap", True))
        mb, ub = model.load_backbone_from_ssl(model_cfg["ssl_backbone_ckpt"], allow_autoswap=allow_swap)
        mh, uh = model.load_head_from_probe(model_cfg["ssl_head_ckpt"])
        log.info(f"SSL backbone loaded (missing={mb}, unexpected={ub}); head loaded (missing={mh}, unexpected={uh})")
    else:
        if model_cfg.get("module") and model_cfg.get("class_name"):
            try:
                model = load_model_from_repo(
                    model_cfg["repo_root"], model_cfg["module"], model_cfg["class_name"],
                    model_cfg.get("checkpoint", ""), model_cfg.get("strict_load", False), log
                )
            except Exception as e:
                log.warning(f"Import repo fallito ({e}). Fallback ResNet50.")
        if model is None:
            model = load_fallback_resnet50(
                model_cfg.get("checkpoint",""),
                model_cfg.get("strict_load", False),
                num_classes=len(class_names),
                log=log
            )
    model = model.to(device).eval()

    # === EVALUATION LOOP con salvataggi allineati ===
    y_true, y_pred = [], []
    logits_list = []
    rows = []

    with torch.no_grad():
        if backend == "webdataset":
            for sample in test_loader:
                if sample is None:
                    continue
                img, meta_any, key = sample
                meta = _parse_meta(meta_any)
                lab_txt = meta.get("class_label", None)
                if lab_txt is None:
                    continue
                if lab_txt not in class_to_id:
                    continue
                lab = class_to_id[lab_txt]

                x = img.to(device, non_blocking=True)
                out = model(x)
                logits = out[0] if isinstance(out, (list, tuple)) else (out["logits"] if isinstance(out, dict) and "logits" in out else out)
                pred = int(torch.argmax(logits, dim=1).item())

                y_true.append(lab); y_pred.append(pred)
                logits_list.append(logits.detach().cpu().numpy())

                rows.append({
                    "wds_key": key,
                    "patient_id": meta.get("patient_id"),
                    "slide_id": meta.get("wsi_or_roi") or meta.get("slide_id"),
                    "coords": meta.get("coords"),
                    "y_true": int(lab),
                    "y_pred": int(pred),
                })
        else:
            # imagefolder: niente meta/keys → CSV minimale
            for x, lab in test_loader:
                x = x.to(device, non_blocking=True)
                logits = model(x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_true.append(lab.numpy()); y_pred.append(preds)
                logits_list.append(logits.detach().cpu().numpy())

    y_true = np.concatenate([np.atleast_1d(t) for t in y_true]).astype(int)
    y_pred = np.concatenate([np.atleast_1d(p) for p in y_pred]).astype(int)
    logits_np = np.concatenate(logits_list, axis=0) if len(logits_list)>0 else None

    # === metriche ===
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        if logits_np is not None:
            probs = torch.from_numpy(logits_np)
            probs = softmax_logits(probs).numpy()
            y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
            metrics["macro_auc_ovr"] = float(roc_auc_score(y_bin, probs, average="macro", multi_class="ovr"))
            metrics["macro_auprc"] = float(average_precision_score(y_bin, probs, average="macro"))
            # ECE
            confidences = probs.max(axis=1); predictions = probs.argmax(axis=1)
            accuracies = (predictions == y_true)
            bins = np.linspace(0.0, 1.0, 16)
            ece = 0.0
            for i in range(15):
                in_bin = (confidences > bins[i]) & (confidences <= bins[i+1])
                prop = in_bin.mean()
                if prop > 0:
                    ece += abs(accuracies[in_bin].mean() - confidences[in_bin].mean()) * prop
            metrics["ece"] = float(ece)
    except Exception as e:
        log.warning(f"AUC/AUPRC/ECE failed: {e}")

    # confusion matrix + report
    cm = confusion_matrix(y_true, y_pred)
    plot_confmat(cm, class_names, out_root / f"cm_{model_tag}.png")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(out_root / f"report_per_class.json", "w") as f:
        json.dump(report, f, indent=2)

    # salvataggi
    with open(out_root / f"metrics_{model_tag}.json", "w") as f:
        json.dump({
            "experiment": cfg["experiment"],
            "model": cfg["model"],
            "metrics": metrics,
            "class_names": class_names
        }, f, indent=2)

    if logits_np is not None:
        np.save(out_root / "logits_test.npy", logits_np)

    # >>>>>>> CSV con lo STESSO NOME di prima, ma ARRICCHITO <<<<<<<
    with open(out_root / "predictions.csv", "w", newline="") as f:
        if backend == "webdataset":
            fieldnames = ["wds_key", "patient_id", "slide_id", "coords", "y_true", "y_pred"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        else:
            w = csv.writer(f)
            w.writerow(["y_true","y_pred"])
            for t,p in zip(y_true, y_pred):
                w.writerow([int(t), int(p)])

    log.info(f"[TEST] Acc={metrics['accuracy']:.4f}  BalAcc={metrics['balanced_accuracy']:.4f}  MacroF1={metrics['macro_f1']:.4f}")
    if "macro_auc_ovr" in metrics:
        log.info(f"[TEST] MacroAUC(OvR)={metrics['macro_auc_ovr']:.4f}")
    log.info("FINITO.")

    # UMAP opzionale
    if cfg["evaluation"]["umap"]["enabled"] and HAVE_UMAP and logits_np is not None:
        source = cfg["evaluation"]["umap"].get("source","logits")
        data_umap = logits_np
        plot_umap_logits(
            data_umap, y_true, class_names,
            out_root / f"embedding_{model_tag}_{source}.png",
            cfg["evaluation"]["umap"]
        )

if __name__ == "__main__":
    main()
