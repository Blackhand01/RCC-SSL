#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.io import append_row_csv, ensure_dir, prefixed
try:
    from ..utils.torch_ops import move_to
except ModuleNotFoundError:  # pragma: no cover - fallback for namespace issues
    import importlib.util
    import sys
    from pathlib import Path as _Path

    _torch_ops_path = _Path(__file__).resolve().parent.parent / "utils" / "torch_ops.py"
    spec = importlib.util.spec_from_file_location("src.training.utils.torch_ops", _torch_ops_path)
    module = importlib.util.module_from_spec(spec) if spec and spec.loader else None
    if module and spec and spec.loader:
        spec.loader.exec_module(module)
        sys.modules.setdefault("src.training.utils.torch_ops", module)
        move_to = module.move_to  # type: ignore[attr-defined]
    else:  # pragma: no cover
        raise
from ..utils.viz import plot_confusion

__all__ = [
    "extract_features",
    "save_features",
    "visualize_features_umap_pca",
    "train_linear_probe_torch",
    "extract_split",
    "save_parquet",
]

try:  # pragma: no cover - optional dependency
    from sklearn.metrics import confusion_matrix
except Exception:  # pragma: no cover
    confusion_matrix = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    PCA = None  # type: ignore


@torch.no_grad()
def extract_features(backbone: torch.nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract normalized features from backbone for entire loader.
    Returns tuple (features, labels) as NumPy arrays.
    """
    backbone.eval().to(device)
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for batch in loader:
        batch = move_to(batch, device)
        inputs = batch["inputs"]
        targets = batch["targets"]
        if hasattr(backbone, "forward_global"):
            embeddings = backbone.forward_global(inputs)
        else:
            embeddings = backbone(inputs)
        embeddings = F.normalize(embeddings, dim=-1)
        feats.append(embeddings.detach().cpu().numpy())
        labels.append(targets.detach().cpu().numpy())

    if not feats:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    features = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(labels, axis=0).astype(np.int64, copy=False)
    return features, y


def save_features(backbone, loaders, device: torch.device, out_dir: Path, tag: str) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    feature_dir = ensure_dir(Path(out_dir) / "features")
    for split_name in ("train", "val"):
        loader = loaders.get(split_name)
        if loader is None:
            continue
        X, y = extract_features(backbone, loader, device)
        np.save(feature_dir / f"{tag}_{split_name}_X.npy", X)
        np.save(feature_dir / f"{tag}_{split_name}_y.npy", y)
        paths[f"{split_name}_X"] = str(feature_dir / f"{tag}_{split_name}_X.npy")
        paths[f"{split_name}_y"] = str(feature_dir / f"{tag}_{split_name}_y.npy")
    return paths


def visualize_features_umap_pca(X: np.ndarray, y: np.ndarray, out_png: Path, labels=None) -> None:
    if X.size == 0 or y.size == 0:
        return
    emb = None
    try:  # pragma: no cover - optional dependency
        import warnings
        import umap  # type: ignore
        warnings.filterwarnings(
            "ignore",
            message="n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.",
            category=UserWarning,
            module="umap.umap_"
        )
        emb = umap.UMAP(n_components=2, random_state=1337).fit_transform(X)
    except Exception:
        try:
            from sklearn.decomposition import PCA
            emb = PCA(n_components=2, random_state=1337).fit_transform(X)
        except Exception:
            return
    if emb is None:
        return
    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    fig = plt.figure()
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=y, s=4, alpha=0.7, cmap="tab10")
    if labels is not None and len(labels) > 0:
        # create a legend with class names
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(i)), label=labels[i]) for i in range(len(labels))]
        plt.legend(handles=handles, title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _as_numpy(arr: np.ndarray) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray(arr)


def train_linear_probe_torch(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    *,
    n_epochs: int,
    lr: float,
    wd: float,
    batch_size: int,
    out_dirs: Dict[str, Path],
    model_key: str,
) -> Tuple[Dict[str, float], str]:
    Xtr = _as_numpy(Xtr).astype(np.float32, copy=False)
    Xva = _as_numpy(Xva).astype(np.float32, copy=False)
    ytr = _as_numpy(ytr).astype(np.int64, copy=False)
    yva = _as_numpy(yva).astype(np.int64, copy=False)

    if Xtr.size == 0 or Xva.size == 0:
        return {"val_acc": float("nan")}, ""

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    in_dim = Xtr.shape[1]
    classes = np.unique(np.concatenate([ytr, yva])) if ytr.size or yva.size else np.arange(Xtr.shape[1])
    num_classes = int(len(classes)) if len(classes) > 0 else 1

    head = nn.Linear(in_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    # Class weights = inverse frequency (normalized)
    _, counts = np.unique(ytr, return_counts=True)
    w = counts.max() / np.maximum(counts, 1)
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)

    def _iter_batches(X: np.ndarray, y: np.ndarray):
        N = X.shape[0]
        if N == 0:
            return
        for start in range(0, N, batch_size):
            stop = min(N, start + batch_size)
            xb = torch.from_numpy(X[start:stop]).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(y[start:stop]).to(device=device, dtype=torch.long)
            yield xb, yb

    best_acc = -math.inf
    best_path = str(prefixed(out_dirs["checkpoints"], model_key, "ssl_linear_best", "pt"))
    csv_path = prefixed(out_dirs["metrics"], model_key, "ssl_linear_timeseries", "csv")

    for epoch in range(max(1, n_epochs)):
        # train
        head.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in _iter_batches(Xtr, ytr):
            optimizer.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * yb.size(0)
            total_count += yb.size(0)

        # validation
        head.eval()
        val_loss = 0.0
        val_count = 0
        correct = 0
        preds: list[int] = []
        with torch.no_grad():
            for xb, yb in _iter_batches(Xva, yva):
                logits = head(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.detach()) * yb.size(0)
                val_count += yb.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum())
                preds.extend(pred.detach().cpu().tolist())

        train_loss = total_loss / max(1, total_count)
        val_loss_avg = val_loss / max(1, val_count)
        val_acc = correct / max(1, val_count)
        append_row_csv(
            csv_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss_avg,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0].get("lr", 0.0),
            },
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"state_dict": head.state_dict(), "in_dim": in_dim, "num_classes": num_classes},
                best_path,
            )
            if confusion_matrix is not None and val_count > 0:
                cm = confusion_matrix(yva.tolist(), preds)  # type: ignore[arg-type]
                labels = [str(cls) for cls in sorted(set(int(c) for c in classes))]
                plot_confusion(
                    cm,
                    labels,
                    prefixed(out_dirs["plots"], model_key, "ssl_linear_confusion_val", "png"),
                )

    return {"val_acc": best_acc}, best_path


# ------------------------------------------------------------------ retro-compatibility helpers
@torch.no_grad()
def extract_split(backbone: torch.nn.Module, loader, device: torch.device) -> Dict[str, np.ndarray]:
    X, y = extract_features(backbone, loader, device)
    return {"X": X, "y": y}


def save_parquet(features: np.ndarray, labels: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(features.astype("float32"))
    df["label"] = labels.astype("int32")
    df.to_parquet(out_path)
    return out_path
