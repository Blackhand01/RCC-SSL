#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn

try:  # pragma: no cover - optional dependency
    from sklearn.metrics import f1_score
except ImportError:  # pragma: no cover
    def f1_score(y_true, y_pred, average="macro"):
        return 0.0

__all__ = ["train_linear_head"]


def _load_parquet(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a parquet file produced by save_parquet and return tensors (features, labels).
    """
    df = pd.read_parquet(path)
    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    features = torch.tensor(df.drop(columns=["label"]).to_numpy(), dtype=torch.float32)
    return features, labels


def train_linear_head(
    train_pq: Path,
    val_pq: Path,
    num_classes: int,
    epochs: int = 50,
    lr: float = 1e-2,
) -> Dict[str, object]:
    """
    Train a linear head (torch.nn.Linear) on saved features.
    Returns statistics for logging and state_dict of the best model.
    """
    Xtr, Ytr = _load_parquet(train_pq)
    Xva, Yva = _load_parquet(val_pq)

    head = nn.Linear(Xtr.shape[1], num_classes)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    history: list[Dict[str, float]] = []
    best = {"epoch": -1, "val_acc": -1.0, "val_loss": float("inf"), "val_f1_macro": 0.0, "state_dict": None}

    for epoch in range(epochs):
        head.train()
        optimizer.zero_grad()
        logits = head(Xtr)
        loss = criterion(logits, Ytr)
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(Xva)
            val_loss = float(criterion(val_logits, Yva))
            val_pred = val_logits.argmax(dim=1)
            val_acc = float((val_pred == Yva).float().mean())
            y_true_np = Yva.cpu().numpy()
            y_pred_np = val_pred.cpu().numpy()
            val_f1 = float(f1_score(y_true_np, y_pred_np, average="macro")) if y_true_np.size else 0.0

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.detach()),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
            }
        )

        if val_acc > best["val_acc"]:
            best["epoch"] = epoch
            best["val_acc"] = val_acc
            best["val_loss"] = val_loss
            best["val_f1_macro"] = val_f1
            best["state_dict"] = {k: v.detach().cpu() for k, v in head.state_dict().items()}

    if best["state_dict"] is None:
        best["state_dict"] = {k: v.detach().cpu() for k, v in head.state_dict().items()}

    head.load_state_dict(best["state_dict"], strict=False)
    head.eval()
    with torch.no_grad():
        logits = head(Xva)
        val_pred = logits.argmax(dim=1)
        predictions = val_pred.cpu().numpy()
        targets = Yva.cpu().numpy()
        best["val_acc"] = float((val_pred == Yva).float().mean())
        best["val_f1_macro"] = float(f1_score(targets, predictions, average="macro")) if targets.size else 0.0

    return {
        "rows": history,
        "best": best,
        "state_dict": best["state_dict"],
        "val_targets": targets,
        "val_predictions": predictions,
    }
