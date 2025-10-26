# utils/probe.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
except ImportError:  # pragma: no cover
    def accuracy_score(y_true, y_pred):
        return 0.0

    def f1_score(y_true, y_pred, average="macro"):
        return 0.0

    def confusion_matrix(y_true, y_pred):
        return [[0]]

from .data import class_labels_from_cfg
from .io import append_row_csv, dump_json, prefixed
from .viz import plot_acc, plot_confusion, write_derived_csv


def extract_features(backbone, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    backbone.eval().to(device)
    feats: List[np.ndarray] = []
    labs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            targets = batch["targets"].cpu().numpy()
            normalized = F.normalize(backbone.forward_global(inputs), dim=-1)
            feats.append(normalized.cpu().numpy())
            labs.append(targets)
    if not feats:
        dim = getattr(backbone, "output_dim", 0)
        return np.zeros((0, dim)), np.zeros((0,))
    return np.concatenate(feats, axis=0), np.concatenate(labs, axis=0)


def _fit_logreg(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    val_feats: np.ndarray,
    val_labels: np.ndarray,
    C: float = 1.0,
    n_jobs: int = 4,
    seed: int = 1337,
):
    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            max_iter=2000,
            n_jobs=n_jobs,
            C=C,
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(train_feats, train_labels)
        pred = clf.predict(val_feats)
        metrics = {
            "val_acc": float(accuracy_score(val_labels, pred)),
            "val_f1_macro": float(f1_score(val_labels, pred, average="macro")),
        }
        return clf, metrics, (val_labels, pred)
    except ImportError:
        zeros = np.zeros(len(val_labels))
        return None, {"val_acc": 0.0, "val_f1_macro": 0.0}, (val_labels, zeros)


def _run_knn(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    val_feats: np.ndarray,
    val_labels: np.ndarray,
    k: int = 20,
    n_jobs: int = 4,
):
    try:
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=n_jobs, weights="distance")
        knn.fit(train_feats, train_labels)
        pred = knn.predict(val_feats)
        metrics = {
            "val_acc": float(accuracy_score(val_labels, pred)),
            "val_f1_macro": float(f1_score(val_labels, pred, average="macro")),
        }
        return metrics, (val_labels, pred)
    except ImportError:
        zeros = np.zeros(len(val_labels))
        return {"val_acc": 0.0, "val_f1_macro": 0.0}, (val_labels, zeros)


def _dump_sklearn(clf, run_dirs, model_key: str):
    try:
        import joblib

        path = prefixed(run_dirs["artifacts"], model_key, "best_classifier_sklearn", "joblib")
        joblib.dump(clf, path)
        return path
    except Exception:
        return None


def _dump_torch_linear_from_logreg(clf, in_dim: int, n_classes: int, run_dirs, model_key: str):
    head = nn.Linear(in_dim, n_classes)
    weight = torch.as_tensor(clf.coef_, dtype=head.weight.dtype)
    bias = torch.as_tensor(clf.intercept_, dtype=head.bias.dtype)
    head.weight.data.copy_(weight)
    head.bias.data.copy_(bias)
    path = prefixed(run_dirs["artifacts"], model_key, "best_classifier", "pt")
    torch.save(head.state_dict(), path)
    dump_json(
        prefixed(run_dirs["artifacts"], model_key, "best_classifier_meta", "json"),
        {
            "type": "linear",
            "in_dim": int(in_dim),
            "num_classes": int(n_classes),
            "source": "sklearn.LogisticRegression",
        },
    )
    return path


def _log_probe_outputs(
    run_dirs,
    model_key: str,
    class_labels: List[str],
    linear_metrics: Optional[Dict[str, float]],
    linear_conf: Optional[Tuple[np.ndarray, np.ndarray]],
    knn_metrics: Optional[Dict[str, float]],
    knn_conf: Optional[Tuple[np.ndarray, np.ndarray]],
) -> None:
    csv_path = prefixed(run_dirs["metrics"], model_key, "ssl_probe_timeseries", "csv")
    row = {
        "epoch": 0,
        "lr": 0.0,
        "train_loss": float("nan"),
        "val_loss": float("nan"),
        "val_acc": float("nan"),
        "val_f1_macro": float("nan"),
    }
    if linear_metrics:
        row["val_acc"] = linear_metrics["val_acc"]
        row["val_f1_macro"] = linear_metrics["val_f1_macro"]
    append_row_csv(csv_path, row)

    derived_csv = prefixed(run_dirs["plots"], model_key, "ssl_probe_timeseries_derived", "csv")
    write_derived_csv(csv_path, derived_csv)
    if linear_metrics:
        plot_acc(derived_csv, prefixed(run_dirs["figures"], model_key, "ssl_probe_acc", "png"), model_key)
        if linear_conf:
            y_true, y_pred = linear_conf
            cm = np.asarray(confusion_matrix(y_true, y_pred))
            labels = class_labels or list(range(cm.shape[0]))
            plot_confusion(cm, labels, prefixed(run_dirs["figures"], model_key, "ssl_probe_linear_confusion_val", "png"))
    if knn_conf:
        y_true, y_pred = knn_conf
        cm = np.asarray(confusion_matrix(y_true, y_pred))
        labels = class_labels or list(range(cm.shape[0]))
        plot_confusion(cm, labels, prefixed(run_dirs["figures"], model_key, "ssl_probe_knn_confusion_val", "png"))

    payload: Dict[str, Any] = {"best_epoch": 0}
    if linear_metrics:
        payload["linear"] = linear_metrics
    if knn_metrics:
        payload["knn"] = knn_metrics
    dump_json(prefixed(run_dirs["metrics"], model_key, "ssl_probe_final_metrics", "json"), payload)


def _resolve_probe_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    resolved = (cfg.get("train", {}).get("ssl", {}) or {}).get("probe", {}) or {}
    if "enabled" in resolved and "do_linear_probe" not in resolved:
        resolved["do_linear_probe"] = bool(resolved["enabled"])
    resolved.setdefault("do_linear_probe", True)
    resolved.setdefault("do_knn", False)
    return resolved


def fit_probe_and_log(
    backbone,
    loaders,
    run_dirs,
    model_key: str,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    probe_cfg = _resolve_probe_cfg(cfg)
    if not (probe_cfg.get("do_linear_probe", True) or probe_cfg.get("do_knn", False)):
        return {}

    train_feats, train_labels = extract_features(backbone, loaders["train"], device)
    val_feats, val_labels = extract_features(backbone, loaders["val"], device)
    if train_feats.size == 0 or val_feats.size == 0:
        return {}

    results: Dict[str, float] = {}
    linear_metrics: Optional[Dict[str, float]] = None
    linear_conf: Optional[Tuple[np.ndarray, np.ndarray]] = None

    if probe_cfg.get("do_linear_probe", True):
        C = float(probe_cfg.get("C", 1.0))
        n_jobs = int(probe_cfg.get("n_jobs", 4))
        seed = int(cfg.get("experiment", {}).get("seed", 1337))
        clf, metrics, (y_true, y_pred) = _fit_logreg(
            train_feats,
            train_labels,
            val_feats,
            val_labels,
            C=C,
            n_jobs=n_jobs,
            seed=seed,
        )
        if clf is not None and train_feats.shape[1] > 0:
            _dump_sklearn(clf, run_dirs, model_key)
            _dump_torch_linear_from_logreg(
                clf,
                in_dim=train_feats.shape[1],
                n_classes=int(len(np.unique(np.concatenate([train_labels, val_labels])))),
                run_dirs=run_dirs,
                model_key=model_key,
            )
        linear_metrics = metrics
        linear_conf = (y_true, y_pred)
        results["probe_linear_val_acc"] = metrics["val_acc"]
        results["probe_linear_val_f1_macro"] = metrics["val_f1_macro"]

    knn_metrics: Optional[Dict[str, float]] = None
    knn_conf: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if probe_cfg.get("do_knn", False):
        k = int(probe_cfg.get("k", 20))
        n_jobs = int(probe_cfg.get("knn_n_jobs", probe_cfg.get("n_jobs", 4)))
        metrics, (y_true, y_pred) = _run_knn(
            train_feats,
            train_labels,
            val_feats,
            val_labels,
            k=k,
            n_jobs=n_jobs,
        )
        knn_metrics = metrics
        knn_conf = (y_true, y_pred)
        results["probe_knn_val_acc"] = metrics["val_acc"]
        results["probe_knn_val_f1_macro"] = metrics["val_f1_macro"]

    _log_probe_outputs(run_dirs, model_key, class_labels_from_cfg(cfg), linear_metrics, linear_conf, knn_metrics, knn_conf)
    return results
