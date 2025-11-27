#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch patient-level aggregation for RCC subtype classification.

- Scans --mlruns-root (experiment with exp_* or a single run).
- For each run, finds the latest eval dir containing predictions.csv.
- Aggregates at patient level, ALWAYS excluding NOT_TUMOR and using ALL patches.
- Output is written under: <run>/eval/<model>/<timestamp>/per_patient/

Aggregation:
- 'prob_sum' (default): sum per-class softmax across patches; zero-out NOT_TUMOR before argmax.
- 'vote'            : majority vote over patch predictions ignoring NOT_TUMOR.

Per run outputs (inside per_patient/):
- patient_predictions.csv
- metrics_patient.json
- cm_patient_<model>.png
- info_patient.json (counts: total patients, tumor-evaluable, skipped non_tumor_only)
Also updates a global runs_summary_patient.csv at experiment root.
"""

from __future__ import annotations
import os, sys, json, argparse, glob, csv, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------- small utils -------------------------
def _read_json(p: Path) -> dict:
    try:
        return json.load(open(p))
    except Exception:
        return {}

def _softmax_logits(x: torch.Tensor) -> torch.Tensor:
    """Numerically-stable softmax over last dim."""
    x = x - x.max(dim=1, keepdim=True).values
    return torch.exp(x) / torch.exp(x).sum(dim=1, keepdim=True)

def _mode_excluding(items, exclude_value=None):
    """Most frequent item excluding a specific value; returns None if empty after exclusion."""
    vals = [v for v in items if v != exclude_value]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]

def _plot_confmat(cm: np.ndarray, labels: List[str], out_png: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Patient-level Confusion Matrix (tumor-only)")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            plt.text(j, i, str(val), ha="center", va="center",
                     color="white" if val > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# ------------------------- core aggregation -------------------------
def aggregate_patients(
    rows: List[Dict],                 # parsed predictions.csv
    logits_np: Optional[np.ndarray],  # (N, C) or None
    class_names: List[str],
    *,
    method: str = "prob_sum",         # "prob_sum" (default) or "vote"
) -> Tuple[List[Dict], Dict]:
    """
    Always exclude NOT_TUMOR and use ALL patches.

    Patient GT rule:
      - If any tumor labels exist among patch GTs: patient GT = mode over tumor-only labels.
      - If NO tumor labels (all NOT_TUMOR): mark as non_tumor_only → y_true_patient = -1 (excluded from metrics).

    Patient prediction:
      - prob_sum: sum softmax across patches; set NOT_TUMOR score=0; argmax over tumor classes.
      - vote   : majority vote on patch predictions, ignoring NOT_TUMOR; fallback to overall mode if empty.
    """
    n_classes = len(class_names)
    if "NOT_TUMOR" not in class_names:
        raise ValueError("Class 'NOT_TUMOR' is required in class_names.")
    excl_id = class_names.index("NOT_TUMOR")

    # Group patch indices by patient
    by_pat: Dict[str, List[int]] = defaultdict(list)
    for idx, r in enumerate(rows):
        pid = r.get("patient_id")
        if pid is not None and pid != "":
            by_pat[pid].append(idx)

    # Precompute probabilities for prob_sum
    probs = None
    if logits_np is not None and method == "prob_sum":
        t = torch.from_numpy(logits_np)
        probs = _softmax_logits(t).numpy()

    patient_rows: List[Dict] = []
    y_true_pat: List[int] = []
    y_pred_pat: List[int] = []

    # tumor-only mapping for metrics/plots
    keep_idx = [i for i, n in enumerate(class_names) if n != "NOT_TUMOR"]
    keep_labels = [class_names[i] for i in keep_idx]
    idx_remap = {c: i for i, c in enumerate(keep_idx)}

    n_total_pat, n_tumor_eval, n_non_tumor_only = 0, 0, 0

    for pid, idxs in by_pat.items():
        n_total_pat += 1

        # --------- ground-truth (tumor-only mode) ----------
        y_true_items = [int(rows[i]["y_true"]) for i in idxs]
        gt_pat = _mode_excluding(y_true_items, exclude_value=excl_id)
        if gt_pat is None:
            gt_status = "non_tumor_only"
            n_non_tumor_only += 1
        else:
            gt_status = "tumor"
            n_tumor_eval += 1

        # --------- prediction ----------
        if method == "prob_sum" and probs is not None:
            score = np.zeros((n_classes,), dtype=np.float64)
            for i in idxs:  # ALL patches
                vec = probs[i].copy()
                vec[excl_id] = 0.0  # zero-out NOT_TUMOR
                score += vec
            score[excl_id] = -1.0  # make sure NOT_TUMOR cannot be chosen
            pred_pat = int(np.argmax(score))
            tumor_mass = score[score >= 0].sum() if score[score >= 0].size else 1.0
            confidence = float(score[pred_pat] / max(tumor_mass, 1e-9))
            support = {class_names[c]: float(score[c]) for c in range(n_classes) if c != excl_id}
        else:
            votes = [int(rows[i]["y_pred"]) for i in idxs if int(rows[i]["y_pred"]) != excl_id]
            if votes:
                pred_pat = Counter(votes).most_common(1)[0][0]
            else:
                # fall back to mode over all predictions (including NOT_TUMOR) or first
                pred_pat = _mode_excluding([int(rows[i]["y_pred"]) for i in idxs], exclude_value=None)
                if pred_pat is None:
                    pred_pat = int(rows[idxs[0]]["y_pred"])
            confidence = 1.0
            support = {}

        patient_rows.append({
            "patient_id": pid,
            "gt_status": gt_status,  # "tumor" | "non_tumor_only"
            "y_true_patient": (int(gt_pat) if gt_pat is not None else -1),
            "y_pred_patient": int(pred_pat),
            "n_patches": int(len(idxs)),
            "n_used_patches": int(len(idxs)),  # all patches
            "confidence": confidence,
            "support_sum_by_class": support,
        })

        if gt_pat is not None:
            y_true_pat.append(int(gt_pat))
            y_pred_pat.append(int(pred_pat))

    # ----- Metrics @ patient-level (tumor-only) -----
    metrics: Dict = {
        "n_patients_total": int(n_total_pat),
        "n_patients_tumor_eval": int(n_tumor_eval),
        "n_patients_non_tumor_only_skipped": int(n_non_tumor_only),
    }

    if y_true_pat:
        y_true_arr = np.asarray(y_true_pat, dtype=int)
        y_pred_arr = np.asarray(y_pred_pat, dtype=int)

        metrics.update({
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
            "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
        })

        # Tumor-only confusion matrix + classification report
        yt = np.array([idx_remap[y] for y in y_true_arr if y in idx_remap], dtype=int)
        yp = np.array([idx_remap[y] for y in y_pred_arr if y in idx_remap], dtype=int)
        if yt.size and yp.size:
            cm = confusion_matrix(yt, yp, labels=list(range(len(keep_idx))))
            metrics["_cm"] = cm.tolist()
            metrics["_labels"] = keep_labels
            report = classification_report(yt, yp, target_names=keep_labels, output_dict=True, zero_division=0)
            metrics["_report"] = report

        # AUC/AUPRC and Top-2 (only for prob_sum with logits)
        if method == "prob_sum" and logits_np is not None:
            tumor_scores = []
            tumor_targets = []
            for r in patient_rows:
                gt = r["y_true_patient"]
                if gt < 0 or gt == class_names.index("NOT_TUMOR"):
                    continue
                vec = np.zeros((len(keep_idx),), dtype=np.float64)
                for k, v in r["support_sum_by_class"].items():
                    if k != "NOT_TUMOR":
                        j = keep_labels.index(k)
                        vec[j] = float(v)
                s = vec / max(vec.sum(), 1e-9)
                tumor_scores.append(s)
                if gt in keep_idx:
                    tumor_targets.append(keep_idx.index(gt))
            if tumor_scores and tumor_targets and (len(tumor_scores) == len(tumor_targets)):
                S = np.vstack(tumor_scores)
                yb = label_binarize(np.asarray(tumor_targets), classes=list(range(len(keep_idx))))
                try:
                    metrics["macro_auc_ovr"] = float(roc_auc_score(yb, S, average="macro", multi_class="ovr"))
                    metrics["macro_auprc"] = float(average_precision_score(yb, S, average="macro"))
                except Exception:
                    pass
                # top-2 accuracy
                top2_correct = sum(tgt in np.argsort(svec)[-2:] for svec, tgt in zip(S, tumor_targets))
                if len(tumor_targets) > 0:
                    metrics["top2_accuracy"] = float(top2_correct / len(tumor_targets))

    return patient_rows, metrics


# ------------------------- discovery & I/O -------------------------
def _is_run_dir(p: Path) -> bool:
    return p.is_dir() and (p / "metrics" / "final_metrics.json").is_file()

def _discover_runs(root: Path) -> List[Path]:
    if _is_run_dir(root):
        return [root]
    cands = [d for d in root.glob("exp_*") if _is_run_dir(d)]
    if not cands:
        cands = [d for d in root.rglob("exp_*") if _is_run_dir(d)]
    return sorted(cands)

def _find_latest_eval_dir(run_dir: Path) -> Optional[Path]:
    eval_root = run_dir / "eval"
    if not eval_root.is_dir():
        return None
    hits = sorted(eval_root.glob("*/*/predictions.csv"))
    if not hits:
        hits = sorted(eval_root.glob("*/predictions.csv"))
    if not hits:
        return None
    hits.sort(key=lambda p: p.parent.stat().st_mtime, reverse=True)
    return hits[0].parent  # .../eval/<model>/<timestamp>

def _load_eval_artifacts(eval_dir: Path) -> Tuple[List[Dict], Optional[np.ndarray], List[str], str]:
    pred_csv = eval_dir / "predictions.csv"
    logits_npy = eval_dir / "logits_test.npy"
    metrics_json = None
    for cand in eval_dir.glob("metrics_*.json"):
        metrics_json = cand
        break

    # parse predictions.csv (no pandas)
    rows: List[Dict] = []
    with open(pred_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "patient_id": r.get("patient_id"),
                    "y_true": int(r.get("y_true", -1)),
                    "y_pred": int(r.get("y_pred", -1)),
                })
            except Exception:
                continue

    logits_np = None
    if logits_npy.is_file():
        try:
            logits_np = np.load(logits_npy)
        except Exception:
            logits_np = None

    class_names = ["ccRCC","pRCC","CHROMO","ONCO","NOT_TUMOR"]
    model_name = eval_dir.parent.name if eval_dir.parent else "ssl_linear_best"
    if metrics_json and metrics_json.is_file():
        meta = _read_json(metrics_json)
        cn = meta.get("class_names")
        if isinstance(cn, list) and all(isinstance(x, str) for x in cn):
            class_names = cn
        m = meta.get("model", {})
        if isinstance(m, dict):
            maybe = m.get("name")
            if isinstance(maybe, str) and maybe:
                model_name = maybe

    return rows, logits_np, class_names, model_name


# ------------------------- summary I/O -------------------------
def _append_run_summary(summary_csv: Path, run_dir: Path, model_name: str, metrics: Dict) -> None:
    header = [
        "timestamp","run_dir","model_name",
        "n_patients_total","n_patients_tumor_eval","n_patients_non_tumor_only_skipped",
        "accuracy","balanced_accuracy","macro_f1","macro_auc_ovr","macro_auprc","top2_accuracy"
    ]
    row = {
        "timestamp": int(time.time()),
        "run_dir": str(run_dir),
        "model_name": model_name,
        "n_patients_total": metrics.get("n_patients_total",""),
        "n_patients_tumor_eval": metrics.get("n_patients_tumor_eval",""),
        "n_patients_non_tumor_only_skipped": metrics.get("n_patients_non_tumor_only_skipped",""),
        "accuracy": metrics.get("accuracy",""),
        "balanced_accuracy": metrics.get("balanced_accuracy",""),
        "macro_f1": metrics.get("macro_f1",""),
        "macro_auc_ovr": metrics.get("macro_auc_ovr",""),
        "macro_auprc": metrics.get("macro_auprc",""),
        "top2_accuracy": metrics.get("top2_accuracy",""),
    }
    exists = summary_csv.is_file()
    with open(summary_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Aggregate patch-level eval to patient-level across multiple runs.")
    ap.add_argument("--mlruns-root", required=True, help="Experiment folder (with exp_*) or a single run folder")
    ap.add_argument("--method", choices=["prob_sum","vote"], default="prob_sum",
                    help="Aggregation method. NOT_TUMOR always excluded; ALL patches used.")
    args = ap.parse_args()

    root = Path(args.mlruns_root).resolve()
    runs = _discover_runs(root)
    if not runs:
        raise SystemExit(f"No runs found under: {root}")

    # global summary at experiment root (or parent if single run)
    summary_csv = (root / "runs_summary_patient.csv") if not _is_run_dir(root) else (root.parent / "runs_summary_patient.csv")

    for run in runs:
        eval_dir = _find_latest_eval_dir(run)
        if eval_dir is None:
            print(f"[WARN] No eval with predictions.csv for run: {run}")
            continue

        # ---- load artifacts from the eval dir ----
        rows, logits_np, class_names, model_name = _load_eval_artifacts(eval_dir)

        # ---- aggregate ----
        patients, metrics = aggregate_patients(rows, logits_np, class_names, method=args.method)

        # ---- write into per_patient/ subfolder ----
        per_dir = eval_dir / "per_patient"
        per_dir.mkdir(parents=True, exist_ok=True)

        # 1) patient_predictions.csv (add gt_status)
        pp_csv = per_dir / "patient_predictions.csv"
        with open(pp_csv, "w", newline="") as f:
            fn = ["patient_id","gt_status","y_true_patient","y_pred_patient","n_patches","n_used_patches","confidence","support_sum_by_class"]
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            for r in patients:
                w.writerow(r)

        # 2) metrics_patient.json
        mp_json = per_dir / "metrics_patient.json"
        with open(mp_json, "w") as f:
            json.dump({
                "class_names": class_names,
                "method": args.method,
                "exclude": "NOT_TUMOR (fixed)",
                "metrics": metrics
            }, f, indent=2)

        # 3) confusion matrix plot (tumor-only)
        if metrics.get("_cm") and metrics.get("_labels"):
            cm = np.array(metrics["_cm"])
            labels = metrics["_labels"]
            _plot_confmat(cm, labels, per_dir / f"cm_patient_{model_name}.png")

        # 4) info file with counts
        info_json = per_dir / "info_patient.json"
        with open(info_json, "w") as f:
            json.dump({
                "run_dir": str(run),
                "eval_dir": str(eval_dir),
                "per_patient_dir": str(per_dir),
                "counts": {
                    "n_patients_total": metrics.get("n_patients_total", 0),
                    "n_patients_tumor_eval": metrics.get("n_patients_tumor_eval", 0),
                    "n_patients_non_tumor_only_skipped": metrics.get("n_patients_non_tumor_only_skipped", 0),
                }
            }, f, indent=2)

        # 5) append run summary (at experiment root)
        try:
            _append_run_summary(summary_csv, run, model_name, metrics)
        except Exception as e:
            print(f"[WARN] Could not append summary: {e}")

        print(f"[OK] {run.name}: patient aggregation → {per_dir}")

    print(f"[DONE] Updated summary: {summary_csv}")


if __name__ == "__main__":
    main()
