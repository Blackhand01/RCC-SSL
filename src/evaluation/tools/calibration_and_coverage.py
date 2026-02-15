#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ---------- helpers ----------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def load_true_labels_from_predictions_csv(pred_csv: Path) -> np.ndarray:
    """
    Try to read the true label column from predictions.csv.
    Adapt candidate names if your pipeline uses other headers.
    """
    candidates = ["y_true", "true", "label", "target", "gt", "y"]
    with pred_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"{pred_csv} has no CSV header.")
        cols = [c.strip() for c in reader.fieldnames]
        col = None
        for c in candidates:
            if c in cols:
                col = c
                break
        if col is None:
            raise RuntimeError(
                f"Cannot find true label column in {pred_csv}. "
                f"Columns found: {cols}. "
                f"Rename/add one of: {candidates}"
            )
        y = []
        for row in reader:
            y.append(int(float(row[col])))
    return np.asarray(y, dtype=np.int64)

def compute_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE top-label: conf = max prob; correct = (pred==true)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        w = mask.sum() / n
        ece += w * abs(acc_bin - conf_bin)
    return float(ece)

def plot_reliability(conf: np.ndarray, correct: np.ndarray, out_png: Path, n_bins: int = 15, title: str = "") -> None:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    accs, confs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            accs.append(np.nan); confs.append((lo+hi)/2); counts.append(0)
            continue
        accs.append(correct[mask].mean())
        confs.append(conf[mask].mean())
        counts.append(mask.sum())

    ece = compute_ece(conf, correct, n_bins=n_bins)

    plt.figure()
    xs = np.linspace(0, 1, 100)
    plt.plot(xs, xs)  # perfect calibration line
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title + f"  (ECE={ece:.3f})")
    plt.ylim(0, 1); plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def plot_risk_coverage(conf: np.ndarray, correct: np.ndarray, out_png: Path, title: str = "") -> None:
    """
    Coverage = fraction of cases accepted (higher confidence)
    Risk = 1 - accuracy on accepted cases
    """
    order = np.argsort(-conf)  # descending confidence
    conf_s = conf[order]
    corr_s = correct[order]

    n = len(conf_s)
    coverages = []
    risks = []
    accs = []

    # points at 1% granularity (you can change)
    for k in range(1, 101):
        m = max(1, int(n * (k / 100.0)))
        acc = corr_s[:m].mean()
        coverages.append(k / 100.0)
        accs.append(acc)
        risks.append(1.0 - acc)

    plt.figure()
    plt.plot(coverages, risks)
    plt.xlabel("Coverage (fraction kept)")
    plt.ylabel("Risk (1 - accuracy)")
    plt.title(title)
    plt.ylim(0, 1); plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Path tipo .../eval/<model>/<timestamp>/")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--title", type=str, default="")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Patch-level: logits + predictions.csv
    logits_path = run_dir / "logits_test.npy"
    pred_csv = run_dir / "predictions.csv"
    if not logits_path.exists():
        raise SystemExit(f"Missing {logits_path}")
    if not pred_csv.exists():
        raise SystemExit(f"Missing {pred_csv}")

    logits = np.load(logits_path)               # [N, C]
    probs = softmax(logits, axis=1)             # [N, C]
    y_true = load_true_labels_from_predictions_csv(pred_csv)

    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (y_pred == y_true).astype(np.float32)

    # Plots
    plot_reliability(
        conf, correct,
        out_png=out_dir / "reliability_patch.png",
        n_bins=args.n_bins,
        title=(args.title or "Patch-level Reliability")
    )
    plot_risk_coverage(
        conf, correct,
        out_png=out_dir / "risk_coverage_patch.png",
        title=(args.title or "Patch-level Risk–Coverage")
    )

    ece = compute_ece(conf, correct, n_bins=args.n_bins)
    (out_dir / "calibration_summary_patch.txt").write_text(
        f"N={len(conf)}\nECE={ece:.6f}\n",
        encoding="utf-8"
    )

    print(f"[OK] Wrote plots to: {out_dir}")

if __name__ == "__main__":
    main()
