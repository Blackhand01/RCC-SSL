#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
import re
import json

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("viz_ablation_dashboard")


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, class_names):
    n_classes = len(class_names)
    eps = 1e-12

    recalls = []
    f1s = []

    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())

        recall = tp / (tp + fn + eps)
        prec = tp / (tp + fp + eps)
        f1 = 2 * prec * recall / (prec + recall + eps)

        recalls.append(recall)
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    balanced_acc = float(np.mean(recalls))

    tumor_idxs = [i for i, name in enumerate(class_names) if name != "NOT_TUMOR"]
    min_recall_tumors = float(np.min([recalls[i] for i in tumor_idxs])) if tumor_idxs else float(np.min(recalls))

    out = {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "min_recall_tumors": min_recall_tumors,
    }
    for i, name in enumerate(class_names):
        out[f"recall__{name}"] = float(recalls[i])

    # Optional AUROC/PR-AUC (if sklearn available + probs not dummy)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if probs is not None and probs.shape[1] == n_classes:
            y_true_1h = np.eye(n_classes)[y_true]
            out["macro_auroc_ovr"] = float(roc_auc_score(y_true_1h, probs, average="macro", multi_class="ovr"))
            out["macro_pr_auc_ovr"] = float(average_precision_score(y_true_1h, probs, average="macro"))
        else:
            out["macro_auroc_ovr"] = None
            out["macro_pr_auc_ovr"] = None
    except Exception:
        out["macro_auroc_ovr"] = None
        out["macro_pr_auc_ovr"] = None

    return out


def make_confusion(y_true, y_pred, class_names):
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t >= 0 and p >= 0:
            cm[t, p] += 1
    return cm


TS_RE = re.compile(r"^\d{8}_\d{6}$")


def parse_ts_dirname(name: str):
    # returns sortable tuple (YYYYMMDD, HHMMSS) or None
    if TS_RE.match(name):
        d, t = name.split("_")
        return (d, t)
    return None


def discover_ablation_runs(experiment_dir: Path):
    # run dirs look like exp_ibot_abl01, exp_ibot_abl02, ...
    runs = sorted([p for p in experiment_dir.iterdir() if p.is_dir() and "abl" in p.name])
    return runs


def discover_eval_leaf(run_dir: Path):
    """
    Find candidate leaves containing predictions.csv under:
      run_dir/eval/*/*/predictions.csv
    Return the leaf directory (the one containing predictions.csv) chosen by rule:
      - prefer latest timestamp folder if timestamp present
      - otherwise prefer lexicographically last
    """
    candidates = []
    for pred in run_dir.glob("eval/*/*/predictions.csv"):
        leaf = pred.parent  # .../<timestamp>/
        ts = parse_ts_dirname(leaf.name)
        candidates.append((ts, str(leaf), leaf))

    if not candidates:
        return None

    # sort: timestamped last by actual ts; non-timestamped go first (ts=None)
    # We want "latest", so we pick max with key where None is very small.
    def keyfun(item):
        ts, _, leaf = item
        if ts is None:
            return ("00000000", "000000")  # lowest
        return ts

    best = max(candidates, key=keyfun)[2]
    return best


def load_eval_leaf(leaf: Path):
    pred_csv = leaf / "predictions.csv"
    logits_npy = leaf / "logits_test.npy"

    dfp = pd.read_csv(pred_csv)
    if "y_true" not in dfp.columns or "y_pred" not in dfp.columns:
        raise ValueError(f"{pred_csv} must contain y_true and y_pred columns")

    y_true = dfp["y_true"].to_numpy().astype(int)
    y_pred = dfp["y_pred"].to_numpy().astype(int)

    probs = None
    if logits_npy.exists():
        logits = np.load(logits_npy)
        probs = softmax(logits)

    # optional metrics json (if present)
    metrics_json = None
    mj = list(leaf.glob("metrics_*.json"))
    if mj:
        metrics_json = mj[0]

    return dfp, y_true, y_pred, probs, metrics_json


def main():
    logger = setup_logger()

    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", type=str, required=True,
                    help="Path to exp_YYYYMMDD_HHMMSS_xxx directory that contains exp_*_ablXX subfolders.")
    ap.add_argument("--class_names", type=str, required=True,
                    help="Comma-separated class names in index order.")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--primary_metric", type=str, default="macro_f1",
                    choices=["macro_f1", "balanced_accuracy", "min_recall_tumors"])
    ap.add_argument("--run_glob", type=str, default="*abl*",
                    help="Override run folder selection if needed (default: *abl*).")
    args = ap.parse_args()

    experiment_dir = Path(args.experiment_dir).resolve()
    # DEFAULT: post_processing/outputs/<experiment_name>
    if args.out_dir is None:
        this_dir = Path(__file__).resolve().parent
        out_dir = this_dir / "outputs" / experiment_dir.name
    else:
        out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = [c.strip() for c in args.class_names.split(",")]

    # discover run dirs
    run_dirs = sorted([p for p in experiment_dir.glob(args.run_glob) if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {experiment_dir} with glob={args.run_glob}")

    rows = []
    cms = {}

    for run_dir in run_dirs:
        run_name = run_dir.name
        leaf = discover_eval_leaf(run_dir)
        if leaf is None:
            logger.warning(f"[SKIP] No eval leaf found for {run_name}")
            continue

        logger.info(f"Run: {run_name} | Eval leaf: {leaf}")
        dfp, y_true, y_pred, probs, metrics_json = load_eval_leaf(leaf)

        m = compute_basic_metrics(y_true, y_pred, probs, class_names)
        m["run_name"] = run_name
        m["eval_leaf"] = str(leaf)

        # optional: parse experiment_snapshot.yaml tags later if you want
        if metrics_json is not None:
            m["metrics_json"] = str(metrics_json)
            try:
                mj = json.loads(Path(metrics_json).read_text())
                # keep a couple of common keys if present
                for k in ["accuracy", "macro_f1", "balanced_accuracy"]:
                    if k in mj and m.get(k) is None:
                        m[k] = mj[k]
            except Exception:
                pass

        rows.append(m)

        cm = make_confusion(y_true, y_pred, class_names)
        cms[run_name] = cm

    if not rows:
        raise RuntimeError("No runs had usable eval artifacts (predictions.csv).")

    dfm = pd.DataFrame(rows).sort_values(args.primary_metric, ascending=False)
    dfm.to_csv(out_dir / "ablation_summary.csv", index=False)

    best_run = dfm.iloc[0]["run_name"]
    logger.info(f"Best run by {args.primary_metric}: {best_run}")

    # Plot A: summary metrics
    bar_metrics = ["macro_f1", "balanced_accuracy", "min_recall_tumors", "macro_auroc_ovr", "macro_pr_auc_ovr"]
    cols = ["run_name"] + [c for c in bar_metrics if c in dfm.columns]
    df_bar = dfm[cols].copy()

    fig_bar = px.bar(
        df_bar.melt(id_vars=["run_name"], var_name="metric", value_name="value"),
        x="run_name", y="value", color="metric",
        barmode="group",
        title="Ablations – Summary metrics (higher is better)"
    )
    fig_bar.update_layout(xaxis_title="", yaxis_title="")
    fig_bar.write_html(out_dir / "slides_ablation_summary.html")

    # Plot B: per-class recall top-2
    top2 = dfm["run_name"].head(2).tolist()
    recall_cols = [c for c in dfm.columns if c.startswith("recall__")]
    df_rec = dfm[dfm["run_name"].isin(top2)][["run_name"] + recall_cols].copy()
    df_rec = df_rec.melt(id_vars=["run_name"], var_name="class", value_name="recall")
    df_rec["class"] = df_rec["class"].str.replace("recall__", "", regex=False)

    fig_rec = px.bar(
        df_rec, x="class", y="recall", color="run_name",
        barmode="group",
        title="Per-class recall (top-2 runs)"
    )
    fig_rec.update_layout(xaxis_title="", yaxis_title="Recall")
    fig_rec.write_html(out_dir / "slides_per_class_recall_top2.html")

    # Plot C: confusion best
    cm = cms[best_run]
    fig_cm = go.Figure(data=go.Heatmap(z=cm, x=class_names, y=class_names))
    fig_cm.update_layout(
        title=f"Confusion matrix – {best_run}",
        xaxis_title="Predicted",
        yaxis_title="True"
    )
    fig_cm.write_html(out_dir / "slides_confusion_best.html")

    logger.info(f"[OK] Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
