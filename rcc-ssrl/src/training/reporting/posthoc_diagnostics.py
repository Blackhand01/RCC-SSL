#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


KNOWN_MODELS = ("moco_v3", "dino_v3", "ibot", "i_jepa")


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc diagnostics for SSL ablations: "
            "build training and evaluation CSV reports."
        )
    )
    parser.add_argument(
        "--exp-root",
        type=str,
        required=True,
        help=(
            "Experiment root path, e.g. "
            "/beegfs-scratch/.../outputs/mlruns/exp_20251109_181540_moco_v3."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=KNOWN_MODELS,
        default=None,
        help="Optional model name override (moco_v3, dino_v3, ibot, i_jepa).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write any file, only print a short summary.",
    )
    return parser.parse_args()


def infer_model_name(exp_root: Path, explicit: Optional[str]) -> str:
    """Infer model name from explicit CLI argument or from exp_root name."""
    if explicit is not None:
        return explicit

    name = exp_root.name
    for m in KNOWN_MODELS:
        if m in name:
            return m

    raise ValueError(
        f"Could not infer model name from experiment root '{name}'. "
        f"Supported models: {', '.join(KNOWN_MODELS)}. Use --model to override."
    )


def infer_experiment_token(exp_root: Path, model_name: str) -> str:
    """
    Infer experiment datetime token from exp_root name.

    Example:
        exp_root.name = 'exp_20251109_181540_moco_v3'
        model_name = 'moco_v3'
        -> '20251109_181540'
    """
    name = exp_root.name
    core = name[4:] if name.startswith("exp_") else name
    suffix = f"_{model_name}"
    if core.endswith(suffix):
        return core[: -len(suffix)]
    return core


def discover_run_dirs(exp_root: Path) -> List[Path]:
    """
    Discover ablation run directories under exp_root.

    Assumption: ablation runs are direct children of exp_root with names:
      'exp_*_abl*'  (e.g., exp_moco_v3_abl01, exp_dino_v3_abl12, ...)
    """
    run_dirs: List[Path] = []
    for child in exp_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("exp_") and "_abl" in name:
            run_dirs.append(child)
    return sorted(run_dirs, key=lambda p: p.name)


# ---------- Training diagnostics: loss series + helpers ----------


def find_ssl_timeseries_file(metrics_dir: Path, model_name: str) -> Optional[Path]:
    """Find the SSL timeseries CSV file for the given model."""
    primary = metrics_dir / f"{model_name}__ssl_timeseries.csv"
    if primary.exists():
        return primary

    # Fallback: any file ending with 'ssl_timeseries.csv'
    candidates = list(metrics_dir.glob("*ssl_timeseries.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Choose the lexicographically last one as "latest"
        return sorted(candidates)[-1]
    return None


def load_ssl_loss_series(model_name: str, csv_path: Path) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Load SSL loss series for a given model, normalizing to a single scalar per step.

    Returns:
        loss_series: pd.Series of floats (L_t)
        df: original DataFrame (for n_epochs, etc.)
    """
    df = pd.read_csv(csv_path)

    # Model-specific logic to build a scalar loss per step.
    if model_name == "moco_v3":
        # Simple contrastive loss, already scalar.
        for candidate in ("loss_ssl", "ssl_loss", "loss"):
            if candidate in df.columns:
                loss = df[candidate]
                break
        else:
            loss_cols = [c for c in df.columns if "loss" in c.lower()]
            if not loss_cols:
                raise ValueError(f"No loss column found in {csv_path}")
            loss = df[loss_cols[0]]

    elif model_name == "dino_v3":
        # DINO v3: loss = loss_global + loss_local (or sum of loss_g*, loss_l*).
        if "loss" in df.columns:
            loss = df["loss"]
        else:
            cols = [
                c
                for c in df.columns
                if c.startswith("loss_g")
                or c.startswith("loss_l")
                or c in ("loss_global", "loss_local")
            ]
            if not cols:
                cols = [c for c in df.columns if c.lower().startswith("loss")]
            if not cols:
                raise ValueError(f"No DINO-style loss columns found in {csv_path}")
            loss = df[cols].sum(axis=1)

    elif model_name == "ibot":
        # iBOT: ssl_loss = loss_patch + loss_token (+ loss_global if present).
        cols = [c for c in ("loss_patch", "loss_token", "loss_global") if c in df.columns]
        if cols:
            loss = df[cols].sum(axis=1)
        else:
            cols = [c for c in df.columns if c.lower().startswith("loss")]
            if not cols:
                raise ValueError(f"No iBOT-style loss columns found in {csv_path}")
            loss = df[cols].sum(axis=1) if len(cols) > 1 else df[cols[0]]

    elif model_name == "i_jepa":
        # i-JEPA: L2 loss, either 'loss' or mean of 'loss_target_*'.
        if "loss" in df.columns:
            loss = df["loss"]
        else:
            target_cols = [c for c in df.columns if c.startswith("loss_target")]
            if target_cols:
                loss = df[target_cols].mean(axis=1)
            else:
                cols = [c for c in df.columns if c.lower().startswith("loss")]
                if not cols:
                    raise ValueError(f"No i-JEPA-style loss columns found in {csv_path}")
                loss = df[cols[0]]
    else:
        # Generic fallback: pick the first 'loss*' column.
        cols = [c for c in df.columns if "loss" in c.lower()]
        if not cols:
            raise ValueError(f"No loss column found in {csv_path}")
        loss = df[cols[0]]

    loss = pd.to_numeric(loss, errors="coerce")
    return loss, df


def compute_loss_trend(loss: pd.Series) -> Dict[str, Any]:
    """
    Compute loss trend indicators:
      - pct_drop
      - slope_last (linear regression over last K steps)
      - spikes (>3σ)
      - plateau
      - collapse
      - trend label and a short comment
    """
    values = loss.to_numpy(dtype=float)
    finite_mask = np.isfinite(values)
    values = values[finite_mask]

    if values.size < 2:
        return {
            "trend": "collapse" if values.size == 0 else "plateau",
            "pct_drop": 0.0,
            "slope_last": 0.0,
            "spikes": 0,
            "plateau": True,
            "collapse": values.size == 0,
            "comment": "insufficient data to estimate trend",
        }

    L0 = float(values[0])
    L_last = float(values[-1])
    denom0 = max(abs(L0), 1e-8)
    pct_drop = (L0 - L_last) / denom0 * 100.0

    # Linear regression on last K steps
    K = min(20, values.size)
    y = values[-K:]
    x = np.arange(K, dtype=float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(np.sum((x - x_mean) ** 2))
    if denom > 0:
        slope_last = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    else:
        slope_last = 0.0

    # Spikes: deviations > 3 sigma from global mean
    mu = float(values.mean())
    sigma = float(values.std(ddof=0))
    if sigma > 0:
        z = np.abs(values - mu) / sigma
        spikes = int(np.sum(z > 3.0))
    else:
        spikes = 0

    plateau = abs(slope_last) < 1e-3 and abs(pct_drop) < 1.0
    collapse = (not np.all(np.isfinite(values))) or (L_last > L0 * 1.2)

    if collapse:
        trend_label = "collapse"
    elif plateau:
        trend_label = "plateau"
    elif pct_drop > 0:
        if spikes > 0 and pct_drop < 3.0:
            trend_label = "unstable"
        else:
            trend_label = "decreasing"
    else:
        trend_label = "unstable"

    comments: List[str] = []
    comments.append(f"loss drop ≈{pct_drop:.1f}% from start to end")
    if spikes > 0:
        comments.append(f"{spikes} spike(s) >3σ")
    if plateau:
        comments.append("plateau on last part")
    if collapse:
        comments.append("possible collapse (loss increasing or NaN/inf)")

    return {
        "trend": trend_label,
        "pct_drop": float(pct_drop),
        "slope_last": float(slope_last),
        "spikes": int(spikes),
        "plateau": bool(plateau),
        "collapse": bool(collapse),
        "comment": "; ".join(comments),
    }


def load_final_metrics(metrics_dir: Path) -> Dict[str, Any]:
    """Load final_metrics.json if present, else return empty dict."""
    final_path = metrics_dir / "final_metrics.json"
    if not final_path.exists():
        return {}
    with open(final_path, "r") as f:
        return json.load(f)


def analyze_training_run(run_dir: Path, model_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Analyze one ablation run for training:
      - summary row (high-level training info)
      - loss trend row (loss dynamics)
    """
    metrics_dir = run_dir / "metrics"
    timeseries_path = find_ssl_timeseries_file(metrics_dir, model_name)
    if timeseries_path is None:
        raise FileNotFoundError(f"No SSL timeseries CSV found under {metrics_dir}")

    loss_series, df_ts = load_ssl_loss_series(model_name, timeseries_path)
    n_steps = int(loss_series.shape[0])

    if "epoch" in df_ts.columns:
        n_epochs = int(df_ts["epoch"].max()) + 1
    else:
        n_epochs = None

    final_metrics = load_final_metrics(metrics_dir)
    ssl_best_epoch = final_metrics.get("ssl_best_epoch", None)
    probe_val_acc = (
        final_metrics.get("probe_linear_val_acc", None)
        if "probe_linear_val_acc" in final_metrics
        else final_metrics.get("probe_val_acc", None)
    )

    trend_info = compute_loss_trend(loss_series)

    summary_row: Dict[str, Any] = {
        "ablation_name": run_dir.name,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "ssl_best_epoch": ssl_best_epoch,
        "probe_val_acc": probe_val_acc,
    }

    trend_row: Dict[str, Any] = {
        "ablation_name": run_dir.name,
        "trend": trend_info["trend"],
        "pct_drop": trend_info["pct_drop"],
        "slope_last": trend_info["slope_last"],
        "spikes": trend_info["spikes"],
        "plateau": trend_info["plateau"],
        "collapse": trend_info["collapse"],
        "comment": trend_info["comment"],
    }

    return summary_row, trend_row


# ---------- Evaluation diagnostics: predictions + helpers ----------


def find_eval_dir_for_run(run_dir: Path, model_name: str) -> Optional[Path]:
    """
    Find the eval directory for a given run and model.

    Expected pattern:
      <run_dir>/eval/<model_name>_ssl_linear_best/<timestamp>/
    """
    model_eval_root = run_dir / "eval" / f"{model_name}_ssl_linear_best"
    if not model_eval_root.exists():
        return None

    # Prefer timestamp-named directories inside model_eval_root
    subdirs = [d for d in model_eval_root.iterdir() if d.is_dir()]
    if not subdirs:
        # Maybe predictions/report are directly inside model_eval_root
        return model_eval_root if model_eval_root.exists() else None

    # Use the lexicographically last folder as "latest" eval
    return sorted(subdirs)[-1]


def load_report_per_class(report_path: Path) -> Dict[str, Any]:
    """Load per-class metrics JSON produced by evaluation."""
    with open(report_path, "r") as f:
        return json.load(f)


def load_predictions(pred_path: Path) -> pd.DataFrame:
    """
    Load predictions CSV with required columns:
      wds_key, patient_id, slide_id, y_true, y_pred
    """
    df = pd.read_csv(pred_path)
    required_cols = {"wds_key", "patient_id", "slide_id", "y_true", "y_pred"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions.csv: {missing}")
    return df


def compute_global_stats(df: pd.DataFrame, report_per_class: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute global patch-level statistics and enrich with JSON aggregates.
    """
    n_patches = int(len(df))
    n_patients = int(df["patient_id"].nunique())
    n_slides = int(df["slide_id"].nunique())

    accuracy_from_df = float((df["y_true"] == df["y_pred"]).mean())
    accuracy_json = report_per_class.get("accuracy", None)
    macro_avg = report_per_class.get("macro avg", {})
    weighted_avg = report_per_class.get("weighted avg", {})

    return {
        "n_patches": n_patches,
        "n_patients": n_patients,
        "n_slides": n_slides,
        "accuracy": accuracy_from_df,
        "accuracy_json": float(accuracy_json) if accuracy_json is not None else None,
        "macro_avg": {
            "precision": macro_avg.get("precision", None),
            "recall": macro_avg.get("recall", None),
            "f1": macro_avg.get("f1-score", None),
        },
        "weighted_avg": {
            "precision": weighted_avg.get("precision", None),
            "recall": weighted_avg.get("recall", None),
            "f1": weighted_avg.get("f1-score", None),
        },
    }


def build_notes_from_per_class(report_per_class: Dict[str, Any]) -> List[str]:
    """
    Generate human-readable notes from per-class metrics,
    highlighting obvious failures and asymmetries.
    """
    notes: List[str] = []
    class_names = [
        k
        for k in report_per_class.keys()
        if k not in {"accuracy", "macro avg", "weighted avg"}
    ]

    for cls in class_names:
        cls_metrics = report_per_class[cls]
        prec = float(cls_metrics.get("precision", 0.0) or 0.0)
        rec = float(cls_metrics.get("recall", 0.0) or 0.0)
        f1 = float(cls_metrics.get("f1-score", 0.0) or 0.0)
        support = float(cls_metrics.get("support", 0.0) or 0.0)

        if rec == 0.0 and support > 0:
            notes.append(
                f"class '{cls}' has recall=0.0 (never correctly predicted, complete failure)."
            )
        elif f1 < 0.3 and support > 0:
            notes.append(
                f"class '{cls}' has very low f1≈{f1:.3f} with support={support:.0f}."
            )

        if rec > 0.95 and prec < 0.6:
            notes.append(
                f"class '{cls}' has recall≈{rec:.3f} and precision≈{prec:.3f} "
                f"(model over-calls this class)."
            )

        if prec > 0.8 and rec < 0.4:
            notes.append(
                f"class '{cls}' has precision≈{prec:.3f} and recall≈{rec:.3f} "
                f"(model too conservative for this class)."
            )

    return notes


def per_class_short_dict(report_per_class: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact {class: {precision, recall, f1, support}} dictionary."""
    out: Dict[str, Any] = {}
    for cls, metrics in report_per_class.items():
        if cls in {"accuracy", "macro avg", "weighted avg"}:
            continue
        out[cls] = {
            "precision": metrics.get("precision", None),
            "recall": metrics.get("recall", None),
            "f1": metrics.get("f1-score", None),
            "support": metrics.get("support", None),
        }
    return out


def analyze_eval_run(
    exp_root: Path, run_dir: Path, model_name: str
) -> Optional[Dict[str, Any]]:
    """
    Build a compact evaluation summary for one ablation run.

    Returns an object with:
      - ablation_name
      - paths (predictions_csv, report_per_class_json)
      - eval_summary (global stats, per-class short metrics, notes)
    or None if no evaluation artefacts are found.
    """
    eval_dir = find_eval_dir_for_run(run_dir, model_name)
    if eval_dir is None:
        return None

    pred_path = eval_dir / "predictions.csv"
    report_path = eval_dir / "report_per_class.json"

    if not pred_path.exists() or not report_path.exists():
        return None

    df_pred = load_predictions(pred_path)
    report_per_class = load_report_per_class(report_path)
    global_stats = compute_global_stats(df_pred, report_per_class)
    notes = build_notes_from_per_class(report_per_class)
    per_class = per_class_short_dict(report_per_class)

    return {
        "run_id": run_dir.name,
        "ablation_name": run_dir.name,
        "paths": {
            "predictions_csv": str(pred_path.relative_to(exp_root)),
            "report_per_class_json": str(report_path.relative_to(exp_root)),
        },
        "eval_summary": {
            "global": global_stats,
            "per_class": per_class,
            "notes": notes,
        },
    }


def build_eval_llm_row(eval_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a flat row for eval_llm_report.csv from eval_info structure.
    """
    global_stats = eval_info["eval_summary"]["global"]
    notes = eval_info["eval_summary"]["notes"]

    return {
        "ablation_name": eval_info["ablation_name"],
        "n_patches": global_stats["n_patches"],
        "n_patients": global_stats["n_patients"],
        "n_slides": global_stats["n_slides"],
        "accuracy": global_stats["accuracy"],
        "macro_precision": global_stats["macro_avg"]["precision"],
        "macro_recall": global_stats["macro_avg"]["recall"],
        "macro_f1": global_stats["macro_avg"]["f1"],
        "weighted_precision": global_stats["weighted_avg"]["precision"],
        "weighted_recall": global_stats["weighted_avg"]["recall"],
        "weighted_f1": global_stats["weighted_avg"]["f1"],
        "notes": " | ".join(notes) if notes else "",
    }


# ---------- Orchestrator: training + evaluation ----------


def main() -> None:
    args = parse_args()

    exp_root = Path(args.exp_root).resolve()
    if not exp_root.exists():
        raise SystemExit(f"Experiment root does not exist: {exp_root}")

    model_name = infer_model_name(exp_root, args.model)
    run_dirs = discover_run_dirs(exp_root)
    if not run_dirs:
        raise SystemExit(f"No ablation runs found under {exp_root}")

    exp_token = infer_experiment_token(exp_root, model_name)
    report_dir = exp_root / "reporting"

    if not args.dry_run:
        report_dir.mkdir(parents=True, exist_ok=True)

    # Rows for CSVs
    training_report_rows: List[Dict[str, Any]] = []
    eval_llm_rows: List[Dict[str, Any]] = []
    diagnostics_rows: List[Dict[str, Any]] = []

    # For warnings / summary
    skipped_training: List[Dict[str, str]] = []
    skipped_eval: List[Dict[str, str]] = []

    for run_dir in run_dirs:
        run_id = run_dir.name

        training_summary: Optional[Dict[str, Any]] = None
        loss_trend: Optional[Dict[str, Any]] = None
        eval_summary: Optional[Dict[str, Any]] = None

        training_error: Optional[str] = None
        eval_error: Optional[str] = None

        # --- Training diagnostics ---
        try:
            training_summary, loss_trend = analyze_training_run(run_dir, model_name)

            # Merge summary + loss trend into a single row, avoiding duplicate ablation_name
            trend_no_ablation = dict(loss_trend)
            trend_no_ablation.pop("ablation_name", None)
            training_row = {**training_summary, **trend_no_ablation}
            training_report_rows.append(training_row)
        except Exception as e:  # noqa: BLE001
            training_error = str(e)
            skipped_training.append({"run_id": run_id, "reason": training_error})

        # --- Evaluation diagnostics ---
        try:
            eval_summary = analyze_eval_run(exp_root, run_dir, model_name)
            if eval_summary is not None:
                eval_llm_rows.append(build_eval_llm_row(eval_summary))
            else:
                eval_error = "no eval predictions/report found"
                skipped_eval.append({"run_id": run_id, "reason": eval_error})
        except Exception as e:  # noqa: BLE001
            eval_error = str(e)
            skipped_eval.append({"run_id": run_id, "reason": eval_error})

        diagnostics_rows.append(
            {
                "ablation_name": run_id,
                "training_ok": training_error is None and training_summary is not None and loss_trend is not None,
                "training_error": training_error,
                "eval_ok": eval_error is None and eval_summary is not None,
                "eval_error": eval_error,
            }
        )

    # ---------- Write outputs ----------
    prefix = f"{exp_token}_{model_name}"

    if not args.dry_run:
        warnings: List[str] = []

        # 1) <datetime>_<model>_training_report.csv
        training_report_path = report_dir / f"{prefix}_training_report.csv"
        df_training = pd.DataFrame(training_report_rows)
        df_training.to_csv(training_report_path, index=False)
        if df_training.empty:
            warnings.append("training_report is empty (no successful training diagnostics).")

        # 2) <datetime>_<model>_eval_llm_report.csv
        eval_llm_report_path = report_dir / f"{prefix}_eval_llm_report.csv"
        df_eval_llm = pd.DataFrame(eval_llm_rows)
        df_eval_llm.to_csv(eval_llm_report_path, index=False)
        if df_eval_llm.empty:
            warnings.append("eval_llm_report is empty (no evaluation artefacts found).")

        # 3) <datetime>_<model>_diagnostics.csv
        diagnostics_path = report_dir / f"{prefix}_diagnostics.csv"
        df_diag = pd.DataFrame(diagnostics_rows)
        df_diag.to_csv(diagnostics_path, index=False)
        if df_diag.empty:
            warnings.append("diagnostics is empty (no runs discovered?).")

        print(f"[posthoc_diagnostics] Written reports to: {report_dir}")
        print(f"  - {training_report_path.name}")
        print(f"  - {eval_llm_report_path.name}")
        print(f"  - {diagnostics_path.name}")

        if skipped_training:
            warnings.append(f"{len(skipped_training)} training run(s) skipped.")
        if skipped_eval:
            warnings.append(f"{len(skipped_eval)} eval run(s) skipped.")

        for w in warnings:
            print(f"[posthoc_diagnostics][WARNING] {w}")
    else:
        print("[posthoc_diagnostics] DRY RUN")
        print(f"  Exp root: {exp_root}")
        print(f"  Model: {model_name}")
        print(f"  Discovered runs: {[p.name for p in run_dirs]}")
        print(f"  Training rows (training_report): {len(training_report_rows)}")
        print(f"  Eval rows (eval_llm_report): {len(eval_llm_rows)}")
        print(f"  Diagnostics rows: {len(diagnostics_rows)}")
        if skipped_training:
            print(f"  Skipped training runs: {skipped_training}")
        if skipped_eval:
            print(f"  Skipped eval runs: {skipped_eval}")


if __name__ == "__main__":
    main()
