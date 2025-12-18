#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TUMOR_CLASSES = ("ccRCC", "pRCC", "CHROMO", "ONCO")
NOT_TUMOR = "NOT_TUMOR"


# ------------------------- utils -------------------------

def read_json(path: Path) -> Dict:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def is_nan(x: float) -> bool:
    return x != x


def fmt_csv(x: float, nd: int = 3) -> str:
    """CSV formatting: blank if NaN."""
    if is_nan(x):
        return ""
    return f"{x:.{nd}f}"


def fmt_tex(x: float, nd: int = 3) -> str:
    """LaTeX formatting: '--' if NaN."""
    if is_nan(x):
        return "--"
    return f"{x:.{nd}f}"


def fmt_int_tex(x: float) -> str:
    if is_nan(x):
        return "--"
    return str(int(x))


def infer_model_family(run_dir: Path) -> str:
    """
    Infer model family from folder names.
    In your layout you have:
      ablation_final/exp_YYYY..._ibot/exp_ibot_abl01
      ablation_final/exp_YYYY..._dino_v3/exp_dino_v3_abl01
      ...
    """
    s = str(run_dir).lower()
    for k in ("moco_v3", "dino_v3", "ibot", "i_jepa", "ijepa"):
        if k in s:
            return "i_jepa" if k in ("ijepa", "i_jepa") else k
    # fallback: exp_<name>_ablXX
    m = re.search(r"exp_([a-z0-9_]+)_abl\d+", run_dir.name.lower())
    if m:
        return m.group(1)
    return "unknown"


def extract_ablation_id(run_dir: Path) -> str:
    m = re.search(r"(abl\d+)", run_dir.name.lower())
    return m.group(1) if m else ""


def find_latest_eval_dir(run_dir: Path) -> Optional[Path]:
    """
    Find latest eval subdir:
      run_dir/eval/<model_name>/<timestamp>/
    Picks most recently modified timestamp folder that contains metrics_*.json.
    """
    eval_root = run_dir / "eval"
    if not eval_root.is_dir():
        return None

    hits = list(eval_root.glob("*/*/metrics_*.json"))
    if not hits:
        return None

    hits.sort(key=lambda p: p.parent.stat().st_mtime, reverse=True)
    return hits[0].parent


def min_recall_from_report(report: Dict) -> float:
    """
    Compute min recall across tumor classes from a classification_report dict.
    """
    recalls = []
    for c in TUMOR_CLASSES:
        if c in report and isinstance(report[c], dict):
            recalls.append(safe_float(report[c].get("recall")))
    if not recalls:
        return float("nan")
    return min(recalls)


def worst_tumor_class_from_report(report: Dict) -> str:
    items: List[Tuple[str, float]] = []
    for c in TUMOR_CLASSES:
        if c in report and isinstance(report[c], dict) and "recall" in report[c]:
            items.append((c, safe_float(report[c]["recall"])))
    if not items:
        return "--"
    items = [x for x in items if not is_nan(x[1])]
    if not items:
        return "--"
    return min(items, key=lambda x: x[1])[0]


# ------------------------- data model -------------------------

@dataclass
class Row:
    model_family: str
    run_name: str
    ablation: str

    # Patch-level (evaluation, 5 classes incl. NOT_TUMOR)
    patch_accuracy: float
    patch_macro_f1: float
    patch_bal_acc: float
    patch_macro_auc: float
    patch_macro_auprc: float
    patch_min_recall_tumor: float
    patch_worst_tumor_class: str

    # Patient-level (evaluation, tumor-only; NOT_TUMOR excluded)
    patient_accuracy: float
    patient_macro_f1: float
    patient_bal_acc: float
    patient_macro_auc: float
    patient_macro_auprc: float
    patient_min_recall_tumor: float
    patient_worst_tumor_class: str
    n_patients_tumor_eval: float

    # Val side (optional context)
    val_probe_acc: float


def load_one_run(run_dir: Path) -> Optional[Row]:
    # validation metrics (optional)
    final_metrics = read_json(run_dir / "metrics" / "final_metrics.json")
    val_probe_acc = safe_float(final_metrics.get("probe_linear_val_acc"), float("nan"))

    eval_dir = find_latest_eval_dir(run_dir)
    if eval_dir is None:
        return None

    # ---------- PATCH-LEVEL ----------
    patch_metrics_json = next(iter(eval_dir.glob("metrics_*.json")), None)
    patch_report_json = eval_dir / "report_per_class.json"
    patch_metrics = read_json(patch_metrics_json) if patch_metrics_json else {}
    patch_report = read_json(patch_report_json)

    pm = patch_metrics.get("metrics", {}) if isinstance(patch_metrics.get("metrics", {}), dict) else {}
    patch_accuracy = safe_float(pm.get("accuracy"), float("nan"))
    patch_bal_acc = safe_float(pm.get("balanced_accuracy"), float("nan"))
    patch_macro_f1 = safe_float(pm.get("macro_f1"), float("nan"))
    patch_macro_auc = safe_float(pm.get("macro_auc_ovr"), float("nan"))
    patch_macro_auprc = safe_float(pm.get("macro_auprc"), float("nan"))
    patch_min_recall_tumor = min_recall_from_report(patch_report)
    patch_worst_tumor_class = worst_tumor_class_from_report(patch_report)

    # ---------- PATIENT-LEVEL ----------
    patient_json = eval_dir / "per_patient" / "metrics_patient.json"
    patient = read_json(patient_json)
    mpat = patient.get("metrics", {}) if isinstance(patient.get("metrics", {}), dict) else {}

    patient_accuracy = safe_float(mpat.get("accuracy"), float("nan"))
    patient_bal_acc = safe_float(mpat.get("balanced_accuracy"), float("nan"))
    patient_macro_f1 = safe_float(mpat.get("macro_f1"), float("nan"))
    patient_macro_auc = safe_float(mpat.get("macro_auc_ovr"), float("nan"))
    patient_macro_auprc = safe_float(mpat.get("macro_auprc"), float("nan"))
    n_patients_tumor_eval = safe_float(mpat.get("n_patients_tumor_eval"), float("nan"))

    # patient report is under mpat["_report"] (tumor-only)
    patient_report = mpat.get("_report", {}) if isinstance(mpat.get("_report", {}), dict) else {}
    patient_min_recall_tumor = min_recall_from_report(patient_report)
    patient_worst_tumor_class = worst_tumor_class_from_report(patient_report)

    model_family = infer_model_family(run_dir)

    return Row(
        model_family=model_family,
        run_name=run_dir.name,
        ablation=extract_ablation_id(run_dir),

        patch_accuracy=patch_accuracy,
        patch_macro_f1=patch_macro_f1,
        patch_bal_acc=patch_bal_acc,
        patch_macro_auc=patch_macro_auc,
        patch_macro_auprc=patch_macro_auprc,
        patch_min_recall_tumor=patch_min_recall_tumor,
        patch_worst_tumor_class=patch_worst_tumor_class,

        patient_accuracy=patient_accuracy,
        patient_macro_f1=patient_macro_f1,
        patient_bal_acc=patient_bal_acc,
        patient_macro_auc=patient_macro_auc,
        patient_macro_auprc=patient_macro_auprc,
        patient_min_recall_tumor=patient_min_recall_tumor,
        patient_worst_tumor_class=patient_worst_tumor_class,
        n_patients_tumor_eval=n_patients_tumor_eval,

        val_probe_acc=val_probe_acc,
    )


def is_run_dir(p: Path) -> bool:
    return p.is_dir() and (p / "metrics" / "final_metrics.json").is_file()


def discover_runs(experiments_root: Path) -> List[Path]:
    """
    Discover run dirs under:
      ablation_final/exp_YYYY..._model/exp_model_ablXX
    """
    if is_run_dir(experiments_root):
        return [experiments_root]

    runs: List[Path] = []
    for exp in experiments_root.glob("exp_*"):
        if not exp.is_dir():
            continue
        for run in exp.glob("exp_*"):
            if is_run_dir(run):
                runs.append(run)

    if not runs:
        for run in experiments_root.rglob("exp_*"):
            if is_run_dir(run):
                runs.append(run)

    return sorted(runs)


def better_patch_selected_by_macro_f1(a: Row, b: Row) -> bool:
    """
    True if a is better than b by PATCH-level ranking.
    Selection requested: choose best ablation by patch-level Macro-F1.
    Tie-breakers: AUROC, PR-AUC, MinRecall(T), BalAcc, Accuracy.
    """
    key_a = (
        a.patch_macro_f1,
        a.patch_macro_auc,
        a.patch_macro_auprc,
        a.patch_min_recall_tumor,
        a.patch_bal_acc,
        a.patch_accuracy,
    )
    key_b = (
        b.patch_macro_f1,
        b.patch_macro_auc,
        b.patch_macro_auprc,
        b.patch_min_recall_tumor,
        b.patch_bal_acc,
        b.patch_accuracy,
    )
    return key_a > key_b


def select_best_per_model(rows: List[Row]) -> List[Row]:
    best: Dict[str, Row] = {}
    for r in rows:
        if r.model_family not in best:
            best[r.model_family] = r
        else:
            if better_patch_selected_by_macro_f1(r, best[r.model_family]):
                best[r.model_family] = r
    return [best[k] for k in sorted(best.keys())]


def write_csv(rows: List[Row], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "model_family", "run_name", "ablation",
        "patch_accuracy", "patch_macro_f1", "patch_bal_acc", "patch_macro_auc_ovr", "patch_macro_auprc", "patch_min_recall_tumor", "patch_worst_tumor_class",
        "patient_accuracy", "patient_macro_f1", "patient_bal_acc", "patient_macro_auc_ovr", "patient_macro_auprc", "patient_min_recall_tumor", "patient_worst_tumor_class",
        "n_patients_tumor_eval",
        "val_probe_acc",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                r.model_family, r.run_name, r.ablation,
                fmt_csv(r.patch_accuracy), fmt_csv(r.patch_macro_f1), fmt_csv(r.patch_bal_acc),
                fmt_csv(r.patch_macro_auc), fmt_csv(r.patch_macro_auprc),
                fmt_csv(r.patch_min_recall_tumor), r.patch_worst_tumor_class,
                fmt_csv(r.patient_accuracy), fmt_csv(r.patient_macro_f1), fmt_csv(r.patient_bal_acc),
                fmt_csv(r.patient_macro_auc), fmt_csv(r.patient_macro_auprc),
                fmt_csv(r.patient_min_recall_tumor), r.patient_worst_tumor_class,
                fmt_csv(r.n_patients_tumor_eval, 0),
                fmt_csv(r.val_probe_acc),
            ])


def write_latex_patch_table(rows: List[Row], out_tex: Path) -> None:
    """
    Patch-only LaTeX table (booktabs + resizebox), with Accuracy as first metric.
    Columns: Model | Acc | Macro-F1 | BalAcc | AUROC | PR-AUC | MinRecall(T)
    """
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Model & Acc & Macro-F1 & BalAcc & AUROC & PR-AUC & MinRecall(T) \\\\")
    lines.append("\\midrule")

    for r in rows:
        lines.append(
            f"{r.model_family} & "
            f"{fmt_tex(r.patch_accuracy)} & "
            f"{fmt_tex(r.patch_macro_f1)} & "
            f"{fmt_tex(r.patch_bal_acc)} & "
            f"{fmt_tex(r.patch_macro_auc)} & "
            f"{fmt_tex(r.patch_macro_auprc)} & "
            f"{fmt_tex(r.patch_min_recall_tumor)} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\caption{Patch-level results. Best ablation per model selected by patch-level Macro-F1 (tie-breakers: AUROC, PR-AUC, MinRecall(T), BalAcc, Acc).}")
    lines.append("\\label{tab:main_results_patch}")
    lines.append("\\end{table}")

    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_patient_table(rows: List[Row], out_tex: Path) -> None:
    """
    Patient-only LaTeX table (booktabs + resizebox), with Accuracy as first metric.
    Columns: Model | Acc | Macro-F1 | AUROC | PR-AUC | MinRecall(T) | N_pat(T)
    (BalAcc patient-level puoi aggiungerla se vuoi, ma così di solito ci sta bene.)
    """
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Model & Acc & Macro-F1 & AUROC & PR-AUC & MinRecall(T) & $N_{pat}(T)$ \\\\")
    lines.append("\\midrule")

    for r in rows:
        lines.append(
            f"{r.model_family} & "
            f"{fmt_tex(r.patient_accuracy)} & "
            f"{fmt_tex(r.patient_macro_f1)} & "
            f"{fmt_tex(r.patient_macro_auc)} & "
            f"{fmt_tex(r.patient_macro_auprc)} & "
            f"{fmt_tex(r.patient_min_recall_tumor)} & "
            f"{fmt_int_tex(r.n_patients_tumor_eval)} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\caption{Patient-level results (tumor-only aggregation; NOT\\_TUMOR excluded). Rows correspond to the same best ablations selected by patch-level Macro-F1.}")
    lines.append("\\label{tab:main_results_patient}")
    lines.append("\\end{table}")

    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--select-best-per-model", action="store_true")
    args = ap.parse_args()

    runs = discover_runs(args.experiments_root)
    if not runs:
        raise SystemExit(f"No runs found under: {args.experiments_root}")

    rows: List[Row] = []
    for run in runs:
        r = load_one_run(run)
        if r is not None:
            rows.append(r)

    if not rows:
        raise SystemExit("Found runs, but no usable eval artifacts (missing eval/*/*/metrics_*.json).")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # All rows (debug / supplementary)
    write_csv(rows, args.out_dir / "main_results_all.csv")

    if args.select_best_per_model:
        best = select_best_per_model(rows)
        write_csv(best, args.out_dir / "main_results_best.csv")

        # Split LaTeX in two tables
        write_latex_patch_table(best, args.out_dir / "main_results_best_patch.tex")
        write_latex_patient_table(best, args.out_dir / "main_results_best_patient.tex")

    print(f"[OK] Wrote outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
