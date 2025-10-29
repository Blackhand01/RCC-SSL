#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-hoc diagnostics for the *new* pipeline (no validate_outputs.py).
- Input: path to an experiment directory that already contains model subfolders.
- Output: diagnostics_report.md + diagnostics_report.json created inside that directory.

It supports both layouts:
  A) <EXP>/<exp_name>/<model_key>/{checkpoints,metrics,plots,configuration}/...
  B) <EXP>/<model_key>/{...}    plus optional runs_summary_{ssl,sl}.csv at <EXP> level

CSV stems may differ (new pipeline allows custom stem via config):
we auto-detect timeseries CSVs for SSL/SL and fall back to canonical names.
"""
from __future__ import annotations
import argparse, json, os, sys, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

try:
    import pandas as pd  # optional: robust CSV
except Exception:
    pd = None  # type: ignore


# --------------------------- I/O helpers --------------------------------------
def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    if pd is not None:
        try:
            df = pd.read_csv(path)
            return df.to_dict(orient="records")
        except Exception:
            pass
    import csv
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


# --------------------------- discovery logic ----------------------------------
def _detect_mode(model_dir: Path) -> str:
    ck = model_dir / "checkpoints"
    if any(p.name.endswith("ssl_best.pt") for p in ck.glob("*ssl_best.pt")):
        return "ssl"
    if any(p.name.endswith("sl_best_classifier.pt") for p in ck.glob("*sl_best_classifier.pt")):
        return "sl"
    # Heuristic: look at metrics CSV names
    mt = model_dir / "metrics"
    if any(mt.glob("*__ssl_*timeseries*.csv")):
        return "ssl"
    if any(mt.glob("*__sl_*timeseries*.csv")):
        return "sl"
    # Default to SSL if nothing matches explicitly
    return "ssl"

def _first_match(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def _find_timeseries_csv(mt_dir: Path, model_key: str, mode: str) -> Optional[Path]:
    """
    New pipeline supports a configurable CSV stem; we try common patterns:
      <model>__ssl_timeseries.csv
      <model>__<stem>.csv
      any file containing "__ssl_" (or "__sl_") and "timeseries"
    """
    # Canonical names
    candidates = [
        mt_dir / f"{model_key}__ssl_timeseries.csv" if mode == "ssl" else mt_dir / f"{model_key}__sl_timeseries.csv",
    ]
    # Any custom stem that still contains mode + "timeseries"
    glob_pat = f"{model_key}__*{mode}*timeseries*.csv"
    candidates.extend(sorted(mt_dir.glob(glob_pat)))
    return _first_match(candidates)

def _find_summary_json(mt_dir: Path, model_key: str, mode: str) -> Optional[Path]:
    candidates = [
        mt_dir / f"{model_key}__ssl_summary.json" if mode == "ssl" else mt_dir / f"{model_key}__sl_summary.json",
    ]
    # fallbacks
    candidates.extend(sorted(mt_dir.glob(f"{model_key}__*{mode}*summary*.json")))
    return _first_match(candidates)

def _find_plot(mt_dir: Path, plots_dir: Path, model_key: str, names: List[str]) -> bool:
    for stem in names:
        p = plots_dir / f"{model_key}__{stem}.png"
        if not p.exists():
            return False
    return True

def _read_runs_summary(exp_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Optionally merge elapsed seconds, etc., from runs_summary_{ssl,sl}.csv at exp_root.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for mode in ("ssl", "sl"):
        p = exp_root / f"runs_summary_{mode}.csv"
        rows = _read_csv_rows(p)
        for r in rows:
            mk = r.get("model") or r.get("model_key") or r.get("run_name") or ""
            if mk:
                out.setdefault(mk, {}).update(r)
    return out

def _gather_models(exp_path: Path) -> Tuple[str, Path, List[Path]]:
    """
    Return (exp_name, exp_root_for_summaries, model_dirs)
    It accepts both <EXP>/<exp_name>/* and <EXP>/* layouts.
    """
    exp_name = exp_path.name
    subdirs = [p for p in exp_path.iterdir() if p.is_dir()]

    # Case A: inside <EXP>/<exp_name>/ we directly see model folders (they have 'metrics')
    if subdirs and all((p / "metrics").exists() for p in subdirs):
        return exp_name, exp_path.parent if (exp_path.parent.exists()) else exp_path, subdirs

    # Case B: one deeper level holds model folders
    for p in subdirs:
        inner = [q for q in p.iterdir() if q.is_dir()]
        if inner and any((q / "metrics").exists() for q in inner):
            return p.name, exp_path, inner

    # Fallback: assume current level holds models
    return exp_name, exp_path, subdirs


# --------------------------- per-mode gather ----------------------------------
def _gather_ssl(model_dir: Path, model_key: str) -> Dict[str, Any]:
    mt = model_dir / "metrics"
    ck = model_dir / "checkpoints"
    plots = model_dir / "plots"

    ts_csv = _find_timeseries_csv(mt, model_key, "ssl")
    ts_rows = _read_csv_rows(ts_csv) if ts_csv else []
    summary = _read_json(_find_summary_json(mt, model_key, "ssl") or Path("__missing__"))

    # Probe CSV may or may not exist
    probe_csv = _first_match([
        mt / f"{model_key}__ssl_linear_timeseries.csv",
        *sorted(mt.glob(f"{model_key}__*linear*timeseries*.csv")),
    ])
    probe_rows = _read_csv_rows(probe_csv) if probe_csv else []
    probe_best = None
    if probe_rows:
        # try common keys
        for key in ("val_acc", "val_accuracy", "acc_val"):
            try:
                vals = [float(r.get(key, "nan")) for r in probe_rows if key in r]
                if vals:
                    probe_best = max(v for v in vals if v == v)  # filter NaNs
                    break
            except Exception:
                pass

    features_dir = ck / "features"
    features_ok = all((features_dir / f"{model_key}_{split}_X.npy").exists()
                      for split in ("train", "val")) and all(
                      (features_dir / f"{model_key}_{split}_y.npy").exists()
                      for split in ("train", "val"))

    issues: List[str] = []
    if not (ck / f"{model_key}__ssl_best.pt").exists():
        issues.append("missing checkpoints/*__ssl_best.pt")
    if not ts_rows:
        issues.append("missing metrics/*__ssl_timeseries*.csv")
    if not (plots / f"{model_key}__ssl_losses.png").exists():
        issues.append("missing plots/*__ssl_losses.png")
    if not summary:
        issues.append("missing or empty metrics/*__ssl_summary.json")

    return {
        "mode": "ssl",
        "model_key": model_key,
        "ssl_best_epoch": summary.get("best_epoch"),
        "ssl_loss": summary.get("ssl_loss"),
        "ssl_backbone_path": summary.get("ssl_backbone_path", ""),
        "probe_linear_val_acc": probe_best,
        "has_linear_ckpt": (ck / f"{model_key}__ssl_linear_best.pt").exists(),
        "features_ok": features_ok,
        "plots_ok": (plots / f"{model_key}__ssl_losses.png").exists(),
        "issues": issues,
    }

def _gather_sl(model_dir: Path, model_key: str) -> Dict[str, Any]:
    mt = model_dir / "metrics"
    ck = model_dir / "checkpoints"
    plots = model_dir / "plots"

    ts_csv = _find_timeseries_csv(mt, model_key, "sl")
    ts_rows = _read_csv_rows(ts_csv) if ts_csv else []
    summary = _read_json(_find_summary_json(mt, model_key, "sl") or Path("__missing__"))

    last_val = None
    if ts_rows:
        for key in ("val_acc", "val_accuracy", "acc_val"):
            try:
                val = float(ts_rows[-1].get(key, "nan"))
                if val == val:
                    last_val = val
                    break
            except Exception:
                pass

    issues: List[str] = []
    if not (ck / f"{model_key}__sl_best_classifier.pt").exists():
        issues.append("missing checkpoints/*__sl_best_classifier.pt")
    if not ts_rows:
        issues.append("missing metrics/*__sl_timeseries*.csv")
    needed_plots = ["sl_losses", "sl_acc", "sl_confusion_val"]
    if not _find_plot(mt, plots, model_key, needed_plots):
        issues.append("missing one or more plots/*__{sl_losses,sl_acc,sl_confusion_val}.png")
    if not summary:
        issues.append("missing or empty metrics/*__sl_summary.json")

    return {
        "mode": "sl",
        "model_key": model_key,
        "sl_best_epoch": summary.get("best_epoch"),
        "sl_val_acc": summary.get("val_acc", last_val),
        "sl_val_f1_macro": summary.get("val_f1_macro"),
        "sl_val_loss": summary.get("val_loss"),
        "sl_classifier_path": summary.get("sl_classifier_path", ""),
        "plots_ok": _find_plot(mt, plots, model_key, needed_plots),
        "issues": issues,
    }


# --------------------------- report builders ----------------------------------
def _mk_table(models: List[Dict[str, Any]]) -> str:
    if not models:
        return "*Nessun modello trovato.*"
    lines = []
    lines.append("| model | mode | key metrics | elapsed_min | issues |")
    lines.append("|---|---|---|---:|---|")
    for m in models:
        if m["mode"] == "ssl":
            km = f"ssl_loss={m.get('ssl_loss')} · best_epoch={m.get('ssl_best_epoch')} · probe_val_acc={m.get('probe_linear_val_acc')}"
        else:
            km = f"val_acc={m.get('sl_val_acc')} · f1={m.get('sl_val_f1_macro')} · loss={m.get('sl_val_loss')}"
        issues = ", ".join(m.get("issues", [])[:3]) if m.get("issues") else ""
        elapsed_s = m.get('elapsed_s')
        elapsed_min = f"{float(elapsed_s)/60:.2f}" if elapsed_s and elapsed_s != '' else ''
        lines.append(f"| {m['model_key']} | {m['mode']} | {km} | {elapsed_min} | {issues} |")
    return "\n".join(lines)

def _mk_suggestions(models: List[Dict[str, Any]]) -> List[str]:
    sug: List[str] = []
    for m in models:
        key = m["model_key"]
        if m["mode"] == "ssl":
            if not m.get("features_ok", True):
                sug.append(f"- **{key}**: riesegui sola **feature extraction** (no training) per abilitare linear probe.")
            if m.get("probe_linear_val_acc") in (None, float("nan")):
                sug.append(f"- **{key}**: lancia **linear probe** per stimare la qualità del backbone.")
        else:
            if not m.get("plots_ok", False):
                sug.append(f"- **{key}**: rigenera i plot SL (loss/acc/confusion) dal CSV timeseries.")
    return sug


# --------------------------- main --------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Generate diagnostics report for an existing experiment folder (new pipeline).")
    ap.add_argument("--exp-path", required=True, type=Path, help="Path all'esperimento (livello che contiene i modelli o la sotto-cartella <exp_name>/).")
    ap.add_argument("--out-md", default="diagnostics_report.md", type=str, help="Nome file Markdown di output.")
    ap.add_argument("--out-json", default="diagnostics_report.json", type=str, help="Nome file JSON di output.")
    args = ap.parse_args()

    exp_path = args.exp_path.resolve()
    if not exp_path.exists():
        print(f"[ERR] exp-path non esiste: {exp_path}", file=sys.stderr)
        return 2

    exp_name, exp_root_for_summaries, model_dirs = _gather_models(exp_path)
    runs_index = _read_runs_summary(exp_root_for_summaries)

    models: List[Dict[str, Any]] = []
    for model_dir in sorted(model_dirs):
        model_key = model_dir.name
        mode = _detect_mode(model_dir)
        info = _gather_ssl(model_dir, model_key) if mode == "ssl" else _gather_sl(model_dir, model_key)
        # merge runs summary info (elapsed_s, etc.)
        info.update(runs_index.get(model_key, {}))
        models.append(info)

    # Markdown
    md_lines: List[str] = []
    md_lines.append(f"# Diagnostics report — {exp_name}")
    md_lines.append("")
    md_lines.append(f"_Directory:_ `{exp_path}`")
    md_lines.append("")
    md_lines.append("## Riepilogo")
    md_lines.append(_mk_table(models))
    md_lines.append("")
    md_lines.append("## Suggerimenti rapidi")
    sugs = _mk_suggestions(models)
    md_lines.extend(sugs if sugs else ["- Nessun intervento necessario."])
    md_lines.append("")
    md_lines.append("## Dettaglio per modello")
    for m in models:
        md_lines.append(f"### {m['model_key']}  ·  {m['mode']}")
        if m["mode"] == "ssl":
            md_lines.append(f"- ssl_loss: **{m.get('ssl_loss')}**  ·  best_epoch: **{m.get('ssl_best_epoch')}**")
            md_lines.append(f"- probe_linear_val_acc: **{m.get('probe_linear_val_acc')}**")
            md_lines.append(f"- backbone: `{m.get('ssl_backbone_path','')}`  ·  linear_ckpt: {m.get('has_linear_ckpt')}")
            md_lines.append(f"- features_ok: {m.get('features_ok')}  ·  plots_ok: {m.get('plots_ok')}")
        else:
            md_lines.append(f"- val_acc: **{m.get('sl_val_acc')}**  ·  f1_macro: **{m.get('sl_val_f1_macro')}**  ·  val_loss: {m.get('sl_val_loss')}")
            md_lines.append(f"- best_epoch: {m.get('sl_best_epoch')}  ·  ckpt: `{m.get('sl_classifier_path','')}`  ·  plots_ok: {m.get('plots_ok')}")
        if m.get("issues"):
            md_lines.append("- **Issues:**")
            for it in m["issues"]:
                md_lines.append(f"  - {it}")
        md_lines.append("")

    (exp_path / args.out_md).write_text("\n".join(md_lines), encoding="utf-8")
    (exp_path / args.out_json).write_text(json.dumps({"exp": exp_name, "models": models}, indent=2), encoding="utf-8")
    print(f"[OK] Wrote: {exp_path / args.out_md}")
    print(f"[OK] Wrote: {exp_path / args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
