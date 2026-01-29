# utils/viz.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Union

import pandas as pd

from .io import prefixed

def _skip_plot(out_png: Union[str, Path], reason: str) -> Path:
    """Skip plot generation and return the output path with a warning."""
    print(f"[viz] Skipping plot {out_png}: {reason}")
    return Path(out_png)

def _save_figure(fig, out_png: Union[str, Path], plt=None, *, dpi: int = 144,
                 tight: bool = True, transparent: bool = False,
                 also_svg: bool = False, close: bool = True):
    """
    Save a matplotlib Figure to disk (PNG only by default) and return the figure.
    - Creates parent directories if missing.
    - Applies tight_layout if available.
    - Optionally closes the figure to free memory (for long runs).
    """
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    # Save PNG
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=transparent)
    # Optionally save SVG
    if also_svg:
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", transparent=transparent)
    # Optionally close
    if close and plt is not None:
        try:
            plt.close(fig)
        except Exception:
            pass
    return fig

__all__ = [
    "write_derived_csv",
    "plot_ssl_losses",
    "plot_lr",
    "plot_sl_losses",
    "plot_acc",
    "plot_confusion",
    "render_all_ssl",
    "render_all_sl",
    "render_ssl_classifier",
    "tta_predict_simple",
]


def _lineplot(df: pd.DataFrame, x_key: str, y_keys: list[str], title: str, out_png: Path) -> Path:
    plt = _resolve_backend()
    if plt is None:
        return _skip_plot(out_png, "matplotlib missing")
    sns = _load_seaborn()
    if sns is None:
        return _skip_plot(out_png, "seaborn missing")
    # Check if any y_keys are present in df
    available_y_keys = [y for y in y_keys if y in df.columns and not df[y].isna().all()]
    if not available_y_keys:
        return _skip_plot(out_png, f"no data for y_keys {y_keys}")
    fig = plt.figure()
    for y in available_y_keys:
        sns.lineplot(data=df, x=x_key, y=y, label=y)
    plt.title(title)
    return _save_figure(fig, out_png, plt)


def _load_df(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def plot_ssl_losses(derived_csv: Path | str, out_png: Path | str, model_key: str) -> Path:
    df = _load_df(derived_csv)
    x_key = "step" if "step" in df.columns else df.columns[0] if len(df.columns) else "step"
    y_keys = [c for c in df.columns if c.startswith("loss")]
    return _lineplot(df, x_key, y_keys, f"{model_key} · SSL losses", Path(out_png))


def plot_lr(derived_csv: Path | str, out_png: Path | str, model_key: str) -> Path:
    df = _load_df(derived_csv)
    if df.empty:
        return Path(out_png)
    lr_key = None
    for col in df.columns:
        if col == "lr" or col.endswith("_lr"):
            lr_key = col
            break
        if "lr" in col:
            lr_key = col
    if not lr_key:
        return Path(out_png)
    x_key = "step" if "step" in df.columns else df.columns[0]
    return _lineplot(df, x_key, [lr_key], f"{model_key} · learning rate", Path(out_png))


def plot_sl_losses(derived_csv: Path | str, out_png: Path | str, model_key: str) -> Path:
    df = _load_df(derived_csv)
    return _lineplot(df, "epoch", ["train_loss", "val_loss"], f"{model_key} · Train/val loss", Path(out_png))


def plot_acc(derived_csv: Path | str, out_png: Path | str, model_key: str) -> Path:
    df = _load_df(derived_csv)
    return _lineplot(df, "epoch", ["val_acc"], f"{model_key} · Validation accuracy", Path(out_png))


def render_all_ssl(csv_path: Path | str, plots_dir: Path | str, model_key: str) -> Dict[str, Path]:
    plots_root = Path(plots_dir)
    figures: Dict[str, Path] = {}
    figures["ssl_losses"] = plot_ssl_losses(
        csv_path,
        prefixed(plots_root, model_key, "ssl_losses", "png"),
        model_key,
    )
    # Extra diagnostic plots
    figures["ssl_similarities"] = _lineplot(
        _load_df(csv_path),
        "step",
        ["pos_sim", "neg_sim"],
        f"{model_key} · SSL similarities",
        prefixed(plots_root, model_key, "ssl_similarities", "png"),
    )
    figures["ssl_temperature"] = _lineplot(
        _load_df(csv_path),
        "step",
        ["t_teacher"],
        f"{model_key} · SSL temperature",
        prefixed(plots_root, model_key, "ssl_temperature", "png"),
    )
    figures["ssl_lr"] = _lineplot(
        _load_df(csv_path),
        "step",
        ["lr"],
        f"{model_key} · SSL lr",
        prefixed(plots_root, model_key, "ssl_lr", "png"),
    )
    return figures


def render_all_sl(csv_path: Path | str, plots_dir: Path | str, model_key: str) -> Dict[str, Path]:
    plots_root = Path(plots_dir)
    figures: Dict[str, Path] = {}
    figures["sl_losses"] = plot_sl_losses(
        csv_path,
        prefixed(plots_root, model_key, "sl_losses", "png"),
        model_key,
    )
    figures["sl_acc"] = plot_acc(
        csv_path,
        prefixed(plots_root, model_key, "sl_acc", "png"),
        model_key,
    )
    return figures


def render_ssl_classifier(csv_path: Path | str, plots_dir: Path | str, model_key: str) -> Dict[str, Path]:
    plots_root = Path(plots_dir)
    figures: Dict[str, Path] = {}
    df = _load_df(csv_path)
    figures["ssl_clf_losses"] = _lineplot(
        df,
        "epoch",
        ["train_loss", "val_loss"],
        f"{model_key} · SSL linear head loss",
        prefixed(plots_root, model_key, "ssl_linear_loss", "png"),
    )
    figures["ssl_clf_acc"] = _lineplot(
        df,
        "epoch",
        ["val_acc"],
        f"{model_key} · SSL linear head acc",
        prefixed(plots_root, model_key, "ssl_linear_acc", "png"),
    )
    return figures


def tta_predict_simple(model, img: "torch.Tensor", *, rotations=(0,90,180,270), flips=("h","v")):
    """
    Light TTA: 90°k rotations and H/V flips; average softmax of predictions.
    img: (1,C,H,W) or (C,H,W)
    """
    import torch
    if img.ndim == 3: img = img.unsqueeze(0)
    outs = []
    for r in rotations:
        x = img
        if r != 0:
            k = (r // 90) % 4
            x = torch.rot90(x, k, dims=(2,3))
        for f in [None, "h", "v"]:
            xf = x
            if f == "h": xf = torch.flip(xf, dims=(3,))
            if f == "v": xf = torch.flip(xf, dims=(2,))
            with torch.no_grad():
                logits = model(xf)
                probs = torch.softmax(logits, dim=-1)
            outs.append(probs)
    return torch.stack(outs, dim=0).mean(dim=0)


def plot_confusion(cm, labels, out_png: Path | str) -> Path:
    plt = _resolve_backend()
    if plt is None:
        return _skip_plot(out_png, "matplotlib missing")
    sns = _load_seaborn()
    if sns is None:
        return _skip_plot(out_png, "seaborn missing")
    import numpy as np

    fig = plt.figure()
    sns.heatmap(np.asarray(cm), annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    return _save_figure(fig, out_png, plt)


def write_derived_csv(csv_in: str, csv_out: str | None = None,
                      target_col: str = "ssl_loss",
                      sma_window: int = 50,
                      ema_m: float | None = None) -> str:
    """
    Read a time-series CSV and write a derived CSV with smoothed metrics:
    - SMA over `target_col` with window `sma_window`
    - Optional EMA over `target_col` if `ema_m` is provided (0<ema_m<1)
    Returns output path.
    """
    import csv, math, os
    import numpy as np
    if csv_out is None:
        root, ext = os.path.splitext(csv_in)
        csv_out = f"{root}__derived{ext}"

    xs, ys, rows = [], [], []
    with open(csv_in, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for r in reader:
            rows.append(r)
            xs.append(float(r.get("step", len(xs))))
            try:
                ys.append(float(r.get(target_col, "nan")))
            except ValueError:
                ys.append(float("nan"))
    y = np.asarray(ys, dtype=np.float64)
    # SMA (naive)
    if sma_window > 1 and len(y) >= 1:
        k = min(sma_window, max(1, len(y)))
        cumsum = np.cumsum(np.nan_to_num(y, nan=0.0))
        sma = (cumsum - np.concatenate(([0.0], cumsum[:-k]))) / k
        # pad first k-1 with nan for alignment
        sma[:k-1] = np.nan
    else:
        sma = np.full_like(y, np.nan)
    # EMA
    if ema_m is not None and 0.0 < float(ema_m) < 1.0:
        ema = np.empty_like(y)
        ema[:] = np.nan
        alpha = 1.0 - float(ema_m)
        acc = None
        for i, v in enumerate(y):
            if math.isfinite(v):
                acc = (v if acc is None else float(ema_m) * acc + alpha * v)
                ema[i] = acc
    else:
        ema = np.full_like(y, np.nan)

    # write out
    out_fields = list(rows[0].keys()) if rows else ["step", target_col]
    if "sma" not in out_fields: out_fields += [f"{target_col}_sma_{sma_window}"]
    if "ema" not in out_fields: out_fields += [f"{target_col}_ema"]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for i, r in enumerate(rows):
            r[f"{target_col}_sma_{sma_window}"] = ("" if math.isnan(sma[i]) else f"{sma[i]:.8g}")
            r[f"{target_col}_ema"] = ("" if math.isnan(ema[i]) else f"{ema[i]:.8g}")
            writer.writerow(r)
    return csv_out


def _resolve_backend():
    try:
        import matplotlib

        if os.environ.get("DISPLAY", "") == "":
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa

        return matplotlib.pyplot  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[viz] WARNING: matplotlib unavailable ({exc}); skipping plot generation.")
        return None


def _load_seaborn():
    try:
        import seaborn as sns

        return sns
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[viz] WARNING: seaborn unavailable ({exc}); skipping plot generation.")
        return None



