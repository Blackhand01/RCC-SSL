# utils/viz.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pandas as pd

from .io import prefixed

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
]


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


def write_derived_csv(in_csv: Path | str, out_csv: Path | str) -> Path:
    src = Path(in_csv)
    dst = Path(out_csv)
    df = pd.read_csv(src) if src.exists() else pd.DataFrame()
    if df.empty:
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst, index=False)
        return dst

    for col in [c for c in df.columns if c not in ("epoch", "step", "elapsed_s")]:
        window = max(2, len(df) // 10)
        df[f"{col}_smoothed"] = df[col].rolling(window, min_periods=1).mean()
        df[f"{col}_d1"] = df[col].diff().fillna(0.0)

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    return dst


def _skip_plot(out_png: Path | str, reason: str) -> Path:
    path = Path(out_png)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass
    print(f"[viz] skip {path.name}: {reason}")
    return path


def _save_figure(fig, out_png: Path | str, plt_module) -> Path:
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt_module.close(fig)
    return out_path


def _lineplot(df: pd.DataFrame, x_key: str, y_keys: list[str], title: str, out_png: Path) -> Path:
    plt = _resolve_backend()
    if plt is None:
        return _skip_plot(out_png, "matplotlib missing")
    sns = _load_seaborn()
    if sns is None:
        return _skip_plot(out_png, "seaborn missing")
    fig = plt.figure()
    for y in y_keys:
        if y in df.columns:
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


def render_all_ssl(csv_path: Path | str, plots_dir: Path | str, model_key: str) -> Dict[str, Path]:
    plots_root = Path(plots_dir)
    figures: Dict[str, Path] = {}
    figures["ssl_losses"] = plot_ssl_losses(
        csv_path,
        prefixed(plots_root, model_key, "ssl_losses", "png"),
        model_key,
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
