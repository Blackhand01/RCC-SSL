#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate RCC pipeline outputs (no CLI args).
Env (optional):
  - EXPERIMENTS_ROOT (preferred) or OUTPUTS_ROOT -> .../experiments
  - EXP_ID, EXP_NAME  (to pin a specific experiment)
Layout expected per model:
  experiments/<exp_id>/<exp_name>/<model_key>/{configuration,checkpoints,metrics,plots}
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- soft deps (all optional) ---
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pq = None  # type: ignore


def _experiments_root() -> Path:
    eroot = os.environ.get("EXPERIMENTS_ROOT")
    if eroot:
        return Path(eroot).expanduser().resolve()
    out = os.environ.get("OUTPUTS_ROOT")
    if out:
        return (Path(out) / "experiments").expanduser().resolve()
    return (Path.cwd() / "outputs" / "mlruns" / "experiments").resolve()


def _latest_exp(root: Path) -> Tuple[Path, str]:
    candidates = [p for p in root.iterdir() if p.is_dir() and re.match(r"^exp_\d{8}-\d{6}$", p.name)]
    if not candidates:
        raise FileNotFoundError(f"Nessun exp_* in {root}")
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0], candidates[0].name


def _select_exp(root: Path) -> Tuple[Path, str, str]:
    exp_id = os.environ.get("EXP_ID")
    exp_name = os.environ.get("EXP_NAME")
    if exp_id:
        exp_dir = root / exp_id
        if not exp_dir.exists():
            raise FileNotFoundError(f"EXP_ID non trovato: {exp_dir}")
        if exp_name:
            return exp_dir / exp_name, exp_id, exp_name
        subdirs = [p for p in exp_dir.iterdir() if p.is_dir()]
        if not subdirs:
            raise FileNotFoundError(f"Nessun exp_name dentro {exp_dir}")
        if len(subdirs) > 1:
            subdirs.sort(key=lambda p: p.name, reverse=True)
        return subdirs[0], exp_id, subdirs[0].name
    latest_root, latest_id = _latest_exp(root)
    subdirs = [p for p in latest_root.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"Nessun exp_name dentro {latest_root}")
    subdirs.sort(key=lambda p: p.name, reverse=True)
    return subdirs[0], latest_id, subdirs[0].name


def _list_models(exp_path: Path) -> List[Path]:
    return [p for p in exp_path.iterdir() if p.is_dir()]


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _load_json(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"JSON invalid ({exc.__class__.__name__})"


def _check_png(path: Path) -> Optional[str]:
    if not _exists(path):
        return "missing or empty"
    if Image is None:
        return None
    try:
        image = Image.open(path)
        image.verify()
        return None
    except Exception as exc:  # pragma: no cover - image corruption
        return f"invalid PNG ({exc.__class__.__name__})"


def _read_csv(path: Path) -> Tuple[Optional["pd.DataFrame"], Optional[str]]:
    if not _exists(path):
        return None, "missing or empty"
    if pd is None:
        return None, "pandas not available"
    try:
        dataframe = pd.read_csv(path)  # type: ignore[arg-type]
        if dataframe.empty:
            return None, "empty dataframe"
        return dataframe, None
    except Exception as exc:  # pragma: no cover - pandas errors
        return None, f"csv read error ({exc.__class__.__name__})"


def _no_nan_inf_df(dataframe) -> Optional[str]:
    if pd is None or np is None:
        return None
    if dataframe.isna().values.any():
        return "contains NaN"
    numeric = dataframe.select_dtypes(include=["number"])
    if not numeric.empty and not np.isfinite(numeric.to_numpy()).all():
        return "contains Inf/-Inf/NaN in numeric cols"
    return None


def _check_monotonic_index(dataframe) -> Optional[str]:
    columns = [col for col in ("step", "epoch") if col in dataframe.columns]
    if not columns:
        return None
    for column in columns:
        try:
            values = dataframe[column].to_numpy()
            if len(values) >= 2 and (values[1:] < values[:-1]).any():
                return f"column '{column}' not non-decreasing"
        except Exception:  # pragma: no cover - tolerate dtype surprises
            continue
    return None


def _load_npy(path: Path) -> Tuple[Optional["np.ndarray"], Optional[str]]:
    if np is None:
        return None, "numpy not available"
    try:
        array = np.load(path, allow_pickle=False)
        return array, None
    except Exception as exc:  # pragma: no cover - numpy errors
        return None, f"npy read error ({exc.__class__.__name__})"


def _load_parquet(path: Path) -> Tuple[Optional["pd.DataFrame"], Optional[str]]:
    if pd is not None and hasattr(pd, "read_parquet"):
        try:
            dataframe = pd.read_parquet(path)  # type: ignore[arg-type]
            return dataframe, None
        except Exception:
            pass
    if pq is not None:
        try:
            table = pq.read_table(path)
            return table.to_pandas(), None
        except Exception as exc:  # pragma: no cover - pyarrow errors
            return None, f"parquet read error ({exc.__class__.__name__})"
    return None, "parquet reader not available"


def _check_features(feature_dir: Path, model_key: str) -> List[str]:
    issues: List[str] = []

    def _feature_candidates(prefix: str, suffix: str, extension: str) -> List[Path]:
        base = feature_dir / f"{prefix}_{suffix}.{extension}"
        alt = feature_dir / f"{model_key}_{prefix}_{suffix}.{extension}"
        return [base, alt]

    def _resolve_path(prefix: str, suffix: str, extensions: List[str]) -> Tuple[Optional[Path], List[str]]:
        tried: List[str] = []
        for ext in extensions:
            for candidate in _feature_candidates(prefix, suffix, ext):
                tried.append(candidate.name)
                if candidate.exists():
                    return candidate, []
        return None, tried

    def _pair(prefix: str) -> Tuple[Optional["np.ndarray"], Optional["np.ndarray"], List[str]]:
        errors: List[str] = []
        x_path, tried_x = _resolve_path(prefix, "X", ["npy", "parquet"])
        y_path, tried_y = _resolve_path(prefix, "y", ["npy", "parquet"])
        if x_path and y_path:
            if x_path.suffix == ".npy" and y_path.suffix == ".npy":
                feats, err_x = _load_npy(x_path)
                labels, err_y = _load_npy(y_path)
            else:
                if pd is None:
                    return None, None, [f"pandas missing for parquet"]
                feats_df, err_x = _load_parquet(x_path)
                labels_df, err_y = _load_parquet(y_path)
                feats = None if feats_df is None else feats_df.to_numpy()
                labels = None if labels_df is None else labels_df.squeeze().to_numpy()
            if err_x:
                errors.append(f"{x_path.name}: {err_x}")
            if err_y:
                errors.append(f"{y_path.name}: {err_y}")
            return feats, labels, errors
        if not x_path:
            errors.append(f"missing {prefix}_X ({', '.join(tried_x)})")
        if not y_path:
            errors.append(f"missing {prefix}_y ({', '.join(tried_y)})")
        return None, None, errors

    for split in ("train", "val"):
        feats, labels, errors = _pair(split)
        issues.extend(errors)
        if feats is None or labels is None:
            if pd is None:
                # Already captured missing files; skip consistency checks
                continue
            continue
        if feats.shape[0] != labels.shape[0]:
            issues.append(f"{split}: X/y length mismatch ({feats.shape[0]} vs {labels.shape[0]})")
        if np is not None and (not np.isfinite(feats).all()):
            issues.append(f"{split}: features contain non-finite values")
        if np is not None and (not np.isfinite(labels).all()):
            issues.append(f"{split}: labels contain non-finite values")
        if feats.size == 0 or labels.size == 0:
            issues.append(f"{split}: empty features or labels")
    return issues


def _torch_load_ok(path: Path) -> Optional[str]:
    if not _exists(path):
        return "missing or empty"
    if torch is None:
        return None
    try:
        torch.load(str(path), map_location="cpu")
        return None
    except Exception as exc:  # pragma: no cover - checkpoint corruption
        return f"torch.load failed ({exc.__class__.__name__})"


def _expect_ssl(model_dir: Path) -> Dict[str, List[Path]]:
    model_key = model_dir.name
    ck_dir = model_dir / "checkpoints"
    mt_dir = model_dir / "metrics"
    pl_dir = model_dir / "plots"
    cfg_dir = model_dir / "configuration"
    cfg_root = cfg_dir / model_key
    return {
        "must_exist": [
            cfg_root / "experiment_snapshot.yaml",
            ck_dir / f"{model_key}__ssl_best.pt",
            ck_dir / "features" / f"{model_key}_train_X.npy",
            ck_dir / "features" / f"{model_key}_train_y.npy",
            ck_dir / "features" / f"{model_key}_val_X.npy",
            ck_dir / "features" / f"{model_key}_val_y.npy",
            mt_dir / f"{model_key}__ssl_timeseries.csv",
            mt_dir / f"{model_key}__ssl_linear_timeseries.csv",
            pl_dir / f"{model_key}__ssl_losses.png",
        ],
        "optional": [
            cfg_dir / "experiment_config.yaml",
            cfg_dir / "resolved_config.json",
            ck_dir / f"{model_key}__ssl_linear_best.pt",
            ck_dir / "features" / "train_X.npy",
            ck_dir / "features" / "train_y.npy",
            ck_dir / "features" / "val_X.npy",
            ck_dir / "features" / "val_y.npy",
            pl_dir / f"{model_key}__ssl_linear_confusion_val.png",
            pl_dir / f"{model_key}__ssl_features_umap.png",
        ],
    }


def _expect_sl(model_dir: Path) -> Dict[str, List[Path]]:
    model_key = model_dir.name
    ck_dir = model_dir / "checkpoints"
    mt_dir = model_dir / "metrics"
    pl_dir = model_dir / "plots"
    cfg_dir = model_dir / "configuration"
    cfg_root = cfg_dir / model_key
    return {
        "must_exist": [
            cfg_root / "experiment_snapshot.yaml",
            ck_dir / f"{model_key}__sl_best_classifier.pt",
            mt_dir / f"{model_key}__sl_timeseries.csv",
            pl_dir / f"{model_key}__sl_losses.png",
            pl_dir / f"{model_key}__sl_acc.png",
            pl_dir / f"{model_key}__sl_confusion_val.png",
        ],
        "optional": [
            cfg_dir / "experiment_config.yaml",
            cfg_dir / "resolved_config.json",
        ],
    }


def _model_kind(model_dir: Path) -> str:
    ck_dir = model_dir / "checkpoints"
    if any(ck_dir.glob("*ssl_best.pt")) or any(ck_dir.glob("*ssl_linear_best.pt")):
        return "ssl"
    if any(ck_dir.glob("*sl_best_classifier.pt")):
        return "sl"
    # Heuristic fallback: prefer SSL if there are SSL metrics present
    mt_dir = model_dir / "metrics"
    if any(mt_dir.glob("*__ssl_timeseries.csv")):
        return "ssl"
    return "sl"


def validate() -> int:
    root = _experiments_root()
    if not root.exists():
        print(f"[ERR] Experiments root not found: {root}", file=sys.stderr)
        return 2
    try:
        exp_path, exp_id, exp_name = _select_exp(root)
    except Exception as exc:
        print(f"[ERR] Unable to select experiment: {exc}", file=sys.stderr)
        return 2
    if not exp_path.exists():
        print(f"[ERR] Experiment path not found: {exp_path}", file=sys.stderr)
        return 2

    models_section: List[Dict[str, object]] = []
    report: Dict[str, object] = {
        "exp_root": str(root),
        "exp_id": exp_id,
        "exp_name": exp_name,
        "models": models_section,
    }
    print(f"[INFO] Validating: {exp_path}")

    failures = 0
    for model_dir in _list_models(exp_path):
        kind = _model_kind(model_dir)
        schema = _expect_ssl(model_dir) if kind == "ssl" else _expect_sl(model_dir)
        issues: List[str] = []
        warnings: List[str] = []

        for required in schema["must_exist"]:
            if not _exists(required):
                issues.append(f"missing: {required.relative_to(model_dir)}")
        for optional in schema["optional"]:
            if not optional.exists():
                warnings.append(f"optional missing: {optional.relative_to(model_dir)}")

        cfg_json = model_dir / "configuration" / "resolved_config.json"
        if cfg_json.exists():
            _, err = _load_json(cfg_json)
            if err:
                issues.append(f"resolved_config.json: {err}")

        ck_dir = model_dir / "checkpoints"
        for filename in ("ssl_best.pt", "ssl_linear_best.pt", "sl_best_classifier.pt"):
            candidate = ck_dir / filename
            if candidate.exists():
                err = _torch_load_ok(candidate)
                if err:
                    issues.append(f"{filename}: {err}")

        pl_dir = model_dir / "plots"
        for png_path in pl_dir.glob("*.png"):
            err = _check_png(png_path)
            if err:
                issues.append(f"{png_path.name}: {err}")

        mt_dir = model_dir / "metrics"
        for csv_path in mt_dir.glob("*.csv"):
            dataframe, err = _read_csv(csv_path)
            if err:
                issues.append(f"{csv_path.name}: {err}")
                continue
            err_nan = _no_nan_inf_df(dataframe)
            if err_nan:
                issues.append(f"{csv_path.name}: {err_nan}")
            err_monotonic = _check_monotonic_index(dataframe)
            if err_monotonic:
                warnings.append(f"{csv_path.name}: {err_monotonic}")

        if kind == "ssl":
            feat_dir = ck_dir / "features"
            if feat_dir.exists():
                issues.extend(_check_features(feat_dir, model_dir.name))
            else:
                issues.append("missing: checkpoints/features/")

        status = "ok" if not issues else "fail"
        if status != "ok":
            failures += 1
        models_section.append(
            {
                "model_key": model_dir.name,
                "mode": kind,
                "status": status,
                "issues": issues,
                "warnings": warnings,
            }
        )
        print(f"[{status.upper()}] {model_dir.name} — issues:{len(issues)} warnings:{len(warnings)}")

    out_json = exp_path / "validation_report.json"
    out_md = exp_path / "validation_report.md"
    try:
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        lines = [f"# Validation report — {report['exp_id']} / {report['exp_name']}", ""]
        for model in models_section:
            lines.append(f"## {model['model_key']}  ·  {model['mode']}  ·  **{model['status']}**")
            if model["issues"]:
                lines.append("- **Issues:**")
                for item in model["issues"]:
                    lines.append(f"  - {item}")
            if model["warnings"]:
                lines.append("- Warnings:")
                for item in model["warnings"]:
                    lines.append(f"  - {item}")
            lines.append("")
        out_md.write_text("\n".join(lines), encoding="utf-8")
    except Exception:  # pragma: no cover - filesystem issues
        print("[WARN] Unable to write validation_report.{json,md}", file=sys.stderr)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(validate())
