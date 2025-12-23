#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single source of truth for explainability filesystem layout + central configs.

Goals:
  - One canonical layout (no timestamped runs) with optional env override.
  - Centralised config directory: src/explainability/configs/
  - Outputs stay under src/explainability/output/ by default.
  - Avoid hard-coded absolute paths inside code; compute relative to repo.

Back-compat:
  - Default artifact root is src/explainability/output unless XAI_ROOT is set.

New (unified pipeline):
  - Light outputs (stats-only) remain under src/explainability/output/...
  - Heavy per-patch artifacts (input/rollout/ROI/overlays) live under each model root on scratch:
      <MODEL_ROOT>/attention_rollout_concept/run_<RUN_ID>/
  - Experiment discovery helpers for scratch model runs live here as the single source of truth.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# Repo / package roots
# ---------------------------------------------------------------------

EXPLAINABILITY_DIR = Path(__file__).resolve().parent  # .../src/explainability
SRC_DIR = EXPLAINABILITY_DIR.parent                   # .../src
REPO_ROOT = SRC_DIR.parent                            # .../ (repo root)

# Centralised configs directory
CONFIG_DIR = EXPLAINABILITY_DIR / "configs"
# Canonical output root (default)
OUTPUT_DIR = EXPLAINABILITY_DIR / "output"

# Scratch model root (defaults for the RCC cluster)
MODELS_ROOT_DEFAULT = Path(
    "/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/models"
)


def _env_path(key: str) -> Optional[Path]:
    v = os.getenv(key, "").strip()
    if not v:
        return None
    return Path(v)


# Canonical artifacts root.
# Default keeps outputs under src/explainability/output unless XAI_ROOT is set.
XAI_ROOT = _env_path("XAI_ROOT") or OUTPUT_DIR


def resolve_config(path_or_name: Union[str, Path]) -> Path:
    """
    Resolve a config file path.
      - If an existing path is provided -> return it.
      - Else interpret it as a filename under CONFIG_DIR.
    """
    p = Path(path_or_name)
    if p.exists():
        return p
    return CONFIG_DIR / str(path_or_name)


def resolve_models_root(models_root: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve the scratch models root:
      - explicit models_root arg wins
      - else env MODELS_ROOT
      - else MODELS_ROOT_DEFAULT
    """
    if models_root is not None:
        return Path(models_root).expanduser()
    return (_env_path("MODELS_ROOT") or MODELS_ROOT_DEFAULT).expanduser()


# ---------------------------------------------------------------------
# Layout dataclasses
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationLayout:
    root_dir: Path
    metadata_dir: Path
    analysis_dir: Path
    report_dir: Path
    shortlist_dir: Path
    shortlist_yaml: Path
    shortlist_json: Path

    @property
    def configs_dir(self) -> Path:
        # Legacy alias (shortlist artifacts no longer live under configs/).
        return self.shortlist_dir


@dataclass(frozen=True)
class NoRoiLayout:
    root_dir: Path
    artifacts_dir: Path
    plots_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class SpatialLayout:
    """
    Model-dependent spatial XAI layout.
    Stored under <XAI_ROOT>/spatial/<MODEL_ID>/...
    """
    root_dir: Path
    artifacts_dir: Path
    plots_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class RoiConceptLayout:
    """
    Model-dependent concept XAI with ROI masks (depends on spatial outputs).
    Light artifacts (arrays/JSON) are stored under <XAI_ROOT>/roi/<MODEL_ID>/...
    Heavy ROI crops/overlays are stored under <MODEL_ROOT>/xai/roi/...
    """
    root_dir: Path
    artifacts_dir: Path
    rois_dir: Path
    figures_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class ComparisonLayout:
    root_dir: Path
    figures_dir: Path
    summary_csv: Path
    report_md: Path


@dataclass(frozen=True)
class SpatialConceptHeavyLayout:
    """
    Heavy, per-patch artifacts for unified spatial+concept XAI.
    Stored under: <MODEL_ROOT>/attention_rollout_concept/run_<RUN_ID>/
    """
    root_dir: Path
    selection_dir: Path
    items_dir: Path
    selection_json: Path
    summary_csv: Path
    summary_json: Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Canonical layout builders
# ---------------------------------------------------------------------

def calibration_layout(root: Optional[Path] = None) -> CalibrationLayout:
    """
    Canonical calibration + deep validation layout.
    shortlist_dir is under the analysis output (no artifacts in configs/).
    """
    base = (Path(root) if root is not None else XAI_ROOT) / "calibration"
    meta = base / "metadata"
    analysis = base / "analysis"
    report = analysis / "report"
    shortlist_dir = analysis
    return CalibrationLayout(
        root_dir=base,
        metadata_dir=meta,
        analysis_dir=analysis,
        report_dir=report,
        shortlist_dir=shortlist_dir,
        shortlist_yaml=shortlist_dir / "concepts_shortlist.yaml",
        shortlist_json=shortlist_dir / "concepts_shortlist.json",
    )


def no_roi_layout(root: Optional[Path] = None) -> NoRoiLayout:
    """
    Canonical NO-ROI concept scoring on TEST (model-independent).
    """
    base = (Path(root) if root is not None else XAI_ROOT) / "no_roi"
    return NoRoiLayout(
        root_dir=base,
        artifacts_dir=base / "artifacts",
        plots_dir=base / "plots",
        logs_dir=base / "logs",
    )

def _model_id(model_root: Union[str, Path]) -> str:
    return Path(model_root).name


def model_xai_root(model_root: Union[str, Path]) -> Path:
    """
    Legacy helper retained for compatibility with older code.
    Prefer spatial_layout/roi_concept_layout for canonical outputs under XAI_ROOT.
    """
    return Path(model_root) / "xai"


def spatial_layout(model_root: Union[str, Path]) -> SpatialLayout:
    base = XAI_ROOT / "spatial" / _model_id(model_root)
    return SpatialLayout(
        root_dir=base,
        artifacts_dir=base / "artifacts",
        plots_dir=base / "plots",
        logs_dir=base / "logs",
    )


def roi_concept_layout(model_root: Union[str, Path]) -> RoiConceptLayout:
    model_root_p = Path(model_root)
    base = XAI_ROOT / "roi" / _model_id(model_root_p)
    # Heavy outputs (per-sample crops/overlays) must not live under the repo output.
    heavy = model_root_p / "xai" / "roi"
    return RoiConceptLayout(
        root_dir=base,
        artifacts_dir=base / "artifacts",
        rois_dir=heavy / "rois",
        figures_dir=heavy / "figures",
        logs_dir=base / "logs",
    )


def comparison_layout(model_id: str) -> ComparisonLayout:
    base = XAI_ROOT / "roi-no_roi-comparision" / str(model_id)
    tables_dir = base / "tables"
    return ComparisonLayout(
        root_dir=base,
        figures_dir=base / "figures",
        summary_csv=tables_dir / "roi_vs_no_roi_summary.csv",
        report_md=base / "report.md",
    )


def spatial_concept_heavy_layout(model_root: Union[str, Path], run_id: str) -> SpatialConceptHeavyLayout:
    """
    Heavy artifacts layout for unified spatial+concept XAI (per model root).
    """
    mr = Path(model_root)
    root = mr / "attention_rollout_concept" / f"run_{str(run_id)}"
    selection_dir = root / "selection"
    items_dir = root / "items"
    return SpatialConceptHeavyLayout(
        root_dir=root,
        selection_dir=selection_dir,
        items_dir=items_dir,
        selection_json=selection_dir / "xai_selection.json",
        summary_csv=root / "xai_summary.csv",
        summary_json=root / "xai_summary.json",
    )


def ensure_spatial_concept_heavy_layout(layout: SpatialConceptHeavyLayout) -> SpatialConceptHeavyLayout:
    _ensure_dir(layout.root_dir)
    _ensure_dir(layout.selection_dir)
    _ensure_dir(layout.items_dir)
    return layout


# ---------------------------------------------------------------------
# Ensure helpers (used by runners)
# ---------------------------------------------------------------------

def ensure_calibration_layout(layout: Optional[CalibrationLayout] = None) -> CalibrationLayout:
    l = layout or calibration_layout()
    _ensure_dir(l.root_dir)
    _ensure_dir(l.metadata_dir)
    _ensure_dir(l.analysis_dir)
    _ensure_dir(l.report_dir)
    _ensure_dir(l.shortlist_dir)
    return l


def ensure_no_roi_layout(layout: Optional[NoRoiLayout] = None) -> NoRoiLayout:
    l = layout or no_roi_layout()
    _ensure_dir(l.root_dir)
    _ensure_dir(l.artifacts_dir)
    _ensure_dir(l.plots_dir)
    _ensure_dir(l.logs_dir)
    return l


def ensure_spatial_layout(layout: SpatialLayout) -> SpatialLayout:
    _ensure_dir(layout.artifacts_dir)
    _ensure_dir(layout.plots_dir)
    _ensure_dir(layout.logs_dir)
    return layout


def ensure_roi_concept_layout(model_root: Union[str, Path, RoiConceptLayout]) -> RoiConceptLayout:
    if isinstance(model_root, RoiConceptLayout):
        layout = model_root
    else:
        layout = roi_concept_layout(model_root)
    _ensure_dir(layout.artifacts_dir)
    # Heavy dirs live under model_root (scratch) - still ensure them.
    _ensure_dir(layout.rois_dir)
    _ensure_dir(layout.figures_dir)
    _ensure_dir(layout.logs_dir)
    return layout


def ensure_roi_layout(model_root: Union[str, Path]) -> RoiConceptLayout:
    """
    Backward-compatible alias used by run_spatial-concept.py.
    """
    return ensure_roi_concept_layout(roi_concept_layout(model_root))


def ensure_comparison_layout(model_id: str) -> ComparisonLayout:
    l = comparison_layout(model_id)
    _ensure_dir(l.figures_dir)
    _ensure_dir(l.summary_csv.parent)
    _ensure_dir(l.report_md.parent)
    return l


def get_heavy_xai_dir(model_root: Union[str, Path], run_id: str, *, kind: str = "spatial_concept") -> Path:
    """
    Resolve heavy XAI directory under a model root.
    kind:
      - spatial_concept -> <MODEL_ROOT>/attention_rollout_concept/run_<RUN_ID>/
    """
    kind = str(kind).strip().lower()
    mr = Path(model_root)
    if kind in ("spatial_concept", "attention_rollout_concept", "roi"):
        return mr / "attention_rollout_concept" / f"run_{str(run_id)}"
    # Default fallback: keep heavy XAI under model_root/xai/<kind>/run_<id>
    return mr / "xai" / kind / f"run_{str(run_id)}"


def get_item_out_dir(model_root: Union[str, Path], run_id: str, idx: int, *, kind: str = "spatial_concept") -> Path:
    """
    Item output dir for a single selected sample under the heavy layout.
    Canonical name uses 8 digits: idx_00001234
    """
    base = get_heavy_xai_dir(model_root, run_id, kind=kind)
    return Path(base) / "items" / f"idx_{int(idx):08d}"


def get_light_stats_dir(kind: str, model_id: str) -> Path:
    """
    Resolve the canonical light (stats-only) output directory inside the repo.
    """
    kind_norm = str(kind).strip().lower()
    mid = str(model_id)
    if kind_norm in ("spatial", "spatial_stats", "stats_spatial"):
        return OUTPUT_DIR / "spatial" / mid
    if kind_norm in ("roi", "roi_stats", "stats_roi"):
        return OUTPUT_DIR / "roi" / mid
    if kind_norm in ("roi-no_roi-comparision", "comparision", "comparison"):
        return OUTPUT_DIR / "roi-no_roi-comparision" / mid
    if kind_norm in ("no_roi", "no-roi"):
        return OUTPUT_DIR / "no_roi"
    if kind_norm in ("calibration",):
        return OUTPUT_DIR / "calibration"
    return OUTPUT_DIR / kind_norm / mid


# ---------------------------------------------------------------------
# Experiment discovery + resolvers (scratch models)
# ---------------------------------------------------------------------

def iter_exp_roots(models_root: Union[str, Path], exp_prefix: str) -> Iterator[Path]:
    """
    Iterate experiment roots under models_root matching exp_prefix (sorted by name).
    """
    root = Path(models_root)
    if not root.exists() or not root.is_dir():
        return iter(())
    exps = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(str(exp_prefix))]
    exps = sorted(exps, key=lambda p: p.name)
    return iter(exps)


def iter_ablation_dirs(exp_root: Union[str, Path]) -> Iterator[Path]:
    """
    Iterate ablation dirs under an exp root (sorted by name).
    Expected pattern: exp_*_ablXX
    """
    er = Path(exp_root)
    if not er.exists() or not er.is_dir():
        return iter(())
    abls = [p for p in er.iterdir() if p.is_dir() and ("_abl" in p.name)]
    abls = sorted(abls, key=lambda p: p.name)
    return iter(abls)


def resolve_checkpoints(ablation_dir: Union[str, Path]) -> Optional[Dict[str, Path]]:
    """
    Resolve required checkpoints under an ablation dir.
    Returns dict with keys:
      - ssl_backbone_ckpt
      - ssl_head_ckpt
    """
    ad = Path(ablation_dir)
    ckpt_dir = ad / "checkpoints"
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        return None

    # Backbone: *_ssl_best.pt but NOT *_ssl_linear_best.pt
    backbone = sorted(
        [p for p in ckpt_dir.glob("*_ssl_best.pt") if "linear" not in p.name.lower()],
        key=lambda p: p.name,
    )
    head = sorted(list(ckpt_dir.glob("*_ssl_linear_best.pt")), key=lambda p: p.name)
    if not backbone or not head:
        return None

    return {
        "ssl_backbone_ckpt": backbone[-1],
        "ssl_head_ckpt": head[-1],
    }


def resolve_latest_eval_dir(ablation_dir: Union[str, Path], pattern: str = "*_ssl_linear_best*") -> Optional[Path]:
    """
    Resolve latest eval dir for an ablation:
      <ablation_dir>/eval/<something matching pattern>/<TIMESTAMP>/
    Chooses latest TIMESTAMP (lexicographic), and if multiple parents match, chooses
    the latest (parent, timestamp) lexicographically.
    """
    ad = Path(ablation_dir)
    eval_root = ad / "eval"
    if not eval_root.exists() or not eval_root.is_dir():
        return None

    parents = sorted(
        [p for p in eval_root.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, pattern)],
        key=lambda p: p.name,
    )
    candidates: List[Tuple[str, str, Path]] = []
    for par in parents:
        ts_dirs = sorted([d for d in par.iterdir() if d.is_dir()], key=lambda p: p.name)
        if not ts_dirs:
            continue
        ts = ts_dirs[-1]
        candidates.append((par.name, ts.name, ts))
    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda t: (t[0], t[1]))
    return candidates[-1][2]


# ---------------------------------------------------------------------
# Canonical exported constants
# ---------------------------------------------------------------------

CALIBRATION_PATHS = calibration_layout()
NO_ROI_PATHS = no_roi_layout()

# Central config file defaults (optional convenience)
CALIBRATION_CONFIG_YAML = CONFIG_DIR / "calibration.yaml"
NO_ROI_CONFIG_YAML = CONFIG_DIR / "no_roi.yaml"
SPATIAL_CONFIG_YAML = CONFIG_DIR / "spatial.yaml"
SPATIAL_CONCEPT_CONFIG_YAML = CONFIG_DIR / "roi.yaml"
CONCEPT_PLIP_CONFIG_YAML = CONFIG_DIR / "config_concept_plip.yaml"
CONCEPTS_LIST_YAML = CONFIG_DIR / "concepts_list.yaml"
CONCEPT_SHORTLIST_YAML_CFG = CONFIG_DIR / "concepts_shortlist.yaml"
CONCEPT_SHORTLIST_JSON_CFG = CONFIG_DIR / "concepts_shortlist.json"
CONCEPT_SHORTLIST_FLAT_CSV_CFG = CONFIG_DIR / "concepts_shortlist_flat.csv"


__all__ = [
    "EXPLAINABILITY_DIR",
    "SRC_DIR",
    "REPO_ROOT",
    "MODELS_ROOT_DEFAULT",
    "resolve_models_root",
    "XAI_ROOT",
    "CONFIG_DIR",
    "OUTPUT_DIR",
    "resolve_config",
    "CalibrationLayout",
    "NoRoiLayout",
    "SpatialLayout",
    "RoiConceptLayout",
    "ComparisonLayout",
    "SpatialConceptHeavyLayout",
    "CALIBRATION_PATHS",
    "NO_ROI_PATHS",
    "CALIBRATION_CONFIG_YAML",
    "NO_ROI_CONFIG_YAML",
    "SPATIAL_CONFIG_YAML",
    "SPATIAL_CONCEPT_CONFIG_YAML",
    "CONCEPT_PLIP_CONFIG_YAML",
    "CONCEPTS_LIST_YAML",
    "CONCEPT_SHORTLIST_YAML_CFG",
    "CONCEPT_SHORTLIST_JSON_CFG",
    "CONCEPT_SHORTLIST_FLAT_CSV_CFG",
    "ensure_calibration_layout",
    "ensure_no_roi_layout",
    "model_xai_root",
    "spatial_layout",
    "roi_concept_layout",
    "ensure_spatial_layout",
    "ensure_roi_concept_layout",
    "ensure_roi_layout",
    "comparison_layout",
    "ensure_comparison_layout",
    "spatial_concept_heavy_layout",
    "ensure_spatial_concept_heavy_layout",
    "get_heavy_xai_dir",
    "get_item_out_dir",
    "get_light_stats_dir",
    "iter_exp_roots",
    "iter_ablation_dirs",
    "resolve_checkpoints",
    "resolve_latest_eval_dir",
]
