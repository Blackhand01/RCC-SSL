"""
Calibration + deep validation for PLIP concept prompts.

Fixed layout (no runs/):
  - output/calibration/metadata/: unified calibration artifacts (TRAIN+VAL)
  - output/calibration/analysis/: deep validation outputs + audits
  - output/calibration/analysis/report/: paper-ready report (md/tables/figures)
  - output/calibration/analysis/concepts_shortlist.yaml: final shortlist for test
"""

from typing import Optional

from ...paths import CALIBRATION_PATHS, CalibrationLayout, ensure_calibration_layout

# Canonical layout (pulled directly from explainability.paths)
METADATA_DIR = CALIBRATION_PATHS.metadata_dir
ANALYSIS_DIR = CALIBRATION_PATHS.analysis_dir
REPORT_DIR = CALIBRATION_PATHS.report_dir
SHORTLIST_DIR = CALIBRATION_PATHS.shortlist_dir
SHORTLIST_JSON = CALIBRATION_PATHS.shortlist_json
SHORTLIST_YAML = CALIBRATION_PATHS.shortlist_yaml


def ensure_layout(layout: Optional[CalibrationLayout] = None) -> CalibrationLayout:
    """
    Ensure canonical calibration directories exist and return the resolved layout.
    """
    return ensure_calibration_layout(layout or CALIBRATION_PATHS)


__all__ = [
    "ANALYSIS_DIR",
    "SHORTLIST_DIR",
    "METADATA_DIR",
    "REPORT_DIR",
    "SHORTLIST_JSON",
    "SHORTLIST_YAML",
    "CALIBRATION_PATHS",
    "ensure_layout",
]
