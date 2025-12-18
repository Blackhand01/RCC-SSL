"""
Canonical (fixed) filesystem layout for PLIP concept calibration + validation.

Paths (single source of truth, see explainability.paths):
  A) Calibration unified metadata (TRAIN+VAL scored separately, output merged):
     outputs/xai/concept/calibration/metadata/

  B) Deep validation (analysis of the calibration):
     outputs/xai/concept/calibration/analysis/

  C) Paper-ready report:
     outputs/xai/concept/calibration/analysis/report/

  D) Final shortlist for test (no-ROI and ROI):
     src/explainability/configs/concepts_shortlist.yaml

Additional canonical pipelines:
  E) Concept XAI on TEST (NO-ROI, model-independent, computed once):
     outputs/xai/concept/no_roi/

  F) Concept XAI on TEST (ROI, model-dependent, uses spatial masks from a model):
     <MODEL_ROOT>/xai/concept/roi/

  G) Comparison ROI vs NO-ROI (paper-ready, per model):
     outputs/xai/concept/comparision/<MODEL_ID>/

NOTE:
  Folder name is intentionally "comparision" (typo kept for backward compatibility with your request).
"""
