"""
Canonical (fixed) filesystem layout for PLIP concept calibration + validation.

Paths (single source of truth, see explainability.paths):
  A) Calibration unified metadata (TRAIN+VAL scored separately, output merged):
     src/explainability/output/calibration/metadata/

  B) Deep validation (analysis of the calibration):
     src/explainability/output/calibration/analysis/

  C) Paper-ready report:
     src/explainability/output/calibration/analysis/report/

  D) Final shortlist for test (no-ROI and ROI):
     src/explainability/output/calibration/analysis/concepts_shortlist.yaml

Additional canonical pipelines:
  E) Concept XAI on TEST (NO-ROI, model-independent, computed once):
     src/explainability/output/no_roi/

  F) Concept XAI on TEST (ROI, model-dependent, uses spatial masks from a model):
     src/explainability/output/roi/<MODEL_ID>/

  G) Comparison ROI vs NO-ROI (paper-ready, per model):
     src/explainability/output/roi-no_roi-comparision/<MODEL_ID>/

NOTE:
  Folder name is intentionally "comparision" (typo kept for backward compatibility with your request).
"""
