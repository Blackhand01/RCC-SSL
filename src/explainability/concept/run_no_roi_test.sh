#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# Run PLIP Concept XAI on TEST WITHOUT ROI (NO-ROI)
# Canonical output under:
#   $PROJECT_ROOT/src/explainability/output/no_roi/
# ------------------------------------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-/home/mla_group_01/rcc-ssrl}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venvs/xai}"

CONFIG_NO_ROI="${CONFIG_NO_ROI:-$PROJECT_ROOT/src/explainability/configs/no_roi.yaml}"

# Canonical calibration outputs
CALIB_METADATA_DIR_DEFAULT="$PROJECT_ROOT/src/explainability/output/calibration/metadata"
SHORTLIST_YAML_DEFAULT="$PROJECT_ROOT/src/explainability/output/calibration/analysis/concepts_shortlist.yaml"

CALIBRATION_METADATA_DIR="${CALIBRATION_METADATA_DIR:-$CALIB_METADATA_DIR_DEFAULT}"
CONCEPT_SHORTLIST_YAML="${CONCEPT_SHORTLIST_YAML:-$SHORTLIST_YAML_DEFAULT}"

# Optional: override TEST dir via env (runner can also read it from YAML)
WDS_TEST_DIR="${WDS_TEST_DIR:-}"

# Subset knobs (optional). They map to CLI flags supported by run_no_roi.py.
MAX_PATCHES="${MAX_PATCHES:-}"
SUBSET_PROB="${SUBSET_PROB:-}"
SUBSET_KEYS="${SUBSET_KEYS:-}"
SUBSET_SEED="${SUBSET_SEED:-0}"

# Overwrite canonical no_roi artifacts (optional)
OVERWRITE="${OVERWRITE:-0}"

# --- CRITICAL: do not leak ~/.local into Python ---
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Make imports robust regardless of repo layout
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Hugging Face cache (prefer HF_HOME; avoids TRANSFORMERS_CACHE warning)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
# HF_HOME is the supported knob for cache root :contentReference[oaicite:2]{index=2}

# ------------------------------------------------------------------
# Hard preflight (fail fast)
# ------------------------------------------------------------------
if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "[FATAL] PROJECT_ROOT not found: $PROJECT_ROOT" >&2
  exit 2
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[FATAL] Venv not found: $VENV_DIR" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[FATAL] PYTHON_BIN not executable: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$PROJECT_ROOT/src/explainability/output/no_roi/logs"

# Check canonical inputs exist
if [[ ! -d "$CALIBRATION_METADATA_DIR" ]]; then
  echo "[FATAL] Calibration metadata dir not found: $CALIBRATION_METADATA_DIR" >&2
  exit 2
fi
for req in "concepts.json" "text_features.pt"; do
  if [[ ! -f "$CALIBRATION_METADATA_DIR/$req" ]]; then
    echo "[FATAL] Missing calibration artifact: $CALIBRATION_METADATA_DIR/$req" >&2
    exit 2
  fi
done
if [[ ! -f "$CONCEPT_SHORTLIST_YAML" ]]; then
  echo "[FATAL] Shortlist YAML not found: $CONCEPT_SHORTLIST_YAML" >&2
  exit 2
fi
if [[ -n "$WDS_TEST_DIR" ]] && [[ ! -d "$WDS_TEST_DIR" ]]; then
  echo "[FATAL] WDS_TEST_DIR does not exist: $WDS_TEST_DIR" >&2
  exit 2
fi

# ------------------------------------------------------------------
# Binary-compat sanity checks (this is your actual failure)
# ------------------------------------------------------------------
"$PYTHON_BIN" - <<'PY'
import sys
print("[INFO] sys.executable =", sys.executable)
try:
    import numpy as np
    print("[INFO] numpy =", np.__version__)
    major = int(np.__version__.split(".", 1)[0])
    if major >= 2:
        raise SystemExit(
            "[FATAL] Detected NumPy >= 2 in the active env. "
            "Your stack (torch/sklearn wheels) is very likely built against NumPy 1.x.\n"
            "Fix inside the venv:\n"
            "  python -m pip install -U --force-reinstall 'numpy<2'\n"
            "  python -m pip install -U --force-reinstall scikit-learn\n"
        )
except Exception as e:
    raise SystemExit(f"[FATAL] NumPy import failed: {e}")

# sklearn import is where your traceback dies
try:
    import sklearn
    print("[INFO] sklearn =", sklearn.__version__)
except Exception as e:
    raise SystemExit(
        "[FATAL] scikit-learn import failed (binary mismatch vs NumPy).\n"
        "Fix inside the venv:\n"
        "  python -m pip install -U --force-reinstall 'numpy<2'\n"
        "  python -m pip install -U --force-reinstall scikit-learn\n"
        f"Original error: {e}"
    )

# torch sanity
try:
    import torch
    print("[INFO] torch =", torch.__version__)
    print("[INFO] cuda available =", torch.cuda.is_available())
except Exception as e:
    raise SystemExit(f"[FATAL] torch import failed: {e}")
PY

# ------------------------------------------------------------------
# Build CLI args
# ------------------------------------------------------------------
args=()
args+=( "--config" "$CONFIG_NO_ROI" )
args+=( "--calibration-metadata-dir" "$CALIBRATION_METADATA_DIR" )
args+=( "--shortlist-yaml" "$CONCEPT_SHORTLIST_YAML" )

if [[ -n "$WDS_TEST_DIR" ]]; then
  args+=( "--test-dir" "$WDS_TEST_DIR" )
fi
if [[ -n "$MAX_PATCHES" ]]; then
  args+=( "--max-patches" "$MAX_PATCHES" )
fi
if [[ -n "$SUBSET_PROB" ]]; then
  args+=( "--subset-prob" "$SUBSET_PROB" "--subset-seed" "$SUBSET_SEED" )
fi
if [[ -n "$SUBSET_KEYS" ]]; then
  args+=( "--subset-keys" "$SUBSET_KEYS" )
fi
if [[ "$OVERWRITE" == "1" ]]; then
  args+=( "--overwrite" )
fi

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
echo "[INFO] CONFIG_NO_ROI=$CONFIG_NO_ROI"
echo "[INFO] CALIBRATION_METADATA_DIR=$CALIBRATION_METADATA_DIR"
echo "[INFO] CONCEPT_SHORTLIST_YAML=$CONCEPT_SHORTLIST_YAML"
if [[ -n "$WDS_TEST_DIR" ]]; then echo "[INFO] WDS_TEST_DIR=$WDS_TEST_DIR"; fi
if [[ -n "$MAX_PATCHES" ]]; then echo "[INFO] MAX_PATCHES=$MAX_PATCHES"; fi
if [[ -n "$SUBSET_PROB" ]]; then echo "[INFO] SUBSET_PROB=$SUBSET_PROB (seed=$SUBSET_SEED)"; fi
if [[ -n "$SUBSET_KEYS" ]]; then echo "[INFO] SUBSET_KEYS=$SUBSET_KEYS"; fi
echo "[INFO] OVERWRITE=$OVERWRITE"

cd "$PROJECT_ROOT"

# ------------------------------------------------------------------
# Execute robustly (try both common module layouts; fallback to file)
# ------------------------------------------------------------------
if "$PYTHON_BIN" -c "import importlib; importlib.import_module('src.explainability.concept.run_no_roi')" >/dev/null 2>&1; then
  exec "$PYTHON_BIN" -u -m src.explainability.concept.run_no_roi "${args[@]}"
elif "$PYTHON_BIN" -c "import importlib; importlib.import_module('explainability.concept.run_no_roi')" >/dev/null 2>&1; then
  exec "$PYTHON_BIN" -u -m explainability.concept.run_no_roi "${args[@]}"
else
  exec "$PYTHON_BIN" -u "$PROJECT_ROOT/src/explainability/concept/run_no_roi.py" "${args[@]}"
fi
