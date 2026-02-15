#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# User-tunable (override via env)
# -----------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/home/mla_group_01/rcc-ssrl}"
VENV_PATH="${VENV_PATH:-/home/mla_group_01/.venvs/xai}"
PYTHON_BIN="${PYTHON_BIN:-python}"

ATTN_SCRIPT="${ATTN_SCRIPT:-$PROJECT_ROOT/src/explainability/spatial/attention_rollout.py}"

# Eval artifacts
EVAL_RUN_DIR="${EVAL_RUN_DIR:-}"                         # REQUIRED
PREDICTIONS_CSV="${PREDICTIONS_CSV:-predictions.csv}"
LOGITS_NPY="${LOGITS_NPY:-logits_test.npy}"

# Checkpoints
MODEL_NAME="${MODEL_NAME:-ssl_vit}"
BACKBONE_NAME="${BACKBONE_NAME:-vit_small_patch16_224}"
SSL_BACKBONE_CKPT="${SSL_BACKBONE_CKPT:-}"               # REQUIRED
SSL_HEAD_CKPT="${SSL_HEAD_CKPT:-}"                       # REQUIRED

# Data
DATA_BACKEND="${DATA_BACKEND:-webdataset}"               # webdataset | imagefolder
TEST_WDS_DIR="${TEST_WDS_DIR:-}"                         # REQUIRED if webdataset
WDS_PATTERN="${WDS_PATTERN:-*.tar}"
# FIX: dataset uses multi-extension keys (e.g. "img.jpg", "meta.json")
WDS_IMAGE_KEY="${WDS_IMAGE_KEY:-img.jpg;jpg}"
WDS_META_KEY="${WDS_META_KEY:-meta.json;json}"
TEST_IMAGEFOLDER_DIR="${TEST_IMAGEFOLDER_DIR:-}"         # REQUIRED if imagefolder

IMG_SIZE="${IMG_SIZE:-224}"
IMAGENET_NORM="${IMAGENET_NORM:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Runtime
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1337}"

# Outputs
OUTPUTS_ROOT="${OUTPUTS_ROOT:-$PROJECT_ROOT/outputs/xai_spatial}"
RUN_ID="${RUN_ID:-}"

# XAI
XAI_METHODS="${XAI_METHODS:-attn_rollout}"
ATNN_DISCARD_RATIO="${ATNN_DISCARD_RATIO:-0.9}"
GRADCAM_TARGET_LAYER="${GRADCAM_TARGET_LAYER:-backbone.model.blocks.11}"
IG_STEPS="${IG_STEPS:-32}"

# Selection
FULL_TEST="${FULL_TEST:-0}"
TOPK_TP="${TOPK_TP:-6}"
TOPK_FN="${TOPK_FN:-6}"
TOPK_FP="${TOPK_FP:-6}"
TOPK_LOWCONF="${TOPK_LOWCONF:-20}"

# Labels
CLASS_ORDER_JSON="${CLASS_ORDER_JSON:-[\"ccRCC\",\"pRCC\",\"chRCC\",\"oncocytoma\",\"unclassified\"]}"

# -----------------------------
# Basic checks
# -----------------------------
if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "[FATAL] PROJECT_ROOT not found: $PROJECT_ROOT" >&2
  exit 2
fi
if [[ ! -f "$ATTN_SCRIPT" ]]; then
  echo "[FATAL] attention_rollout.py not found: $ATTN_SCRIPT" >&2
  exit 2
fi
if [[ -z "${EVAL_RUN_DIR}" || ! -d "${EVAL_RUN_DIR}" ]]; then
  echo "[FATAL] Set EVAL_RUN_DIR to a valid eval directory containing predictions/logits." >&2
  echo "        Current EVAL_RUN_DIR: ${EVAL_RUN_DIR:-<empty>}" >&2
  echo "        Expected files: ${PREDICTIONS_CSV} and ${LOGITS_NPY}" >&2
  exit 2
fi
if [[ -z "${SSL_BACKBONE_CKPT}" || ! -f "${SSL_BACKBONE_CKPT}" ]]; then
  echo "[FATAL] Set SSL_BACKBONE_CKPT to an existing file." >&2
  exit 2
fi
if [[ -z "${SSL_HEAD_CKPT}" || ! -f "${SSL_HEAD_CKPT}" ]]; then
  echo "[FATAL] Set SSL_HEAD_CKPT to an existing file." >&2
  exit 2
fi

if [[ "${DATA_BACKEND}" == "webdataset" ]]; then
  if [[ -z "${TEST_WDS_DIR}" || ! -d "${TEST_WDS_DIR}" ]]; then
    echo "[FATAL] DATA_BACKEND=webdataset requires TEST_WDS_DIR (directory with .tar shards)." >&2
    exit 2
  fi
elif [[ "${DATA_BACKEND}" == "imagefolder" ]]; then
  if [[ -z "${TEST_IMAGEFOLDER_DIR}" || ! -d "${TEST_IMAGEFOLDER_DIR}" ]]; then
    echo "[FATAL] DATA_BACKEND=imagefolder requires TEST_IMAGEFOLDER_DIR." >&2
    exit 2
  fi
else
  echo "[FATAL] Unknown DATA_BACKEND=${DATA_BACKEND} (use webdataset|imagefolder)." >&2
  exit 2
fi

# -----------------------------
# Optional tee logs
# -----------------------------
if [[ -n "${LOG_DIR:-}" ]]; then
  mkdir -p "${LOG_DIR}"
  exec > >(tee -a "${LOG_DIR}/run_attention_rollout.${RUN_ID:-local}.$(date +%Y%m%d_%H%M%S).out") 2>&1
fi

# -----------------------------
# Env / venv
# -----------------------------
cd "$PROJECT_ROOT"
if [[ -n "${VENV_PATH}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# -----------------------------
# Build config YAML (generated)
# -----------------------------
mkdir -p "${OUTPUTS_ROOT}"
CFG_DIR="${OUTPUTS_ROOT}/_configs"
mkdir -p "${CFG_DIR}"

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

CFG_PATH="${CFG_DIR}/attn_rollout_${MODEL_NAME}_${RUN_ID}.yaml"

# EXPORT: everything that the Python block reads from os.environ
export PROJECT_ROOT DATA_BACKEND XAI_METHODS FULL_TEST TOPK_TP TOPK_FN TOPK_FP TOPK_LOWCONF
export CLASS_ORDER_JSON SEED RUN_ID OUTPUTS_ROOT DEVICE MODEL_NAME BACKBONE_NAME
export SSL_BACKBONE_CKPT SSL_HEAD_CKPT EVAL_RUN_DIR PREDICTIONS_CSV LOGITS_NPY
export IMG_SIZE IMAGENET_NORM BATCH_SIZE NUM_WORKERS
export TEST_WDS_DIR WDS_PATTERN WDS_IMAGE_KEY WDS_META_KEY TEST_IMAGEFOLDER_DIR
export ATNN_DISCARD_RATIO GRADCAM_TARGET_LAYER IG_STEPS
export CFG_PATH

"${PYTHON_BIN}" - <<'PY'
import json, os
from pathlib import Path
import yaml

def env(name, default=None):
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default

def env_int(name, default):
    return int(env(name, str(default)))

def env_float(name, default):
    return float(env(name, str(default)))

def env_bool01(name, default):
    v = str(env(name, str(default))).strip().lower()
    return v in ("1","true","yes","y","on")

data_backend = str(env("DATA_BACKEND", "webdataset")).lower()
xai_methods = [m.strip() for m in str(env("XAI_METHODS", "attn_rollout")).split(",") if m.strip()]

full_test = env_bool01("FULL_TEST", 0)
if full_test:
    topk_tp = 10**9
    topk_fn = 10**9
    topk_fp = 10**9
    topk_low = 0
else:
    topk_tp = env_int("TOPK_TP", 6)
    topk_fn = env_int("TOPK_FN", 6)
    topk_fp = env_int("TOPK_FP", 6)
    topk_low = env_int("TOPK_LOWCONF", 20)

class_order = json.loads(str(env("CLASS_ORDER_JSON", "[]")))
if not class_order:
    raise SystemExit("CLASS_ORDER_JSON is empty; set it to the correct class order.")

cfg = {
  "experiment": {
    "seed": env_int("SEED", 1337),
    "run_id": str(env("RUN_ID")),
    "outputs_root": str(env("OUTPUTS_ROOT")),
  },
  "runtime": {"device": str(env("DEVICE", "cuda"))},
  "model": {
    "name": str(env("MODEL_NAME", "ssl_vit")),
    "arch_hint": "ssl_linear",
    "backbone_name": str(env("BACKBONE_NAME", "vit_small_patch16_224")),
    "ssl_backbone_ckpt": str(env("SSL_BACKBONE_CKPT")),
    "ssl_head_ckpt": str(env("SSL_HEAD_CKPT")),
  },
  "evaluation_inputs": {
    "eval_run_dir": str(env("EVAL_RUN_DIR")),
    "predictions_csv": str(env("PREDICTIONS_CSV", "predictions.csv")),
    "logits_npy": str(env("LOGITS_NPY", "logits_test.npy")),
  },
  "labels": {"class_order": class_order},
  "data": {
    "backend": data_backend,
    "img_size": env_int("IMG_SIZE", 224),
    "imagenet_norm": env_bool01("IMAGENET_NORM", 1),
    "batch_size": env_int("BATCH_SIZE", 1),
    "num_workers": env_int("NUM_WORKERS", 8),
  },
  "selection": {
    "per_class": {"topk_tp": topk_tp, "topk_fn": topk_fn, "topk_fp": topk_fp},
    "global_low_conf": {"topk": topk_low},
  },
  "xai": {
    "methods": xai_methods,
    "attn_rollout": {"discard_ratio": env_float("ATNN_DISCARD_RATIO", 0.9)},
    "gradcam": {"target_layer": str(env("GRADCAM_TARGET_LAYER", "backbone.model.blocks.11"))},
    "ig": {"steps": env_int("IG_STEPS", 32)},
  },
}

if data_backend == "webdataset":
    cfg["data"]["webdataset"] = {
      "test_dir": str(env("TEST_WDS_DIR")),
      "pattern": str(env("WDS_PATTERN", "*.tar")),
      "image_key": str(env("WDS_IMAGE_KEY", "img.jpg;jpg")),
      "meta_key": str(env("WDS_META_KEY", "meta.json;json")),
    }
else:
    cfg["data"]["imagefolder"] = {"test_dir": str(env("TEST_IMAGEFOLDER_DIR"))}

out = Path(str(env("CFG_PATH")))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(str(out))
PY

echo "[INFO] Config written: ${CFG_PATH}"
echo "[INFO] Running: ${PYTHON_BIN} ${ATTN_SCRIPT} --config ${CFG_PATH}"
exec "${PYTHON_BIN}" "${ATTN_SCRIPT}" --config "${CFG_PATH}"
