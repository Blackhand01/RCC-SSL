#!/usr/bin/env bash
#SBATCH --job-name=attn_rollout
#SBATCH -o /home/mla_group_01/src/logs/xai/attn_rollout.%j.out
#SBATCH -e /home/mla_group_01/src/logs/xai/attn_rollout.%j.err
#SBATCH -p gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --exclude=compute-5-14,compute-5-11,compute-3-12

set -euo pipefail

PROJECT_ROOT="/home/mla_group_01/rcc-ssrl"
mkdir -p "${PROJECT_ROOT}/src/logs/xai" || true

module purge || true

# -----------------------------
# REQUIRED: eval + ckpts (i-JEPA abl01)
# -----------------------------
ABL_ROOT="/mnt/beegfs-compat/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/models/ablation_final/exp_20251213_121650_i_jepa/exp_i_jepa_abl01"

EVAL_RUN_DIR="${EVAL_RUN_DIR:-${ABL_ROOT}/eval/i_jepa_ssl_linear_best/20251213_121650}"
SSL_BACKBONE_CKPT="${SSL_BACKBONE_CKPT:-${ABL_ROOT}/checkpoints/i_jepa__ssl_best.pt}"
SSL_HEAD_CKPT="${SSL_HEAD_CKPT:-${ABL_ROOT}/checkpoints/i_jepa__ssl_linear_best.pt}"

# -----------------------------
# DATA (WebDataset test)
# -----------------------------
DATA_BACKEND="${DATA_BACKEND:-webdataset}"

TEST_WDS_DIR="${TEST_WDS_DIR:-/mnt/beegfs-compat/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/data/processed/rcc_webdataset_final/test}"
WDS_PATTERN="${WDS_PATTERN:-*.tar}"
# FIX: le chiavi reali nel tar sono "img.jpg" e "meta.json" (fallback compat: "jpg"/"json")
WDS_IMAGE_KEY="${WDS_IMAGE_KEY:-img.jpg;jpg}"
WDS_META_KEY="${WDS_META_KEY:-meta.json;json}"

# -----------------------------
# MODEL
# -----------------------------
MODEL_NAME="${MODEL_NAME:-i_jepa_ssl_vit}"
BACKBONE_NAME="${BACKBONE_NAME:-vit_small_patch16_224}"

# -----------------------------
# OUTPUTS
# -----------------------------
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs/xai_spatial}"
RUN_ID="${RUN_ID:-i_jepa_abl01_$(date +%Y%m%d_%H%M%S)}"

# -----------------------------
# XAI controls
# -----------------------------
XAI_METHODS="${XAI_METHODS:-attn_rollout}"
ATNN_DISCARD_RATIO="${ATNN_DISCARD_RATIO:-0.9}"

# subset vs full-test
FULL_TEST="${FULL_TEST:-0}"
TOPK_TP="${TOPK_TP:-6}"
TOPK_FN="${TOPK_FN:-6}"
TOPK_FP="${TOPK_FP:-6}"
TOPK_LOWCONF="${TOPK_LOWCONF:-20}"

# perf/runtime
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMG_SIZE="${IMG_SIZE:-224}"
IMAGENET_NORM="${IMAGENET_NORM:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-1337}"

# venv (opzionale)
VENV_PATH="${VENV_PATH:-/home/mla_group_01/.venvs/xai}"
if [[ -n "${VENV_PATH}" && -d "${VENV_PATH}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

# -----------------------------
# Sanity checks (fail fast)
# -----------------------------
if [[ ! -d "${EVAL_RUN_DIR}" ]]; then
  echo "[FATAL] EVAL_RUN_DIR non esiste: ${EVAL_RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_RUN_DIR}/predictions.csv" ]] || [[ ! -f "${EVAL_RUN_DIR}/logits_test.npy" ]]; then
  echo "[FATAL] EVAL_RUN_DIR deve contenere predictions.csv e logits_test.npy: ${EVAL_RUN_DIR}" >&2
  ls -la "${EVAL_RUN_DIR}" || true
  exit 3
fi
if [[ ! -f "${SSL_BACKBONE_CKPT}" ]]; then
  echo "[FATAL] SSL_BACKBONE_CKPT non trovato: ${SSL_BACKBONE_CKPT}" >&2
  exit 4
fi
if [[ ! -f "${SSL_HEAD_CKPT}" ]]; then
  echo "[FATAL] SSL_HEAD_CKPT non trovato: ${SSL_HEAD_CKPT}" >&2
  exit 5
fi
if [[ "${DATA_BACKEND}" != "webdataset" ]]; then
  echo "[FATAL] Questo job e' configurato per webdataset. DATA_BACKEND=${DATA_BACKEND}" >&2
  exit 6
fi
if [[ ! -d "${TEST_WDS_DIR}" ]]; then
  echo "[FATAL] TEST_WDS_DIR non esiste: ${TEST_WDS_DIR}" >&2
  exit 7
fi

# -----------------------------
# Run
# -----------------------------
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

export PROJECT_ROOT VENV_PATH
export EVAL_RUN_DIR SSL_BACKBONE_CKPT SSL_HEAD_CKPT
export DATA_BACKEND TEST_WDS_DIR WDS_PATTERN WDS_IMAGE_KEY WDS_META_KEY
export MODEL_NAME BACKBONE_NAME
export OUTPUTS_ROOT RUN_ID
export XAI_METHODS ATNN_DISCARD_RATIO
export FULL_TEST TOPK_TP TOPK_FN TOPK_FP TOPK_LOWCONF
export DEVICE NUM_WORKERS IMG_SIZE IMAGENET_NORM BATCH_SIZE SEED

exec "${PROJECT_ROOT}/src/explainability/spatial/run_attention_rollout.sh"
