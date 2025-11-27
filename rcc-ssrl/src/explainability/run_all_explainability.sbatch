#!/usr/bin/env bash
#SBATCH -J xai_all
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/xai/run_all.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/xai/run_all.%j.err
#SBATCH -p gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# This job wraps the *existing* run_full_xai.sh orchestrator and runs it inside an allocation.

set -euo pipefail

# --------- user-configurable defaults (can be overridden via sbatch --export=ALL,VAR=...) ----------
# N.B.: questi valori sono solo fallback; i reali possono essere esportati prima di chiamare sbatch.

EXP_ROOT="${EXP_ROOT:-/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3}"
MODEL_NAME="${MODEL_NAME:-moco_v3_ssl_linear_best}"
BACKBONE_NAME="${BACKBONE_NAME:-vit_small_patch16_224}"

ONLY_SPATIAL="${ONLY_SPATIAL:-0}"      # 1 to enable (non ancora usato in run_full_xai.sh)
ONLY_CONCEPT="${ONLY_CONCEPT:-0}"      # 1 to enable (non ancora usato in run_full_xai.sh)

WITH_VLM_AGG="${WITH_VLM_AGG:-0}"      # legacy; non usato dall’orchestratore attuale
MIN_VLM_CONF="${MIN_VLM_CONF:-0.7}"

# venv per rcc-ssrl (train + explainability)
VENV_PATH="${VENV_PATH:-}"

echo "[INFO] Host: $(hostname)"
echo "[INFO] EXP_ROOT=${EXP_ROOT}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] BACKBONE_NAME=${BACKBONE_NAME}"
echo "[INFO] ONLY_SPATIAL=${ONLY_SPATIAL} ONLY_CONCEPT=${ONLY_CONCEPT}"
echo "[INFO] WITH_VLM_AGG=${WITH_VLM_AGG} MIN_VLM_CONF=${MIN_VLM_CONF}"
echo "[INFO] VENV_PATH=${VENV_PATH}"

# Orchestrator reale (esiste già)
SCRIPT="/home/mla_group_01/rcc-ssrl/src/explainability/run_full_xai.sh"

if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERROR] Orchestrator script not found: $SCRIPT" >&2
  exit 1
fi

chmod +x "$SCRIPT"

# srun runs the job step inside the allocation provided by sbatch.
SRUN_ARGS=( srun --ntasks=1 )

echo "[INFO] Command: ${SRUN_ARGS[*]} $SCRIPT"
"${SRUN_ARGS[@]}" "$SCRIPT"
