#!/usr/bin/env bash
#SBATCH -J xai_all
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/xai/run_all.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/xai/run_all.%j.err
#SBATCH -p gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# This job wraps the .sh orchestrator and runs it inside an allocation.
# SBATCH directives must appear before any non-comment lines and control the job submission. 

set -euo pipefail

# --------- user-configurable defaults (override at submission with --export=ALL,VAR=...) ----------
EXP_ROOT="${EXP_ROOT:-/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251123_220420_moco_v3}"
MODEL_NAME="${MODEL_NAME:-moco_v3_ssl_linear_best}"
BACKBONE_NAME="${BACKBONE_NAME:-vit_small_patch16_224}"
ONLY_SPATIAL="${ONLY_SPATIAL:-0}"      # 1 to enable
ONLY_CONCEPT="${ONLY_CONCEPT:-0}"      # 1 to enable
WITH_VLM_AGG="${WITH_VLM_AGG:-0}"      # 1 to enable
MIN_VLM_CONF="${MIN_VLM_CONF:-0.7}"
VENV_PATH="${VENV_PATH:-}"             # e.g., /home/mla_group_01/rcc-ssrl/.venvs/train

echo "[INFO] Host: $(hostname)"
echo "[INFO] EXP_ROOT=${EXP_ROOT}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] BACKBONE_NAME=${BACKBONE_NAME}"
echo "[INFO] ONLY_SPATIAL=${ONLY_SPATIAL} ONLY_CONCEPT=${ONLY_CONCEPT}"
echo "[INFO] WITH_VLM_AGG=${WITH_VLM_AGG} MIN_VLM_CONF=${MIN_VLM_CONF}"
echo "[INFO] VENV_PATH=${VENV_PATH}"

SCRIPT="/home/mla_group_01/rcc-ssrl/src/explainability/run_all_explainability.sh"
chmod +x "$SCRIPT"

# srun runs the job step inside the allocation provided by sbatch. 
SRUN_ARGS=( srun --ntasks=1 )

ARGS=( --experiment-root "$EXP_ROOT"
       --model-name "$MODEL_NAME"
       --backbone-name "$BACKBONE_NAME" )

if [[ "$ONLY_SPATIAL" == "1" ]]; then ARGS+=( --only-spatial ); fi
if [[ "$ONLY_CONCEPT" == "1" ]]; then ARGS+=( --only-concept ); fi
if [[ "$WITH_VLM_AGG" == "1" ]]; then ARGS+=( --with-vlm-aggregate --min-vlm-confidence "$MIN_VLM_CONF" ); fi
if [[ -n "$VENV_PATH" ]]; then ARGS+=( --venv "$VENV_PATH" ); fi

echo "[INFO] Command: ${SRUN_ARGS[*]} $SCRIPT ${ARGS[*]}"
"${SRUN_ARGS[@]}" "$SCRIPT" "${ARGS[@]}"
