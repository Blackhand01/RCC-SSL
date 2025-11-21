#!/bin/bash
#SBATCH -p gpu_a40
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
# The job name and log paths are overridden by --job-name/--output/--error from sbatch

set -euo pipefail

# -------- User/Repo paths -----------------------------------------------------
ROOT="/home/mla_group_01/rcc-ssrl"
TRAIN_DIR="${ROOT}/src/training"

cd "${TRAIN_DIR}"

# -------- Config argument (REQUIRED) ------------------------------------------
# Accept config YAML as first CLI arg; fail fast if missing
CONFIG_PATH="${1:-}"
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "[sbatch] ERROR: missing CONFIG_PATH argument (YAML file)." >&2
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[sbatch] ERROR: config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

# -------- Environment (venv/modules) ------------------------------------------
if [[ -d "${ROOT}/.venv" ]]; then
  source "${ROOT}/.venv/bin/activate"
fi

# Optional: user can set GPUS_PER_NODE externally (defaults to 1)
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"

echo "[sbatch] Host: $(hostname)"
echo "[sbatch] GPUs per node: ${GPUS_PER_NODE}"
echo "[sbatch] Config: ${CONFIG_PATH}"
echo "[sbatch] RUN_NAME=${RUN_NAME:-}"
echo "[sbatch] EXP_GROUP=${EXP_GROUP:-}"
echo "[sbatch] OUTPUTS_GROUP_DIR=${OUTPUTS_GROUP_DIR:-}"

# Export several conventional env vars so Python can locate the config
export CONFIG_PATH="${CONFIG_PATH}"
export TRAIN_CONFIG="${CONFIG_PATH}"
export RUN_CFG="${CONFIG_PATH}"

# -------- Launch ---------------------------------------------------------------
# torchrun is used as in your previous runs; --standalone for single-node
command -v torchrun >/dev/null 2>&1 || { echo "[sbatch] ERROR: torchrun not found"; exit 3; }

# Prefer an explicit --config ARG if your launcher supports it; otherwise the env var above is used.
torchrun --nproc_per_node="${GPUS_PER_NODE}" --standalone \
  "${TRAIN_DIR}/launch_training.py" --config "${CONFIG_PATH}"
