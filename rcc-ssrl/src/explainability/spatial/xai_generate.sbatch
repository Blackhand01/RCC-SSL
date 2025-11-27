#!/usr/bin/env bash
#SBATCH -J xai_spatial_vit
#SBATCH -o ${LOG_DIR}/xai_spatial.%j.out
#SBATCH -e ${LOG_DIR}/xai_spatial.%j.err
#SBATCH -p gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-/home/mla_group_01/rcc-ssrl/src/explainability/spatial/config_xai.yaml}"
WORKDIR="/home/mla_group_01/rcc-ssrl/src/explainability/spatial"

echo "[INFO] Host: $(hostname)"
echo "[INFO] CONFIG_PATH=${CONFIG_PATH}"
echo "[INFO] WORKDIR=${WORKDIR}"

module purge || true

# Attiva eventuale venv (passata via env VENV_PATH, propagata da run_full_xai)
if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

cd "$WORKDIR"
srun python3 xai_generate.py --config "$CONFIG_PATH"
