#!/usr/bin/env bash
#SBATCH -J rcc-train
#SBATCH -A mla_group_01
#SBATCH -p gpu_a40
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/%x.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/%x.%j.err


set -euo pipefail

# 1) Porta la working dir nel punto di submit (SLURM copia gli script nello spool)
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"

# 2) Ambiente software
module purge
module load miniconda3/3.13.25
eval "$(conda shell.bash hook)"
conda activate train || { echo "[ERR] conda env 'train' mancante"; exit 2; }

# 3) Backend headless e logging locale alla repo
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export SKIP_VENV=1

# Fondamentale per far vedere "src/" come package
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${WORKDIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${WORKDIR}"
fi

JOB_LOG_DIR="${WORKDIR}/logs"
mkdir -p "${JOB_LOG_DIR}"

# 4) Sanity log GPU
nvidia-smi -L || true
python - <<'PY'
import torch
print("Torch", torch.__version__, "CUDA", torch.cuda.is_available(), "NGPU", torch.cuda.device_count())
PY

# 5) Config centralizzate
DEFAULT_PROJECT_ROOT="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project"
export PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs/mlruns}"
export WEB_DATASET_ROOT="${WEB_DATASET_ROOT:-$PROJECT_ROOT}"
export EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-${WORKDIR}/configs/exp_01.yaml}"
export RUN_INDEX="${RUN_INDEX:--1}"

# 6) Avvio dal path corretto
cd "${WORKDIR}"
exec python -u launch_training.py
