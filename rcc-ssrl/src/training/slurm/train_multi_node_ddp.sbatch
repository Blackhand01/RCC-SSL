#!/usr/bin/env bash
#SBATCH -J rcc-train-ddp
#SBATCH -A mla_group_01
#SBATCH -p gpu_a40
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/%x.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/%x.%j.err

# Set EXP_DATETIME for log renaming
export EXP_DATETIME="${EXP_DATETIME:-$(date +%Y%m%d-%H%M%S)}"

set -euo pipefail

# 1) Allinea la working dir allo submit
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
# Ensure WORKDIR is the repo root (handle submission from subdirs)
if [[ "$WORKDIR" == *"/src/training" ]]; then
  WORKDIR="$(dirname "$(dirname "$WORKDIR")")"
fi
cd "$WORKDIR"
mkdir -p "${WORKDIR}/logs"

# 2) Ambiente software
source "${WORKDIR}/src/training/scripts/env/bootstrap_env.sh"

# 3) Backend headless e variabili comuni
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export SKIP_VENV=1

# 4) Sanity log
nvidia-smi -L || true
python - <<'PY'
import torch
print("Torch", torch.__version__, "CUDA", torch.cuda.is_available(), "NGPU", torch.cuda.device_count())
PY

echo "[SLURM] NODES=$SLURM_NNODES GPUS/Node=$SLURM_GPUS_PER_NODE"

# 5) torchrun rendezvous
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}

export EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-${WORKDIR}/src/training/configs/exp_10h.yaml}"
export RUN_INDEX="${RUN_INDEX:--1}"
DEFAULT_PROJECT_ROOT="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project"
export PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs/mlruns}"
export WEB_DATASET_ROOT="${WEB_DATASET_ROOT:-$PROJECT_ROOT}"

# 6) Lancio distribuito
cd "${WORKDIR}"
srun bash -lc '
  torchrun \
    --nnodes='"$SLURM_NNODES"' \
    --nproc_per_node='"$SLURM_GPUS_PER_NODE"' \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
    python -u '"$WORKDIR"'/src/training/launch_training.py
'

# 7) Rename logs with experiment datetime
if [[ -n "${EXP_DATETIME:-}" ]]; then
  LOG_OUT_ORIG="/home/mla_group_01/rcc-ssrl/src/logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"
  LOG_ERR_ORIG="/home/mla_group_01/rcc-ssrl/src/logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err"
  LOG_OUT_NEW="/home/mla_group_01/rcc-ssrl/src/logs/${SLURM_JOB_NAME}.${EXP_DATETIME}.out"
  LOG_ERR_NEW="/home/mla_group_01/rcc-ssrl/src/logs/${SLURM_JOB_NAME}.${EXP_DATETIME}.err"
  mv "$LOG_OUT_ORIG" "$LOG_OUT_NEW" 2>/dev/null || true
  mv "$LOG_ERR_ORIG" "$LOG_ERR_NEW" 2>/dev/null || true
fi
