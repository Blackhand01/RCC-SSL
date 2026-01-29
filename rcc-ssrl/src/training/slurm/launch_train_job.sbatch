#!/usr/bin/env bash
#SBATCH --job-name=rcc-train
#SBATCH --account=mla_group_01
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1                    # 3 max that you can use now
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/train/%x.%j.out

#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/train/%x.%j.err

set -euo pipefail

# Standardized roots for HPC
export HOME_ROOT="${HOME_ROOT:-/home/mla_group_01}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/beegfs-scratch/mla_group_01/workspace/mla_group_01}"
export PROJECT_ROOT="${PROJECT_ROOT:-${HOME_ROOT}/rcc-ssrl}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs/mlruns}"
export RCC_DATASET_ROOT="${RCC_DATASET_ROOT:-${SCRATCH_ROOT}/wsi-ssrl-rcc_project/data/processed}"
export RCC_WDS_ROOT="${RCC_WDS_ROOT:-${RCC_DATASET_ROOT}/rcc_webdataset_final}"
export WDS_ROOT="${WDS_ROOT:-${RCC_WDS_ROOT}}"
export WEB_DATASET_ROOT="${WEB_DATASET_ROOT:-${RCC_WDS_ROOT}}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# 1) Workdir
WORKDIR="${SLURM_SUBMIT_DIR:-$PROJECT_ROOT}"
if [[ "$WORKDIR" == *"/src/training" ]]; then
  WORKDIR="$(dirname "$(dirname "$WORKDIR")")"
fi
cd "$WORKDIR"

# 2) Env
source "${WORKDIR}/src/training/scripts/bootstrap_env.sh"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="${WORKDIR}${PYTHONPATH:+:$PYTHONPATH}"

# NCCL (single-node, A40)
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker0

# 3) Stage dataset to local scratch (if available)
STAGING_DIR=""
if [[ -n "${SLURM_TMPDIR:-}" && -d "$SLURM_TMPDIR" ]]; then
  STAGING_DIR="$SLURM_TMPDIR"
elif [[ -n "${TMPDIR:-}" && -d "$TMPDIR" ]]; then
  STAGING_DIR="$TMPDIR"
elif [[ -n "${SCRATCH:-}" && -d "${SCRATCH:-}" ]]; then
  STAGING_DIR="${SCRATCH:-}/$USER"
fi


# 4) Sanity log
nvidia-smi -L || true
python - <<'PY'
import torch
print("Torch", torch.__version__, "CUDA", torch.cuda.is_available(), "NGPU", torch.cuda.device_count())
PY

# Dataset visibility check on the allocated node
srun -N1 -n1 bash -lc 'hostname; ls -ld ${RCC_DATASET_ROOT}/rcc_webdataset_final/train' || {
  echo "[sbatch] Dataset path not visible on this node" >&2
  exit 4
}

# Fail-fast preflight (dataset presence + shards)
PYTHONPATH="${PYTHONPATH}" python -m src.training.utils.preflight || {
  echo "[sbatch] Preflight failed" >&2
  exit 5
}

# 5) Project paths
# Change this variable if you want a different config
export EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-${WORKDIR}/src/training/configs/exp_debug_pipeline.yaml}"
export RUN_INDEX="${RUN_INDEX:--1}"

# 6) DDP: launch 1 process per GPU (3 proc)
NPROC="${SLURM_GPUS_ON_NODE:-3}"

# 7) Run with torchrun (standalone on single node)
exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  src/training/launch_training.py
