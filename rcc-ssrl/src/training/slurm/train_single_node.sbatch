#!/usr/bin/env bash
#SBATCH --job-name=rcc-train
#SBATCH --account=mla_group_01
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1                    # 3 max che puoi usare ora
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/%x.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/%x.%j.err
set -euo pipefail

# 1) Workdir
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
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

# 3) Staging dataset su scratch locale (se disponibile)
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

# 5) Project paths
DEFAULT_PROJECT_ROOT="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project"
export PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs/mlruns}"
# Cambia questa variabile se vuoi un'altra config
export EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-${WORKDIR}/src/training/configs/exp_debug_moco_v3_vit.yaml}"
export RUN_INDEX="${RUN_INDEX:-0}"

# 6) DDP: lancia 1 processo per GPU (3 proc)
NPROC="${SLURM_GPUS_ON_NODE:-3}"

# 7) Run con torchrun (standalone su singolo nodo)
exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  src/training/launch_training.py