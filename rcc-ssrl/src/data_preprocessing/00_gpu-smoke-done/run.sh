#!/usr/bin/env bash
set -euo pipefail

export HOME_ROOT="${HOME}/${USER:-user}/project"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/beegfs-scratch/${USER}/project}"

mkdir -p "$SCRATCH_ROOT/logs/gpu-smoke" "$SCRATCH_ROOT/outputs/gpu-smoke"

case "${1:-batch}" in
  batch)
    echo "[RUN] sbatch gpu_smoke.sbatch"
    sbatch gpu_smoke.sbatch
    ;;
  interactive)
    # Debug session on GPU node (close with exit)
    srun -p <gpu_partition> -A <your_account> --gpus=1 --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash -l
    ;;
  *)
    echo "Usage: $0 [batch|interactive]"
    exit 1
    ;;
esac
