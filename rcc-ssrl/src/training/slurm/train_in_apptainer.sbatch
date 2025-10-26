#!/usr/bin/env bash
#SBATCH -J rcc-appt
#SBATCH -A mla_group_01
#SBATCH -p gpu_a40
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/training/logs/%x.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/training/logs/%x.%j.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"
mkdir -p "${WORKDIR}/logs"

module purge
module load apptainer

SIF="${SIF:-/path/to/pytorch-cu121.sif}"

export EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-${WORKDIR}/configs/experiment_debug.yaml}"
export RUN_INDEX="${RUN_INDEX:--1}"
DEFAULT_PROJECT_ROOT="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project"
export PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs/mlruns}"
export WEB_DATASET_ROOT="${WEB_DATASET_ROOT:-$PROJECT_ROOT}"

apptainer exec --nv --bind "$WORKDIR":"$WORKDIR" "$SIF" bash -lc "
  cd $WORKDIR
  export MPLBACKEND=Agg
  export PYTHONUNBUFFERED=1
  export SKIP_VENV=1
  python -u launch_training.py
"
