#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Usage:
# MODE=ssl RUN_INDEX=0 ./scripts/run_training.sh
export MODE="${MODE:-ssl}"  # ssl|sl (solo per tagging log)
export EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-${TRAINING_ROOT}/configs/experiment_debug.yaml}"
export RUN_INDEX="${RUN_INDEX:--1}"  # -1 => tutti i runs
export PYTHONUNBUFFERED=1
DEFAULT_PROJECT_ROOT="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project"
export PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs/mlruns}"
export WEB_DATASET_ROOT="${WEB_DATASET_ROOT:-$PROJECT_ROOT}"

cd "${TRAINING_ROOT}"

# DDP layout (single-node default)
NGPUS=${NGPUS:-${SLURM_GPUS_PER_NODE:-1}}
torchrun --nproc_per_node="${NGPUS}" launch_training.py
