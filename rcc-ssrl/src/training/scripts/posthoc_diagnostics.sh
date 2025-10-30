#!/usr/bin/env bash
set -euo pipefail

EXP_PATH=/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/exp_20251029-220906

# Use your training venv/conda if available
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate train || true
fi

python -m src.training.reporting.posthoc_diagnostics --exp-path "$EXP_PATH"

#