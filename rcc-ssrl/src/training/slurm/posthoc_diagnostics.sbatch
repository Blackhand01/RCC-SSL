#!/usr/bin/env bash
#SBATCH -J rcc-posthoc
#SBATCH -A mla_group_01
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

set -euo pipefail

EXP_PATH="${EXP_PATH:?Set EXP_PATH to the experiment directory}"

# Reuse the training environment if present
module purge || true
if command -v conda >/dev/null 2>&1; then
  module load miniconda3/3.13.25 || true
  eval "$(conda shell.bash hook)"
  conda activate train || true
fi

python -m src.training.reporting.posthoc_diagnostics --exp-path "$EXP_PATH"
