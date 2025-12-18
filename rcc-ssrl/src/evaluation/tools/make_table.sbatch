#!/usr/bin/env bash
#SBATCH -J rcc-main-table
#SBATCH -A mla_group_01
#SBATCH -p cpu_sapphire
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/eval/%x.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/eval/%x.%j.err

set -euo pipefail

REPO="/home/mla_group_01/rcc-ssrl"
SCRIPT="$REPO/src/evaluation/tools/build_main_results_table.py"

python3 "$SCRIPT" \
  --experiments-root "/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/ablation_final" \
  --out-dir "$REPO/src/evaluation/results" \
  --select-best-per-model
