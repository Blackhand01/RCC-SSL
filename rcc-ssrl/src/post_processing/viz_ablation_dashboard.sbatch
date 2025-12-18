#!/usr/bin/env bash
#SBATCH -J viz_ablation
#SBATCH -A mla_group_01
#SBATCH -p cpu_sapphire
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/post_process/%x.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/post_process/%x.err

set -euo pipefail

EXPERIMENT_DIR="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project/outputs/mlruns/experiments/ablation_final/exp_20251209_234722_ibot"

python3 /home/mla_group_01/rcc-ssrl/src/post_processing/viz_ablation_dashboard.py \
  --experiment_dir "$EXPERIMENT_DIR" \
  --class_names "ccRCC,pRCC,CHROMO,ONCO,NOT_TUMOR" \
  --primary_metric macro_f1
