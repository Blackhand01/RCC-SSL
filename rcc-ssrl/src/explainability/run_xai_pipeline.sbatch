#!/usr/bin/env bash
#SBATCH --job-name=xai_pipeline
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/xai/xai_pipeline.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/xai/xai_pipeline.%j.err
#SBATCH -p gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --nodelist=compute-6-11

set -euo pipefail

cd /home/mla_group_01/rcc-ssrl
mkdir -p /home/mla_group_01/rcc-ssrl/src/logs/xai
bash /home/mla_group_01/rcc-ssrl/src/explainability/run_xai_pipeline.sh
