#!/usr/bin/env bash
#SBATCH --job-name=xai_pipeline
#SBATCH -p gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/xai/xai_pipeline.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/xai/xai_pipeline.%j.err
## evita --nodelist a meno che non sia obbligatorio

set -euo pipefail

export PROJ=/home/mla_group_01/rcc-ssrl
export PYTHONNOUSERSITE=1

cd "$PROJ"
mkdir -p "$PROJ/src/logs/xai"

# Optional overrides via sbatch --export=...
#   RUN_ID=latest|<tag>
#   EXP_PREFIX=exp_20251109_
#   MODELS_ROOT=/beegfs-scratch/...
#   XAI_ROOT=/path/for/canonical/artifacts
#   WDS_TEST_DIR=/path/to/test
#   PYTHON_BIN=/path/to/python
#   LOG_LEVEL=INFO|DEBUG

bash "$PROJ/src/explainability/run_xai_pipeline.sh"
