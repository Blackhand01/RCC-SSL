#!/bin/bash
#SBATCH --job-name=gen_cand_v2
#SBATCH --partition=cpu_sapphire
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --chdir=/home/mla_group_01/rcc-ssrl
#SBATCH --output=/home/mla_group_01/logs/03_webdataset_v2/gen_cand_%j.log
#SBATCH --error=/home/mla_group_01/logs/03_webdataset_v2/gen_cand_%j.err

set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# OpenSlide (C lib) via micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate libslide
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Project Python venv
source /home/mla_group_01/.venvs/rcc/bin/activate

# EXAMPLE: only VAL (you can remove --subset-only for all subsets)
python -u /home/mla_group_01/scripts/03_webdataset_v2/generate_candidates.py \
  --config /home/mla_group_01/scripts/03_webdataset_v2/config.yaml \
  "$@"
