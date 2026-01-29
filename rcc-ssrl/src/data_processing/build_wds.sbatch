#!/bin/bash
#SBATCH --job-name=build_wds_v2
#SBATCH --partition=cpu_sapphire
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --chdir=/home/mla_group_01/rcc-ssrl
#SBATCH --output=/home/mla_group_01/rcc-ssrl/logs/03_webdataset_v2/build_wds_%j.log
#SBATCH --error=/home/mla_group_01/rcc-ssrl/logs/03_webdataset_v2/build_wds_%j.err

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
source /home/mla_group_01/rcc-ssrl/.venvs/rcc/bin/activate

# Build WDS: remove --limit for full run; add --skip-existing if you want to protect already created shards
python -u /home/mla_group_01/rcc-ssrl/scripts/03_webdataset_v2/build_webdataset_balanced.py \
  --config /home/mla_group_01/rcc-ssrl/scripts/03_webdataset_v2/config.yaml \
  --splits train val test \
  --limit 0 \
  "$@"