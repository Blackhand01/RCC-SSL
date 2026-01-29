#!/bin/bash
#SBATCH --job-name=roi_masks_v3
#SBATCH --partition=cpu_sapphire
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --chdir=/home/mla_group_01/rcc-ssrl
#SBATCH --output=/home/mla_group_01/rcc-ssrl/logs/03_webdataset_v3/roi_masks_%j.out
#SBATCH --error=/home/mla_group_01/rcc-ssrl/logs/03_webdataset_v3/roi_masks_%j.err

set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# OpenSlide (C lib) with micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate libslide
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# venv Python
source /home/mla_group_01/rcc-ssrl/.venvs/rcc/bin/activate

python -u /home/mla_group_01/rcc-ssrl/scripts/03_webdataset_v3/roi_index_and_masks_v3.py \
  --config /home/mla_group_01/rcc-ssrl/scripts/03_webdataset_v3/config_v3.yaml \
  --resume "$@"
