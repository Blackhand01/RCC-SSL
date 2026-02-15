#!/bin/bash
#SBATCH --job-name=balance_v2
#SBATCH --partition=cpu_sapphire
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/mla_group_01/rcc-ssrl
#SBATCH --output=/home/mla_group_01/logs/03_webdataset_v2/balance_%j.log
#SBATCH --error=/home/mla_group_01/logs/03_webdataset_v2/balance_%j.err

set -euo pipefail

# Activate environments
eval "$(micromamba shell hook -s bash)"
micromamba activate libslide
source /home/mla_group_01/.venvs/rcc/bin/activate

python -u /home/mla_group_01/scripts/03_webdataset_v2/balance_and_select.py \
  --config /home/mla_group_01/scripts/03_webdataset_v2/config.yaml
