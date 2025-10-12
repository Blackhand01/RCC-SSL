#!/bin/bash
#SBATCH --job-name=ingest_all
#SBATCH --partition=cpu_sapphire
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/home/mla_group_01/rcc-ssrl
#SBATCH --output=/beegfs-scratch/mla_group_01/rcc-ssrl/logs/%x_%j.out
#SBATCH --error=/beegfs-scratch/mla_group_01/rcc-ssrl/logs/%x_%j.err

module purge
module load python/3.10
source $HOME/.venvs/rcc/bin/activate || true

export PYTHONPATH=/home/mla_group_01/rcc-ssrl/src:$PYTHONPATH
export SCRATCH=/beegfs-scratch/mla_group_01/rcc-ssrl

python -m ingest.run_all_ingest
