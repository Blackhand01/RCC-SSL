#!/bin/bash
#SBATCH --job-name=<JOBNAME>
#SBATCH --partition=gpu_a100            # o cpu_sapphire
#SBATCH --gres=gpu:1                    # rimuovi se CPU
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --chdir=/home/mla_group_01/rcc-ssrl
#SBATCH --output=/beegfs-scratch/mla_group_01/rcc-ssrl/logs/%x_%j.out
#SBATCH --error=/beegfs-scratch/mla_group_01/rcc-ssrl/logs/%x_%j.err

export SCRATCH="/beegfs-scratch/mla_group_01/rcc-ssrl"
export PYTHONPATH=$HOME/rcc-ssrl/src:$PYTHONPATH

module purge
module load python/3.10 cuda/12.1

source $HOME/.venvs/rcc/bin/activate  # o container Apptainer

echo "[INFO] Host: $(hostname)  Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "[INFO] Start: $(date -Is)"

srun python src/<script>.py  # es. ingest_governance.py

echo "[INFO] End: $(date -Is)"
