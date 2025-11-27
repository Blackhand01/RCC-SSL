#!/usr/bin/env bash
#SBATCH -J gpu-smoke
#SBATCH -A <your_account>
#SBATCH -p <gpu_partition>
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

set -euo pipefail

# (opzione 1) Ambiente nativo via module:
module avail >/dev/null 2>&1 || true    # elenco moduli (guida 'module')
module load cuda/12.2 || true
module load anaconda/2024 || true
source activate pytorch || true  # oppure your venv

# (opzione 2) Container Apptainer (commenta la 1 e scommenta questa):
# module load apptainer || true
# export APPTAINER_BIND="$PWD:$PWD"
# export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
# apptainer exec --nv <your_pytorch_gpu.sif> python3 gpu_smoke.py  # --nv abilita GPU
# exit 0

echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python3 -u gpu_smoke.py
