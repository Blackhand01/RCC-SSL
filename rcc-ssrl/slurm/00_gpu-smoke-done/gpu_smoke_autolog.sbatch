#!/usr/bin/env bash
#SBATCH -J gpu-smoke
#SBATCH -p gpu_a40                  # verrà sovrascritto dal wrapper con --partition
#SBATCH --gpus=1                    # oppure: --gres=gpu:1 (vedi wrapper)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
# #SBATCH -A mla_group_01           # decommenta se richiesto

set -euo pipefail

echo "==[META]===================================================="
echo "JOBID=${SLURM_JOB_ID}  USER=${SLURM_JOB_USER}  NAME=${SLURM_JOB_NAME}"
echo "PARTITION=${SLURM_JOB_PARTITION}  NODELIST=${SLURM_JOB_NODELIST:-unknown}"
echo "SUBMIT_DIR=${SLURM_SUBMIT_DIR}    WORK_DIR=$(pwd)"
echo "START_AT=$(date -Is)"
echo "============================================================="

echo "==[SLURM JOB DESCR]========================================="
scontrol show job "${SLURM_JOB_ID}" | egrep -i "JobId=|UserId=|JobState=|Partition=|Nodes=|NodeList=|TRES=|Gres=|GresUsed=|Mem="
echo "============================================================="

# Ambiente modulo (adatta alle versioni del cluster)
module purge
module load miniconda3/3.13.25

# Attiva conda in modo portabile
eval "$(conda shell.bash hook)"

# Fail-fast se l'env non esiste (NON creare env sul nodo GPU!)
if ! conda env list | awk '{print $1}' | grep -qx train; then
  echo "[ERR] Conda env 'train' non trovato. Esegui il setup sul login node (vedi istruzioni)."
  exit 2
fi
conda activate train

echo "==[CUDA/DRIVER/GPU]========================================="
echo "[ENV] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
command -v nvidia-smi >/dev/null 2>&1 && {
  nvidia-smi -L || true
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
} || echo "nvidia-smi non disponibile"
python3 -c "import torch; print('PyTorch', torch.__version__, 'CUDA available=', torch.cuda.is_available(), 'GPU count=', torch.cuda.device_count())"
echo "============================================================="

# --- esecuzione test ---
python3 -u gpu_smoke.py

echo "==[POST] scontrol (final)==================================="
scontrol show job "${SLURM_JOB_ID}" | egrep -i "JobState=|Reason="
echo "END_AT=$(date -Is)"
echo "============================================================="
