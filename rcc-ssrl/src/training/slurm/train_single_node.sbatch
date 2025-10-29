#!/usr/bin/env bash
#SBATCH -J rcc-train
#SBATCH -A mla_group_01
#SBATCH -p gpu_a40
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/%x.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/%x.%j.err


set -euo pipefail

# 1) Porta la working dir nel punto di submit (SLURM copia gli script nello spool)
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"

# 2) Ambiente software
module purge
module load miniconda3/3.13.25
eval "$(conda shell.bash hook)"
conda activate train || { echo "[ERR] conda env 'train' mancante"; exit 2; }

# 3) Backend headless e logging locale alla repo
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export SKIP_VENV=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Fondamentale per far vedere "src/" come package
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${WORKDIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${WORKDIR}"
fi

JOB_LOG_DIR="${WORKDIR}/logs"
mkdir -p "${JOB_LOG_DIR}"

# 4) Sanity log GPU
nvidia-smi -L || true
python - <<'PY'
import torch
print("Torch", torch.__version__, "CUDA", torch.cuda.is_available(), "NGPU", torch.cuda.device_count())
PY

# 5) Config centralizzate
DEFAULT_PROJECT_ROOT="/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project"
export PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-${PROJECT_ROOT}/outputs/mlruns}"
# ---- Staging su disco locale di nodo ($SLURM_TMPDIR) ----
DATA_ROOT_ON_NET="${WEB_DATASET_ROOT:-$PROJECT_ROOT}/data/processed/rcc_webdataset_v2"
DATA_ROOT_ON_LOCAL="${SLURM_TMPDIR:-/tmp}/data/processed/rcc_webdataset_v2"
mkdir -p "$(dirname "$DATA_ROOT_ON_LOCAL")"
rsync -a --info=progress2 "$DATA_ROOT_ON_NET"/ "$DATA_ROOT_ON_LOCAL"/
# Punta la pipeline allo scratch locale (stesso layout relativi)
export WEB_DATASET_ROOT="${SLURM_TMPDIR:-/tmp}"
export EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-${WORKDIR}/configs/exp_10h.yaml}"
export RUN_INDEX="${RUN_INDEX:--1}"

# 6) Avvio dal path corretto
cd "${WORKDIR}"
exec python -u launch_training.py
