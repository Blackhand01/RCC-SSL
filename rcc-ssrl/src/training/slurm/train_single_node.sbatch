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

# Set EXP_DATETIME for log renaming
export EXP_DATETIME="${EXP_DATETIME:-$(date +%Y%m%d-%H%M%S)}"


set -euo pipefail

# 1) Porta la working dir nel punto di submit (SLURM copia gli script nello spool)
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
# Ensure WORKDIR is the repo root (handle submission from subdirs)
if [[ "$WORKDIR" == *"/src/training" ]]; then
  WORKDIR="$(dirname "$(dirname "$WORKDIR")")"
fi
cd "$WORKDIR"

# 2) Ambiente software
source "${WORKDIR}/src/training/scripts/bootstrap_env.sh"

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
export EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-${WORKDIR}/src/training/configs/exp_24h.yaml}"
export RUN_INDEX="${RUN_INDEX:--1}"

# 6) Avvio dal path corretto
cd "${WORKDIR}"
exec python -u src/training/launch_training.py

# 7) Rename logs with experiment datetime
if [[ -n "${EXP_DATETIME:-}" ]]; then
  LOG_OUT_ORIG="/home/mla_group_01/rcc-ssrl/src/logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"
  LOG_ERR_ORIG="/home/mla_group_01/rcc-ssrl/src/logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err"
  LOG_OUT_NEW="/home/mla_group_01/rcc-ssrl/src/logs/${SLURM_JOB_NAME}.${EXP_DATETIME}.out"
  LOG_ERR_NEW="/home/mla_group_01/rcc-ssrl/src/logs/${SLURM_JOB_NAME}.${EXP_DATETIME}.err"
  mv "$LOG_OUT_ORIG" "$LOG_OUT_NEW" 2>/dev/null || true
  mv "$LOG_ERR_ORIG" "$LOG_ERR_NEW" 2>/dev/null || true
fi
