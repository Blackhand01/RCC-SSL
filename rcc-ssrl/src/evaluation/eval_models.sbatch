#!/usr/bin/env bash
#SBATCH -J rcc-eval
#SBATCH -A mla_group_01
#SBATCH -p gpu_a40
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o /home/mla_group_01/rcc-ssrl/src/logs/eval/eval.%j.out
#SBATCH -e /home/mla_group_01/rcc-ssrl/src/logs/eval/eval.%j.err
#SBATCH --exclude=compute-5-14,compute-5-11,compute-3-12

set -euo pipefail

WORKDIR="/home/mla_group_01/rcc-ssrl/src/evaluation"
VENV="/home/mla_group_01/rcc-ssrl/.venvs/rcc-eval"
PYTHON="$VENV/bin/python"

mkdir -p /home/mla_group_01/rcc-ssrl/src/logs/eval
cd "$WORKDIR"

# venv bootstrap (idempotente)
[[ -f "$VENV/bin/activate" ]] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip -q install --upgrade pip
pip -q install -r /home/mla_group_01/rcc-ssrl/src/evaluation/requirements_eval.txt

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# === Config obbligatoria: deve arrivare da auto_eval (CFG_PATH) ===
CFG_PATH="${CFG_PATH:-}"

if [[ -z "$CFG_PATH" ]]; then
  echo "[ERROR] CFG_PATH non impostato." >&2
  echo "[HINT] Lancia tramite auto_eval.py --submit oppure esporta CFG_PATH a mano." >&2
  exit 1
fi

CFG_USE="$CFG_PATH"

ONLY_ONE_SHARD="${ONLY_ONE_SHARD:-0}"
SHARD_EXAMPLE="${SHARD_EXAMPLE:-shard-000000.tar}"

TMP_CFG=""
cleanup() { [[ -n "${TMP_CFG}" && -f "${TMP_CFG}" ]] && rm -f "${TMP_CFG}"; }
trap cleanup EXIT

echo "[INFO] Host=$(hostname)  CFG_USE=${CFG_USE}  SMOKE=${ONLY_ONE_SHARD:-0}"

exec "$PYTHON" eval.py --config "$CFG_USE"
