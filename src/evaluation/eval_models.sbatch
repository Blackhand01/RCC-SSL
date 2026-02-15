#!/usr/bin/env bash
#SBATCH -J eval
#SBATCH -A mla_group_01
#SBATCH -p gpu_a40
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o /home/mla_group_01/src/logs/eval/eval.%j.out
#SBATCH -e /home/mla_group_01/src/logs/eval/eval.%j.err
#SBATCH --exclude=compute-5-14,compute-5-11,compute-3-12

set -euo pipefail

WORKDIR="/home/mla_group_01/src/evaluation"
VENV="/home/mla_group_01/.venvs/eval"
PYTHON="$VENV/bin/python"

mkdir -p /home/mla_group_01/src/logs/eval
cd "$WORKDIR"

# venv bootstrap (idempotente)
[[ -f "$VENV/bin/activate" ]] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip -q install --upgrade pip
pip -q install -r /home/mla_group_01/src/evaluation/requirements_eval.txt

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# === Required config: must come from auto_eval (CFG_PATH) ===
CFG_PATH="${CFG_PATH:-}"

if [[ -z "$CFG_PATH" ]]; then
  echo "[ERROR] CFG_PATH not set." >&2
  echo "[HINT] Run via auto_eval.py --submit or export CFG_PATH manually." >&2
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
