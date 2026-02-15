#!/usr/bin/env bash
#SBATCH --job-name=no_roi_test
#SBATCH -o /home/mla_group_01/src/logs/xai/no_roi.%j.out
#SBATCH -e /home/mla_group_01/src/logs/xai/no_roi.%j.err
#SBATCH -p gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --exclude=compute-5-14,compute-5-11,compute-3-12

set -euo pipefail

export PROJECT_ROOT="/home/mla_group_01/rcc-ssrl"
export VENV_DIR="/home/mla_group_01/.venvs/xai"

mkdir -p "$PROJECT_ROOT/src/logs/xai" \
         "$PROJECT_ROOT/src/explainability/output/no_roi/logs"

# --- CRITICAL: do not leak ~/.local into Python ---
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Keep imports robust regardless of repo layout
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Hugging Face cache (use HF_HOME; TRANSFORMERS_CACHE is deprecated)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# Activate the *only* environment you should be using
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[FATAL] Venv not found: $VENV_DIR" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# Extra safety: ensure the venv python is actually used
export PYTHON_BIN="$VENV_DIR/bin/python"
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] VENV_DIR=$VENV_DIR"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -c 'import sys; print("[INFO] sys.executable=", sys.executable)'

# Delegate to the deterministic runner
bash "$PROJECT_ROOT/src/explainability/concept/run_no_roi_test.sh"
