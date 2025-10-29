#!/usr/bin/env bash
set -euo pipefail

# --- Python env bootstrap (idempotent) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
REQUIREMENTS_FILE="${TRAINING_DIR}/requirements.txt"
# 1) Respect SKIP_VENV if provided
if [[ "${SKIP_VENV:-0}" != "1" ]]; then
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ ! -d ".venv" ]]; then
      python3 -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install --upgrade pip
  fi
fi

# 2) Dependencies (pin only critical ones if needed)
if [[ "${INSTALL_DEPS:-1}" == "1" ]]; then
  pip install -r "$REQUIREMENTS_FILE"
fi

# 3) CUDA/cuDNN env hints (customize if your cluster needs it)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

echo "[bootstrap_env] ready: venv=${VIRTUAL_ENV:-none}, deps=${INSTALL_DEPS:-1}"
