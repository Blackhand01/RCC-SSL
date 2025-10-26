#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-"${PROJECT_ROOT}/outputs/mlruns"}"

export CONDA_DIR="${CONDA_DIR:-$HOME/micromamba}"                  # micromamba dir
export VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"                # venv python

# -------- Load modules (esempio; adatta a HPC Polito) --------
# module load cuda/12.1 cudnn/9.1 gcc/12.2  # <- se necessario
# module load apptainer                       # <- se usi container

# -------- OpenSlide (separazione C vs Py) --------
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib:/usr/local/lib"

# -------- Python venv attivazione --------
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  pip install -r "$PROJECT_ROOT/requirements.txt"
else
  source "$VENV_DIR/bin/activate"
fi

# -------- WebDataset perf flags --------
export WDS_VERBOSE=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}   # CPU dataload
