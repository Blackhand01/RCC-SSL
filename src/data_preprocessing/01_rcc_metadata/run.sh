#!/usr/bin/env bash
set -euo pipefail
HOME_ROOT="${HOME_ROOT:-/home/mla_group_01/rcc-ssrl}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/beegfs-scratch/mla_group_01/workspace/mla_group_01/wsi-ssrl-rcc_project}"
RAW_DATA_DIR="${RAW_DATA_DIR:-$SCRATCH_ROOT/data/raw}"
REPORT_DIR="${REPORT_DIR:-$HOME_ROOT/reports/0_phase}"
METADATA_CSV="${METADATA_CSV:-$HOME_ROOT/configs/metadata.csv}"

VENV_DIR="${VENV_DIR:-$HOME_ROOT/.venvs/rcc}"
VENV_ACT="${VENV_ACT:-$VENV_DIR/bin/activate}"

mkdir -p "$REPORT_DIR"
module purge >/dev/null 2>&1 || true
module load python/3.10 >/dev/null 2>&1 || true

if [[ ! -f "$VENV_ACT" ]]; then
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_ACT"
  python -m pip install -U pip wheel setuptools
  python -m pip install -r "$(dirname "$0")/requirements.txt"
else
  # shellcheck disable=SC1090
  source "$VENV_ACT"
fi

echo "[INFO] RAW_DATA_DIR=$RAW_DATA_DIR"
echo "[INFO] REPORT_DIR=$REPORT_DIR"
echo "[INFO] METADATA_CSV=$METADATA_CSV"
python "$(dirname "$0")/rcc_metadata_enrich.py" \
  --raw-dir "$RAW_DATA_DIR" \
  --report-dir "$REPORT_DIR" \
  --metadata-csv "$METADATA_CSV"
